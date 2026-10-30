"""Provider receipt capture during the original native tool invocation.

This journal records billing facts, not successful tool completion or settlement.
The host can reconcile a paid failure without repeating the external request.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar

from .jobs import JobError, _hash, _json
from .model_exit import content_reference
from .pricing import BillingReceipt, PricingError

_CURRENT = ContextVar('power_loop_billing_receipt', default=None)


def tool_billing_managed(name: str) -> bool:
    """Whether this exact callback's fee is owned by the native budget path."""
    slot = _CURRENT.get()
    attempt = slot.budget_attempt if slot is not None else None
    return attempt is not None and attempt.call.name == name


def tool_billing_active() -> bool:
    """Whether this send requires native billing, including internal calls."""
    from .task_runtime import _SEND
    binding = _SEND.get()
    return binding is not None and binding.runtime.budget_config is not None


@contextmanager
def receipt_context(slot):
    token = _CURRENT.set(slot)
    try:
        yield
    finally:
        _CURRENT.reset(token)


async def record_billing_receipt(receipt: BillingReceipt) -> None:
    """Save provider facts before subsequent decoding/business logic can fail.

    Await from the original async tool callback. A successful return means the
    receipt is durable; it does not mean that the charge has been settled.
    """
    from .task_runtime import _SEND
    slot = _CURRENT.get()
    if slot is None:
        raise JobError('tool_billing_checkpoint_unavailable')
    cache = slot.binding.runtime._native_returns
    cache.require(slot)
    if _SEND.get() is not slot.binding or asyncio.current_task() is not slot.task or slot.finished:
        raise JobError('tool_billing_original_required')
    if not isinstance(receipt, BillingReceipt):
        raise PricingError('billing_receipt_required')
    if slot.billing_receipt is not None and slot.billing_receipt != receipt:
        raise PricingError('tool_billing_receipt_changed')
    slot.billing_receipt = receipt  # Retain before the first cancellable await.
    await persist_receipt(slot.binding.runtime, slot)


def identity(return_key):
    return 'tool_billing_' + _hash(return_key)[:48]


async def read_receipt(manager, tx, source, ref, entry, plan, *, optional=False):
    from .native_returns import _LIMIT, _basis
    from .native_returns import identity as return_identity
    from .pending_calls import _object
    from .tool_call_plans import _read
    key = identity(return_identity(manager, source, ref, entry['call_id']))
    data = await _object(manager, tx, manager.events, 'payload_json', 'event_id=?', (key,), _LIMIT)
    anchor = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key + '_a',))
    if data is None:
        if optional and anchor is None:
            return None
        raise JobError('tool_billing_receipt_missing')
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key,))
    start, _ = await _basis(manager, tx, source, ref, entry, plan)
    expected = dict(version=1, scope=plan['scope'], call_ref=ref.to_dict(), call_id=entry['call_id'],
                    plan_hash=_hash(plan), entry_hash=_hash(entry), started=start,
                    billing_receipt=data.get('billing_receipt'))
    if (data != expected or type(data.get('version')) is not int or entry['recovery'] is not None
            or event['job_id'] != source['job_id'] or event['event_type'] != 'tool_billing_receipt'
            or event['delivery_status'] != 'not_applicable' or event['request_hash'] != _hash(data)):
        raise JobError('tool_billing_receipt_changed')
    try:
        BillingReceipt.from_dict(data['billing_receipt'])
    except PricingError as exc:
        raise JobError('tool_billing_receipt_invalid') from exc
    saved_anchor = await _read(manager, tx, key + '_a', source, 'tool_billing_receipt_anchor')
    if saved_anchor != {'version': 1, 'receipt': content_reference(event)}:
        raise JobError('tool_billing_receipt_anchor_changed')
    return event, data


async def persist_receipt(runtime, slot):
    if slot.nested_parent is not None:
        from .nested_tools import persist_receipt as persist_nested_receipt
        return await persist_nested_receipt(slot)
    from .native_returns import _LIMIT, _basis
    from .tool_call_plans import _event
    from .tool_call_plans import read as read_plan
    cache, manager = runtime._native_returns, runtime.manager
    try:
        async with cache.lock:
            cache.check()
            if slot.billing_saved and slot.binding.runtime is runtime:
                return
            cache.require(slot)
            if slot.billing_receipt is None:
                raise JobError('tool_billing_receipt_missing')
            async with manager._write() as tx:
                source = await manager._authorize(tx, slot.binding.actor, slot.binding.lease.job_id)
                manager._check_proof(source, slot.binding.lease)
                plan = await read_plan(manager, tx, source, slot.ref)
                entries = [e for e in plan['entries'] if e['call_id'] == slot.call_id]
                if len(entries) != 1 or _hash(plan) != slot.plan_hash or _hash(entries[0]) != slot.entry_hash:
                    raise JobError('tool_billing_original_changed')
                start, _ = await _basis(manager, tx, source, slot.ref, entries[0], plan)
                value = dict(version=1, scope=plan['scope'], call_ref=slot.ref.to_dict(), call_id=slot.call_id,
                             plan_hash=slot.plan_hash, entry_hash=slot.entry_hash, started=start,
                             billing_receipt=slot.billing_receipt.to_dict())
                if len(_json(value).encode()) > _LIMIT:
                    raise JobError('tool_billing_receipt_too_large')
                old = await read_receipt(manager, tx, source, slot.ref, entries[0], plan, optional=True)
                if old is not None:
                    if old[1] != value:
                        raise JobError('tool_billing_receipt_changed')
                else:
                    key = identity(cache.key(slot))
                    await _event(manager, tx, key, source, 'tool_billing_receipt', value)
                    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key,))
                    await _event(manager, tx, key + '_a', source, 'tool_billing_receipt_anchor',
                                 {'version': 1, 'receipt': content_reference(event)})
            slot.billing_saved = True
    except BaseException:
        slot.failed = True
        raise
