"""Actual native callback returns, separate from work completion or a tool row.

Only the original pipeline can reserve a receipt before invoking its frozen
handler. A returned value is data, not proof that an asynchronous child stopped.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field

from .jobs import JobError, _hash, _json
from .model_exit import content_reference

_LIMIT = 16 * 1024 * 1024


class NativeReturnPending(JobError):
    """Do not replace an actual returned value with a synthetic tool error."""

    def __init__(self):
        super().__init__('native_return_checkpoint_pending')


def identity(manager, source, ref, call_id):
    return 'native_return_' + _hash([manager.store.store_id, source['job_id'], source['generation'], ref.to_dict(), call_id])[:48]


def anchor_id(key):
    return 'native_return_anchor_' + _hash(key)[:42]


@dataclass(eq=False)
class _Receipt:
    binding: object
    ref: object
    call_id: str
    plan_hash: str
    entry_hash: str
    task: object
    text: str | None = None
    failed: bool = False
    saved: bool = False


@dataclass
class ReturnCache:
    runtime: object = field(repr=False)

    def __post_init__(self):
        from .root_context import _authority
        self.authority = _authority(self.runtime)
        self.slots, self.lock = {}, asyncio.Lock()

    def check(self):
        from .root_context import _authority
        actual = _authority(self.runtime)
        if (any(a is not b for a, b in zip(actual[:3], self.authority[:3], strict=True))
                or actual[3:] != self.authority[3:]):
            raise JobError('native_return_runtime_changed')

    def key(self, slot):
        return identity(self.runtime.manager, asdict(slot.binding.lease), slot.ref, slot.call_id)

    def require(self, slot):
        self.check()
        if type(slot) is not _Receipt or self.slots.get(self.key(slot)) is not slot:
            raise JobError('native_return_original_required')

    def reserve(self, binding, ref, plan, entry):
        from .task_runtime import _SEND
        self.check()
        if _SEND.get() is not binding or binding.runtime is not self.runtime or entry['recovery'] is not None:
            raise JobError('native_return_original_required')
        slot = _Receipt(binding, ref, entry['call_id'], _hash(plan), _hash(entry), asyncio.current_task())
        key = self.key(slot)
        if key in self.slots or len(self.slots) >= 1000:
            raise JobError('native_return_requires_reconciliation')
        self.slots[key] = slot
        return slot

    def capture(self, slot, text):
        from .task_runtime import _SEND
        self.require(slot)
        if (_SEND.get() is not slot.binding or asyncio.current_task() is not slot.task
                or type(text) is not str or slot.text is not None):
            raise JobError('native_return_original_required')
        # No await between original callback return/serialization and retention.
        slot.text = text

    def release_unreturned(self, slot):
        if slot.text is None:
            self.slots.pop(self.key(slot), None)

    def pending_count(self):
        return sum(s.text is not None for s in self.slots.values())

    def failed_count(self):
        return sum(s.failed for s in self.slots.values())


async def _basis(manager, tx, source, ref, entry, plan):
    from .tool_call_plans import _read
    from .tool_call_plans import identity as plan_id
    key = plan_id(manager, source, ref) + '_start_' + _hash(entry['call_id'])[:24]
    started = await _read(manager, tx, key, source, 'tool_call_dispatch_started')
    if started != {'version': 1, 'scope': plan['scope'], 'plan_id': plan_id(manager, source, ref),
                   'plan_hash': _hash(plan), 'entry_hash': _hash(entry)}:
        raise JobError('native_return_dispatch_changed')
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key,))
    # Indexed caller/type scope, then exact raw-call comparison. No subquery.
    operations = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE caller_session_id=? AND tool_call_id=? "
        "AND event_type='operation' ORDER BY event_id LIMIT 1001", (ref.session_id, entry['call_id']))
    if len(operations) > 1000:
        raise JobError('native_return_operations_too_large')
    refs = []
    from .jobs import JobActor
    actor = JobActor(source['tenant_id'], source['principal_id'], source['session_id'])
    for operation in operations:
        if json.loads(operation['call_ref_json'] or 'null') != ref.to_dict():
            continue
        await manager._authorize(tx, actor, operation['job_id'])
        refs.append(content_reference(operation))
    return content_reference(event), refs


async def read(manager, tx, source, ref, entry, plan, *, optional=False):
    from .pending_calls import _object
    from .tool_call_plans import _read
    key = identity(manager, source, ref, entry['call_id'])
    data = await _object(manager, tx, manager.events, 'payload_json', 'event_id=?', (key,), _LIMIT)
    if data is None:
        if optional:
            return None
        raise JobError('pending_native_return_required')
    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (key,))
    anchor = await _read(manager, tx, anchor_id(key), source, 'native_callback_return_anchor')
    start, operations = await _basis(manager, tx, source, ref, entry, plan)
    if (set(data) != {'version', 'scope', 'call_ref', 'call_id', 'plan_hash', 'entry_hash', 'started', 'operations', 'return_text'}
            or type(data['version']) is not int or data['version'] != 1
            or data['scope'] != plan['scope'] or data['call_ref'] != ref.to_dict()
            or data['call_id'] != entry['call_id'] or data['plan_hash'] != _hash(plan)
            or data['entry_hash'] != _hash(entry) or entry['recovery'] is not None
            or data['started'] != start or data['operations'] != operations or type(data['return_text']) is not str
            or event['event_type'] != 'native_callback_returned' or event['job_id'] != source['job_id']
            or event['delivery_status'] != 'not_applicable' or event['request_hash'] != _hash(data)):
        raise JobError('native_return_checkpoint_changed')
    if anchor != {'version': 1, 'return': content_reference(event)}:
        raise JobError('native_return_anchor_changed')
    return event, data


async def persist(runtime, slot):
    from .tool_call_plans import read as read_plan
    cache, manager = runtime._native_returns, runtime.manager
    try:
        async with cache.lock:
            cache.check()
            if slot.saved and slot.binding.runtime is runtime:
                return
            cache.require(slot)
            if slot.text is None:
                raise JobError('native_return_unconfirmed')
            async with manager._write() as tx:
                source = await manager._authorize(tx, slot.binding.actor, slot.binding.lease.job_id)
                manager._check_proof(source, slot.binding.lease)
                plan = await read_plan(manager, tx, source, slot.ref)
                entries = [e for e in plan['entries'] if e['call_id'] == slot.call_id]
                if len(entries) != 1 or _hash(plan) != slot.plan_hash or _hash(entries[0]) != slot.entry_hash:
                    raise JobError('native_return_original_changed')
                start, operations = await _basis(manager, tx, source, slot.ref, entries[0], plan)
                value = {'version': 1, 'scope': plan['scope'], 'call_ref': slot.ref.to_dict(), 'call_id': slot.call_id,
                         'plan_hash': slot.plan_hash, 'entry_hash': slot.entry_hash, 'started': start,
                         'operations': operations, 'return_text': slot.text}
                document = _json(value)
                if len(document.encode()) > _LIMIT:
                    raise JobError('native_return_too_large')
                old = await read(manager, tx, source, slot.ref, entries[0], plan, optional=True)
                if old is not None:
                    if _hash(old[1]) != _hash(value):
                        raise JobError('native_return_checkpoint_changed')
                else:
                    await tx.execute(f'INSERT INTO {manager.events} '
                        '(event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)',
                        (cache.key(slot), source['job_id'], 'native_callback_returned', source['version'],
                         _hash(value), document, 'not_applicable', manager.now()))
                    event = await tx.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (cache.key(slot),))
                    from .tool_call_plans import _event
                    await _event(manager, tx, anchor_id(cache.key(slot)), source, 'native_callback_return_anchor',
                                 {'version': 1, 'return': content_reference(event)})
            del cache.slots[cache.key(slot)]
            slot.saved = True
    except BaseException:
        slot.failed = True
        raise


async def invoke(pipeline, registered, arguments, slot):
    """Called by the pipeline after original dispatch, before formatting/hooks."""
    cache = slot.binding.runtime._native_returns if slot is not None else None
    try:
        from .tool_contracts import validate_registration
        validate_registration(pipeline.tool_registry, registered.definition.name, registered)
        result = await pipeline.tool_registry.invoke_async(registered.definition.name, arguments)
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        if slot is not None:
            cache.capture(slot, text)
            try:
                await persist(slot.binding.runtime, slot)
            except Exception as exc:
                raise NativeReturnPending() from exc
        return text
    finally:
        if slot is not None:
            cache.release_unreturned(slot)


async def flush_local(runtime, *, failed_only=False):
    cache = runtime._native_returns
    cache.check()
    for slot in list(cache.slots.values()):
        if slot.text is not None and (not failed_only or slot.failed):
            await persist(runtime, slot)
