"""Provider receipt capture during the original native tool invocation.

The live callback retains provider facts until ordinary settlement.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar

from .jobs import JobError
from .pricing import BillingReceipt, PricingError

_CURRENT = ContextVar('power_loop_billing_receipt', default=None)


def tool_billing_managed(name: str) -> bool:
    """Whether this exact callback's fee is owned by the native budget path."""
    slot = _CURRENT.get()
    attempt = slot.budget_attempt if slot is not None else None
    if attempt is None or attempt.call.name != name:
        return False
    from .control_runtime import current
    return current(attempt.binding).get("attempt", attempt.attempt_id)["budget_required"]


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
    """Retain provider facts for settlement before subsequent decoding can fail.

    Await from the original async tool callback. Settlement reports the charge
    after the callback completes or raises.
    """
    from .task_runtime import _SEND
    slot = _CURRENT.get()
    if slot is None:
        raise JobError('tool_billing_checkpoint_unavailable')
    from .background_execution import BackgroundReceipt
    if isinstance(slot, BackgroundReceipt):
        slot.record(receipt)
        return
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


async def persist_receipt(runtime, slot):
    if slot.nested_parent is not None:
        from .nested_tools import persist_receipt as persist_nested_receipt
        return await persist_nested_receipt(slot)
    from .execution_tools import persist_billing_receipt
    return await persist_billing_receipt(runtime, slot)
