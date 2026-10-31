"""Internal priced calls reuse the original task's budget and durable delivery.

Each supplier invocation has its own prepared/dispatched/receipt/finished events.
Recovery only finishes bookkeeping; it never invokes the supplier again.
"""
from __future__ import annotations

import asyncio
import copy

from .jobs import JobError, _hash
from .pricing import PricingError


def require_parent(parent):
    from .task_runtime import _SEND
    if parent is None or parent.budget_attempt is None or _SEND.get() is not parent.binding:
        raise JobError('nested_tool_original_required')
    cache = parent.binding.runtime._native_returns
    source = parent
    for _ in range(17):
        cache.require(source)
        if source.finished or source.task.done():
            raise JobError('nested_tool_original_required')
        source = source.nested_parent
        if source is None:
            return
    raise JobError('tool_attempt_nesting_limit')


async def join_children(slot, *, cancel=False):
    """The source callback cannot finish while its internal supplier tasks run."""
    while tasks := [task for task in slot.children if not task.done()]:
        if cancel:
            for task in tasks:
                task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if not cancel:
            for result in results:
                if isinstance(result, BaseException):
                    raise result


async def persist_receipt(slot):
    from .control_runtime import current, queue_event
    from .execution_tools import event_reference
    from .tool_attempts import event_id
    cache, attempt = slot.binding.runtime._native_returns, slot.budget_attempt
    try:
        async with cache.lock:
            cache.require(slot)
            if slot.billing_saved:
                return
            if slot.billing_receipt is None:
                raise JobError("tool_billing_receipt_missing")
            control = current(slot.binding)
            saved = control.get("attempt", attempt.attempt_id)
            if saved is None or saved["phase"] not in ("dispatched", "finished") or saved["prepared"] != attempt.prepared:
                raise JobError("tool_attempt_original_mismatch")
            key = event_id(slot.binding.runtime.manager, "receipt", attempt.attempt_id)
            queue_event(slot.binding, key, "tool_attempt_receipt", dict(version=1, scope=attempt.prepared["scope"],
                prepared_hash=_hash(attempt.prepared), billing_receipt=slot.billing_receipt.to_dict()))
            queue_event(slot.binding, key + "_a", "tool_attempt_anchor", {"version": 1, "source": event_reference(slot.binding, key)})
            await control.checkpoint(force=True)
            slot.billing_saved = True
    except BaseException:
        slot.failed = True
        raise


async def flush_slot(slot):
    """Retain exact original fees; a failed delivery never repeats its handler."""
    from .control_runtime import current
    from .execution_budget import close_unreserved, release_unstarted
    from .tool_attempts import finish
    cache, attempt = slot.binding.runtime._native_returns, slot.budget_attempt
    cache.require(slot)
    control = current(slot.binding)
    try:
        if slot.billing_receipt is not None and not slot.billing_saved:
            await persist_receipt(slot)
        saved = control.get("attempt", attempt.attempt_id)
        if saved is None or saved["prepared"] != attempt.prepared:
            raise JobError("tool_attempt_original_mismatch")
        if saved["phase"] in ("prepared", "aborted"):
            if await close_unreserved(attempt) is None:
                await release_unstarted(attempt)
            cache.slots.pop(cache.key(slot))
            return
        if attempt.observation is None:
            raise JobError("nested_tool_observation_required")
        try:
            await finish(attempt)
        except Exception:
            # The failed delivery path seals its immutable fee outbox first.
            await control.checkpoint(force=True)
            if control.get("attempt", attempt.attempt_id)["phase"] != "finished":
                raise
        cache.slots.pop(cache.key(slot))
    except BaseException:
        slot.failed = True
        raise


async def invoke_priced_tool(definition, arguments, handler):
    """Invoke one trusted internal tool inside an active original native callback."""
    from power_loop.contracts.errors import ToolValidationError
    from power_loop.contracts.tools import ToolResult, validate_tool_args
    from power_loop.tools.registry import ToolRegistry

    from . import tool_attempts as tools
    from .native_returns import _Receipt
    from .tool_billing import _CURRENT, receipt_context

    parent = _CURRENT.get()
    require_parent(parent)
    task = asyncio.current_task()
    if task is not parent.task:
        parent.children.add(task)
        runtime = parent.binding.runtime
        if task not in runtime._adapter_tasks:
            runtime.track_task(task, job_id=parent.binding.lease.job_id)
    registry = ToolRegistry()
    registry.register(definition, handler)
    arguments = copy.deepcopy(dict(arguments))
    error = validate_tool_args(definition, arguments)
    if error is not None:
        raise ToolValidationError(definition.name, error)
    attempt = await tools.prepare_nested(parent, registry.get(definition.name), arguments)
    cache = parent.binding.runtime._native_returns
    slot = _Receipt(parent.binding, parent.ref, parent.call_id, parent.plan_hash, parent.entry_hash,
        asyncio.current_task(), budget_attempt=attempt, nested_parent=parent)
    cache.require(parent)
    if len(cache.slots) >= 1000:
        raise JobError('native_return_requires_reconciliation')
    cache.slots[cache.key(slot)] = slot
    try:
        if not await tools.reserve(attempt):
            raise JobError('budget_tool_adapter_unavailable')
        from .control_runtime import source
        require_parent(parent)
        await tools.authorize_dispatch(attempt, None, source(slot.binding))
        try:
            with receipt_context(slot):
                result = await registry.invoke_async(definition.name, arguments)
            await join_children(slot)
            if isinstance(result, ToolResult):
                if result.billing_receipt is not None:
                    if slot.billing_receipt is not None and slot.billing_receipt != result.billing_receipt:
                        raise PricingError('tool_billing_receipt_changed')
                    slot.billing_receipt = result.billing_receipt
                result = result.content
            tools.capture(attempt, slot.billing_receipt, None)
        except BaseException as exc:
            await join_children(slot, cancel=True)
            if attempt.observation is None:
                tools.capture(attempt, slot.billing_receipt, exc)
            raise
        finally:
            slot.finished = True
            await flush_slot(slot)
        return result
    except BaseException:
        slot.finished = True
        if cache.slots.get(cache.key(slot)) is slot:
            try:
                await flush_slot(slot)
            except BaseException:
                pass  # The owned slot blocks admission/close until its facts can be saved.
        raise
