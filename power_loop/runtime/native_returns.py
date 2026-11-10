"""Actual native callback returns, separate from work completion or a tool row.

Only the original pipeline can reserve a receipt before invoking its frozen
handler. A returned value is data, not proof that an asynchronous child stopped.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field

from .jobs import JobError, _hash


class NativeReturnPending(JobError):
    """Do not replace an actual returned value with a synthetic tool error."""

    def __init__(self):
        super().__init__('native_return_checkpoint_pending')


def identity(manager, source, ref, call_id):
    return 'native_return_' + _hash([manager.store.store_id, source['job_id'], source['generation'], ref.to_dict(), call_id])[:48]


@dataclass(eq=False)
class _Receipt:
    binding: object
    ref: object
    call_id: str
    plan_hash: str
    entry_hash: str
    task: object
    text: str | None = None
    children: set = field(default_factory=set, repr=False)
    finished: bool = False
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

    def reserve(self, binding, ref, plan, entry, *, plan_hash=None):
        from .task_runtime import _SEND
        self.check()
        if _SEND.get() is not binding or binding.runtime is not self.runtime or entry['recovery'] is not None:
            raise JobError('native_return_original_required')
        slot = _Receipt(binding, ref, entry['call_id'], plan_hash or _hash(plan), _hash(entry), asyncio.current_task())
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
        slot.finished = True
        if slot.text is None:
            self.slots.pop(self.key(slot), None)

    def pending_count(self):
        return sum(s.text is not None for s in self.slots.values())

    def failed_count(self):
        return sum(s.failed for s in self.slots.values())


async def persist(runtime, slot):
    from .execution_tools import persist_return
    return await persist_return(runtime, slot)


async def invoke(pipeline, registered, arguments, slot, *, attempt=None, dispatch=None):
    """Called by the pipeline after original dispatch, before formatting/hooks."""
    cache = slot.binding.runtime._native_returns if slot is not None else None
    recorded = False
    try:
        from .tool_contracts import validate_registration
        validate_registration(pipeline.tool_registry, registered.definition.name, registered)
        result = (await dispatch() if dispatch is not None else
                  await pipeline.tool_registry.invoke_async(registered.definition.name, arguments))
        if slot is not None:
            from .nested_tools import join_children
            await join_children(slot)
        from power_loop.contracts.tools import ToolResult
        if isinstance(result, ToolResult):
            result = result.content
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        if attempt is not None and attempt.observation is None:
            from .tool_attempts import capture
            capture(attempt, None)
        if slot is not None:
            cache.capture(slot, text)
            try:
                await persist(slot.binding.runtime, slot)
            except Exception as exc:
                raise NativeReturnPending() from exc
        if attempt is not None:
            from .tool_attempts import finish
            recorded = True
            await finish(attempt)
        return text
    except BaseException as exc:
        if slot is not None:
            from .nested_tools import join_children
            await join_children(slot, cancel=True)
        if attempt is not None and not recorded:
            from .tool_attempts import capture, finish
            try:
                if attempt.observation is None:
                    capture(attempt, exc)
                await finish(attempt)
            except BaseException:
                import logging
                logging.getLogger(__name__).warning('Tool observation remains unresolved',
                    extra={'attempt_id': attempt.attempt_id})
        raise
    finally:
        if slot is not None:
            cache.release_unreturned(slot)


async def flush_local(runtime, *, failed_only=False):
    cache = runtime._native_returns
    cache.check()
    for slot in list(cache.slots.values()):
        if slot.text is not None and (not failed_only or slot.failed):
            await persist(runtime, slot)
