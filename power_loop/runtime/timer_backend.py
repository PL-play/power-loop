"""Host-selected timer storage. Native tasks require an explicit host backend."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Protocol, runtime_checkable

from .jobs import JobError
from .store.types import TimerRow


@runtime_checkable
class TimerBackend(Protocol):
    async def create(self, session_id: str, *, delay_s: float | None,
                     due_at_ms: int | None, note: str, interval_s: int | None,
                     operation_id: str | None) -> TimerRow: ...
    async def list(self, session_id: str, *, statuses: tuple[str, ...]) -> list[TimerRow]: ...
    async def cancel(self, session_id: str, timer_id: int, *, operation_id: str | None) -> bool: ...


class StoreTimerBackend:
    """Standalone SDK timers owned by its SessionStore and TimerRunner."""
    def __init__(self, store):
        self.store = store

    async def create(self, session_id, *, delay_s, due_at_ms, note, interval_s, operation_id):
        due = due_at_ms if due_at_ms is not None else int(time.time()*1000 + delay_s*1000)
        return await self.store.create_timer(session_id, due_at=due, note=note, interval_s=interval_s)

    async def list(self, session_id, *, statuses):
        return await self.store.list_timers(session_id, statuses=statuses)

    async def cancel(self, session_id, timer_id, *, operation_id):
        return await self.store.transition_timer(
            session_id, timer_id, from_status='armed', to_status='cancelled',
        )


def backend_for(loop, store) -> TimerBackend:
    backend = getattr(loop, 'timer_backend', None)
    if backend is not None:
        return backend
    if getattr(loop, 'task_runtime', None) is not None:
        raise JobError('task_adapter_unavailable:timer')
    return StoreTimerBackend(store)


def operation_id(loop, session_id, explicit=None):
    if explicit is not None:
        if type(explicit) is not str or not explicit or len(explicit)>255:
            raise ValueError('timer operation_id must be a nonempty string of at most 255 characters')
        return explicit
    from .runtime_state import get_tool_runtime_context
    ctx = get_tool_runtime_context()
    if (ctx.loop is not loop or ctx.session_id != session_id
            or ctx.assistant_seq is None or not ctx.tool_call_id):
        return None  # Hosted backends enforce their required original-source contract.
    body = [ctx.store.store_id, session_id, ctx.assistant_seq, ctx.tool_call_id]
    return 'timer-tool-' + hashlib.sha256(json.dumps(body,separators=(',',':')).encode()).hexdigest()
