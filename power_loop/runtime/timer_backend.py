"""One timer tool, with explicit trusted identity and a deployment-owned backend."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from power_loop.agent.runtime_types import RuntimeRequestError, TargetRef, TrustedActor

from .store.types import TimerRow


@dataclass(frozen=True)
class TimerContext:
    store_id: str
    session_id: str
    actor: TrustedActor
    job_ref: object = None
    tool_call_id: str | None = None
    operation_id: str | None = None
    host_context: Mapping = field(default_factory=dict)


async def timer_context() -> TimerContext:
    from power_loop.core.execution_context import get_execution_context

    from .runtime_state import get_tool_runtime_context
    execution = get_execution_context()
    call = get_tool_runtime_context(required=True)
    body = [execution.store_id, execution.session_id, call.assistant_seq, call.tool_call_id]
    operation = 'timer-tool-' + hashlib.sha256(json.dumps(body, separators=(',', ':')).encode()).hexdigest()
    ref = await call.store.message_ref(execution.session_id, call.assistant_seq)
    actor = replace(execution.actor, call_ref=ref, tool_call_id=call.tool_call_id)
    return TimerContext(execution.store_id, execution.session_id, actor,
        execution.job_ref, call.tool_call_id, operation, execution.host_context)


@runtime_checkable
class TimerBackend(Protocol):
    async def create(self, source: TimerContext, *, delay_s: float | None,
                     due_at_ms: int | None, note: str, interval_s: int | None) -> TimerRow: ...
    async def list(self, source: TimerContext, *, statuses: tuple[str, ...]) -> list[TimerRow]: ...
    async def cancel(self, source: TimerContext, timer_id: int) -> bool: ...


@dataclass(frozen=True)
class TimerDriverStatus:
    """Driver readiness and the last observed error.

    The default driver retains its last error across scans until restarted.
    """

    mode: str
    ready: bool
    last_scan_ms: int | None = None
    last_error: str | None = None
    pending: int = 0
    oldest_due_ms: int | None = None


class TimerDriver(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def status(self) -> TimerDriverStatus: ...


class StoreTimerBackend:
    """Standalone SDK timer storage. The runtime owns its TimerRunner."""
    def __init__(self, store):
        self.store = store

    async def _authorize(self, source, tx):
        from power_loop.agent.runtime_store import RuntimeInputStore
        port = RuntimeInputStore(self.store)
        target = TargetRef(source.store_id, source.session_id)
        port.check_store(target)
        row = await tx.fetchone(f'SELECT parent_session_id,metadata_json FROM {self.store.t.sessions} WHERE session_id=?', (source.session_id,))
        metadata = await port._authorize_session(tx, target.session_id, source.actor, row)
        return metadata, row['parent_session_id']

    async def create(self, source, *, delay_s=None, due_at_ms=None, note, interval_s=None):
        if (delay_s is None) == (due_at_ms is None):
            raise RuntimeRequestError('invalid_options', 'provide exactly one of delay_s / due_at_ms')
        if not note.strip() or (interval_s is not None and interval_s < 1):
            raise RuntimeRequestError('invalid_options', 'note and positive interval required')
        due = due_at_ms if due_at_ms is not None else int(time.time()*1000 + delay_s*1000)
        # Creation and the live-timer limit share the existing session-state lock.
        async with self.store._db.transaction() as tx:
            meta, parent = await self._authorize(source, tx)
            state = await self.store._db.dialect.lock_state(tx, self.store.t.session_state, source.session_id)
            if state['lifecycle'] != 'open':
                raise RuntimeRequestError('session_closed')
            if parent is not None or meta.get('execution_role', 'main') != 'main':
                raise RuntimeRequestError('forbidden', 'only main agents may schedule timers')
            count = await tx.fetchone(f"SELECT COUNT(*) AS n FROM {self.store.t.timers} WHERE session_id=? AND status IN ('armed','firing')", (source.session_id,))
            if count['n'] >= 10:
                raise RuntimeRequestError('timer_limit_reached')
            return await self.store.create_timer(source.session_id, due_at=due, note=note.strip(), interval_s=interval_s, tx=tx)

    async def list(self, source, *, statuses=('armed', 'firing')):
        await self._authorize(source, self.store._db)
        return await self.store.list_timers(source.session_id, statuses=statuses)

    async def cancel(self, source, timer_id):
        await self._authorize(source, self.store._db)
        # Cancelling a claimed recurring timer prevents rearming; the accepted
        # occurrence remains governed by the normal input cancellation API.
        return bool(await self.store._db.execute(f"UPDATE {self.store.t.timers} SET status='cancelled',updated_at=? "
            "WHERE session_id=? AND timer_id=? AND (status='armed' OR (status='firing' AND interval_s IS NOT NULL))",
            (int(time.time()*1000), source.session_id, timer_id)))


def backend_for(loop, store=None) -> TimerBackend:
    if loop.timer_backend is None:
        raise RuntimeRequestError('capability_unavailable', 'timer backend is disabled')
    return loop.timer_backend
