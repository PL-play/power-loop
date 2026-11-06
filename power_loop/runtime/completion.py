"""One bounded completion collector, shared by owning engines and host adapters.

Waiting, model receipt processing, and final text are execution work, not
physical-stop or business-delivery proof. Persistent scope admission remains
the task manager's responsibility. No platform imports live here.
"""
from __future__ import annotations

import asyncio
import math
from contextvars import ContextVar
from dataclasses import dataclass, field

from .background import ACTIVE_STATUSES as ACTIVE
from .cancellation import CancellationToken

FINALIZE_PROMPT = (
    "Your workflow task ended without a final answer. Using only the work and evidence already "
    "in this session, write the requested final result now as a non-empty plain-text assistant "
    "message. Do not call tools and do not describe this recovery instruction."
)
WINDOW: ContextVar[CompletionWindow | None] = ContextVar("power_loop_completion_window", default=None)


@dataclass
class CompletionWindow:
    timeout_s: float = 1800.0
    max_rounds: int = 30
    round_ceiling: int | None = None
    multi_session: bool = False
    remaining: int = field(init=False)
    deadline: float | None = field(default=None, init=False)
    utc_deadline_at_ms: int | None = field(default=None, init=False)
    session_id: str | None = field(default=None, init=False)
    _loop: object = field(default=None, init=False, repr=False)
    _collecting: bool = field(default=False, init=False, repr=False)
    answer_attempted: bool = field(default=False, init=False)
    _unreported_rounds: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if (type(self.timeout_s) not in (int, float) or not math.isfinite(self.timeout_s) or self.timeout_s <= 0 or
                type(self.max_rounds) is not int or self.max_rounds < 0 or type(self.multi_session) is not bool or
                self.round_ceiling is not None and (type(self.round_ceiling) is not int or self.round_ceiling < 0)):
            raise ValueError("invalid_completion_limits")
        self.remaining = self.max_rounds

    def bind(self, sid, *, deadline_at_ms=None, now_ms=None):
        loop = asyncio.get_running_loop()
        if self._loop is not None and (self._loop is not loop or not self.multi_session and self.session_id != sid):
            raise ValueError("completion_scope_changed")
        bound = None
        if deadline_at_ms is not None or now_ms is not None:
            if (type(deadline_at_ms) is not int or not 0 < deadline_at_ms < 2**63
                    or type(now_ms) is not int or not 0 < now_ms < 2**63):
                raise ValueError("invalid_completion_deadline")
            if self.utc_deadline_at_ms is not None and self.utc_deadline_at_ms != deadline_at_ms:
                raise ValueError("completion_deadline_changed")
            self.utc_deadline_at_ms = deadline_at_ms
            bound = loop.time() + max(0, (deadline_at_ms - now_ms) / 1000)
        if self.deadline is None:
            self._loop, self.session_id, self.deadline = loop, sid, loop.time() + self.timeout_s
        if bound is not None:
            self.deadline = min(self.deadline, bound)

    def allowance(self, raw):
        if self.round_ceiling is None:
            return self.remaining
        return min(self.remaining, max(0, self.round_ceiling - int(raw.get("rounds") or 0) - self._unreported_rounds))

    def consume(self, more, allowance):
        reported = int(more.rounds or 0)
        if reported < 0:
            raise ValueError("invalid_completion_round_evidence")
        used = max(1, reported)
        self.remaining -= used
        self._unreported_rounds += used - reported
        return used <= allowance

    def require_time(self):
        if self.deadline is None or asyncio.get_running_loop().time() >= self.deadline:
            raise TimeoutError("completion deadline reached")


async def running_background(store, sid):
    return [row for row in await store.list_background_tasks(sid) if str(row.status) in ACTIVE]


async def wait_background(store, sid, ids, timeout_s, stop_event=None):
    token = CancellationToken.from_any(stop_event)
    deadline = asyncio.get_running_loop().time() + max(0, timeout_s)
    rows = []
    while not token.is_cancelled():
        rows = [row for row in [await store.get_background_task(sid, tid) for tid in ids] if row is not None]
        if len(rows) != len(ids):
            raise RuntimeError("background task disappeared before workflow completion")
        if all(str(row.status) not in ACTIVE for row in rows) or asyncio.get_running_loop().time() >= deadline:
            return rows
        await asyncio.sleep(min(0.1, max(0, deadline - asyncio.get_running_loop().time())))
    return rows


def _merge(raw, more):
    usage = dict(raw.get("usage") or {})
    for key, value in (more.usage or {}).items():
        usage[key] = usage.get(key, 0) + value
    raw.update(status=more.status, final_text=more.final_text,
        rounds=int(raw.get("rounds") or 0) + int(more.rounds or 0), usage=usage,
        new_input_after_pass=bool(getattr(more, "new_input_after_pass", False)))
