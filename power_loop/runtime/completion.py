"""One bounded completion collector, shared by owning engines and host adapters.

Waiting, model receipt processing, and final text are execution work, not
physical-stop or business-delivery proof. Persistent scope admission remains
the task manager's responsibility. No platform imports live here.
"""
from __future__ import annotations

import asyncio
import contextlib
import math
from contextvars import ContextVar
from dataclasses import dataclass, field

from .cancellation import CancellationToken

ACTIVE = frozenset({"queued", "pending", "running"})
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


async def collect_required(loop, raw, stop_event=None, *, window=None, resolve_loop=None, drain_on_error=False):
    """Collect to a fixed point using one shared time/round window.

    A caller-owned window survives repeated collection within the same node.
    A continuation resolver is a trusted host port, never a model argument.
    """
    from power_loop.workflow.admission import idle

    from . import delegation
    from .spec import workflow_continuation_loop
    raw = dict(raw)
    sid = raw.get("session_id")
    if not sid:
        return raw
    window = window or WINDOW.get() or CompletionWindow()
    from .completion_windows import bind_window
    await bind_window(loop, window, sid)
    token = CancellationToken.from_any(stop_event)
    if token.is_cancelled():
        return {**raw, "status": "cancelled", "error": "cancelled during completion settlement"}
    if not hasattr(loop, "ensure_store"):
        return raw
    if window._collecting:
        return {**raw, "status": "failed", "completion_unresolved": True, "error": "completion collection already running"}
    window._collecting = True
    try:
        async with asyncio.timeout(max(0, window.deadline - asyncio.get_running_loop().time())):
            window.require_time()
            store = await loop.ensure_store()
            while True:
                if drain_on_error or raw.get("status") == "completed":
                    await delegation.await_children(store, sid, stop_event,
                        max(0, window.deadline - asyncio.get_running_loop().time()))
                if token.is_cancelled():
                    return {**raw, "status": "cancelled", "error": "cancelled during completion settlement"}
                if raw.get("status") != "completed":
                    return raw
                prompts = []
                running = await running_background(store, sid) if hasattr(store, "list_background_tasks") else []
                if running:
                    async with idle():
                        rows = await wait_background(store, sid, [r.task_id for r in running],
                            max(0, window.deadline - asyncio.get_running_loop().time()), stop_event)
                    if token.is_cancelled():
                        return {**raw, "status": "cancelled", "error": "cancelled while waiting for background tasks"}
                    if any(str(row.status) in ACTIVE for row in rows):
                        raise TimeoutError("background tasks still pending")
                    lines = [f"- {r.task_id} [{r.status}] {str(r.command or '')[:120]}\n  {str(r.output_tail or '')[-1500:]}" for r in rows]
                    prompts.append("Your background tasks have settled. Read and use these results to finish the original task:\n" + "\n".join(lines))
                if not prompts and hasattr(store, "list_unseen_background_updates"):
                    unseen = await store.list_unseen_background_updates(sid)
                    if any(str(row.status) not in ACTIVE for row in unseen):
                        prompts.append("Final background results are available. Process them and finish the original task.")
                if raw.get("new_input_after_pass"):
                    prompts.append("Process the new input already in this session and finish the original task. Do not repeat completed actions.")
                if not prompts:
                    from .owner_waits import wait_ready
                    resumed = await wait_ready(loop, raw, stop_event, deadline=window.deadline)
                    if resumed == "cancelled":
                        return {**raw, "status": "cancelled", "error": "cancelled during original task wait"}
                    if resumed == "history":
                        prompts.append("Process the accepted original input already in this session and finish the original task without repeating completed actions.")
                pending = (await loop.inbox_pending(sid)).get("pending", 0) if hasattr(loop, "inbox_pending") else 0
                if not prompts and not pending:
                    return raw
                allowance = window.allowance(raw)
                if allowance <= 0:
                    return {**raw, "status": "hit_round_limit", "completion_unresolved": True, "error": "completion settlement round budget exhausted"}
                window.require_time()
                child = await (resolve_loop or workflow_continuation_loop)(loop, sid)
                window.require_time()
                with contextlib.ExitStack() as guards:
                    for _, guard in getattr(loop, "_child_run_guards", ()) or ():
                        guards.enter_context(guard())
                    async with delegation.readmit_session(loop, sid):
                        more = (await child.follow_up("\n\n".join(prompts), session_id=sid,
                            stop_event=stop_event, max_rounds=allowance) if prompts else
                            await child.flush_follow_ups(sid, stop_event=stop_event, max_rounds=allowance))
                if more is None or not hasattr(more, "status"):
                    return {**raw, "status": "failed", "completion_unresolved": True, "error": "completion settlement returned no result"}
                _merge(raw, more)
                within_limit = window.consume(more, allowance)
                if not within_limit and raw["status"] == "completed":
                    return {**raw, "status": "failed", "completion_unresolved": True, "error": "completion round allowance exceeded"}
                if raw["status"] != "completed":
                    raw["completion_unresolved"] = True
    except TimeoutError as exc:
        from .owner_waits import WaitDeadline
        if isinstance(exc, WaitDeadline):
            return {**raw, "status": "timed_out", "completion_unresolved": True, "error": str(exc)}
        return {**raw, "status": "failed", "completion_unresolved": True, "error": "completion settlement timed out"}
    except Exception as exc:
        expired = getattr(exc, "code", None) == "owner_wait_deadline_exceeded"
        return {**raw, "status": "cancelled" if token.is_cancelled() else "timed_out" if expired else "failed",
            "completion_unresolved": True, "error": "completion check failed: " + str(exc)}
    finally:
        window._collecting = False


async def finish_text(loop, raw, stop_event=None, *, window=None, resolve_loop=None):
    """Direct host-adapter completion; managed engines own this work themselves."""
    from .spec import workflow_continuation_loop
    window = window or CompletionWindow()
    raw = await collect_required(loop, raw, stop_event, window=window, resolve_loop=resolve_loop)
    if raw.get("status") != "completed" or (raw.get("final_text") or "").strip():
        return raw
    sid = raw.get("session_id")
    if not sid:
        return {**raw, "status": "failed", "completion_unresolved": True, "error": "workflow leaf completed without a final result or recoverable session"}
    token = CancellationToken.from_any(stop_event)
    if token.is_cancelled():
        return {**raw, "status": "cancelled"}
    allowance = min(2, window.allowance(raw))
    if allowance <= 0 or window.answer_attempted:
        return {**raw, "status": "failed", "completion_unresolved": True, "error": "workflow leaf finalization round budget exhausted"}
    window.answer_attempted = True
    try:
        async with asyncio.timeout(max(0, window.deadline - asyncio.get_running_loop().time())):
            window.require_time()
            child = await (resolve_loop or workflow_continuation_loop)(loop, sid)
            window.require_time()
            with contextlib.ExitStack() as guards:
                for _, guard in getattr(loop, "_child_run_guards", ()) or ():
                    guards.enter_context(guard())
                extra = await child.send(FINALIZE_PROMPT, session_id=sid, stop_event=stop_event,
                    tools=[], max_rounds=allowance)
            _merge(raw, extra)
            within_limit = window.consume(extra, allowance)
            if not within_limit and raw["status"] == "completed":
                return {**raw, "status": "failed", "completion_unresolved": True, "error": "completion round allowance exceeded"}
    except Exception as exc:
        return {**raw, "status": "cancelled" if token.is_cancelled() else "failed",
            "completion_unresolved": True, "error": "workflow leaf finalization failed: " + type(exc).__name__ + ": " + str(exc)}
    if raw["status"] == "completed" and (raw.get("final_text") or "").strip():
        return raw
    error = "workflow leaf finalization returned no final result" if raw["status"] == "completed" else "workflow leaf finalization ended with status " + str(raw["status"])
    return {**raw, "status": "failed", "completion_unresolved": True, "error": error}
