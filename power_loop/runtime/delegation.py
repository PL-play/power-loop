"""Asynchronous subagent sessions, durable results and serialized continuation.

The host owns idle-parent wake routing. No notification callback starts a bare parent loop.
Execution is process-local; records and undelivered inbox items survive the process.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import weakref
from dataclasses import dataclass, field, replace
from typing import Any

from power_loop.agent.follow_up import InboxItem
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.recall import authorize, dumps
from power_loop.runtime.spec import (
    _ON_CHILD_SESSION,
    _USAGE_SINK,
    AgentSpec,
    run_agent_spec,
    workflow_continuation_loop,
)

log = logging.getLogger(__name__)
RESULT_PREFIX = "[subagent_result]\n"


@dataclass
class Running:
    run_id: str
    task: asyncio.Task | None = None
    token: Any = None
    parent: str = ""
    title: str = ""
    suppress_wake: bool = False
    phase: str = "running"
    descendant_usage: dict = field(default_factory=dict)
    projection_expected: bool = True


_RUNNING: dict[tuple[int, str], Running] = {}
_TASKS: set[asyncio.Task] = set()


def _retain(task):
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


async def _save_result(store, parent, sid, run, raw, after_seq, lifecycle):
    messages = await store.history_bounds(sid)
    messages = [max(after_seq + 1, messages[0]), messages[1]] if messages and messages[1] > after_seq else None
    projection, projection_status = None, "empty"
    try:
        projection = await store.history_bounds(sid, source="projection", after_seq=after_seq)
        projection_status = "ready" if projection else "failed" if run.projection_expected else "disabled"
    except ValueError:
        projection_status = "disabled"
    usage = dict(raw.get("usage") or {})
    for name, value in run.descendant_usage.items():
        usage[name] = usage.get(name, 0) + value
    result = {**raw, "usage": usage, "session_id": sid, "run_id": run.run_id, "messages": messages,
              "projection": projection, "history_status": "available" if messages else "empty",
              "projection_status": projection_status}
    # Persist ownership/prepared result BEFORE any destructive cleanup.
    key = "subagent:run:" + run.run_id
    await store.set_runtime_state(parent, key, {"phase": "prepared", "result": result})
    if lifecycle == "ephemeral" and raw.get("status") == "completed":
        await store.close_session(sid, cascade=True)
        result.update(messages=None, projection=None, history_status="deleted", projection_status="deleted")
        await store.set_runtime_state(parent, "subagent:child:" + sid,
                                      {"deleted": True, "lifecycle": lifecycle})
    await store.set_runtime_state(parent, key, {"phase": "completed", "result": result, "notified": False})
    return result


async def _finish(loop, parent, sid, running, raw, after_seq, lifecycle, finalize, on_complete):
    store = await loop.ensure_store()
    running.phase = "settling"
    children = [r.task for (key, _), r in list(_RUNNING.items())
                if key == id(store) and r.parent == sid and r.task is not None]
    if children:
        await asyncio.gather(*children)
    if finalize is not None:
        raw = await finalize(sid, raw)
    # A nudge racing finalization is already durable. Consume it in this run before returning.
    continuation = None
    while (await store.inbox_counts(sid))["pending"]:
        if running.token.is_cancelled():
            break
        if continuation is None:
            continuation = await workflow_continuation_loop(loop, sid)
        with contextlib.ExitStack() as guards:
            for _, guard in getattr(loop, "_child_run_guards", ()):
                guards.enter_context(guard())
            extra = await continuation.flush_follow_ups(sid, stop_event=running.token)
        if extra is None:
            break
        raw["final_text"] = str(extra.final_text or "")
        raw["status"] = extra.status
        raw["rounds"] = int(raw.get("rounds", 0)) + int(extra.rounds)
        usage = dict(raw.get("usage") or {})
        for key, value in (extra.usage or {}).items():
            usage[key] = usage.get(key, 0) + value
        raw["usage"] = usage
        if finalize is not None:
            raw = await finalize(sid, raw)
    # From the final pending check through cleanup, nudge uses our gate (below).
    async with _gate(store, sid):
        child_active = any(key == id(store) and r.parent == sid for (key, _), r in _RUNNING.items())
        if ((await store.inbox_counts(sid))["pending"] or child_active) and not running.token.is_cancelled():
            # Re-enter outside the gate; callers can enqueue without losing accepted input.
            again = True
        else:
            again = False
            result = await _save_result(store, parent, sid, running, raw, after_seq, lifecycle)
            def accumulate(value):
                totals = dict(value or {})
                for name, count in result.get("usage", {}).items():
                    if isinstance(count, (int, float)) and not isinstance(count, bool):
                        totals[name] = totals.get(name, 0) + count
                return totals
            await store.mutate_runtime_state(parent, "subagent:usage:total", accumulate, default={})
            parent_run = _RUNNING.get((id(store), parent))
            if parent_run is not None:
                for name, value in result.get("usage", {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        parent_run.descendant_usage[name] = parent_run.descendant_usage.get(name, 0) + value
            _RUNNING.pop((id(store), sid), None)
            _publish_completion(loop, parent, sid, result)
    if again:
        return await _finish(loop, parent, sid, running, raw, after_seq, lifecycle, finalize, on_complete)
    try:
        if running.suppress_wake:
            return result
        # Inbox delivery is durable and deduplicated before asking the host to wake an idle
        # parent. A failed/skipped wake still leaves the result for its next normal send.
        await loop.deliver(InboxItem(RESULT_PREFIX + completion_view(loop, result), kind="task_done",
                                    item_id="subagent:" + running.run_id), parent, enqueue_only=True)
        if on_complete is not None:
            await on_complete(parent, result)
        record = await store.get_runtime_state(parent, "subagent:run:" + running.run_id)
        await store.set_runtime_state(parent, "subagent:run:" + running.run_id, {**record, "notified": True})
    except Exception:
        log.exception("subagent completion not delivered; durable result retained: %s", running.run_id)
    return result


_GATES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _gate(store, sid):
    return _GATES.setdefault((id(store), sid), asyncio.Lock())


async def _complete(loop, parent, sid, run, raw, after_seq, lifecycle, finalize, on_complete):
    """Finalization errors/cancellation must not leave an immortal running job."""
    store = await loop.ensure_store()
    try:
        return await _finish(loop, parent, sid, run, raw, after_seq, lifecycle, finalize, on_complete)
    except BaseException as exc:
        failed = {**raw, "status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                  "error": f"subagent finalization: {type(exc).__name__}: {exc}"}
        try:
            result = await _save_result(store, parent, sid, run, failed, after_seq, lifecycle)
            _publish_completion(loop, parent, sid, result)
            if not run.suppress_wake:
                await loop.deliver(InboxItem(RESULT_PREFIX + completion_view(loop, result), kind="task_done",
                                            item_id="subagent:" + run.run_id), parent, enqueue_only=True)
                if on_complete is not None:
                    await on_complete(parent, result)
            return result
        except Exception:
            log.exception("subagent finalization failed; original records retained: %s", run.run_id)
    finally:
        if _RUNNING.get((id(store), sid)) is run:
            _RUNNING.pop((id(store), sid), None)


async def start(loop, spec: AgentSpec, task: str, *, on_complete=None, finalize=None,
                admission: asyncio.Semaphore | None = None) -> dict[str, Any]:
    parent = get_session_id()
    if parent is None:
        raise ValueError("spawn_agent requires an active parent session")
    store = await loop.ensure_store()
    parent_run = _RUNNING.get((id(store), parent))
    ready = asyncio.get_running_loop().create_future()
    run = Running(secrets.token_hex(8), token=CancellationToken(), parent=parent, title=spec.name)
    if parent_run is not None:
        run.token = parent_run.token.child()

    admitted = False

    async def created(sid):
        nonlocal admitted
        session = await store.get_session(sid)
        run.projection_expected = session.config.get("context_projection", True)
        _RUNNING[(id(store), sid)] = run
        await store.set_runtime_state(parent, "subagent:child:" + sid,
                                      {"deleted": False, "lifecycle": spec.lifecycle})
        await store.set_runtime_state(parent, "subagent:run:" + run.run_id,
                                      {"phase": "running", "session_id": sid, "run_id": run.run_id})
        await store.merge_session_metadata(sid, {"standalone_subagent": True})
        ready.set_result(sid)
        if admission is not None:
            run.phase = "queued"
            await admission.acquire()
            admitted = True
            run.phase = "running"

    async def execute():
        token = _ON_CHILD_SESSION.set(created)
        usage = _USAGE_SINK.set(None)
        sid = None
        try:
            # Manager owns cleanup, after repair/projection/range collection.
            raw = await run_agent_spec(replace(spec, lifecycle="linked", metadata={**spec.metadata, "managed_subagent": True}), task,
                                       parent_loop=loop, stop_event=run.token)
            sid = ready.result() if ready.done() and not ready.cancelled() else None
            if sid is None:
                raise ValueError(raw.get("final_text") or "child session was not created")
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
                return
            sid = ready.result()
            raw = {"status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                   "final_text": "", "rounds": 0, "usage": {}, "error": str(exc)}
        finally:
            _ON_CHILD_SESSION.reset(token)
            _USAGE_SINK.reset(usage)
            if admitted:
                admission.release()
        await _complete(loop, parent, sid, run, raw, 0, spec.lifecycle, finalize, on_complete)

    run.task = asyncio.create_task(execute(), name="subagent-" + run.run_id)
    _retain(run.task)
    sid = await asyncio.shield(ready)
    return {"session_id": sid, "run_id": run.run_id, "mode": "create", "status": "queued"}


async def nudge(loop, sid: str, message: str, *, on_complete=None, finalize=None,
                admission: asyncio.Semaphore | None = None) -> dict[str, Any]:
    parent = get_session_id()
    if parent is None:
        raise ValueError("nudge requires an active parent session")
    store = await loop.ensure_store()
    _, unavailable = await authorize(store, parent, sid)
    if unavailable:
        return {"status": "unavailable", "reason": unavailable, "session_id": sid}
    row = await store.get_session(sid)
    if not row.metadata.get("standalone_subagent"):
        raise ValueError("not a standalone subagent; use the owning workflow's nudge")
    # Build using the saved child contract and current parent policy, never the parent's full tools.
    child_loop = await workflow_continuation_loop(loop, sid)
    async with _gate(store, sid):
        running = _RUNNING.get((id(store), sid))
        if running is not None:
            await child_loop.deliver(InboxItem(message, mode="steer"), sid, enqueue_only=True)
            return {"session_id": sid, "run_id": running.run_id, "mode": "steer", "status": "accepted"}
        if child_loop._lock_for(sid).locked():
            raise ValueError("session is owned by another execution; nudge was not accepted")
        record = await store.get_runtime_state(parent, "subagent:child:" + sid)
        lifecycle = record.get("lifecycle", "linked")
        before = await store.history_bounds(sid)
        after_seq = before[1] if before else 0
        run = Running(secrets.token_hex(8), token=CancellationToken(), parent=parent, title=str(row.metadata.get("spec_name") or "subagent"))
        run.projection_expected = child_loop.config.projection_representation is not None
        parent_run = _RUNNING.get((id(store), parent))
        if parent_run is not None:
            run.token = parent_run.token.child()
        _RUNNING[(id(store), sid)] = run
        await store.set_runtime_state(parent, "subagent:run:" + run.run_id,
                                      {"phase": "queued", "session_id": sid, "run_id": run.run_id})

        async def execute():
            from power_loop.contracts.event_payloads import SubagentTaskStartPayload
            from power_loop.contracts.events import AgentEventType
            from power_loop.runtime.spec import _publish_subagent_event
            from power_loop.workflow.engine import InProcessExecutor
            _publish_subagent_event(loop, parent, AgentEventType.SUBAGENT_TASK_START,
                                   SubagentTaskStartPayload(task=message, preset=run.title,
                                                           sub_session_id=sid, depth=row.spawn_depth))
            admitted = False
            try:
                if admission is not None:
                    run.phase = "queued"
                    await admission.acquire()
                    admitted = True
                run.phase = "running"
                raw = await InProcessExecutor().continue_agent(
                    sid, message, parent_loop=loop, extra_rounds=child_loop.config.max_rounds,
                    stop_event=run.token)
            except BaseException as exc:
                raw = {"status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                       "final_text": "", "rounds": 0, "usage": {}, "error": str(exc)}
            finally:
                if admitted:
                    admission.release()
            await _complete(loop, parent, sid, run, raw, after_seq, lifecycle, finalize, on_complete)

        run.task = asyncio.create_task(execute(), name="subagent-" + run.run_id)
        _retain(run.task)
        return {"session_id": sid, "run_id": run.run_id, "mode": "continue", "status": "queued"}


def result_view(text: str, cap: int = 600) -> str | None:
    """Tool-context rule: only final_text may change, never envelope metadata."""
    import json
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or "session_id" not in value or not value.get("run_id"):
        return None
    if isinstance(value.get("final_text"), str) and len(value["final_text"]) > cap:
        value["final_text"] = value["final_text"][:max(0, cap)] + "…[truncated]"
    return dumps(value)


def running_jobs(store, parent):
    return [{"id": "sub:" + run.run_id, "kind": "subagent", "title": run.title,
             "status": run.phase, "session_id": sid, "started_at": None, "origin": ""}
            for (key, sid), run in _RUNNING.items() if key == id(store) and run.parent == parent]


async def stop(store, parent, run_id, grace_s):
    run = next((run for (key, _), run in _RUNNING.items()
                if key == id(store) and run.parent == parent and run.run_id == run_id), None)
    if run is None:
        return "finished"
    run.suppress_wake = True
    run.token.cancel("stopped")
    try:
        await asyncio.wait_for(asyncio.shield(run.task), timeout=grace_s)
        return "stopped"
    except TimeoutError:
        run.task.cancel()
        return "forced"


async def await_children(store, parent, stop_event=None, timeout_s=1800):
    """Owning executors must collect asynchronous descendants before settling a leaf."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    token = CancellationToken.from_any(stop_event)
    while True:
        runs = [run for (key, _), run in _RUNNING.items()
                if key == id(store) and run.parent == parent and run.task is not None]
        if not runs:
            return
        if token.is_cancelled():
            for run in runs:
                run.token.cancel("parent cancelled")
            raise ValueError("cancelled while waiting for subagents")
        if asyncio.get_running_loop().time() >= deadline:
            raise ValueError("subagents still pending after completion wait")
        await asyncio.wait([run.task for run in runs], timeout=0.1)


def completion_view(loop, result):
    config = getattr(loop, "config", None)
    policy = getattr(config, "tool_context", None)
    projection = getattr(getattr(config, "representation", None), "projection_cfg", {}) or {}
    overrides = getattr(policy, "overrides", {}) or {}
    cap = overrides.get("spawn_agent", {}).get("result_max_chars") or projection.get("final_text_cap") or 600
    return result_view(dumps(result), int(cap))


def _publish_completion(loop, parent, sid, result):
    from power_loop.contracts.event_payloads import SubagentCompletedPayload, SubagentTextPayload
    from power_loop.contracts.events import AgentEventType
    from power_loop.runtime.spec import _publish_subagent_event
    status = str(result.get("status") or "failed")
    fields = {"sub_session_id": sid, "status": status, "rounds": int(result.get("rounds") or 0),
              "final_text": str(result.get("final_text") or "")}
    if status == "completed":
        _publish_subagent_event(loop, parent, AgentEventType.SUBAGENT_TEXT, SubagentTextPayload(**fields))
    _publish_subagent_event(loop, parent, AgentEventType.SUBAGENT_COMPLETED, SubagentCompletedPayload(**fields))
