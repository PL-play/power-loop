"""Middle-tier tests: durable journal (D2), detached execution + wake (D3),
introspection (D4). Fake LLM only; no real provider."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from power_loop import AgentEventType, AgentLoopConfig, StatefulAgentLoop, TimerRunner
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.contracts.hook_contexts import TimerFireCtx
from power_loop.contracts.hooks import HookDirective
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.workflow import (
    WorkflowRunError,
    WorkflowRunHandle,
    create_workflow,
    get_workflow,
    journal,
    list_workflows,
    register_wake_guard,
)
from power_loop.workflow.result import AgentResult, WorkflowResult
from power_loop.workflow.runner import make_wake_guard
from power_loop.workflow.tool import _handle_create_workflow, _handle_workflow_status

pytestmark = pytest.mark.unit


@dataclass
class _FakeLLM(LLMService):
    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        sp = (request.system_prompt or "").lower()
        txt = '{"subtopics": ["a", "b"]}' if "subtopic" in sp else "ok"
        return LLMResponse(raw_text=txt, content_text=txt)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _loop() -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=_FakeLLM(),
        db_path=tempfile.mktemp(suffix=".db"),
        config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=3, compactor=None),
    )


SINGLE = {"name": "single", "root": {"type": "agent", "id": "a",
          "spec": {"name": "a", "system_prompt": "Summarize."}}}
SEQ = {"name": "seq", "root": {"type": "sequence", "steps": [
    {"type": "agent", "id": "one", "spec": {"name": "a", "system_prompt": "p"}},
    {"type": "agent", "id": "two", "spec": {"name": "b", "system_prompt": "p"}},
]}}
BAD_REF = {"name": "bad", "root": {"type": "sequence", "steps": [
    {"type": "agent", "id": "plan", "spec": {"name": "p", "system_prompt": "List the subtopics."},
     "output_schema": {"name": "P", "schema": {"type": "object", "required": ["subtopics"],
        "properties": {"subtopics": {"type": "array", "items": {"type": "string"}}}}}},
    {"type": "foreach", "id": "f", "items_from": "plan.nope", "as": "i",
     "body": {"type": "agent", "id": "b", "spec": {"name": "b", "system_prompt": "do {{i}}"}}},
]}}


# ── D2: journal ──────────────────────────────────────────────────────────────


async def test_journal_roundtrip():
    loop = _loop()
    psid = await loop.new_session()
    store = loop.store
    await journal.seed(store, psid, "r1", "wf")
    await journal.record_step(store, psid, "r1", node_id="a", status="completed",
                              session_id="s", usage={"total_tokens": 5})
    res = WorkflowResult(name="wf", status="completed",
                         results={"a": AgentResult("a", "completed", "ok")})
    await journal.finalize(store, psid, "r1", res)
    j = await journal.read(store, psid, "r1")
    assert j["status"] == "completed"
    assert len(j["steps"]) == 1 and j["steps"][0]["usage"]["total_tokens"] == 5
    assert j["result"]["name"] == "wf"
    assert "r1" in await journal.list_run_ids(store, psid)


async def test_journal_frozen_after_finalize():
    """H1.3: once a run is terminal, a late record_step / plain update must not
    revert its status or result (defense against the C2 orphan-clobber)."""
    loop = _loop()
    psid = await loop.new_session()
    store = loop.store
    await journal.seed(store, psid, "r1", "wf")
    res = WorkflowResult(name="wf", status="completed",
                         results={"a": AgentResult("a", "completed", "ok")})
    await journal.finalize(store, psid, "r1", res)
    assert (await journal.read(store, psid, "r1"))["status"] == "completed"

    # a late orphan step settling after finalize is ignored — status/result frozen
    await journal.record_step(store, psid, "r1", node_id="late", status="running",
                              session_id="s", text="stale")
    j = await journal.read(store, psid, "r1")
    assert j["status"] == "completed"  # NOT reverted to running
    assert j["result"]["name"] == "wf"  # result preserved
    assert all(s["node_id"] != "late" for s in j["steps"])  # late step dropped

    # a plain update is frozen too — first finalize wins
    await journal.update(store, psid, "r1", status="running", result=None)
    assert (await journal.read(store, psid, "r1"))["status"] == "completed"

    # but allow_terminal=True writes (wake / resume) still land
    await journal.update(store, psid, "r1", allow_terminal=True, woke=True)
    after = await journal.read(store, psid, "r1")
    assert after["woke"] is True and after["status"] == "completed"  # status untouched


# ── D3: detached execution ───────────────────────────────────────────────────


async def test_detached_completes_and_journals():
    loop = _loop()
    psid = await loop.new_session()
    wf = create_workflow(SEQ, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True)
    assert isinstance(handle, WorkflowRunHandle)
    await handle.task  # let the background run finish
    j = await get_workflow(loop, psid, handle.run_id, detail=True)
    assert j["status"] == "completed"
    assert {s["node_id"] for s in j["steps"]} >= {"one", "two"}
    assert handle.run_id in [r["run_id"] for r in await list_workflows(loop, psid)]


async def test_detached_requires_parent_session():
    loop = _loop()
    wf = create_workflow(SINGLE, parent_loop=loop)  # no parent_session_id
    with pytest.raises(WorkflowRunError):
        await wf.start(detached=True)


async def test_detached_failure_is_journaled():
    loop = _loop()
    psid = await loop.new_session()
    wf = create_workflow(BAD_REF, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True)
    await handle.task
    j = await get_workflow(loop, psid, handle.run_id)
    assert j["status"] == "failed"
    assert j["error"]


async def test_sync_run_still_returns_result():
    loop = _loop()
    psid = await loop.new_session()
    wf = create_workflow(SINGLE, parent_loop=loop, parent_session_id=psid)
    res = await wf.start(detached=False)
    assert res.status == "completed"


# ── D3: wake + idempotency ───────────────────────────────────────────────────


async def test_wake_delivers_to_idle_parent_once():
    loop = _loop()
    psid = await loop.new_session()
    register_wake_guard(loop)
    wf = create_workflow(SINGLE, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True)
    await handle.task
    assert await loop.list_timers(psid), "completion timer should be armed"
    assert await loop.get_session_stats(psid) is None  # parent not woken yet

    runner = TimerRunner(loop)
    fired = await runner.scan_once()
    assert fired >= 1
    stats = await loop.get_session_stats(psid)
    assert stats is not None and stats.sends == 1  # parent woken exactly once
    assert (await get_workflow(loop, psid, handle.run_id))["woke"] is True

    # the timer is consumed; a second scan does not wake again
    await runner.scan_once()
    assert (await loop.get_session_stats(psid)).sends == 1


async def test_eager_wake_does_not_double_wake():
    """Regression: eager_wake delivers immediately AND arms a durable timer; the
    eager path bypasses TIMER_FIRE, so it must claim 'woke' or the timer fires a
    SECOND time. Parent must wake exactly once."""
    import asyncio

    loop = _loop()
    psid = await loop.new_session()
    register_wake_guard(loop)
    wf = create_workflow(SINGLE, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True, eager_wake=True)
    await handle.task
    # let the fire-and-forget eager follow_up task run
    for _ in range(25):
        await asyncio.sleep(0.02)
        if await loop.get_session_stats(psid) is not None:
            break
    assert (await loop.get_session_stats(psid)).sends == 1  # eager woke once
    assert (await get_workflow(loop, psid, handle.run_id))["woke"] is True  # claimed

    # the durable timer still fires, but the guard now SKIPs it (no 2nd wake)
    await TimerRunner(loop).scan_once()
    assert (await loop.get_session_stats(psid)).sends == 1


async def test_eager_wake_failure_rearms_durable_timer():
    """Regression (H1.4): if the eager follow_up RAISES after claiming the wake, the
    parent would be lost forever (durable timer suppressed by woke=True). The failure
    must re-open the claim and re-arm the timer so the parent still wakes — once."""
    loop = _loop()
    psid = await loop.new_session()
    register_wake_guard(loop)

    orig_follow_up = loop.follow_up
    calls = {"n": 0}

    async def flaky_follow_up(note, sid, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("eager delivery boom")  # only the eager call fails
        return await orig_follow_up(note, sid, **kwargs)

    loop.follow_up = flaky_follow_up  # type: ignore[assignment]

    wf = create_workflow(SINGLE, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True, eager_wake=True)
    await handle.task

    # let the eager follow_up fail and its done-callback re-open the claim
    for _ in range(50):
        await asyncio.sleep(0.02)
        if (await get_workflow(loop, psid, handle.run_id))["woke"] is False:
            break
    assert (await get_workflow(loop, psid, handle.run_id))["woke"] is False  # claim re-opened
    assert await loop.get_session_stats(psid) is None  # eager failed → parent NOT woken yet
    assert await loop.list_timers(psid)  # a durable wake timer is (re-)armed

    # the durable timer now delivers the wake — exactly once
    await TimerRunner(loop).scan_once()
    stats = await loop.get_session_stats(psid)
    assert stats is not None and stats.sends == 1
    await TimerRunner(loop).scan_once()  # idempotent
    assert (await loop.get_session_stats(psid)).sends == 1


async def test_wake_guard_dedupes_rearmed_timer():
    loop = _loop()
    psid = await loop.new_session()
    store = loop.store
    await journal.seed(store, psid, "rX", "wf")
    guard = make_wake_guard(store)
    note = "workflow:run:rX completed"

    ctx1 = TimerFireCtx(session_id=psid, note=note)
    await guard(ctx1)
    assert ctx1.directive == HookDirective.CONTINUE  # first delivery allowed
    assert (await journal.read(store, psid, "rX"))["woke"] is True

    ctx2 = TimerFireCtx(session_id=psid, note=note)
    await guard(ctx2)
    assert ctx2.directive == HookDirective.SKIP  # at-least-once re-arm deduped


async def test_wake_guard_ignores_non_workflow_timer():
    loop = _loop()
    psid = await loop.new_session()
    guard = make_wake_guard(loop.store)
    ctx = TimerFireCtx(session_id=psid, note="just a normal reminder")
    await guard(ctx)
    assert ctx.directive == HookDirective.CONTINUE


# ── events: reuse SYSTEM_LOG, no new enum ────────────────────────────────────


async def test_progress_events_use_system_log_only():
    loop = _loop()
    psid = await loop.new_session()
    events: list = []
    loop.event_bus.subscribe(AgentEventType.SYSTEM_LOG, lambda e: events.append(e))
    wf = create_workflow(SINGLE, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True)
    await handle.task
    wf_events = [e for e in events if e.source == "workflow"]
    assert wf_events
    assert all(e.type == AgentEventType.SYSTEM_LOG for e in wf_events)
    assert all(e.session_id == psid for e in wf_events)


# ── D4: LLM-facing tools ─────────────────────────────────────────────────────


async def test_workflow_status_tool_lists_and_gets():
    loop = _loop()
    psid = await loop.new_session()
    wf = create_workflow(SEQ, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True)
    await handle.task

    tok, stok = set_current_loop(loop), set_session_id(psid)
    try:
        listing = await _handle_workflow_status()
        assert handle.run_id in listing
        detail = await _handle_workflow_status(run_id=handle.run_id, detail=True)
        assert "completed" in detail
        missing = await _handle_workflow_status(run_id="does-not-exist")
        assert "No workflow run" in missing
    finally:
        reset_session_id(stok)
        reset_current_loop(tok)


async def test_create_workflow_tool_detached_returns_runid():
    loop = _loop()
    psid = await loop.new_session()
    tok, stok = set_current_loop(loop), set_session_id(psid)
    try:
        out = await _handle_create_workflow(spec=SINGLE, detached=True)
    finally:
        reset_session_id(stok)
        reset_current_loop(tok)
    assert "Started detached workflow" in out
    # drain the background task so it doesn't outlive the test
    pending = [t for t in asyncio.all_tasks()
               if t is not asyncio.current_task() and t.get_name().startswith("workflow-")]
    if pending:
        await asyncio.gather(*pending)
