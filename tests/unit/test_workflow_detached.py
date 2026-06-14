"""Middle-tier tests: durable journal (D2), detached execution + wake (D3),
introspection (D4). Fake LLM only; no real provider."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import AgentEventType, AgentLoopConfig, StatefulAgentLoop, TimerRunner
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


def test_journal_roundtrip():
    loop = _loop()
    psid = loop.new_session()
    store = loop.store
    journal.seed(store, psid, "r1", "wf")
    journal.record_step(store, psid, "r1", node_id="a", status="completed",
                        session_id="s", usage={"total_tokens": 5})
    res = WorkflowResult(name="wf", status="completed",
                         results={"a": AgentResult("a", "completed", "ok")})
    journal.finalize(store, psid, "r1", res)
    j = journal.read(store, psid, "r1")
    assert j["status"] == "completed"
    assert len(j["steps"]) == 1 and j["steps"][0]["usage"]["total_tokens"] == 5
    assert j["result"]["name"] == "wf"
    assert "r1" in journal.list_run_ids(store, psid)


# ── D3: detached execution ───────────────────────────────────────────────────


async def test_detached_completes_and_journals():
    loop = _loop()
    psid = loop.new_session()
    wf = create_workflow(SEQ, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True)
    assert isinstance(handle, WorkflowRunHandle)
    await handle.task  # let the background run finish
    j = get_workflow(loop, psid, handle.run_id, detail=True)
    assert j["status"] == "completed"
    assert {s["node_id"] for s in j["steps"]} >= {"one", "two"}
    assert handle.run_id in [r["run_id"] for r in list_workflows(loop, psid)]


async def test_detached_requires_parent_session():
    loop = _loop()
    wf = create_workflow(SINGLE, parent_loop=loop)  # no parent_session_id
    with pytest.raises(WorkflowRunError):
        await wf.start(detached=True)


async def test_detached_failure_is_journaled():
    loop = _loop()
    psid = loop.new_session()
    wf = create_workflow(BAD_REF, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True)
    await handle.task
    j = get_workflow(loop, psid, handle.run_id)
    assert j["status"] == "failed"
    assert j["error"]


async def test_sync_run_still_returns_result():
    loop = _loop()
    psid = loop.new_session()
    wf = create_workflow(SINGLE, parent_loop=loop, parent_session_id=psid)
    res = await wf.start(detached=False)
    assert res.status == "completed"


# ── D3: wake + idempotency ───────────────────────────────────────────────────


async def test_wake_delivers_to_idle_parent_once():
    loop = _loop()
    psid = loop.new_session()
    register_wake_guard(loop)
    wf = create_workflow(SINGLE, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True)
    await handle.task
    assert loop.list_timers(psid), "completion timer should be armed"
    assert loop.get_session_stats(psid) is None  # parent not woken yet

    runner = TimerRunner(loop)
    fired = await runner.scan_once()
    assert fired >= 1
    stats = loop.get_session_stats(psid)
    assert stats is not None and stats.sends == 1  # parent woken exactly once
    assert get_workflow(loop, psid, handle.run_id)["woke"] is True

    # the timer is consumed; a second scan does not wake again
    await runner.scan_once()
    assert loop.get_session_stats(psid).sends == 1


def test_wake_guard_dedupes_rearmed_timer():
    loop = _loop()
    psid = loop.new_session()
    store = loop.store
    journal.seed(store, psid, "rX", "wf")
    guard = make_wake_guard(store)
    note = "workflow:run:rX completed"

    ctx1 = TimerFireCtx(session_id=psid, note=note)
    guard(ctx1)
    assert ctx1.directive == HookDirective.CONTINUE  # first delivery allowed
    assert journal.read(store, psid, "rX")["woke"] is True

    ctx2 = TimerFireCtx(session_id=psid, note=note)
    guard(ctx2)
    assert ctx2.directive == HookDirective.SKIP  # at-least-once re-arm deduped


def test_wake_guard_ignores_non_workflow_timer():
    loop = _loop()
    psid = loop.new_session()
    guard = make_wake_guard(loop.store)
    ctx = TimerFireCtx(session_id=psid, note="just a normal reminder")
    guard(ctx)
    assert ctx.directive == HookDirective.CONTINUE


# ── events: reuse SYSTEM_LOG, no new enum ────────────────────────────────────


async def test_progress_events_use_system_log_only():
    loop = _loop()
    psid = loop.new_session()
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
    psid = loop.new_session()
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
    psid = loop.new_session()
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
