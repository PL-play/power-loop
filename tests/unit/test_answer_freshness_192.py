"""A pre-correction answer cannot satisfy a newer user input after empty pass."""
from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, LLMResponse, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store
from .test_pass_turn_builtin import _call, _Script
from .test_workflow_task_runtime_192 import leaf, managed_workflow, records, settle

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("native", [False, True])
async def test_empty_pass_cannot_recover_text_from_before_the_latest_input(tmp_path, native):
    store = await SessionStore.open(str(tmp_path / "fresh-answer.db"), **({"schema": "tasks"} if native else {}))
    try:
        runtime = TaskRuntime(JobManager(store), tenant_id="freshness", principal_id="host",
            executor_id="executor", boot_id="boot") if native else None
        model = _Script(responses=[_call("old", "correct", {}, text="42"),
            _call("pass", "pass_turn", {"reason": "Computed 43"})])
        registry = ToolRegistry()
        async def correct():
            await get_current_loop().follow_up("Correction: return 43.", get_session_id())
            return "Correction queued."
        registry.register(ToolDefinition(name="correct", description="Queue correction", input_schema={"type": "object"}), correct)
        loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
            config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3))
        sid = await loop.new_session()
        result = await loop.send("Return 42, then apply any correction.", sid)
        assert result.status == "completed" and result.passed and model.calls == 2
        assert result.final_text == ""
        assert result.texts[0]["text"] == "42"  # retained progress is not the current answer
        rows = await store.load_all_messages(sid)
        assert any(m.name == "follow_up" and "43" in str(m.content) for m in rows)
    finally:
        await store.close()


async def test_original_node_supplements_new_input_instead_of_delivering_stale_output(managed_workflow, monkeypatch):
    store, _, _, loop, model, sid, state, _ = managed_workflow
    model.hold = set()
    state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["correct"])
    original = model.complete
    requests = []
    async def correct():
        await get_current_loop().follow_up("Correction: the answer is now 43.", get_session_id())
        return "Correction queued."
    loop.tool_registry.register(ToolDefinition(name="correct", description="Queue correction", input_schema={"type": "object"}), correct)
    async def complete(request, **kwargs):
        if "ROLE-BUILD" in (request.system_prompt or ""):
            requests.append(request)
            if len(requests) == 1:
                return _call("old", "correct", {}, text="42")
            if len(requests) == 2:
                return _call("pass", "pass_turn", {"reason": "Computed 43"})
            return LLMResponse(raw_text="43")
        if "ROLE-REVIEW" in (request.system_prompt or ""):
            assert "43" in str(request.messages)
            return LLMResponse(raw_text="43")
        return await original(request, **kwargs)
    monkeypatch.setattr(model, "complete", complete)
    await loop.send("run", sid)
    await settle(managed_workflow)
    jobs = await records(managed_workflow)
    build = next(j for j in jobs if j.kind == "workflow_node" and j.result["node_id"] == "BUILD")
    assert build.outcome == "succeeded" and build.result["text"] == "43"
    assert len(requests) == 3 and build.generation == 1
    assert {t["function"]["name"] for t in requests[-1].tools or []} <= {"pass_turn"}
    assert len([j for j in jobs if j.kind == "workflow"]) == 1
    assert len([j for j in jobs if j.kind == "workflow_node"]) == 2
    assert (await store.inbox_counts(sid))["pending"] == 1
