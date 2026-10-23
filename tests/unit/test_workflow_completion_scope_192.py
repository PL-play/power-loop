"""One owning engine window across node checks; no paid budget extension."""
from __future__ import annotations

import pytest

from power_loop.core.agent_context import get_session_id
from power_loop.runtime import workflow_tasks
from power_loop.runtime.completion import WINDOW
from power_loop.workflow.engine import InProcessExecutor

from .test_job_registry_192 import registry_store
from .test_workflow_task_runtime_192 import leaf, managed_workflow, records, settle

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_all_checks_in_original_node_reuse_one_window_and_other_nodes_do_not(managed_workflow, monkeypatch):
    _, _, _, loop, _, sid, _, _ = managed_workflow
    original, seen = workflow_tasks.collect_node, {}
    async def collect(*args, **kwargs):
        node = workflow_tasks._NODE.get()
        seen.setdefault(node.binding.lease.job_id, []).append(WINDOW.get())
        return await original(*args, **kwargs)
    monkeypatch.setattr(workflow_tasks, "collect_node", collect)
    await loop.send("run", sid)
    await settle(managed_workflow)
    assert len(seen) == 2
    for windows in seen.values():
        assert len(windows) >= 3 and windows[0] is not None
        assert all(window is windows[0] for window in windows)
    first, second = [windows[0] for windows in seen.values()]
    assert first is not second and WINDOW.get() is None


async def test_late_input_cannot_buy_more_rounds_than_original_node_contract(managed_workflow):
    store, _, _, loop, model, sid, state, _ = managed_workflow
    state["spec"]["root"]["steps"][0]["spec"]["max_rounds"] = 1
    class LateInput(InProcessExecutor):
        async def run_agent(self, spec, user_input, **kwargs):
            raw = await super().run_agent(spec, user_input, **kwargs)
            if spec.name == "BUILD":
                assert raw["rounds"] == 1
                await store.append_message(raw["session_id"], role="user", content="new input after original final round")
                raw["new_input_after_pass"] = True
            return raw
    state["executor"] = LateInput()
    await loop.send("run", sid)
    await settle(managed_workflow)
    workflow = next(job for job in await records(managed_workflow) if job.kind == "workflow")
    assert workflow.outcome == "failed" and model.counts["BUILD"] == 1 and model.counts["REVIEW"] == 0
    assert workflow.result["usage"]["total_tokens"] == 10


@pytest.mark.parametrize("policy", ["retry", "fallback", "continuation"])
async def test_unprocessed_result_never_restarts_business_or_uses_an_automatic_escape_policy(managed_workflow, monkeypatch, policy):
    _, manager, _, loop, model, sid, state, _ = managed_workflow
    model.hold, model.spawn_from_build = {"AUDIT"}, True
    node = leaf("BUILD", tools=["spawn_agent"])
    if policy == "retry":
        node["retry"] = {"max_attempts": 3, "on": ["failed", "hit_round_limit"]}
    elif policy == "fallback":
        node["fallback"] = leaf("FALLBACK")
    else:
        node["continuation"] = {"max_continuations": 2, "extra_rounds": 1, "gate": "always"}
    state["spec"]["root"]["steps"][0] = node
    original = workflow_tasks.collect_node
    async def exhausted(*args, **kwargs):
        return await original(*args, **kwargs, max_rounds=0)
    monkeypatch.setattr(workflow_tasks, "collect_node", exhausted)
    await loop.send("run", sid)
    await settle(managed_workflow)
    workflow = next(job for job in await records(managed_workflow) if job.kind == "workflow")
    assert workflow.outcome == "failed"
    assert model.counts["BUILD"] == 2 and model.counts["FALLBACK"] == model.counts["REVIEW"] == 0
    bindings = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='node_session'")
    assert len(bindings) == 1
    audit = next(job for job in await records(managed_workflow) if job.kind == "subagent")
    assert audit.result_status != "handled"


async def test_final_answer_persistence_failure_cannot_execute_business_fallback(managed_workflow, monkeypatch):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    model.hold = set()
    state["spec"]["root"]["steps"][0]["fallback"] = leaf("FALLBACK")
    original_complete = model.complete
    async def complete(request, **kwargs):
        result = await original_complete(request, **kwargs)
        if get_session_id() != sid and "ROLE-BUILD" in (request.system_prompt or "") and model.counts["BUILD"] == 1:
            result.raw_text = ""
            result.tool_calls = [{"id": "initial-pass", "type": "function", "function": {
                "name": "pass_turn", "arguments": '{"reason":"await final answer"}'}}]
        return result
    monkeypatch.setattr(model, "complete", complete)
    append = store.append_message
    async def broken(session_id, **kwargs):
        if session_id != sid and kwargs.get("role") == "assistant" and model.counts["BUILD"] >= 2:
            raise RuntimeError("final assistant persistence unavailable")
        return await append(session_id, **kwargs)
    monkeypatch.setattr(store, "append_message", broken)
    await loop.send("run", sid)
    await settle(managed_workflow)
    assert model.counts["BUILD"] == 2 and model.counts["FALLBACK"] == model.counts["REVIEW"] == 0
    workflow = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    assert workflow.outcome == "failed"
    bindings = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='node_session'")
    assert len(bindings) == 1
