"""Empty pass is consumed input, not a completed workflow text deliverable."""
from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from power_loop.core.agent_context import get_session_id
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.task_runtime import _OWNER, Execution
from power_loop.runtime.workflow_tasks import ensure_node_answer

from .test_job_registry_192 import registry_store
from .test_workflow_task_runtime_192 import leaf, managed_workflow, settle

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("ending", ["recover", "empty", "provider_error", "repeat_pass", "exhausted", "cancel"])
async def test_text_node_must_answer_on_original_sid_before_review(managed_workflow, monkeypatch, ending):
    store, manager, runtime, loop, model, parent, state, _ = managed_workflow
    model.hold.clear()
    if ending == "exhausted":
        state["spec"]["root"]["steps"][0]["spec"]["max_rounds"] = 1
    complete = model.complete
    requests, sids, guarded = [], [], []
    @contextmanager
    def guard():
        guarded.append("enter")
        try:
            yield
        finally:
            guarded.append("exit")
    loop._child_run_guards = (("node-output", guard),)
    async def responses(request, **kwargs):
        result = await complete(request, **kwargs)
        if "ROLE-BUILD" not in (request.system_prompt or ""):
            return result
        requests.append(request)
        sids.append(get_session_id())
        if len(requests) == 1 or ending == "repeat_pass":
            result.raw_text = ""
            result.tool_calls = [{"id": "pass-"+str(len(requests)), "type": "function", "function": {
                "name": "pass_turn", "arguments": '{"reason":"answer is 42, already replied"}'}}]
        elif ending == "empty":
            result.raw_text, result.tool_calls = "", None
        elif ending == "provider_error":
            raise RuntimeError("node supplement provider failed")
        if ending == "cancel" and len(requests) == 1:
            from power_loop.workflow import runner
            runner._ENGINES[state["handles"][0].run_id]._own_cancel.cancel("stop before answer")
        return result
    monkeypatch.setattr(model, "complete", responses)
    await loop.send("run", parent)
    await settle(managed_workflow)
    nodes = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='workflow_node'")
    build = next(r for r in nodes if json.loads(r["result_json"])["node_id"] == "BUILD")
    result = json.loads(build["result_json"])
    assert len(set(sids)) == 1 and len([r for r in nodes if json.loads(r["result_json"])["node_id"] == "BUILD"]) == 1
    if ending == "recover":
        assert build["outcome"] == "succeeded" and result["text"] == "PROOF-42"
        assert model.counts["REVIEW"] == 1 and len(requests) == 2
        assert result["usage"]["total_tokens"] == 20
    else:
        assert build["outcome"] in ("failed", "cancelled") and model.counts["REVIEW"] == 0
    if ending in ("cancel", "exhausted"):
        # The core's existing round-limit finalizer can make another provider
        # request in the same send; this guard must not grant a second send.
        assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='send' AND session_id=?", (sids[0],))) == 1
    if len(requests) > 1:
        assert {t["function"]["name"] for t in requests[1].tools or []} <= {"pass_turn"}
    assert guarded.count("enter") == guarded.count("exit")
    assert (await manager.get_job(runtime.actor(parent), state["handles"][0].job_id)).outcome == (
        "succeeded" if ending == "recover" else "cancelled" if ending == "cancel" else "failed")


async def test_node_answer_reentry_does_not_grant_another_attempt(managed_workflow, monkeypatch):
    from types import SimpleNamespace

    _, manager, runtime, loop, _, parent, _, _ = managed_workflow
    binding = Execution(runtime, runtime.actor(parent), SimpleNamespace(generation=1))
    token = _OWNER.set(binding)
    calls = []
    async def answer(*args, **kwargs):
        calls.append(1)
        return {**args[3], "status": "failed", "error": "subagent_output_missing"}
    monkeypatch.setattr("power_loop.runtime.subagent_output.ensure_answer", answer)
    raw = {"status": "completed", "final_text": "", "rounds": 1, "session_id": "original-node"}
    try:
        first = await ensure_node_answer(runtime, loop, raw, CancellationToken())
        second = await ensure_node_answer(runtime, loop, raw, CancellationToken())
        assert first["error"] == second["error"] == "node_output_missing" and calls == [1]
    finally:
        _OWNER.reset(token)


async def test_file_delivery_contract_does_not_force_an_extra_text_answer(managed_workflow, monkeypatch):
    _, manager, runtime, loop, model, parent, state, _ = managed_workflow
    model.hold.clear()
    state["spec"]["root"] = leaf("BUILD", output_file="report.md")
    original = model.complete
    async def empty(request, **kwargs):
        result = await original(request, **kwargs)
        if "ROLE-BUILD" in (request.system_prompt or ""):
            result.raw_text = ""
            result.tool_calls = [{"id": "file-pass", "type": "function", "function": {"name": "pass_turn", "arguments": '{"reason":"file written"}'}}]
        return result
    monkeypatch.setattr(model, "complete", empty)
    await loop.send("run", parent)
    await settle(managed_workflow)
    assert model.counts["BUILD"] == 1
    assert (await manager.get_job(runtime.actor(parent), state["handles"][0].job_id)).outcome == "succeeded"


async def test_exhausted_node_has_no_new_answer_allowance(managed_workflow, monkeypatch):
    from types import SimpleNamespace

    _, _, runtime, loop, _, parent, _, _ = managed_workflow
    calls = []
    async def follow(*args, **kwargs):
        calls.append(1)
        raise AssertionError("no fresh allowance")
    async def restore(*args):
        return SimpleNamespace(config=SimpleNamespace(max_rounds=1), follow_up=follow)
    monkeypatch.setattr("power_loop.runtime.subagent_output.workflow_continuation_loop", restore)
    binding = Execution(runtime, runtime.actor(parent), SimpleNamespace(generation=1))
    token = _OWNER.set(binding)
    try:
        value = await ensure_node_answer(runtime, loop, {"status": "completed", "final_text": "", "rounds": 1,
            "session_id": "original"}, CancellationToken())
        assert value["status"] == "failed" and value["error"] == "node_output_missing" and not calls
    finally:
        _OWNER.reset(token)


@pytest.mark.parametrize("mode", ["no_scope", "other_runtime", "later_generation", "nonempty", "failed", "missing_sid", "cancelled", "timeout"])
async def test_node_answer_scope_and_failure_boundaries(managed_workflow, monkeypatch, mode):
    from types import SimpleNamespace

    _, _, runtime, loop, _, parent, _, _ = managed_workflow
    binding = Execution(runtime, runtime.actor(parent), SimpleNamespace(generation=2 if mode == "later_generation" else 1))
    stop = CancellationToken()
    raw = {"status": "failed" if mode == "failed" else "completed", "final_text": "answer" if mode == "nonempty" else "",
        "rounds": 1, "session_id": None if mode == "missing_sid" else "original"}
    if mode == "cancelled":
        stop.cancel("stop")
    calls = []
    async def answer(*args, **kwargs):
        calls.append(1)
        raise TimeoutError()
    monkeypatch.setattr("power_loop.runtime.subagent_output.ensure_answer", answer)
    token = _OWNER.set(None if mode == "no_scope" else binding)
    try:
        value = await ensure_node_answer(object() if mode == "other_runtime" else runtime, loop, raw, stop)
        if mode in ("no_scope", "other_runtime", "later_generation", "nonempty", "failed"):
            assert value is raw and not calls
        else:
            assert value["status"] == ("cancelled" if mode == "cancelled" else "failed")
            assert calls == ([1] if mode == "timeout" else [])
    finally:
        _OWNER.reset(token)


async def test_custom_executor_keeps_its_original_non_text_contract(managed_workflow):
    from power_loop.workflow.engine import InProcessExecutor

    _, manager, runtime, loop, model, parent, state, _ = managed_workflow
    class CustomExecutor(InProcessExecutor):
        async def run_agent(self, *args, **kwargs):
            return {"status": "completed", "final_text": "", "session_id": None, "rounds": 0, "usage": {}}
    state["executor"] = CustomExecutor()
    state["spec"]["root"] = leaf("BUILD")
    await loop.send("run", parent)
    await settle(managed_workflow)
    assert model.counts["BUILD"] == 0
    assert (await manager.get_job(runtime.actor(parent), state["handles"][0].job_id)).outcome == "succeeded"
