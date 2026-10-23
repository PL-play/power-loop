"""Node cancellation stops its own operation, never a shared parent task."""
from __future__ import annotations

import asyncio
import json
from functools import wraps

import pytest

from power_loop import LLMResponse
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import task_control
from power_loop.runtime.jobs import JobError
from power_loop.tools.jobs import register_jobs_tools
from power_loop.workflow import runner
from power_loop.workflow.facade import register_workflow_tool

from .test_job_registry_192 import registry_store
from .test_workflow_control_192 import invoke, legacy
from .test_workflow_restore_runtime_192 import native_restore
from .test_workflow_task_runtime_192 import leaf, managed_workflow

__all__ = ["registry_store", "managed_workflow", "native_restore", "legacy"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def issue(loop, model, sid, monkeypatch, name, arguments):
    original, issued = model.complete, False
    async def control(request, **kwargs):
        nonlocal issued
        if get_session_id() == sid and not issued:
            issued = True
            return LLMResponse(raw_text="", tool_calls=[{"id": "node-control", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)}}])
        return await original(request, **kwargs)
    monkeypatch.setattr(model, "complete", control)
    await loop.send("Stop only the selected node", sid)
    assert issued
    store = await loop._ensure_store()
    records = [m for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == name]
    return records[-1], json.loads(records[-1].content)


@pytest.mark.parametrize("tool", ["workflow", "jobs"])
@pytest.mark.parametrize("layout", ["sequence", "parallel"])
async def test_cancel_exact_node_preserves_parent_and_sibling_after_heartbeat(managed_workflow, monkeypatch, tool, layout):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    model.hold = {"BUILD", "REVIEW"} if layout == "sequence" else {"BUILD", "AUDIT"}
    if layout == "parallel":
        state["spec"]["root"] = {"type": "sequence", "steps": [
            {"type": "parallel", "branches": [leaf("BUILD"), leaf("AUDIT")]}, leaf("REVIEW", inputs_from=["AUDIT"])]}
    await loop.send("Start original workflow", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    if layout == "parallel":
        await asyncio.wait_for(model.entered["AUDIT"].wait(), 3)
    [handle] = state["handles"]
    [node] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='workflow_node' AND title='BUILD'")
    node_locals = [v for v in runtime._local_executions.values() if v["binding"].lease.job_id == node["job_id"]]
    assert node_locals and node_locals[0]["cancel_token"] is not None
    if layout == "sequence":
        assert node_locals[0]["task"] is handle.task
    register_workflow_tool(loop.tool_registry)
    register_jobs_tools(loop.tool_registry)
    args = {"action": "cancel", "run_id": handle.run_id, "node": "BUILD", "expected_generation": 1} if tool == "workflow" else {
        "action": "stop", "job_id": node["job_id"], "expected_generation": 1}
    raw, receipt = await issue(loop, model, sid, monkeypatch, tool, args)
    assert receipt["job_id"] == node["job_id"] and receipt["cancel_requested"]
    assert receipt["control_pending"] and receipt["resource_stop_confirmed"] is False
    operation = await manager.get_operation(runtime.actor(sid), receipt["operation_id"])
    assert operation.receipt_ref["id"] == str(raw.id) and operation.call_ref["id"] != operation.receipt_ref["id"]
    if layout == "sequence":
        await asyncio.wait_for(model.entered["REVIEW"].wait(), 3)
    await asyncio.sleep(.8)
    assert not handle.task.done() and not handle.task.cancelling()
    workflow = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert workflow.cancel_requested_at is None
    cancelled = await manager.get_job(runtime.actor(sid), node["job_id"])
    assert cancelled.outcome == "cancelled" and cancelled.resource_state == "unknown"
    assert cancelled.supervision_status == "open"
    if layout == "parallel":
        [other] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='workflow_node' AND title='AUDIT'")
        assert other["cancel_requested_at"] is None and other["outcome"] is None
        model.release["AUDIT"].set()
    else:
        model.release["REVIEW"].set()
    await asyncio.wait_for(handle.task, 5)
    final = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert final.outcome == "failed" and final.resource_state == "unknown" and final.supervision_status == "open", final
    assert final.result["results"]["REVIEW"]["status"] == "completed" and "BUILD" in final.result["tolerated"]
    assert model.counts["REVIEW"] == 1
    assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")) == 1


async def test_heartbeat_observes_persistent_node_cancel_without_local_notification(managed_workflow, monkeypatch):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    # Leave enough time for real PostgreSQL admission transactions; cancellation
    # is still observed by heartbeat alone within this short test lease.
    runtime.lease_ms = 1500
    model.hold = {"BUILD", "REVIEW"}
    await loop.send("Start original", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    [handle] = state["handles"]
    register_workflow_tool(loop.tool_registry)
    async def no_signal(*args):
        return 0
    monkeypatch.setattr(task_control, "signal_local", no_signal)
    _, receipt = await issue(loop, model, sid, monkeypatch, "workflow", {"action": "cancel", "run_id": handle.run_id, "node": "BUILD"})
    assert receipt["local_signals"] == 0
    await asyncio.wait_for(model.entered["REVIEW"].wait(), 4)
    assert not handle.task.cancelling() and not handle.task.done()
    model.release["REVIEW"].set()
    await asyncio.wait_for(handle.task, 5)
    final = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert final.outcome == "failed" and final.resource_state == "unknown" and final.supervision_status == "open"
    assert final.result["results"]["REVIEW"]["status"] == "completed"


@pytest.mark.parametrize("mode", ["exact_job", "missing", "outside", "both", "blank", "old_generation"])
async def test_node_selector_is_exact_and_invalid_selection_never_stops_whole_run(managed_workflow, monkeypatch, mode):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    model.hold = {"BUILD", "REVIEW"}
    await loop.send("Launch", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    [handle] = state["handles"]
    [node] = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")
    args = {"action": "cancel", "run_id": handle.run_id}
    choices = {
        "exact_job": ({"node_job_id": node["job_id"]}, None),
        "missing": ({"node": "never-started"}, "workflow_node_not_found"),
        "outside": ({"node_job_id": handle.job_id}, "workflow_node_outside_run"),
        "both": ({"node": "BUILD", "node_job_id": node["job_id"]}, "ambiguous_workflow_node_selector"),
        "blank": ({"node": " "}, "invalid_workflow_node"),
        "old_generation": ({"node": "BUILD", "expected_generation": 0}, "stop_generation_changed"),
    }
    selector, error = choices[mode]
    register_workflow_tool(loop.tool_registry)
    _, receipt = await issue(loop, model, sid, monkeypatch, "workflow", {**args, **selector})
    assert (await manager.get_job(runtime.actor(sid), handle.job_id)).cancel_requested_at is None
    if error:
        assert receipt["error"]["code"] == error
        assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")
        assert not handle.task.done() and not model.entered["REVIEW"].is_set()
        model.release["BUILD"].set()
    else:
        assert receipt["job_id"] == node["job_id"] and receipt["local_signals"] > 0
    await asyncio.wait_for(model.entered["REVIEW"].wait(), 3)
    model.release["REVIEW"].set()
    await asyncio.wait_for(handle.task, 5)
    final = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert final.outcome == ("failed" if not error else "succeeded")
    assert final.result["results"]["REVIEW"]["status"] == "completed"


@pytest.mark.parametrize("fault", ["same_call_replay", "commit_ack"])
async def test_node_stop_replays_one_original_intent_and_receipt(managed_workflow, monkeypatch, fault):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    await loop.send("Launch", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    [handle] = state["handles"]
    register_workflow_tool(loop.tool_registry)
    if fault == "commit_ack":
        original = task_control.request_stop
        async def lost(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError("stop committed but acknowledgement lost")
        monkeypatch.setattr(task_control, "request_stop", lost)
    else:
        handler = loop.tool_registry.get("workflow").handler
        @wraps(handler)
        async def repeat(**kwargs):
            first = json.loads(await handler(**kwargs))
            second = json.loads(await handler(**kwargs))
            assert first["operation_id"] == second["operation_id"]
            return json.dumps(second)
        loop.tool_registry.register(loop.tool_registry.get("workflow").definition, repeat, overwrite=True)
    _, receipt = await issue(loop, model, sid, monkeypatch, "workflow", {"action": "cancel", "run_id": handle.run_id, "node": "BUILD"})
    assert receipt["replayed"]
    await asyncio.wait_for(handle.task, 5)
    final = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert final.outcome == "failed" and final.resource_state == "unknown" and final.supervision_status == "open"
    assert final.result["results"]["REVIEW"]["status"] == "completed"
    assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")) == 1


@pytest.mark.parametrize("reverse", [False, True])
async def test_queued_node_stop_releases_its_wait_without_running_provider_or_sibling(managed_workflow, monkeypatch, reverse):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    branches = [leaf("BUILD"), leaf("AUDIT")]
    if reverse:
        branches.reverse()
    state["spec"]["root"] = {"type": "sequence", "steps": [{"type": "parallel", "max_concurrency": 1,
        "concurrency_scope": "leaf", "branches": branches}, leaf("REVIEW")]}
    model.hold = {"BUILD", "REVIEW"}
    # Native registration/claim awaits the DB before competing for a leaf slot.
    # Parallel branch order therefore cannot prove AUDIT is still queued. Start
    # the actual slot holder first, without replacing admission or cancellation.
    register = manager.register_job
    async def holder_first(actor, **options):
        if options.get("kind") == "workflow_node" and options.get("title") == "AUDIT":
            await model.entered["BUILD"].wait()
        return await register(actor, **options)
    monkeypatch.setattr(manager, "register_job", holder_first)
    await loop.send("Launch", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    [handle] = state["handles"]
    engine = runner._ENGINES[handle.run_id]
    async def queued():
        while True:
            rows = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node' AND title='AUDIT'")
            if rows and engine._admission_tokens.get("AUDIT"):
                return rows[0]
            await asyncio.sleep(.01)
    node = await asyncio.wait_for(queued(), 3)
    assert model.counts["AUDIT"] == 0 and not handle.task.done()
    register_workflow_tool(loop.tool_registry)
    _, receipt = await issue(loop, model, sid, monkeypatch, "workflow", {"action": "cancel", "run_id": handle.run_id, "node_job_id": node["job_id"]})
    assert receipt["cancel_requested"] and receipt["job_id"] == node["job_id"]
    async def stopped():
        while (await manager.get_job(runtime.actor(sid), node["job_id"])).outcome is None:
            await asyncio.sleep(.01)
    await asyncio.wait_for(stopped(), 3)
    assert model.counts["AUDIT"] == 0 and not handle.task.done()
    model.release["BUILD"].set()
    await asyncio.wait_for(model.entered["REVIEW"].wait(), 3)
    model.release["REVIEW"].set()
    await asyncio.wait_for(handle.task, 5)
    assert (await manager.get_job(runtime.actor(sid), handle.job_id)).outcome == "succeeded"


async def test_node_heartbeat_backend_failure_does_not_become_tolerated_user_cancellation(managed_workflow, monkeypatch):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    runtime.lease_ms = 1500
    await loop.send("Launch", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    [handle] = state["handles"]
    [node] = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")
    original = manager.renew_execution
    async def unavailable(actor, lease, **kwargs):
        if lease.job_id == node["job_id"]:
            raise JobError("backend_unavailable")
        return await original(actor, lease, **kwargs)
    monkeypatch.setattr(manager, "renew_execution", unavailable)
    await asyncio.wait_for(handle.task, 5)
    workflow = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert workflow.outcome == "failed" and workflow.cancel_requested_at is None
    assert model.counts["REVIEW"] == 0
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")


async def test_node_stop_winning_before_provider_admission_stays_scoped(managed_workflow, monkeypatch):
    from power_loop.workflow.engine import WorkflowEngine

    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    entered, release = asyncio.Event(), asyncio.Event()
    original = WorkflowEngine._exec_agent_body
    async def before_dispatch(engine, node, *args, **kwargs):
        if node.id == "BUILD":
            entered.set()
            await release.wait()
        return await original(engine, node, *args, **kwargs)
    monkeypatch.setattr(WorkflowEngine, "_exec_agent_body", before_dispatch)
    await loop.send("Launch", sid)
    await asyncio.wait_for(entered.wait(), 3)
    [handle] = state["handles"]
    register_workflow_tool(loop.tool_registry)
    try:
        _, receipt = await issue(loop, model, sid, monkeypatch, "workflow", {
            "action": "cancel", "run_id": handle.run_id, "node": "BUILD"})
        assert receipt["cancel_requested"] and not handle.task.cancelling()
    finally:
        release.set()
    await asyncio.wait_for(handle.task, 5)
    assert model.counts["BUILD"] == 0 and model.counts["REVIEW"] == 1
    assert (await manager.get_job(runtime.actor(sid), handle.job_id)).outcome == "succeeded"


async def test_cancelled_node_does_not_start_fallback_or_retry_business_work(managed_workflow, monkeypatch):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    state["spec"]["root"]["steps"][0].update(fallback=leaf("FALLBACK"), retry={"max_attempts": 2, "on": ["failed"]})
    await loop.send("Launch", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    [handle] = state["handles"]
    register_workflow_tool(loop.tool_registry)
    _, receipt = await issue(loop, model, sid, monkeypatch, "workflow", {
        "action": "cancel", "run_id": handle.run_id, "node": "BUILD"})
    assert receipt["cancel_requested"]
    await asyncio.wait_for(handle.task, 5)
    assert model.counts["BUILD"] == 1 and model.counts["FALLBACK"] == 0 and model.counts["REVIEW"] == 1
    final = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert final.outcome == "failed" and final.resource_state == "unknown" and final.supervision_status == "open"
    assert final.result["results"]["REVIEW"]["status"] == "completed"


async def test_settling_node_stop_keeps_original_result_and_does_not_cancel_review(managed_workflow, monkeypatch):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    model.hold = {"REVIEW"}
    await loop.send("Launch", sid)
    await asyncio.wait_for(model.entered["REVIEW"].wait(), 3)
    [handle] = state["handles"]
    [row] = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node' AND title='BUILD'")
    before = await manager.get_job(runtime.actor(sid), row["job_id"])
    assert before.outcome == "succeeded" and before.resource_state == "stopped"
    register_workflow_tool(loop.tool_registry)
    _, receipt = await issue(loop, model, sid, monkeypatch, "workflow", {
        "action": "cancel", "run_id": handle.run_id, "node_job_id": row["job_id"]})
    assert receipt["local_signals"] == 0 and receipt["resource_stop_confirmed"] is False
    after = await manager.get_job(runtime.actor(sid), row["job_id"])
    assert after.result == before.result and after.outcome == before.outcome
    assert not handle.task.done() and not handle.task.cancelling()
    model.release["REVIEW"].set()
    await asyncio.wait_for(handle.task, 5)
    assert (await manager.get_job(runtime.actor(sid), handle.job_id)).outcome == "succeeded"


async def test_shared_node_name_requires_exact_route_job_and_preserves_other_iteration(managed_workflow, monkeypatch):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    state["spec"]["root"] = {"type": "foreach", "id": "each", "as": "item", "items": ["left", "right"],
        "parallel": True, "max_concurrency": 2, "body": leaf("BUILD")}
    await loop.send("Launch", sid)
    [handle] = state["handles"]
    async def entered_both():
        while model.counts["BUILD"] < 2:
            await asyncio.sleep(.01)
    await asyncio.wait_for(entered_both(), 3)
    nodes = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")
    assert len(nodes) == 2
    register_workflow_tool(loop.tool_registry)
    _, invalid = await issue(loop, model, sid, monkeypatch, "workflow", {"action": "cancel", "run_id": handle.run_id, "node": "BUILD"})
    assert invalid["error"]["code"] == "workflow_node_ambiguous_choose_job"
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='stop'")
    _, receipt = await issue(loop, model, sid, monkeypatch, "workflow", {
        "action": "cancel", "run_id": handle.run_id, "node_job_id": nodes[0]["job_id"]})
    assert receipt["job_id"] == nodes[0]["job_id"]
    await asyncio.sleep(.8)
    target = await manager.get_job(runtime.actor(sid), nodes[0]["job_id"])
    sibling = await manager.get_job(runtime.actor(sid), nodes[1]["job_id"])
    assert target.outcome == "cancelled" and sibling.outcome is None and sibling.cancel_requested_at is None
    assert not handle.task.done() and not handle.task.cancelling()
    model.release["BUILD"].set()
    await asyncio.wait_for(handle.task, 5)
    assert (await manager.get_job(runtime.actor(sid), nodes[1]["job_id"])).outcome == "succeeded"


async def test_cancel_with_original_generation_cannot_stop_the_restored_original_node(native_restore, monkeypatch):
    c = native_restore
    c.model.release["BUILD"].clear()
    c.model.entered["BUILD"].clear()
    handle = await c.restore()
    await asyncio.wait_for(c.model.entered["BUILD"].wait(), 4)
    register_workflow_tool(c.loop.tool_registry)
    _, receipt = await issue(c.loop, c.model, c.parent, monkeypatch, "workflow", {
        "action": "cancel", "run_id": handle.run_id, "node": "BUILD", "expected_generation": 1})
    assert receipt["error"]["code"] == "stop_generation_changed"
    assert not handle.task.done() and not handle.task.cancelling()
    [node] = [n for n in c.node_jobs if n["title"] == "BUILD"]
    current = await c.manager.get_job(c.actor, node["job_id"])
    assert current.generation == 2 and current.cancel_requested_at is None
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='stop'")
    c.model.release["BUILD"].set()
    await asyncio.wait_for(handle.task, 5)
    assert (await c.manager.get_job(c.actor, handle.job_id)).outcome == "succeeded"


@pytest.mark.parametrize("fault", ["none", "global_collision", "missing_engine", "missing_handle", "foreign_owner", "generation", "job_selector", "blank"])
async def test_legacy_node_control_uses_only_proven_handle_without_cancelling_parent(legacy, monkeypatch, fault):
    store, loop, model, sid, handle, accepted, notified = legacy
    run_id = accepted["run_id"]
    options = {"action": "cancel", "run_id": run_id, "node": "calculate"}
    if fault == "global_collision":
        class WrongEngine:
            def cancel_node(self, *args):
                pytest.fail("global engine ID lookup controlled the wrong engine")
        monkeypatch.setitem(runner._ENGINES, run_id, WrongEngine())
    elif fault == "missing_engine":
        monkeypatch.delattr(handle, "_origin_engine")
    elif fault == "missing_handle":
        monkeypatch.delitem(runner._HANDLES, run_id)
    elif fault == "foreign_owner":
        monkeypatch.setattr(handle, "_origin_parent_sid", "other")
    elif fault == "generation":
        options["expected_generation"] = 1
    elif fault == "job_selector":
        del options["node"]
        options["node_job_id"] = "job_unknown"
    elif fault == "blank":
        options["node"] = " "
    result = await invoke(loop, sid, options)
    assert not handle.cancel_token.is_cancelled() and run_id not in runner._STOPPED_RUNS
    if fault in ("none", "global_collision"):
        assert result["status"] == "accepted" and result["local_signals"] > 0
        assert not result["durable_intent"] and not result["resource_stop_confirmed"]
    elif fault in ("missing_engine", "missing_handle"):
        assert result["status"] == "control_unavailable" and result["resource_state"] == "unknown"
    else:
        assert result["error"]["code"] == {"foreign_owner": "workflow_control_owner_mismatch",
            "generation": "legacy_generation_unavailable", "job_selector": "legacy_node_job_unavailable", "blank": "invalid_workflow_node"}[fault]
    model.release.set()
    await asyncio.wait_for(handle.task, 5)
    assert len(notified) == 1 and (await runner.journal.read(store, sid, run_id))["status"] == "completed"
