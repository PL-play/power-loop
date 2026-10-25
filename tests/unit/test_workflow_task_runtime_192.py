"""Real workflow/loop/store adapters, scope ownership and failure boundaries."""
from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime import delegation
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow import create_workflow, journal
from power_loop.workflow import runner as workflow_runner

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime

__all__ = ["registry_store", "registry_runtime"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def leaf(name, *, tools=None, **kwargs):
    return {"type": "agent", "id": name, "input": name,
            "spec": {"name": name, "system_prompt": "ROLE-" + name,
                     "tools": tools or [], "max_rounds": 3}, **kwargs}


class WorkflowModel(LLMService):
    def __init__(self):
        self.root = None
        self.counts = Counter()
        self.entered = defaultdict(asyncio.Event)
        self.release = defaultdict(asyncio.Event)
        self.hold = {"BUILD"}
        self.spawn_from_build = False
        self.ending = "natural"

    async def complete(self, request, **kwargs):
        sid = get_session_id()
        role = "MAIN" if sid == self.root else next((r for r in ("BUILD", "REVIEW", "AUDIT", "FALLBACK")
                                                     if "ROLE-" + r in (request.system_prompt or "")), "OTHER")
        self.counts[role] += 1
        self.entered[role].set()
        if role in self.hold:
            await self.release[role].wait()
        text, calls = "PROOF-42", None
        if role == "MAIN" and self.counts[role] == 1:
            text, calls = "", [{"id": "wf-call", "type": "function", "function": {"name": "launch", "arguments": "{}"}}]
        elif role == "MAIN" and self.counts[role] == 2 and self.ending == "pass_turn":
            text, calls = "", [{"id": "yield-call", "type": "function", "function": {
                "name": "pass_turn", "arguments": '{"reason":"waiting for workflow"}'}}]
        elif role == "BUILD" and self.spawn_from_build and self.counts[role] == 1:
            text, calls = "", [{"id": "audit-call", "type": "function", "function": {"name": "spawn_agent", "arguments": json.dumps({
                "name": "audit", "system_prompt": "ROLE-AUDIT", "tools": [], "task": "audit", "max_rounds": 2})}}]
        return LLMResponse(raw_text=text, tool_calls=calls, token_usage=LLMTokenUsage(
            prompt_tokens=7, completion_tokens=3, total_tokens=10))

    def stream(self, request):
        async def empty():
            if False:
                yield None
        return empty()

    async def close(self):
        pass


@pytest.fixture
async def managed_workflow(registry_store):
    store, connect = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="workflow-adapter-test", principal_id="host",
                          executor_id="executor", boot_id="boot", lease_ms=2000)
    model = WorkflowModel()
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=4))
    sid = await loop.new_session()
    model.root = sid
    state = {"spec": {"name": "registered-workflow", "root": {"type": "sequence", "steps": [leaf("BUILD"), leaf("REVIEW")] }},
             "handles": [], "notifications": [], "workflows": [], "repeat": False}

    async def notified(completion):
        job = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
        assert job.result is not None and job.result_status == "pending_delivery"
        assert (await store.inbox_counts(sid))["pending"] >= 1
        state["notifications"].append(completion)

    async def launch():
        workflow = create_workflow(state["spec"], parent_loop=get_current_loop(), parent_session_id=get_session_id(), executor=state.get("executor"))
        state["workflows"].append(workflow)
        handle = await workflow.start(detached=True, on_complete=state.get("on_complete", notified))
        state["handles"].append(handle)
        if state["repeat"]:
            second = await workflow.start(detached=True, on_complete=state.get("on_complete", notified))
            assert second is handle
        return json.dumps({"run_id": handle.run_id, "job_id": handle.job_id, "operation_id": handle.operation_id, "status": "queued"})

    registry.register(ToolDefinition(name="launch", description="Start a workflow", input_schema={"type": "object"}), launch)
    before = set(delegation._TASKS)
    try:
        yield store, manager, runtime, loop, model, sid, state, connect
    finally:
        for event in model.release.values():
            event.set()
        for role in model.hold:
            model.release[role].set()
        tasks = [h.task for h in state["handles"]] + list(set(delegation._TASKS) - before)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 8)


async def records(fixture):
    _, manager, runtime, _, _, sid, _, _ = fixture
    rows = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs}")
    return [await manager.get_job(runtime.actor(sid), row["job_id"]) for row in rows]


async def settle(fixture):
    _, _, _, _, model, _, state, _ = fixture
    for role in model.hold:
        model.release[role].set()
    await asyncio.wait_for(asyncio.gather(*(h.task for h in state["handles"])), 8)


@pytest.mark.parametrize("ending", ["natural", "pass_turn"])
async def test_workflow_and_nodes_outlive_creator_and_keep_real_receipt_result_and_usage(managed_workflow, ending):
    store, manager, runtime, loop, model, sid, state, connect = managed_workflow
    model.ending = ending
    first = await loop.send("run workflow", sid)
    assert first.status == "completed" and first.passed == (ending == "pass_turn")
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    [handle] = state["handles"]
    actor = runtime.actor(sid)
    workflow = await manager.get_job(actor, handle.job_id)
    assert workflow.status == "running" and workflow.control_owner_job_id is None
    assert (await manager.get_job(actor, workflow.parent_job_id)).status == "model_closed"
    assert model.counts["REVIEW"] == 0 and not state["notifications"]
    operation = await manager.get_operation(actor, handle.operation_id)
    assert operation.call_ref["id"] != operation.receipt_ref["id"]
    assert (await store.message_by_id(sid, operation.receipt_ref["id"])).tool_call_id == "wf-call"
    await settle(managed_workflow)
    restored = JobManager(await connect())
    workflow = await restored.get_job(actor, handle.job_id)
    assert workflow.status == "settling" and workflow.resource_state == "stopped" and workflow.outcome == "succeeded", workflow.result
    assert workflow.result["usage"]["total_tokens"] == 20  # excludes creator's 20 tokens
    nodes = [r for r in await records(managed_workflow) if r.kind == "workflow_node"]
    assert len(nodes) == 2 and all(n.result["usage"]["total_tokens"] == 10 for n in nodes)
    assert all(n.parent_job_id == workflow.job_id and n.control_owner_job_id == workflow.job_id for n in nodes)
    assert await store.get_session(workflow.session_id) is None  # driver was deliberately ephemeral
    bindings = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE event_type='node_session'")
    assert len(bindings) == 2
    for event in bindings:
        session = await store.get_session(json.loads(event["payload_json"])["session_id"])
        assert session.parent_session_id is None and session.metadata["execution_role"] == "workflow_node"
    assert len(state["notifications"]) == 1
    assert (await loop.flush_follow_ups(sid)).status == "completed"


async def test_replayed_actual_create_never_starts_a_second_workflow(managed_workflow):
    _, _, _, loop, model, sid, state, _ = managed_workflow
    state["repeat"] = True
    assert (await loop.send("run once", sid)).status == "completed"
    await settle(managed_workflow)
    assert len([r for r in await records(managed_workflow) if r.kind == "workflow"]) == 1
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 1


async def test_cancel_only_creator_send_keeps_independent_workflow_running(managed_workflow, monkeypatch):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    original = model.complete
    accepted = asyncio.Event()

    async def hold_parent(request, **kwargs):
        if get_session_id() == sid and model.counts["MAIN"] == 1:
            accepted.set()
            await asyncio.Event().wait()
        return await original(request, **kwargs)

    monkeypatch.setattr(model, "complete", hold_parent)
    sending = asyncio.create_task(loop.send("run", sid))
    try:
        await asyncio.wait_for(accepted.wait(), 3)
        sending.cancel()
        await asyncio.gather(sending, return_exceptions=True)
        wf = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
        assert wf.cancel_requested_at is None
        assert (await manager.get_job(runtime.actor(sid), wf.parent_job_id)).status == "model_closed"
        await settle(managed_workflow)
        assert (await manager.get_job(runtime.actor(sid), wf.job_id)).outcome == "succeeded"
    finally:
        if not sending.done():
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)


async def test_default_managed_completion_keeps_durable_inbox_without_implicitly_starting_parent_model(managed_workflow):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    state["on_complete"] = None
    await loop.send("run", sid)
    await settle(managed_workflow)
    saved = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    assert saved.result_status == "pending_delivery" and saved.result["status"] == "completed"
    assert model.counts["MAIN"] == 2 and not state["notifications"]
    assert (await store.inbox_counts(sid))["pending"] == 1
    assert await store.list_timers(sid) == []
    await loop.flush_follow_ups(sid)
    assert model.counts["MAIN"] == 3


async def test_skipped_node_keeps_last_executed_result_and_creates_no_model_session(managed_workflow):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    state["spec"]["root"]["steps"][1]["when"] = {"ref": "BUILD", "op": "eq", "value": "never"}
    await loop.send("run", sid)
    await settle(managed_workflow)
    wf = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    assert wf.outcome == "succeeded" and wf.result["final"]["node_id"] == "BUILD"
    assert wf.result["results"]["REVIEW"]["status"] == "skipped" and model.counts["REVIEW"] == 0
    assert wf.result["usage"]["total_tokens"] == 10


async def test_workflow_executes_the_frozen_admission_contract_when_original_metadata_changes(managed_workflow, monkeypatch):
    store, manager, runtime, loop, _, sid, state, _ = managed_workflow
    state["spec"]["root"]["steps"][0]["spec"]["metadata"] = {"frozen_marker": "original"}
    original = journal.seed

    async def mutate_after_admission(*args, **kwargs):
        state["workflows"][0].spec.root.steps[0].spec.metadata["frozen_marker"] = "changed"
        return await original(*args, **kwargs)

    monkeypatch.setattr(journal, "seed", mutate_after_admission)
    await loop.send("run", sid)
    await settle(managed_workflow)
    node = next(r for r in await records(managed_workflow) if r.kind == "workflow_node" and r.title == "BUILD")
    session = await store.get_session(node.result["session_id"])
    assert session.metadata["frozen_marker"] == "original"
    operation = await manager.get_operation(runtime.actor(sid), state["handles"][0].operation_id)
    assert operation.result["request"]["spec"]["root"]["steps"][0]["spec"]["metadata"]["frozen_marker"] == "original"


async def test_node_child_results_are_processed_before_downstream_and_aggregate_usage_once(managed_workflow):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    model.hold = {"AUDIT"}
    model.spawn_from_build = True
    state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["spawn_agent"])
    assert (await loop.send("run with audit", sid)).status == "completed"
    await asyncio.wait_for(model.entered["AUDIT"].wait(), 3)
    assert model.counts["REVIEW"] == 0 and not state["notifications"]
    rows = await records(managed_workflow)
    [audit] = [r for r in rows if r.kind == "subagent"]
    [build] = [r for r in rows if r.kind == "workflow_node"]
    assert audit.control_owner_job_id == audit.completion_scope_job_id == build.job_id
    # The child can enter before its creator's model phase has actually exited.
    # Wait for that independent event while the child stays held and downstream
    # remains blocked; observing child entry alone is not an exit barrier.
    async with asyncio.timeout(3):
        while (await manager.get_job(runtime.actor(sid), audit.parent_job_id)).status != "model_closed":
            assert model.counts["REVIEW"] == 0 and not model.release["AUDIT"].is_set()
            await asyncio.sleep(0.01)
    assert (await manager.get_job(runtime.actor(sid), audit.parent_job_id)).status == "model_closed"
    await settle(managed_workflow)
    workflow = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    build = await manager.get_job(runtime.actor(sid), build.job_id)
    assert build.result["usage"]["total_tokens"] == 40
    assert workflow.result["usage"]["total_tokens"] == 50
    assert model.counts["BUILD"] == 3 and model.counts["AUDIT"] == model.counts["REVIEW"] == 1
    child_sids = [r.session_id for r in await records(managed_workflow) if r.kind == "send"]
    assert audit.session_id in child_sids
    assert (await store.inbox_counts(sid))["pending"] == 1


@pytest.mark.parametrize("failure", ["before_result", "result_ack", "inbox_write", "history_only", "wrong_ref"])
async def test_disappearing_child_handle_is_not_completion_proof_without_result_delivery_and_model_processing(managed_workflow, monkeypatch, failure):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    model.hold = {"AUDIT"}
    model.spawn_from_build = True
    state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["spawn_agent"])
    await loop.send("run", sid)
    await asyncio.wait_for(model.entered["AUDIT"].wait(), 3)
    if failure in ("before_result", "result_ack"):
        original = manager.record_execution_result

        async def broken(actor, lease, **kwargs):
            if (await manager.get_job(actor, lease.job_id)).kind == "subagent":
                if failure == "result_ack":
                    await original(actor, lease, **kwargs)
                raise RuntimeError("child result acknowledgement unavailable")
            return await original(actor, lease, **kwargs)

        monkeypatch.setattr(manager, "record_execution_result", broken)
    elif failure == "inbox_write":
        original_deliver = StatefulAgentLoop.deliver

        async def no_child_result(self, item, session_id, **kwargs):
            if session_id != sid and item.kind == "task_done":
                raise RuntimeError("child result inbox unavailable")
            return await original_deliver(self, item, session_id, **kwargs)

        monkeypatch.setattr(StatefulAgentLoop, "deliver", no_child_result)
    elif failure == "history_only":
        from power_loop.agent.follow_up import render_inbox_rows
        original_flush = StatefulAgentLoop.flush_follow_ups

        async def copy_without_model(self, session_id, **kwargs):
            if session_id == sid:
                return await original_flush(self, session_id, **kwargs)
            claimed = await self._claim_inbox(session_id)
            assert claimed
            await self._persist_user_input(session_id, render_inbox_rows(claimed, wrap=False)[0])
            return SimpleNamespace(status="completed", final_text="no model ran", rounds=0, usage={}, new_input_after_pass=False)

        monkeypatch.setattr(StatefulAgentLoop, "flush_follow_ups", copy_without_model)
    else:
        original_close = manager.close_model_execution

        async def wrong_reference(actor, lease, **kwargs):
            job = await manager.get_job(actor, lease.job_id)
            if job.execution_role == "workflow_node" and kwargs.get("evidence", {}).get("last_assistant_ref"):
                kwargs["evidence"] = {**kwargs["evidence"], "last_assistant_ref": {
                    **kwargs["evidence"]["last_assistant_ref"], "id": "999999999999"}}
            return await original_close(actor, lease, **kwargs)

        monkeypatch.setattr(manager, "close_model_execution", wrong_reference)
    await settle(managed_workflow)
    wf = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    assert model.counts["REVIEW"] == 0 and wf.outcome == "failed", wf.result
    assert wf.result["usage"]["total_tokens"] >= 30
    [node] = [r for r in await records(managed_workflow) if r.kind == "workflow_node"]
    assert node.outcome == "failed" and node.result["status"] == "failed"
    assert node.result["text"]
    assert wf.resource_state == ("unknown" if failure == "before_result" else "stopped")


async def test_waiting_for_child_releases_leaf_slot_without_completing_workflow(managed_workflow):
    _, _, _, loop, model, sid, state, _ = managed_workflow
    from power_loop.workflow.admission import GATES
    model.hold = {"AUDIT"}
    model.spawn_from_build = True
    state["spec"]["root"] = {"type": "parallel", "branches": [leaf("BUILD", tools=["spawn_agent"]), leaf("REVIEW")]}
    gate = asyncio.Semaphore(1)
    token = GATES.set((gate,))
    try:
        await loop.send("run", sid)
        await asyncio.wait_for(model.entered["AUDIT"].wait(), 3)
        await asyncio.wait_for(model.entered["REVIEW"].wait(), 3)
        assert not state["handles"][0].task.done() and not state["notifications"]
        await settle(managed_workflow)
        assert gate._value == 1
    finally:
        GATES.reset(token)


@pytest.mark.parametrize("limit", ["timeout", "rounds", "missing_result"])
async def test_required_result_collection_failures_never_advance_review_or_hide_child_usage(managed_workflow, monkeypatch, limit):
    _, manager, _, loop, model, sid, state, _ = managed_workflow
    from power_loop.runtime import workflow_tasks
    model.hold = {"AUDIT"} if limit == "timeout" else set()
    model.spawn_from_build = True
    state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["spawn_agent"])
    original = workflow_tasks.collect_node

    async def bounded(*args, **kwargs):
        return await original(*args, **kwargs, timeout_s=0.03 if limit == "timeout" else 3,
                              max_rounds=0 if limit == "rounds" else 3)

    monkeypatch.setattr(workflow_tasks, "collect_node", bounded)
    if limit == "missing_result":
        original_flush = StatefulAgentLoop.flush_follow_ups

        async def missing(self, session_id, **kwargs):
            if session_id != sid:
                return None
            return await original_flush(self, session_id, **kwargs)

        monkeypatch.setattr(StatefulAgentLoop, "flush_follow_ups", missing)
    await loop.send("run", sid)
    await asyncio.wait_for(state["handles"][0].task, 5)
    assert model.counts["REVIEW"] == 0
    workflow = next(r for r in await records(managed_workflow) if r.kind == "workflow")
    assert workflow.outcome == "failed" and workflow.result["usage"]["total_tokens"] >= 20
    assert workflow.resource_state == ("unknown" if limit == "timeout" else "stopped")
    assert not workflow_runner.is_run_live(state["handles"][0].run_id)
    if limit != "timeout":
        assert workflow.result["usage"]["total_tokens"] == 30


async def test_pass_boundary_input_already_in_history_gets_an_original_session_model_turn(managed_workflow):
    store, _, _, loop, model, sid, state, _ = managed_workflow
    from power_loop.workflow.engine import InProcessExecutor
    model.hold = set()

    class LateInputExecutor(InProcessExecutor):
        async def run_agent(self, spec, user_input, **kwargs):
            raw = await super().run_agent(spec, user_input, **kwargs)
            if spec.name == "BUILD":
                await store.append_message(raw["session_id"], role="user", content="late-input-proof")
                raw["new_input_after_pass"] = True
            return raw

    state["executor"] = LateInputExecutor()
    await loop.send("run", sid)
    await settle(managed_workflow)
    assert model.counts["BUILD"] == 2 and model.counts["REVIEW"] == 1
    node = next(r for r in await records(managed_workflow) if r.kind == "workflow_node" and r.title == "BUILD")
    sends = [r for r in await records(managed_workflow) if r.kind == "send" and r.parent_job_id == node.job_id]
    assert len(sends) == 2 and len({r.session_id for r in sends}) == 1
    assert node.result["usage"]["total_tokens"] == 20


@pytest.mark.parametrize("failure", ["workflow_registration", "journal_seed", "driver_binding", "node_registration", "node_binding"])
async def test_required_admission_failures_never_call_a_leaf_provider(managed_workflow, monkeypatch, failure):
    _, manager, _, loop, model, sid, state, _ = managed_workflow
    original = manager.register_job

    async def registration(*args, **kwargs):
        kind = "workflow" if failure == "workflow_registration" else "workflow_node"
        if kwargs.get("kind") == kind:
            raise JobError("injected_" + failure)
        return await original(*args, **kwargs)

    async def broken(*args, **kwargs):
        raise JobError("injected_" + failure)

    if failure in ("workflow_registration", "node_registration"):
        monkeypatch.setattr(manager, "register_job", registration)
    elif failure == "journal_seed":
        monkeypatch.setattr(journal, "seed", broken)
    else:
        monkeypatch.setattr(manager, "bind_workflow_driver" if failure == "driver_binding" else "bind_node_session", broken)
    try:
        await loop.send("fail admission", sid)
    except JobError:
        pass
    await settle(managed_workflow)
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 0
    for record in await records(managed_workflow):
        if record.kind == "workflow" and state["handles"]:
            assert record.outcome == "failed" and all(value == 0 for value in record.result.get("usage", {}).values())
        assert record.outcome != "succeeded"


async def test_node_result_commits_before_step_journal_and_before_completion_wake(managed_workflow, monkeypatch):
    _, manager, _, loop, _, sid, _, _ = managed_workflow
    original = journal.record_step
    seen = []

    async def observed(*args, **kwargs):
        node_id = kwargs["node_id"]
        nodes = [r for r in await records(managed_workflow) if r.kind == "workflow_node" and r.result and r.result["node_id"] == node_id]
        assert len(nodes) == 1 and nodes[0].resource_state == "stopped"
        seen.append(node_id)
        return await original(*args, **kwargs)

    monkeypatch.setattr(journal, "record_step", observed)
    await loop.send("run", sid)
    await settle(managed_workflow)
    assert seen == ["BUILD", "REVIEW"]


@pytest.mark.parametrize("failure", ["before_result", "result_ack", "journal_finalize"])
async def test_final_result_failure_never_rewrites_committed_success_or_fakes_notification(managed_workflow, monkeypatch, failure):
    store, manager, runtime, loop, _, sid, state, _ = managed_workflow
    await loop.send("run", sid)
    original = manager.record_execution_result
    target = state["handles"][0].job_id

    async def broken_result(actor, lease, **kwargs):
        if lease.job_id == target:
            if failure == "result_ack":
                await original(actor, lease, **kwargs)
            raise RuntimeError("injected final result failure")
        return await original(actor, lease, **kwargs)

    async def broken_journal(*args, **kwargs):
        raise RuntimeError("injected journal failure")

    if failure == "journal_finalize":
        monkeypatch.setattr("power_loop.runtime.workflow_results._views", broken_journal)
    else:
        monkeypatch.setattr(manager, "record_execution_result", broken_result)
    await settle(managed_workflow)
    workflow = await manager.get_job(runtime.actor(sid), target)
    if failure == "before_result":
        assert workflow.result is None and workflow.status == "running"
    else:
        assert workflow.outcome == "succeeded" and workflow.result["status"] == "completed"
    assert (await store.inbox_counts(sid))["pending"] == 0 and not state["notifications"]


async def test_stop_run_persists_cancel_before_stopping_and_preserves_cancelled_owner_result(managed_workflow):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    await loop.send("run", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    [handle] = state["handles"]
    await workflow_runner.stop_run(handle.run_id, grace_s=0.01)
    await settle(managed_workflow)
    job = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert job.cancel_requested_at is not None and job.outcome == "cancelled"
    assert model.counts["REVIEW"] == 0 and not state["notifications"]
    assert (await store.inbox_counts(sid))["pending"] == 1


@pytest.mark.parametrize("failure", ["cancel_before_claim", "claim_failure", "lost_claim_ack"])
async def test_unconfirmed_claim_never_dispatches_and_only_provably_unstarted_jobs_can_finish(managed_workflow, monkeypatch, failure):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    entered, release = asyncio.Event(), asyncio.Event()
    original_claim = runtime.claim

    async def gated(actor, registration):
        if registration.job.kind == "workflow":
            entered.set()
            await release.wait()
            if failure == "lost_claim_ack":
                await original_claim(actor, registration)
                raise RuntimeError("claim acknowledgement lost")
            if failure == "claim_failure":
                raise RuntimeError("claim rejected before transaction")
        return await original_claim(actor, registration)

    monkeypatch.setattr(runtime, "claim", gated)
    await loop.send("run", sid)
    await asyncio.wait_for(entered.wait(), 3)
    handle = state["handles"][0]
    if failure == "cancel_before_claim":
        await manager.request_cancel(runtime.actor(sid), handle.job_id, system_key="cancel-before-claim", reason="stop")
    release.set()
    await settle(managed_workflow)
    job = await manager.get_job(runtime.actor(sid), handle.job_id)
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 0
    if failure == "lost_claim_ack":
        assert job.generation == 1 and job.result is None and job.resource_state == "running"
        assert not state["notifications"] and (await store.inbox_counts(sid))["pending"] == 0
    else:
        assert job.generation == 0 and job.resource_state == "stopped"
        assert job.outcome == ("cancelled" if failure == "cancel_before_claim" else "failed")
        assert len(state["notifications"]) == (0 if failure == "cancel_before_claim" else 1)
        assert (await store.inbox_counts(sid))["pending"] == 1


@pytest.mark.parametrize("shape", ["nested_foreach", "fresh_loop", "retry"])
async def test_parallel_iteration_loop_and_retry_keep_distinct_durable_node_and_send_identities(managed_workflow, monkeypatch, shape):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    model.hold = set()
    if shape == "nested_foreach":
        state["spec"]["root"] = {"type": "foreach", "id": "outer", "as": "region", "items": ["north", "south"],
            "body": {"type": "foreach", "id": "inner", "as": "item", "items": [1, 2], "body": leaf("BUILD"), "parallel": True}, "parallel": True}
        expected_nodes, expected_sends = 4, 4
    elif shape == "fresh_loop":
        state["spec"]["root"] = {"type": "loop", "id": "repeat", "body": leaf("BUILD"),
            "until": {"ref": "BUILD", "op": "eq", "value": "not reached"}, "max_iterations": 2, "on_exhausted": "continue"}
        expected_nodes, expected_sends = 2, 2
    else:
        state["spec"]["root"] = leaf("BUILD", retry={"max_attempts": 2, "on": ["empty"]})
        original = model.complete

        async def empty_once(request, **kwargs):
            result = await original(request, **kwargs)
            if "ROLE-BUILD" in (request.system_prompt or "") and model.counts["BUILD"] == 1:
                result.raw_text = ""
                result.tool_calls = [{"id": "empty-pass", "type": "function", "function": {"name": "pass_turn", "arguments": '{"reason":"no output"}'}}]
            return result

        monkeypatch.setattr(model, "complete", empty_once)
        expected_nodes, expected_sends = 1, 2
    assert (await loop.send("run", sid)).status == "completed"
    await settle(managed_workflow)
    nodes = [r for r in await records(managed_workflow) if r.kind == "workflow_node"]
    wf = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    assert wf.outcome == "succeeded", wf.result
    assert len(nodes) == expected_nodes
    bindings = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE event_type='node_session'")
    assert len(bindings) == expected_sends
    assert len({json.loads(b["payload_json"])["session_id"] for b in bindings}) == expected_sends
    assert wf.result["usage"]["total_tokens"] == expected_sends * 10
    if shape != "retry":
        assert len({json.dumps(json.loads(b["payload_json"])["route"], sort_keys=True) for b in bindings}) == expected_nodes


@pytest.mark.parametrize("phase", ["before_commit", "lost_ack"])
async def test_node_result_commit_failure_blocks_downstream_and_never_publishes_node_success(managed_workflow, monkeypatch, phase):
    _, manager, _, loop, model, sid, state, _ = managed_workflow
    await loop.send("run", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    original = manager.record_execution_result

    async def broken(actor, lease, **kwargs):
        job = await manager.get_job(actor, lease.job_id)
        if job.kind == "workflow_node":
            if phase == "lost_ack":
                await original(actor, lease, **kwargs)
            raise RuntimeError("node result acknowledgement unavailable")
        return await original(actor, lease, **kwargs)

    monkeypatch.setattr(manager, "record_execution_result", broken)
    await settle(managed_workflow)
    assert model.counts["REVIEW"] == 0
    wf = next(r for r in await records(managed_workflow) if r.kind == "workflow")
    assert wf.outcome == "failed" and wf.result["usage"]["total_tokens"] == 10
    data = await journal.read(manager.store, sid, state["handles"][0].run_id)
    assert not data.get("steps")
    node = next(r for r in await records(managed_workflow) if r.kind == "workflow_node")
    if phase == "lost_ack":
        assert node.result["status"] == "completed"  # committed evidence stays immutable
    else:
        assert node.result is None and wf.resource_state == "unknown"


async def test_direct_engine_cannot_bypass_required_workflow_admission(managed_workflow):
    _, _, _, loop, model, sid, _, _ = managed_workflow
    from power_loop.workflow.engine import WorkflowEngine
    from power_loop.workflow.spec import WorkflowSpec
    engine = WorkflowEngine(loop)
    with pytest.raises(JobError, match="workflow_admission_required"):
        await engine.run(WorkflowSpec.from_json({"name": "unadmitted", "root": leaf("BUILD")}))
    assert not model.counts
    assert len(await records(managed_workflow)) == 0


@pytest.mark.parametrize("invalid", ["wrong_role", "wrong_job", "wrong_run", "non_root_driver", "foreign_actor"])
async def test_driver_binding_requires_the_actual_admitted_workflow_and_matching_controller(registry_runtime, invalid):
    store, manager, actor, _, root_proof, _, _ = registry_runtime
    wf = await manager.register_job(actor, kind="workflow", parent=root_proof, independent=True, system_key="wf")
    proof = await manager.claim_job(actor, wf.job.job_id, executor_id="executor", boot_id="boot", expected_version=1)
    metadata = {"kind": "wf_driver", "execution_role": "workflow", "task_owner_job_id": wf.job.job_id,
                "workflow_run_id": wf.operation.operation_id[3:]}
    if invalid == "wrong_role":
        metadata["execution_role"] = "main"
    elif invalid == "wrong_job":
        metadata["task_owner_job_id"] = "job_other"
    elif invalid == "wrong_run":
        metadata["workflow_run_id"] = "unrelated-run"
    sid = await store.create_session(parent_session_id=actor.session_id if invalid == "non_root_driver" else None, metadata=metadata)
    caller = JobActor("foreign", actor.principal_id, actor.session_id) if invalid == "foreign_actor" else actor
    with pytest.raises(JobError, match="ownership|workflow_run_mismatch|denied"):
        await manager.bind_workflow_driver(caller, proof, sid)
    assert (await manager.get_job(actor, wf.job.job_id)).session_id is None


async def test_node_model_admission_cannot_be_forged_and_multiple_retry_sessions_have_one_controller(registry_runtime):
    store, manager, actor, _, root_proof, _, _ = registry_runtime
    wf = await manager.register_job(actor, kind="workflow", parent=root_proof, independent=True, system_key="wf")
    wp = await manager.claim_job(actor, wf.job.job_id, executor_id="executor", boot_id="boot", expected_version=1)
    driver = await store.create_session(metadata={"kind": "wf_driver", "execution_role": "workflow",
        "task_owner_job_id": wf.job.job_id, "workflow_run_id": wf.operation.operation_id[3:]})
    await manager.bind_workflow_driver(actor, wp, driver)
    controller = JobActor(actor.tenant_id, actor.principal_id, driver)
    node = await manager.register_job(controller, kind="workflow_node", parent=wp, system_key="node")
    np = await manager.claim_job(controller, node.job.job_id, executor_id="executor", boot_id="boot", expected_version=1)
    sessions = [await store.create_session(parent_session_id=driver, metadata={"execution_role": "workflow_node",
                "task_owner_job_id": node.job.job_id, "workflow_node_id": "BUILD", "attempt": i,
                "workflow_run_id": wf.operation.operation_id[3:]}) for i in (1, 2)]
    with pytest.raises(JobError, match="node_session_admission_required"):
        await manager.register_job(controller, kind="send", parent=np, session_id=sessions[0], system_key="unbound")
    for sid in sessions:
        with pytest.raises(JobError, match="denied|ownership"):
            await manager.bind_node_session(JobActor(actor.tenant_id, actor.principal_id, sid), np, sid)
        await manager.bind_node_session(controller, np, sid)
        await manager.bind_node_session(controller, np, sid)
        registered = await manager.register_job(controller, kind="send", parent=np, session_id=sid, system_key="send:" + sid)
        assert registered.job.execution_role == "workflow_node" and registered.job.session_id == sid
    [count] = await manager.db.fetchall(f"SELECT COUNT(*) AS n FROM {manager.events} WHERE event_type='node_session'")
    assert count["n"] == 2


@pytest.mark.parametrize("wrong", ["workflow_run_id", "workflow_node_id", "task_node_route", "attempt"])
async def test_node_session_descriptor_rejects_wrong_workflow_node_iteration_and_attempt(registry_runtime, wrong):
    store, manager, actor, _, root_proof, _, _ = registry_runtime
    wf = await manager.register_job(actor, kind="workflow", parent=root_proof, independent=True, system_key="wf",
                                    request={"spec": {"name": "frozen"}})
    wp = await manager.claim_job(actor, wf.job.job_id, executor_id="executor", boot_id="boot", expected_version=1)
    driver = await store.create_session(metadata={"kind": "wf_driver", "execution_role": "workflow",
        "task_owner_job_id": wf.job.job_id, "workflow_run_id": wf.operation.operation_id[3:]})
    await manager.bind_workflow_driver(actor, wp, driver)
    controller = JobActor(actor.tenant_id, actor.principal_id, driver)
    route = [{"foreach": "items", "index": 0}]
    descriptor = {"node": {"id": "BUILD", "retry": {"max_attempts": 2}}, "route": route}
    node = await manager.register_job(controller, kind="workflow_node", parent=wp, system_key="node", request=descriptor)
    proof = await manager.claim_job(controller, node.job.job_id, executor_id="executor", boot_id="boot", expected_version=1)
    meta = {"execution_role": "workflow_node", "task_owner_job_id": node.job.job_id,
            "workflow_run_id": wf.operation.operation_id[3:], "workflow_node_id": "BUILD", "task_node_route": route, "attempt": 1}
    meta[wrong] = 999 if wrong == "attempt" else [{"foreach": "items", "index": 1}] if wrong == "task_node_route" else "unrelated"
    sid = await store.create_session(parent_session_id=driver, metadata=meta)
    with pytest.raises(JobError, match="node_route_mismatch"):
        await manager.bind_node_session(controller, proof, sid)
    assert (await manager.get_operation(controller, node.operation.operation_id)).result["request"] == descriptor


@pytest.mark.parametrize("port", ["create", "operation", "result", "model_exit"])
async def test_registry_freezes_mutable_inputs_before_waiting_for_a_transaction(registry_runtime, monkeypatch, port):
    store, manager, actor, _, proof, _, _ = registry_runtime
    from .test_job_registry_192 import call, child
    data = {"payload": {"value": "original"}}
    original_input = {"payload": {"value": "original"}}
    target = origin = target_proof = None
    if port == "operation":
        target = await child(registry_runtime, independent=True)
        target_proof = await manager.claim_job(actor, target.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
        origin = await call(store, actor)
    entered, release = asyncio.Event(), asyncio.Event()
    original_write = manager._write

    @asynccontextmanager
    async def gated():
        entered.set()
        await release.wait()
        async with original_write() as tx:
            yield tx

    monkeypatch.setattr(manager, "_write", gated)
    if port == "create":
        work = manager.register_job(actor, kind="workflow", parent=proof, independent=True, system_key="frozen", request=data)
    elif port == "operation":
        work = manager.register_operation(actor, target_proof, caller=proof, action="steer", call_ref=origin,
                                          tool_call_id="call-1", request=data, result=data)
    elif port == "result":
        work = manager.record_execution_result(actor, proof, outcome="succeeded", result=data, resource_state="stopped")
    else:
        work = manager.close_model_execution(actor, proof, evidence=data)
    running = asyncio.create_task(work)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        data["payload"]["value"] = "changed while awaiting database"
    finally:
        release.set()
    result = await running
    if port == "create":
        assert result.operation.result["request"] == original_input
        assert (await manager.register_job(actor, kind="workflow", parent=proof, independent=True, system_key="frozen", request=original_input)).replayed
    elif port == "operation":
        assert result.result == original_input
    elif port == "result":
        assert (await manager.get_job(actor, proof.job_id)).result == original_input
    else:
        row = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", ("model_closed_" + proof.job_id,))
        assert json.loads(row["payload_json"])["evidence"] == original_input


@pytest.mark.parametrize("case", ["replay", "ancestor_cancel", "claimed", "foreign", "success_forbidden"])
async def test_unstarted_result_is_idempotent_authorized_and_cannot_claim_unproven_success_or_stop(registry_runtime, case):
    _, manager, actor, root, root_proof, _, _ = registry_runtime
    from .test_job_registry_192 import child
    reg = await child(registry_runtime, independent=True)
    payload = {"status": "failed", "error": "not dispatched"}
    if case == "claimed":
        await manager.claim_job(actor, reg.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
    elif case == "ancestor_cancel":
        await manager.request_cancel(actor, root.job.job_id, system_key="stop", reason="stop")
    caller = JobActor("foreign", actor.principal_id, actor.session_id) if case == "foreign" else actor
    if case in ("claimed", "foreign", "success_forbidden"):
        with pytest.raises(JobError, match="conflict|denied|invalid_unstarted"):
            await manager.record_unstarted_result(caller, reg.job.job_id,
                outcome="succeeded" if case == "success_forbidden" else "failed", result=payload)
        assert (await manager.get_job(actor, reg.job.job_id)).result is None
    else:
        result = await manager.record_unstarted_result(actor, reg.job.job_id, outcome="failed", result=payload)
        assert result["status"] == ("cancelled" if case == "ancestor_cancel" else "failed")
        assert await manager.record_unstarted_result(actor, reg.job.job_id, outcome="failed", result=payload) == result
        with pytest.raises(JobError, match="conflict|cancel|closed"):
            await manager.claim_job(actor, reg.job.job_id, executor_id="worker", boot_id="boot", expected_version=3)


@pytest.mark.parametrize("winner", ["result", "child"])
async def test_scope_result_and_late_required_admission_serialize_without_premature_success(registry_runtime, monkeypatch, winner):
    _, manager, actor, _, _, _, _ = registry_runtime
    from .test_job_registry_192 import node_tree
    _, _, node, proof, _, send_proof = await node_tree(registry_runtime)
    await manager.close_model_execution(actor, send_proof)
    if winner == "child":
        await manager.register_job(actor, kind="background_tool", parent=proof, system_key="late")
        with pytest.raises(JobError, match="required_execution_unsettled"):
            await manager.record_execution_result(actor, proof, outcome="succeeded", result={"status": "completed"},
                                                   resource_state="stopped", require_children_stopped=True)
        assert (await manager.get_job(actor, node.job.job_id)).result is None
        return
    checked, release = asyncio.Event(), asyncio.Event()
    original = manager._require_scope_ready

    async def hold_transaction(tx, row):
        await original(tx, row)
        checked.set()
        await release.wait()

    monkeypatch.setattr(manager, "_require_scope_ready", hold_transaction)
    closing = asyncio.create_task(manager.record_execution_result(actor, proof, outcome="succeeded", result={"status": "completed"},
        resource_state="stopped", require_children_stopped=True))
    await asyncio.wait_for(checked.wait(), 3)
    late = asyncio.create_task(manager.register_job(actor, kind="background_tool", parent=proof, system_key="late"))
    await asyncio.sleep(0)
    release.set()
    await closing
    with pytest.raises(JobError, match="closed|not_running|expired"):
        await late
    assert (await manager.get_job(actor, node.job.job_id)).outcome == "succeeded"


async def test_unadapted_old_session_continuation_is_rejected_before_admission_or_journal_mutation(managed_workflow):
    _, _, _, loop, model, sid, state, _ = managed_workflow
    state["spec"]["root"] = {"type": "loop", "id": "repeat", "body": leaf("BUILD"),
        "until": {"ref": "BUILD", "op": "truthy"}, "continue_nodes": ["BUILD"], "max_iterations": 2}
    try:
        await loop.send("run", sid)
    except JobError as exc:
        assert "session_rebind" in str(exc)
    assert not state["handles"] and model.counts["BUILD"] == 0
    assert len(await records(managed_workflow)) == 1


@pytest.mark.parametrize("entry", ["sync_resume", "async_resume", "idle_nudge", "steer"])
async def test_unadapted_resume_and_nudge_do_not_change_journal_or_launch_unregistered_work(managed_workflow, entry):
    _, _, _, loop, model, sid, state, _ = managed_workflow
    await loop.send("run", sid)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    handle = state["handles"][0]
    before = await journal.read(loop.store, sid, handle.run_id)
    from power_loop.workflow.resume import resume_detached, resume_run
    engine = workflow_runner._ENGINES.get(handle.run_id)
    if engine is None:
        for _ in range(20):
            await asyncio.sleep(0.01)
            engine = workflow_runner._ENGINES.get(handle.run_id)
            if engine is not None:
                break
        before = await journal.read(loop.store, sid, handle.run_id)
    with pytest.raises(JobError, match="task_send_required" if entry == "steer" else "adapter_unavailable"):
        if entry == "sync_resume":
            await resume_run(loop, sid, handle.run_id)
        elif entry == "async_resume":
            await resume_detached(loop, sid, handle.run_id)
        elif entry == "idle_nudge":
            engine.start_node_continuation("BUILD", "continue", 3)
        else:
            await engine.steer_node("BUILD", sid, "continue")
    after = await journal.read(loop.store, sid, handle.run_id)
    assert after == before
