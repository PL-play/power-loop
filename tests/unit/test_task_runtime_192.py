"""Actual loop adapters, admission failures, receipt identity and cancellation."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import (
    AgentLoopConfig,
    AgentSpec,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    StatefulAgentLoop,
    ToolDefinition,
)
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime import delegation
from power_loop.runtime.jobs import JobError, JobLease, JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow import create_workflow

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class AdapterModel(LLMService):
    def __init__(self):
        self.root = None
        self.root_calls = 0
        self.child_calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.tool_name = "spawn_agent"

    async def complete(self, request, **kwargs):
        if get_session_id() != self.root:
            self.child_calls += 1
            self.entered.set()
            await self.release.wait()
            text, calls = "CHILD-PROOF-42", None
        else:
            self.root_calls += 1
            text, calls = "done", None
            if self.root_calls == 1:
                text = ""
                calls = [{"id": "actual-create", "type": "function", "function": {
                    "name": self.tool_name, "arguments": json.dumps({"task": "Return CHILD-PROOF-42", "tools": [], "max_rounds": 2})}}]
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
async def managed(registry_store):
    store, connect = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="isolated-adapter", principal_id="host",
                          executor_id="executor", boot_id="boot", lease_ms=1000)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    model = AdapterModel()
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3))
    sid = await loop.new_session()
    model.root = sid
    before = set(delegation._TASKS)
    try:
        yield store, manager, runtime, loop, model, sid, connect
    finally:
        model.release.set()
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)


async def jobs(managed):
    _, manager, runtime, _, _, sid, _ = managed
    rows = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} ORDER BY created_at,job_id")
    return [await manager.get_job(runtime.actor(sid), row["job_id"]) for row in rows]


async def receipt(managed):
    store, _, _, _, _, sid, _ = managed
    return next(json.loads(r.content) for r in await store.load_all_messages(sid) if r.role == "tool")


async def finish(managed):
    _, _, _, _, model, _, _ = managed
    model.release.set()
    await asyncio.wait_for(asyncio.gather(*list(delegation._TASKS)), 5)


async def test_actual_create_receipt_child_send_and_result_survive_creator_close_and_reopen(managed):
    store, manager, runtime, loop, model, sid, connect = managed
    result = await loop.send("delegate", sid)
    assert result.status == "completed"
    await asyncio.wait_for(model.entered.wait(), 3)
    accepted = await receipt(managed)
    records = await jobs(managed)
    root = next(j for j in records if j.kind == "send" and j.session_id == sid)
    child = next(j for j in records if j.kind == "subagent")
    child_send = next(j for j in records if j.kind == "send" and j.session_id != sid)
    assert root.status == "model_closed" and child.status == child_send.status == "running"
    assert child.parent_job_id == root.job_id and child.control_owner_job_id is None
    assert child_send.control_owner_job_id == child.job_id and child_send.execution_role == "subagent"
    operation = await manager.get_operation(runtime.actor(sid), accepted["operation_id"])
    call, tool = operation.call_ref, operation.receipt_ref
    assert call["id"] != tool["id"] and call["tool_call_id"] is None and tool["tool_call_id"] == "actual-create"
    assert call["role"] == "assistant" and tool["role"] == "tool"
    assert json.loads((await store.message_by_id(sid, tool["id"])).content) == accepted
    await finish(managed)
    reopened = JobManager(await connect())
    saved = await reopened.get_job(runtime.actor(sid), child.job_id)
    assert saved.result["final_text"] == "CHILD-PROOF-42" and saved.result["usage"]["total_tokens"] == 10
    assert saved.status == "settling" and saved.supervision_status == "open" and saved.resource_state == "stopped"
    assert (await store.inbox_counts(sid))["pending"] == 1
    assert (await loop.flush_follow_ups(sid)).status == "completed"
    assert len(await jobs(managed)) == 4 and model.child_calls == 1


async def test_duplicate_actual_call_reuses_admission_without_dispatching_twice(managed, monkeypatch):
    _, _, _, loop, model, sid, _ = managed
    original = delegation.start
    receipts = []

    async def twice(*args, **kwargs):
        receipts.append(await original(*args, **kwargs))
        receipts.append(await original(*args, **kwargs))
        return receipts[0]

    monkeypatch.setattr(delegation, "start", twice)
    assert (await loop.send("delegate", sid)).status == "completed"
    assert receipts[0]["job_id"] == receipts[1]["job_id"]
    assert receipts[1]["replayed"] and receipts[0]["session_id"] == receipts[1]["session_id"]
    await finish(managed)
    assert model.child_calls == 1 and len(await jobs(managed)) == 3


@pytest.mark.parametrize("failure", ["registration", "session_binding"])
async def test_required_admission_failure_never_starts_child_provider(managed, monkeypatch, failure):
    store, manager, _, loop, model, sid, _ = managed
    original = manager.register_job

    async def fail_registration(*args, **kwargs):
        if kwargs.get("kind") == "subagent":
            raise JobError("injected_admission_failure")
        return await original(*args, **kwargs)

    async def fail_binding(*args, **kwargs):
        raise JobError("injected_admission_failure")

    monkeypatch.setattr(manager, "register_job" if failure == "registration" else "bind_session",
                        fail_registration if failure == "registration" else fail_binding)
    assert (await loop.send("delegate", sid)).status == "completed"
    await finish(managed)
    assert model.child_calls == 0 and not delegation.running_jobs(store, sid)
    tool = next(r for r in await store.load_all_messages(sid) if r.role == "tool")
    assert "injected_admission_failure" in tool.content
    child_jobs = [j for j in await jobs(managed) if j.kind == "subagent"]
    assert not child_jobs if failure == "registration" else child_jobs[0].status == "cancelled"


async def test_receipt_storage_failure_preserves_pending_and_canonical_child_without_reexecution(managed, monkeypatch):
    store, manager, runtime, loop, model, sid, _ = managed
    original = manager.reconcile_receipt

    async def broken(*args, **kwargs):
        raise JobError("receipt_binding_unavailable")

    monkeypatch.setattr(manager, "reconcile_receipt", broken)
    try:
        await loop.send("delegate", sid)
    except JobError as exc:
        assert exc.code == "receipt_binding_unavailable"
    assert (await store.get_state(sid)).pending
    [tool] = [r for r in await store.load_all_messages(sid) if r.role == "tool"]
    accepted = json.loads(tool.content)
    assert (await manager.get_operation(runtime.actor(sid), accepted["operation_id"])).receipt_ref is None
    # Recovery must not blindly dispatch a non-idempotent pending call.
    with pytest.raises(JobError, match="adapter_unavailable:resume"):
        await loop.resume(sid)
    monkeypatch.setattr(manager, "reconcile_receipt", original)
    await manager.reconcile_receipt(runtime.actor(sid), accepted["operation_id"], await store.message_ref(sid, tool.seq))
    await finish(managed)
    assert model.child_calls == 1 and len([j for j in await jobs(managed) if j.kind == "subagent"]) == 1


async def test_cancel_queued_child_persists_intent_and_result_without_provider_call(managed, monkeypatch):
    store, manager, runtime, loop, model, sid, _ = managed
    original = delegation.start
    semaphore = asyncio.Semaphore(0)

    async def queued(*args, **kwargs):
        return await original(*args, **kwargs, admission=semaphore)

    monkeypatch.setattr(delegation, "start", queued)
    assert (await loop.send("delegate", sid)).status == "completed"
    accepted = await receipt(managed)
    await delegation.stop(store, sid, accepted["run_id"], 0.01)
    await finish(managed)
    child = await manager.get_job(runtime.actor(sid), accepted["job_id"])
    assert child.cancel_requested_at is not None and child.outcome == "cancelled"
    assert child.resource_state == "stopped" and model.child_calls == 0
    assert (await store.inbox_counts(sid))["pending"] == 1


async def test_heartbeat_failure_stops_the_attempt_and_records_failure(managed, monkeypatch):
    _, manager, _, loop, model, sid, _ = managed
    assert (await loop.send("delegate", sid)).status == "completed"
    await asyncio.wait_for(model.entered.wait(), 3)

    async def broken(*args, **kwargs):
        raise JobError("injected_lease_failure")

    monkeypatch.setattr(manager, "renew_execution", broken)
    await asyncio.wait_for(asyncio.gather(*list(delegation._TASKS), return_exceptions=True), 4)
    [child] = [j for j in await jobs(managed) if j.kind == "subagent"]
    assert child.outcome in ("failed", "cancelled") and child.resource_state == "unknown"
    assert child.supervision_status == "open"
    [attempt] = await manager.db.fetchall(f"SELECT e.payload_json FROM {manager.events} e JOIN {manager.jobs} j ON j.job_id=e.job_id "
        "WHERE j.control_owner_job_id=? AND e.event_type='model_attempt_finished'", (child.job_id,))
    evidence = json.loads(attempt["payload_json"])
    assert evidence["status"] == "unknown" and evidence["error_type"] == "CancelledError" and evidence["estimated"]
    assert model.child_calls == 1


@pytest.mark.parametrize("failure", ["before_commit", "lost_commit_ack", "compatibility_projection"])
async def test_result_commit_and_projection_failures_preserve_canonical_evidence_without_fake_success(managed, monkeypatch, failure):
    store, manager, runtime, loop, _, sid, _ = managed
    assert (await loop.send("delegate", sid)).status == "completed"
    accepted = await receipt(managed)
    if failure == "compatibility_projection":
        original_state = store.set_runtime_state
        failed_once = False

        async def failed_projection(owner, key, value):
            nonlocal failed_once
            if key == "subagent:run:" + accepted["run_id"] and value.get("phase") == "prepared" and not failed_once:
                failed_once = True
                raise RuntimeError("compatibility projection unavailable")
            return await original_state(owner, key, value)

        monkeypatch.setattr(store, "set_runtime_state", failed_projection)
    else:
        original_result = manager.record_execution_result

        async def failed_result(*args, **kwargs):
            if failure == "lost_commit_ack":
                await original_result(*args, **kwargs)
            raise RuntimeError("canonical commit acknowledgement unavailable")

        monkeypatch.setattr(manager, "record_execution_result", failed_result)
    await finish(managed)
    await runtime.flush_control_checkpoints()
    child = await manager.get_job(runtime.actor(sid), accepted["job_id"])
    pending = (await store.inbox_counts(sid))["pending"]
    if failure == "before_commit":
        assert child.result is None and child.status == "settling" and pending == 0
        assert child.resource_state == 'unknown' and child.supervision_status == 'open' and child.result_status == 'none'
        original = await manager._row(manager.db, child.job_id)
        assert original['intake'] == 'closed' and original['lease_until_ms'] is None
    else:
        assert child.result["status"] == "completed" and child.result["final_text"] == "CHILD-PROOF-42"
        assert child.outcome == "succeeded" and child.status == "settling"
        assert pending == (1 if failure == "compatibility_projection" else 0)
    assert not delegation.running_jobs(store, sid)


async def test_cancellation_winning_result_commit_retains_actual_output_usage_and_cancelled_notification(managed, monkeypatch):
    store, manager, runtime, loop, _, sid, _ = managed
    assert (await loop.send("delegate", sid)).status == "completed"
    accepted = await receipt(managed)
    original = manager.record_execution_result

    async def race(actor, lease, **kwargs):
        await manager.request_cancel(runtime.actor(sid), lease.job_id, system_key="result-race", reason="stop")
        return await original(actor, lease, **kwargs)

    monkeypatch.setattr(manager, "record_execution_result", race)
    await finish(managed)
    child = await manager.get_job(runtime.actor(sid), accepted["job_id"])
    assert child.outcome == "cancelled" and child.result["status"] == "cancelled"
    assert child.result["execution_status"] == "completed"
    assert child.result["final_text"] == "CHILD-PROOF-42" and child.result["usage"]["total_tokens"] == 10
    compatibility = await store.get_runtime_state(sid, "subagent:run:" + accepted["run_id"])
    assert compatibility["result"] == child.result


async def test_same_store_parallel_main_sessions_have_isolated_job_roots_and_message_origins(managed):
    store, manager, runtime, loop, model, sid, _ = managed
    model.root_calls = 10  # No delegation: both sends are ordinary root sends.
    second = await loop.new_session()
    model.release.set()
    first_result, second_result = await asyncio.gather(loop.send("first", sid), loop.send("second", second))
    assert first_result.status == second_result.status == "completed"
    rows = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
    assert len(rows) == 2 and {r["session_id"] for r in rows} == {sid, second}
    assert all(r["root_job_id"] == r["job_id"] and r["parent_job_id"] is None for r in rows)
    for r in rows:
        unrelated = runtime.actor(second if r["session_id"] == sid else sid)
        with pytest.raises(JobError, match="denied"):
            await manager.get_job(unrelated, r["job_id"])
    assert len(await store.list_children(sid)) == 0


async def test_slow_result_projection_does_not_renew_or_cancel_already_stopped_execution(managed, monkeypatch):
    store, manager, runtime, loop, _, sid, _ = managed
    assert (await loop.send("delegate", sid)).status == "completed"
    accepted = await receipt(managed)
    original = store.set_runtime_state
    renewals_after_result = []
    original_renew = manager.renew_execution

    async def slow(owner, key, value):
        if key == "subagent:run:" + accepted["run_id"] and value.get("phase") == "prepared":
            await asyncio.sleep(0.6)
        return await original(owner, key, value)

    async def observe(actor, lease, **kwargs):
        job = await manager.get_job(actor, lease.job_id)
        if job.resource_state == "stopped":
            renewals_after_result.append(lease.job_id)
        return await original_renew(actor, lease, **kwargs)

    monkeypatch.setattr(store, "set_runtime_state", slow)
    monkeypatch.setattr(manager, "renew_execution", observe)
    await finish(managed)
    assert not renewals_after_result
    saved = await manager.get_job(runtime.actor(sid), accepted["job_id"])
    assert saved.outcome == "succeeded" and saved.result["status"] == "completed"
    assert (await store.inbox_counts(sid))["pending"] == 1


@pytest.mark.parametrize("scope_stop", [False, True])
async def test_cancelled_caller_model_phase_does_not_cancel_already_accepted_independent_child(managed, monkeypatch, scope_stop):
    _, manager, runtime, loop, model, sid, _ = managed
    original = model.complete
    finishing_parent = asyncio.Event()
    stop_observed = asyncio.Event()
    original_renew = manager.renew_execution

    async def renew(*args, **kwargs):
        try:
            return await original_renew(*args, **kwargs)
        except JobError as exc:
            if exc.code == "cancel_requested":
                stop_observed.set()
            raise

    async def gate_parent(request, **kwargs):
        if get_session_id() == sid and model.root_calls == 1:
            finishing_parent.set()
            await asyncio.Event().wait()
        return await original(request, **kwargs)

    monkeypatch.setattr(model, "complete", gate_parent)
    monkeypatch.setattr(manager, "renew_execution", renew)
    sending = asyncio.create_task(loop.send("delegate", sid))
    try:
        await asyncio.wait_for(finishing_parent.wait(), 3)
        # This case exercises cancellation of an already running child model.
        # A separate case covers cancellation at pre-model admission.
        await asyncio.wait_for(model.entered.wait(), 3)
        accepted = await receipt(managed)
        sending.cancel()
        await asyncio.gather(sending, return_exceptions=True)
        child = await manager.get_job(runtime.actor(sid), accepted["job_id"])
        assert (await manager.get_job(runtime.actor(sid), child.parent_job_id)).status == "model_closed"
        assert (await manager.get_job(runtime.actor(sid), child.parent_job_id)).resource_state == "unknown"
        assert child.cancel_requested_at is None
        if scope_stop:
            await manager.request_cancel(runtime.actor(sid), child.parent_job_id, system_key="stop-original-tree", reason="Stop the complete original scope")
            await asyncio.wait_for(stop_observed.wait(), 3)
        await finish(managed)
        child = await manager.get_job(runtime.actor(sid), child.job_id)
        assert child.outcome == ("cancelled" if scope_stop else "succeeded")
        if not scope_stop:
            assert child.result["final_text"] == "CHILD-PROOF-42"
    finally:
        if not sending.done():
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)


async def test_rejected_adoption_of_cancel_resistant_work_is_bounded_and_retains_unknown_resource(managed):
    _, manager, runtime, loop, model, sid, _ = managed
    model.tool_name = "probe"
    entered, release = asyncio.Event(), asyncio.Event()

    async def resistant():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    async def probe(**kwargs):
        from power_loop.tools.default_tools import BG
        work = asyncio.create_task(resistant())
        await entered.wait()
        try:
            with pytest.raises(JobError, match="adapter_unavailable:background_adopt"):
                await asyncio.wait_for(BG.adopt("probe", {}, work), 2)
            assert not work.done()
            return "unconfirmed resource"
        finally:
            release.set()
            await work

    loop.tool_registry.register(ToolDefinition(name="probe", description="test", input_schema={"type": "object"}), probe)
    try:
        await loop.send("probe", sid)
    except JobError as exc:
        assert exc.code == "unresolved_prior_attempt"
    [root] = await jobs(managed)
    assert root.status == "model_closed" and root.resource_state == "unknown"
    assert model.root_calls == 1
    # Closure is evidence, not permission to erase an unknown resource.
    with pytest.raises(JobError, match="unresolved"):
        await manager.register_job(runtime.actor(sid), kind="subagent", system_key="unsafe-new-work",
            parent=JobLease(root.job_id, "executor", "boot", 1))


@pytest.mark.parametrize("entry", ["workflow", "background_command", "background_tool", "background_adopt", "timer", "timer_tool", "nudge", "direct_subagent", "ephemeral", "submit_input"])
async def test_unadapted_entry_is_rejected_before_external_work(managed, entry):
    store, _, _, loop, model, sid, _ = managed
    model.tool_name = "probe"

    async def probe(**kwargs):
        current = get_current_loop()
        with pytest.raises(ValueError, match="adapter_unavailable|owner_required"):
            if entry == "workflow":
                workflow = create_workflow({"name": "test", "root": {"type": "agent", "id": "x", "spec": {
                    "name": "x", "system_prompt": "x"}}}, parent_loop=current, parent_session_id=get_session_id())
                await workflow.run()
            elif entry in ("background_command", "background_tool", "background_adopt", "timer_tool"):
                from power_loop.tools.default_tools import BG, run_schedule_wakeup
                if entry == "background_command":
                    await BG.run("echo should-never-run")
                elif entry == "background_tool":
                    await BG.run_tool("probe", {})
                elif entry == "background_adopt":
                    work = asyncio.create_task(asyncio.Event().wait())
                    try:
                        await BG.adopt("probe", {}, work)
                    finally:
                        assert work.cancelled()
                else:
                    await run_schedule_wakeup("schedule", delay_seconds=60, note="no")
            elif entry == "timer":
                await current.schedule_timer(sid, delay_s=60, note="no")
            elif entry == "nudge":
                await delegation.nudge(current, sid, "no")
            elif entry == "submit_input":
                await current.submit_input(sid, "not-admitted", "no")
            elif entry == "ephemeral":
                await delegation.start(current, AgentSpec(name="no", system_prompt="no"), "no")
            else:
                from power_loop.runtime.spec import run_agent_spec
                await run_agent_spec(AgentSpec(name="no", system_prompt="no"), "no", parent_loop=current)
        return "rejected safely"

    loop.tool_registry.register(ToolDefinition(name="probe", description="test admission", input_schema={"type": "object"}), probe)
    assert (await loop.send("probe", sid)).status == "completed"
    assert len(await jobs(managed)) == 1 and model.child_calls == 0
    assert await store.list_children(sid) == []


@pytest.mark.parametrize("case", ["other_store", "legacy_role", "invalid_lease", "invalid_executor"])
async def test_host_opt_in_rejects_invalid_or_mismatched_configuration(managed, case):
    _, manager, runtime, _, model, _, _ = managed
    with pytest.raises(ValueError):
        if case == "other_store":
            StatefulAgentLoop(llm=model, task_runtime=runtime)
        elif case == "legacy_role":
            StatefulAgentLoop(llm=model, store=manager.store, task_runtime=runtime)
        else:
            TaskRuntime(manager, tenant_id="test", principal_id="host", executor_id="" if case == "invalid_executor" else "exec",
                        boot_id="boot", lease_ms=0 if case == "invalid_lease" else 30000)


@pytest.mark.parametrize("scope_stop", [False, True])
async def test_cancel_before_child_model_admission_uses_admission_fence(managed, monkeypatch, scope_stop):
    _, manager, runtime, loop, model, sid, _ = managed
    runtime.lease_ms = 300000  # No heartbeat is needed to enforce admission.
    before_model, parent_waiting, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    run, complete = StatefulAgentLoop._run_loop, model.complete
    async def gated_run(self, active_sid, **kwargs):
        if active_sid != sid:
            before_model.set()
            await release.wait()
        return await run(self, active_sid, **kwargs)
    async def gated_parent(request, **kwargs):
        if get_session_id() == sid and model.root_calls == 1:
            parent_waiting.set()
            await asyncio.Event().wait()
        return await complete(request, **kwargs)
    monkeypatch.setattr(StatefulAgentLoop, "_run_loop", gated_run)
    monkeypatch.setattr(model, "complete", gated_parent)
    sending = asyncio.create_task(loop.send("delegate", sid))
    try:
        await asyncio.wait_for(parent_waiting.wait(), 5)
        await asyncio.wait_for(before_model.wait(), 5)
        accepted = await receipt(managed)
        sending.cancel()
        await asyncio.gather(sending, return_exceptions=True)
        child = await manager.get_job(runtime.actor(sid), accepted["job_id"])
        assert child.cancel_requested_at is None and model.child_calls == 0
        if scope_stop:
            await manager.request_cancel(runtime.actor(sid), child.parent_job_id,
                system_key="stop-before-model", reason="Stop original scope before model admission")
        release.set()
        await finish(managed)
        child = await manager.get_job(runtime.actor(sid), child.job_id)
        assert child.outcome == ("cancelled" if scope_stop else "succeeded")
        assert model.child_calls == (0 if scope_stop else 1)
        assert child.resource_state == "stopped"
    finally:
        release.set()
        if not sending.done():
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)
