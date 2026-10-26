"""Original decision effects, races, rollback and actual child input consumption."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import timer_waits as waits
from power_loop.runtime import wait_effects as effects
from power_loop.runtime import wait_notifications as notices
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobError, JobManager, _hash, _json
from power_loop.runtime.task_runtime import _SEND, Execution, TaskRuntime
from power_loop.tools.wait_handoff import register_wait_handoff_tool

from .test_wait_handoff_192 import handoff, registry_store, runtime, waiting
from .test_workflow_restore_runtime_192 import native_restore
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ["handoff", "registry_store", "runtime", "waiting", "managed_workflow", "native_restore"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def decide(h, choice="continue"):
    h.model.options = {"event_id": h.event_id, "decision": choice}
    if choice == "defer":
        h.model.options["next_review_at_ms"] = h.c.now[0] + 1000
    await notices.enqueue_notification(h.loop, h.c.main, h.event_id)
    await h.loop.flush_follow_ups(h.c.main.session_id)


class ChildModel(LLMService):
    def __init__(self):
        self.calls, self.requests = 0, []

    async def complete(self, request, **kwargs):
        self.calls += 1
        self.requests.append(request)
        return LLMResponse(raw_text="Original task continued: 42.",
            token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))

    def stream(self, request):
        async def empty():
            if False:
                yield None
        return empty()

    async def close(self):
        pass


async def child_loop(h):
    c = h.c
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed"})
    model = ChildModel()
    # The original child lease belongs to its executor, not the main wake reviewer.
    child_runtime = TaskRuntime(c.manager, tenant_id=c.actor.tenant_id, principal_id=c.actor.principal_id,
        executor_id=c.owner.executor_id, boot_id=c.owner.boot_id)
    loop = StatefulAgentLoop(llm=model, store=c.store, task_runtime=child_runtime,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3, max_tokens=20))
    return loop, model, Execution(child_runtime, c.actor, c.owner)


@pytest.mark.parametrize("choice", ["continue", "defer", "cancel"])
async def test_apply_original_decision_after_main_closes_is_atomic_single_and_recoverable(handoff, choice):
    h, c = handoff, handoff.c
    await decide(h, choice)
    main_calls = h.model.calls
    replies = await asyncio.gather(*(effects.apply(c.manager, c.main, h.event_id) for _ in range(5)))
    assert all(r == replies[0] for r in replies)
    restored = JobManager(await c.connect(), now_ms=lambda: c.now[0])
    assert await effects.apply(restored, c.main, h.event_id) == replies[0]
    assert h.model.calls == main_calls and not replies[0]["physical_stop_confirmed"]
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_effect'")) == 1
    ops = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action IN ('nudge','stop')")
    inbox = await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.follow_up_queue} WHERE kind='relay'")
    if choice == "continue":
        [op], [item] = ops, inbox
        assert op["call_ref_json"] is None and op["receipt_ref_json"] is None and op["tool_call_id"] is None
        assert item["content"] == c.request.continuation and item["item_id"] == "nudge:" + op["event_id"]
    elif choice == "cancel":
        assert len(ops) == 1 and not inbox
        assert (await c.manager.get_job(c.main, c.owner.job_id)).cancel_requested_at is not None
    else:
        assert not ops and not inbox and replies[0]["next_review_at_ms"] == c.now[0] + 1000
    timer = await c.manager.get_job(c.main, h.registration.job.job_id)
    assert timer.outcome is None and timer.resource_state == "not_started"
    assert len(await c.store.list_children(c.main.session_id)) == 1


@pytest.mark.parametrize("choice", ["continue", "defer"])
@pytest.mark.parametrize("fault", ["cancel", "generation", "deadline", "expired", "receipt"])
async def test_new_effect_requires_current_original_authority_and_committed_decision_receipt(handoff, choice, fault):
    h, c = handoff, handoff.c
    await decide(h, choice)
    if fault == "cancel":
        await c.manager.request_cancel(c.main, c.owner.job_id, system_key="stop-before-effect", reason="stop")
    elif fault == "generation":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET generation=2 WHERE job_id=?", (c.owner.job_id,))
    elif fault == "deadline":
        c.now[0] = c.policy.deadline_at_ms
    elif fault == "expired":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET lease_until_ms=1 WHERE job_id=?", (c.owner.job_id,))
    else:
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET receipt_ref_json=NULL WHERE action='wait_decision'")
    with pytest.raises(JobError):
        await effects.apply(c.manager, c.main, h.event_id)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_effect'")
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.follow_up_queue} WHERE kind='relay'")


async def test_cancel_is_mechanical_after_deadline_and_lease_loss_but_never_crosses_generations(handoff):
    h, c = handoff, handoff.c
    await decide(h, "cancel")
    c.now[0] = c.policy.deadline_at_ms + 1
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET lease_until_ms=1 WHERE job_id=?", (c.owner.job_id,))
    reply = await effects.apply(c.manager, c.main, h.event_id)
    assert reply["status"] == "stop_requested" and not reply["physical_stop_confirmed"]
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET generation=2 WHERE job_id=?", (c.owner.job_id,))
    assert await effects.apply(c.manager, c.main, h.event_id) == reply  # Historical replay only.
    [operation] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE action='stop'")
    assert json.loads(operation["payload_json"])["request"]["target"]["generation"] == 1


@pytest.mark.parametrize("point", ["owner", "operation", "inbox", "effect", "decision_status"])
async def test_system_effect_storage_failure_rolls_back_input_operation_owner_and_receipt(handoff, monkeypatch, point):
    h, c = handoff, handoff.c
    await decide(h)
    before = await c.manager.get_job(c.main, c.owner.job_id)
    async with c.manager._write() as tx:
        transaction_type = type(tx)
    execute = transaction_type.execute

    async def fail(tx, sql, params=()):
        result = await execute(tx, sql, params)
        if ((point == "owner" and sql.startswith("UPDATE " + c.manager.jobs) and params[-1] == c.owner.job_id)
                or (point == "operation" and sql.startswith("INSERT INTO " + c.manager.events) and params[2] == "operation")
                or (point == "inbox" and c.store.t.follow_up_queue in sql and sql.startswith("INSERT"))
                or (point == "effect" and sql.startswith("INSERT INTO " + c.manager.events) and params[2] == "wait_effect")
                or (point == "decision_status" and sql.startswith("UPDATE " + c.manager.events + " SET delivery_status='delivered'"))):
            raise RuntimeError("system effect storage failed")
        return result

    monkeypatch.setattr(transaction_type, "execute", fail)
    with pytest.raises(RuntimeError, match="system effect storage failed"):
        await effects.apply(c.manager, c.main, h.event_id)
    assert (await c.manager.get_job(c.main, c.owner.job_id)).version == before.version
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge' OR event_type='wait_effect'")
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.follow_up_queue} WHERE kind='relay'")


async def test_original_child_consumes_system_input_before_required_wait_can_resolve(handoff):
    h, c = handoff, handoff.c
    await decide(h)
    reply = await effects.apply(c.manager, c.main, h.event_id)
    with pytest.raises(JobError, match="required_result_unhandled"):
        await effects.settle_owner(c.manager, c.actor, c.owner)
    loop, model, binding = await child_loop(h)
    async with loop.task_runtime.execution(binding, owner=True):
        result = await loop.flush_follow_ups(c.sid)
        assert result.status == "completed"
        assert c.request.continuation in str(model.requests[0].messages)
        # The runtime's owning-adapter result path performs the mechanical
        # input proof check; a text answer alone cannot resolve this timer.
        saved = await loop.task_runtime.record_scope(binding, {"status": "completed", "final_text": result.final_text})
    assert saved["status"] == "completed"
    timer = await c.manager.get_job(c.main, h.registration.job.job_id)
    assert timer.outcome == "succeeded" and timer.resource_state == "stopped" and timer.generation == 0
    assert (await effects.apply(c.manager, c.main, h.event_id))["effect_status"] == "handled"
    assert reply["effect_status"] == "delivered"
    assert (await c.manager.get_job(c.main, c.owner.job_id)).outcome == "succeeded"
    assert len(await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_resolved'")) == 1
    await effects.settle_owner(c.manager, c.actor, c.owner)


@pytest.mark.parametrize("fault", ["body", "erase_binding", "effect_hash", "missing_effect", "inbox", "fake_restore"])
async def test_tampered_system_input_or_effect_never_dispatches_a_child_model(handoff, fault):
    h, c = handoff, handoff.c
    await decide(h)
    reply = await effects.apply(c.manager, c.main, h.event_id)
    [op] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='nudge'")
    payload = json.loads(op["payload_json"])
    if fault == "body":
        payload["request"]["message"] = "replacement task"
    elif fault == "erase_binding":
        del payload["request"]["handoff"]
    elif fault == "effect_hash":
        payload["request"]["handoff"]["decision_hash"] = "forged"
    elif fault == "missing_effect":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (reply["effect_id"],))
    elif fault == "inbox":
        await c.manager.db.execute(f"UPDATE {c.store.t.follow_up_queue} SET content='replacement task' WHERE kind='relay'")
    else:
        del payload["request"]["handoff"]
        payload.update(job_id=op["job_id"], mode="restore_input")
        del payload["version"]
        payload["request"].update(restore_operation_id="fake-restore", node={}, workflow={})
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?",
        (_json(payload), _hash(payload["request"]), op["event_id"]))
    loop, model, binding = await child_loop(h)
    async with loop.task_runtime.execution(binding, owner=True):
        with pytest.raises(JobError):
            await loop.flush_follow_ups(c.sid)
    assert model.calls == 0
    assert (await c.manager.get_job(c.main, h.registration.job.job_id)).outcome is None


async def test_due_at_deadline_is_rejected_before_any_wait_is_created(waiting):
    c = waiting
    with pytest.raises(JobError, match="wait_deadline_out_of_bounds"):
        await c.register(request=replace(c.request, due_at_ms=c.policy.deadline_at_ms))
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='timer'")


async def test_deadline_is_exclusive_for_new_due_events_and_decisions(handoff):
    h, c = handoff, handoff.c
    c.now[0] = c.policy.deadline_at_ms
    with pytest.raises(JobError, match="wait_deadline_exceeded"):
        await waits.prepare_due(c.manager, c.main, h.registration.job.job_id)
    with pytest.raises(JobError, match="wait_deadline_exceeded"):
        await notices.enqueue_notification(h.loop, c.main, h.event_id)


async def test_cancel_new_effect_cannot_target_a_later_generation(handoff):
    h, c = handoff, handoff.c
    await decide(h, "cancel")
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET generation=2 WHERE job_id=?", (c.owner.job_id,))
    with pytest.raises(JobError, match="stale_execution"):
        await effects.apply(c.manager, c.main, h.event_id)
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='stop'")


async def test_child_cannot_apply_its_own_main_decision_or_redirect_the_effect(handoff):
    h, c = handoff, handoff.c
    await decide(h)
    with pytest.raises(JobError, match="access_denied"):
        await effects.apply(c.manager, c.actor, h.event_id)
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.follow_up_queue} WHERE kind='relay'")


@pytest.mark.parametrize("point", ["resolution", "timer_result", "timer_state", "owner_result"])
async def test_owner_success_and_required_wait_resolution_commit_or_roll_back_together(handoff, monkeypatch, point):
    h, c = handoff, handoff.c
    await decide(h)
    await effects.apply(c.manager, c.main, h.event_id)
    loop, model, binding = await child_loop(h)
    async with loop.task_runtime.execution(binding, owner=True):
        answer = await loop.flush_follow_ups(c.sid)
        async with c.manager._write() as tx:
            transaction_type = type(tx)
        execute = transaction_type.execute

        async def fail(tx, sql, params=()):
            result = await execute(tx, sql, params)
            if ((point == "resolution" and sql.startswith("INSERT INTO " + c.manager.events) and params[2] == "wait_resolved")
                    or (point == "timer_result" and sql.startswith("INSERT INTO " + c.manager.events)
                        and params[2] == "result" and params[1] == h.registration.job.job_id)
                    or (point == "timer_state" and sql.startswith("UPDATE " + c.manager.jobs + " SET status='settling',outcome='succeeded'"))
                    or (point == "owner_result" and sql.startswith("INSERT INTO " + c.manager.events)
                        and params[2] == "result" and params[1] == c.owner.job_id)):
                raise RuntimeError("result transaction failed")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(transaction_type, "execute", fail)
            with pytest.raises(RuntimeError, match="result transaction failed"):
                await c.manager.record_execution_result(c.actor, c.owner, outcome="succeeded",
                    result={"final_text": answer.final_text}, resource_state="stopped")
        assert (await c.manager.get_job(c.main, h.registration.job.job_id)).outcome is None
        assert (await c.manager.get_job(c.main, c.owner.job_id)).outcome is None
        assert not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_resolved'")
        event = await c.manager.record_execution_result(c.actor, c.owner, outcome="succeeded",
            result={"final_text": answer.final_text}, resource_state="stopped")
        assert event == await c.manager.record_execution_result(c.actor, c.owner, outcome="succeeded",
            result={"final_text": answer.final_text}, resource_state="stopped")
        assert (await c.manager.get_job(c.main, h.registration.job.job_id)).outcome == "succeeded"
        assert len(await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_resolved'")) == 1


@pytest.mark.parametrize("point", ["stop", "effect"])
async def test_cancel_storage_failure_keeps_original_owner_and_wait_open(handoff, monkeypatch, point):
    h, c = handoff, handoff.c
    await decide(h, "cancel")
    async with c.manager._write() as tx:
        transaction_type = type(tx)
    execute = transaction_type.execute

    async def fail(tx, sql, params=()):
        result = await execute(tx, sql, params)
        if ((point == "stop" and sql.startswith("INSERT INTO " + c.manager.events)
                and params[2] == "operation" and params[6] == "stop")
                or (point == "effect" and sql.startswith("INSERT INTO " + c.manager.events) and params[2] == "wait_effect")):
            raise RuntimeError("cancel transaction failed")
        return result

    monkeypatch.setattr(transaction_type, "execute", fail)
    with pytest.raises(RuntimeError, match="cancel transaction failed"):
        await effects.apply(c.manager, c.main, h.event_id)
    owner = await c.manager.get_job(c.main, c.owner.job_id)
    [raw_owner] = await c.manager.db.fetchall(f"SELECT intake FROM {c.manager.jobs} WHERE job_id=?", (c.owner.job_id,))
    assert owner.cancel_requested_at is None and raw_owner["intake"] == "open" and owner.status == "running"
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='stop' OR event_type='wait_effect'")


async def test_actual_workflow_node_consumes_original_system_input_before_review_can_start(managed_workflow, monkeypatch):
    store, manager, task_runtime, loop, model, parent, state, _ = managed_workflow
    state["spec"]["root"]["steps"][0]["spec"]["tools"] = ["arm_wait"]
    register_wait_handoff_tool(loop.tool_registry)
    ready, release = asyncio.Event(), asyncio.Event()
    registered, pending = [], {}
    original = model.complete

    async def arm():
        source, sid = _SEND.get(), get_session_id()
        seq, call_id = current_invocation(sid)
        now = manager.now()
        registration = await waits.register_wait(manager, task_runtime.actor(sid), caller=source.lease,
            call_ref=await store.message_ref(sid, seq), tool_call_id=call_id,
            request=waits.WaitRequest(now + 250, "Review this BUILD node.", "Continue original BUILD: PROOF-42."),
            policy=waits.WaitPolicy(now + 10000))
        registered.append(registration)
        return "Original node wait accepted."

    async def complete(request, **kwargs):
        if get_session_id() == parent and pending and not pending.get("decided"):
            pending["decided"] = True
            return LLMResponse(raw_text="", tool_calls=[{"id": "decide-original-node", "type": "function", "function": {
                "name": "wait_handoff", "arguments": json.dumps({"action": "decide", "event_id": pending["event_id"], "decision": "continue"})}}],
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
        result = await original(request, **kwargs)
        if "ROLE-BUILD" in (request.system_prompt or ""):
            if model.counts["BUILD"] == 1:
                return LLMResponse(raw_text="", tool_calls=[{"id": "node-arm", "type": "function", "function": {
                    "name": "arm_wait", "arguments": "{}"}}],
                    token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
            ready.set()
            await release.wait()
        return result

    monkeypatch.setattr(model, "complete", complete)
    loop.tool_registry.register(ToolDefinition(name="arm_wait", description="Register a node wait", input_schema={"type": "object"}), arm)
    try:
        await loop.send("Start workflow", parent)
        model.release["BUILD"].set()
        await asyncio.wait_for(ready.wait(), 5)
        [registration] = registered
        saved = await waits.read_wait(manager, task_runtime.actor(parent), registration.job.job_id)
        await asyncio.sleep(max(0, (saved["request"]["request"]["due_at_ms"] - manager.now()) / 1000) + .01)
        due = await waits.prepare_due(manager, task_runtime.actor(parent), registration.job.job_id)
        pending["event_id"] = due["event_id"]
        await notices.enqueue_notification(loop, task_runtime.actor(parent), due["event_id"])
        summary = await decisions_summary(manager, task_runtime.actor(parent), due["event_id"])
        route = summary["notification"]["original_owner"]
        assert route["kind"] == "workflow_node" and route["node"]["node_id"] == "BUILD"
        assert route["workflow_job_id"] == state["handles"][0].job_id and model.counts["REVIEW"] == 0
        await loop.flush_follow_ups(parent)
        effect = await effects.apply(manager, task_runtime.actor(parent), due["event_id"])
        assert effect["status"] == "input_enqueued" and model.counts["REVIEW"] == 0
        release.set()
        await asyncio.wait_for(state["handles"][0].task, 8)
        assert (await manager.get_job(task_runtime.actor(parent), state["handles"][0].job_id)).outcome == "succeeded"
        assert model.counts["REVIEW"] == 1
        assert (await manager.get_job(task_runtime.actor(parent), registration.job.control_owner_job_id)).outcome == "succeeded"
        assert (await manager.get_job(task_runtime.actor(parent), registration.job.job_id)).outcome == "succeeded"
        assert (await effects.apply(manager, task_runtime.actor(parent), due["event_id"]))["effect_status"] == "handled"
    finally:
        for registration in registered:
            timer = await manager.get_job(task_runtime.actor(parent), registration.job.job_id)
            if timer.outcome is None:
                await manager.request_cancel(task_runtime.actor(parent), registration.job.job_id, system_key="node-effects-cleanup", reason="cleanup")
                await manager.record_unstarted_result(task_runtime.actor(registration.job.created_by_session_id), registration.job.job_id,
                    outcome="cancelled", result={"reason": "test cleanup"})
        release.set()


async def decisions_summary(manager, actor, event_id):
    from power_loop.runtime.wait_decisions import inspect
    return await inspect(manager, actor, event_id)


@pytest.mark.parametrize("fault", ["missing_resolution", "empty_proofs", "missing_result", "projection"])
async def test_handled_effect_replay_requires_original_resolution_not_a_bare_handled_flag(handoff, fault):
    h, c = handoff, handoff.c
    await decide(h)
    await effects.apply(c.manager, c.main, h.event_id)
    loop, _, binding = await child_loop(h)
    async with loop.task_runtime.execution(binding, owner=True):
        result = await loop.flush_follow_ups(c.sid)
        await c.manager.record_execution_result(c.actor, c.owner, outcome="succeeded",
            result={"final_text": result.final_text}, resource_state="stopped")
    [resolved] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_resolved'")
    if fault == "missing_resolution":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (resolved["event_id"],))
    elif fault == "empty_proofs":
        payload = json.loads(resolved["payload_json"])
        payload["proofs"] = []
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?",
            (_json(payload), _hash(payload), resolved["event_id"]))
    elif fault == "missing_result":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (h.registration.job.job_id,))
    else:
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET outcome='cancelled' WHERE job_id=?", (h.registration.job.job_id,))
    with pytest.raises(JobError, match="wait_resolution_evidence_(missing|corrupt)"):
        await effects.apply(c.manager, c.main, h.event_id)
    assert len(await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.follow_up_queue} WHERE kind='relay'")) == 1


async def test_cancelled_wait_cannot_be_rewritten_as_success_after_child_consumes_its_input(handoff):
    h, c = handoff, handoff.c
    await decide(h)
    await effects.apply(c.manager, c.main, h.event_id)
    await c.manager.request_cancel(c.main, h.registration.job.job_id, system_key="cancel-wait", reason="stop wait")
    loop, _, binding = await child_loop(h)
    async with loop.task_runtime.execution(binding, owner=True):
        result = await loop.flush_follow_ups(c.sid)
        with pytest.raises(JobError, match="wait_state_changed"):
            await c.manager.record_execution_result(c.actor, c.owner, outcome="succeeded",
                result={"final_text": result.final_text}, resource_state="stopped")
    timer = await c.manager.get_job(c.main, h.registration.job.job_id)
    assert timer.cancel_requested_at is not None and timer.outcome is None


@pytest.mark.parametrize("fault", ["key", "caller", "target", "restore_operation", "workflow", "node"])
async def test_native_restore_system_input_keeps_its_own_verified_historical_authority(native_restore, fault):
    c = native_restore
    handle = await c.restore()
    await asyncio.wait_for(handle.task, 8)
    assert (await c.manager.get_job(c.actor, handle.job_id)).outcome == "succeeded"
    events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='nudge' AND call_ref_json IS NULL")
    [event] = [e for e in events if json.loads(e["payload_json"])["request"]["node"]["node_id"] == "BUILD"]
    async with c.manager._write() as tx:
        await effects.validate_input(c.manager, tx, c.actor, event)
    payload = json.loads(event["payload_json"])
    if fault == "key":
        event["operation_key"] = "forged"
    elif fault == "caller":
        event["caller_session_id"] = c.parent
    elif fault == "target":
        payload["request"]["target"]["generation"] = True
    elif fault == "restore_operation":
        payload["request"]["restore_operation_id"] = "fake"
    elif fault == "workflow":
        payload["request"]["workflow"]["generation"] = True
    else:
        payload["request"]["node"]["node_id"] = "wrong-node"
    event["payload_json"] = _json(payload)
    event["request_hash"] = _hash(payload["request"])
    async with c.manager._write() as tx:
        with pytest.raises(JobError):
            await effects.validate_input(c.manager, tx, c.actor, event)
