"""Finished-leaf generations remain owned by their original active workflow."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop.runtime import execution_membership
from power_loop.runtime.jobs import JobError
from power_loop.workflow import runner

from .test_job_registry_192 import registry_store
from .test_node_nudge_runtime_192 import client, target
from .test_workflow_task_runtime_192 import leaf, managed_workflow, settle

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def ready_node(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    c.model.hold = {"AUDIT"}
    c.state["spec"]["root"] = {"type": "sequence", "steps": [
        {"type": "parallel", "branches": [leaf("BUILD"), leaf("AUDIT")]},
        leaf("REVIEW", inputs_from=["BUILD"])]}
    await c.loop.send("start parallel workflow", c.parent)
    await asyncio.wait_for(c.model.entered["AUDIT"].wait(), 3)
    c.run_id = c.state["handles"][0].run_id
    c.engine = runner._ENGINES[c.run_id]
    for _ in range(200):
        rows = [(row, data) for row, data in await target(c) if data["node_id"] == "BUILD"]
        if rows:
            row, data = rows[0]
            job = await c.manager.get_job(c.runtime.actor(c.parent), row["job_id"])
            if job.resource_state == "stopped" and any(not f["active"] and f["result"] and f["result"].node_id == "BUILD" for f in c.engine._frames.values()):
                c.sid, c.job_id, c.data, c.old = data["session_id"], row["job_id"], data, job
                break
        await asyncio.sleep(.01)
    else:
        pytest.fail("BUILD did not settle before parallel sibling")
    return c


@pytest.mark.parametrize("fault", ["none", "admission_ack", "dispatch_ack", "replay"])
async def test_finished_leaf_resumes_same_job_sid_before_downstream(ready_node, managed_workflow, monkeypatch, fault):
    c = ready_node
    calls = {"admit": 0, "dispatch": 0}
    for name, key in (("resume_workflow_node", "admit"), ("claim_resume_dispatch", "dispatch")):
        original = getattr(c.manager, name)
        async def wrapped(*args, _fn=original, _key=key, **kwargs):
            calls[_key] += 1
            value = await _fn(*args, **kwargs)
            if fault == ("admission_ack" if _key == "admit" else "dispatch_ack"):
                raise RuntimeError("committed, ACK lost")
            return value
        monkeypatch.setattr(c.manager, name, wrapped)
    async def handler():
        receipt = await runner.steer_node(c.run_id, "BUILD", c.sid, "NUDGED-43")
        if fault == "replay":
            again = await runner.steer_node(c.run_id, "BUILD", c.sid, "NUDGED-43")
            assert again["operation_id"] == receipt["operation_id"] and again["replayed"]
        return receipt
    await c.issue(handler=handler)
    assert len(c.replies) == 1, await c.store.load_all_messages(c.parent)
    receipt = c.replies[0]
    assert receipt["mode"] == "continue" and receipt["generation"] == 2
    assert receipt["job_id"] == c.job_id and receipt["session_id"] == c.sid and receipt["run_id"] == c.run_id
    assert calls == {"admit": 1, "dispatch": 1} and c.model.counts["REVIEW"] == 0
    await settle(managed_workflow)
    actor = c.runtime.actor(c.parent)
    job = await c.manager.get_job(actor, c.job_id)
    assert job.generation == 2 and job.outcome == "succeeded" and job.result["text"] == "NUDGED-43", job
    assert job.result["usage"]["total_tokens"] == 10
    assert (await c.manager.get_result_revision(actor, c.job_id, 1))["payload"]["result"] == c.old.result
    assert (await c.manager.get_result_revision(actor, c.job_id, 2))["payload"]["result"] == job.result
    wf = await c.manager.get_job(actor, c.state["handles"][0].job_id)
    assert wf.outcome == "succeeded" and wf.generation == 1
    assert wf.result["usage"]["total_tokens"] == 40
    assert await c.manager.execution_usage(actor, c.job_id, generation=1) == c.old.result["usage"]
    assert await c.manager.execution_usage(actor, c.job_id, generation=2) == job.result["usage"]
    assert await c.manager.execution_usage(actor, wf.job_id, generation=1) == wf.result["usage"]
    members = await c.manager.get_execution_membership(actor, wf.job_id, 1)
    assert members["sealed"] and len([e for e in members["members"] if e["payload"]["member"]["job_id"] == c.job_id]) == 2
    assert c.model.counts["BUILD"] == 2 and c.model.counts["AUDIT"] == c.model.counts["REVIEW"] == 1
    assert any(sid != c.sid and "ROLE" not in text and "NUDGED-43" in text for sid, text in c.seen)
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='nudge'")
    assert event["delivery_status"] == "handled" and event["receipt_ref_json"]
    assert json.loads(event["payload_json"])["request"]["previous_generation"] == 1


async def test_rejects_new_call_after_workflow_exit(ready_node, managed_workflow):
    c = ready_node
    await settle(managed_workflow)
    before = c.model.counts.copy()
    await c.issue()
    assert not c.replies and c.model.counts["BUILD"] == before["BUILD"]


async def test_direct_resume_still_requires_actual_main_send(ready_node):
    c = ready_node
    with pytest.raises(JobError, match="task_send_required"):
        await runner.steer_node(c.run_id, "BUILD", c.sid, "correction")


async def test_two_successive_generations_preserve_old_inputs_and_results(ready_node, managed_workflow):
    c = ready_node
    for generation in (2, 3):
        await c.issue(message="NUDGED-43")
        assert c.replies[-1]["generation"] == generation
        await asyncio.wait_for(asyncio.shield(next(iter(c.engine._repairs.values()))), 4)
        job = await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)
        assert job.generation == generation and job.outcome == "succeeded"
    await settle(managed_workflow)
    rows = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (c.job_id,))
    assert len(rows) == 3
    assert (await c.manager.execution_usage(c.runtime.actor(c.parent), c.job_id))["total_tokens"] == 30


@pytest.mark.parametrize("fault", ["plan", "result", "contract", "source", "workflow", "pending_tools", "pending_input",
    "failed", "unknown", "open", "generation", "journal", "cancel_node", "cancel_workflow", "scope_missing", "seal_missing", "edge_missing"])
async def test_admission_rechecks_frozen_evidence_and_no_replacement_runs(ready_node, monkeypatch, fault):
    c = ready_node
    original, errors = c.manager.resume_workflow_node, []
    async def invalid(actor, job_id, **kwargs):
        if fault in ("plan", "result", "contract"):
            kwargs[{"plan": "plan", "result": "previous_result", "contract": "contract"}[fault]] = {}
        elif fault in ("source", "workflow"):
            from dataclasses import replace
            key = "caller" if fault == "source" else "workflow"
            kwargs[key] = replace(kwargs[key], generation=999)
        elif fault == "pending_tools":
            await c.manager.db.execute(f"UPDATE {c.store.t.session_state} SET pending_json='{{}}' WHERE session_id=?", (c.sid,))
        elif fault == "pending_input":
            await c.store.inbox_put(c.sid, [{"item_id": "previous-input", "content": "old input"}])
        elif fault == "journal":
            from power_loop.workflow import journal
            await c.store.set_runtime_state(c.parent, journal.run_key(c.run_id), {"status": "running", "steps": []})
        elif fault.startswith("cancel_"):
            await c.manager.request_cancel(actor, job_id if fault == "cancel_node" else kwargs["workflow"].job_id,
                system_key="test-stop", reason="test")
        elif fault in ("scope_missing", "seal_missing", "edge_missing"):
            key = {"scope_missing": execution_membership.scope_id(c.manager, job_id, 1),
                "seal_missing": execution_membership.seal_id(c.manager, job_id, 1)}.get(fault)
            if key:
                await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (key,))
            else:
                await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='execution_member'", (job_id,))
        else:
            column, value = {"failed": ("outcome", "failed"), "unknown": ("resource_state", "unknown"),
                "open": ("intake", "open"), "generation": ("generation", 9)}[fault]
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET {column}=? WHERE job_id=?", (value, job_id))
        try:
            return await original(actor, job_id, **kwargs)
        except JobError as exc:
            errors.append(exc.code)
            raise
    monkeypatch.setattr(c.manager, "resume_workflow_node", invalid)
    await c.issue()
    assert not c.replies and len(errors) == 1
    assert c.model.counts["BUILD"] == 1
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type IN ('execution','dispatch')")
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


@pytest.mark.parametrize("fault", ["binding_insert", "execution_insert", "inbox_insert", "execution_scope", "execution_member"])
async def test_partial_admission_rolls_back_all_records(ready_node, monkeypatch, fault):
    c = ready_node
    # The transaction has its own executor; inject through its transaction port.
    from contextlib import asynccontextmanager
    write = c.manager._write
    @asynccontextmanager
    async def broken():
        async with write() as tx:
            execute = tx.execute
            async def wrapped(sql, params=()):
                if sql.startswith("INSERT") and ((fault == "binding_insert" and "node_session" in params) or
                        (fault == "execution_insert" and "execution" in params) or
                        (fault == "execution_scope" and fault in params and c.job_id in params) or
                        (fault == "execution_member" and fault in params and params[0] == execution_membership.member_id(c.manager, c.job_id, 2)) or
                        (fault == "inbox_insert" and c.store.t.follow_up_queue in sql)):
                    raise RuntimeError("injected write failure")
                return await execute(sql, params)
            with monkeypatch.context() as patch:
                patch.setattr(tx, "execute", wrapped)
                yield tx
    monkeypatch.setattr(c.manager, "_write", broken)
    await c.issue()
    assert not c.replies
    job = await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)
    assert job.generation == 1 and job.result == c.old.result
    assert c.model.counts["BUILD"] == 1
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type IN ('execution','dispatch')")


async def test_parallel_join_waits_for_resumed_leaf(ready_node, managed_workflow, monkeypatch):
    c = ready_node
    entered, release = asyncio.Event(), asyncio.Event()
    original = c.model.complete
    async def held(request, **kwargs):
        from power_loop.core.agent_context import get_session_id
        if get_session_id() == c.sid and "NUDGED-43" in str(request.messages):
            entered.set()
            await release.wait()
        return await original(request, **kwargs)
    monkeypatch.setattr(c.model, "complete", held)
    try:
        await c.issue()
        assert len(c.replies) == 1
        await asyncio.wait_for(entered.wait(), 3)
        c.model.release["AUDIT"].set()
        await asyncio.sleep(.05)
        assert c.model.counts["REVIEW"] == 0 and not c.state["handles"][0].task.done()
        await c.issue(message="SECOND-43")
        assert c.replies[-1]["mode"] == "steer" and c.replies[-1]["generation"] == 2
        release.set()
        await settle(managed_workflow)
        assert c.model.counts["REVIEW"] == 1
        inputs = await c.manager.db.fetchall(f"SELECT delivery_status FROM {c.manager.events} WHERE job_id=? AND action='nudge'", (c.job_id,))
        assert len(inputs) == 2 and all(row["delivery_status"] == "handled" for row in inputs)
    finally:
        release.set()


@pytest.mark.parametrize("fault", ["dispatch_denied", "dispatch_error", "bind_error"])
async def test_admitted_but_undispatched_generation_cannot_leave_old_success(ready_node, managed_workflow, monkeypatch, fault):
    c = ready_node
    if fault == "bind_error":
        bind = c.manager.bind_node_session
        async def broken(actor, proof, sid):
            if proof.generation > 1:
                raise RuntimeError("binding verification failed")
            return await bind(actor, proof, sid)
        monkeypatch.setattr(c.manager, "bind_node_session", broken)
    else:
        async def broken(*args, **kwargs):
            if fault == "dispatch_error":
                raise RuntimeError("dispatch unavailable")
            return False
        monkeypatch.setattr(c.manager, "claim_resume_dispatch", broken)
    await c.issue()
    assert not c.replies
    await settle(managed_workflow)
    actor = c.runtime.actor(c.parent)
    job = await c.manager.get_job(actor, c.job_id)
    assert job.generation == 2
    if fault == "bind_error":
        assert job.outcome == "failed" and job.resource_state == "stopped"
    else:
        # Without our exact dispatch proof, another worker may own execution.
        # Retain a cancellable unresolved ledger, never invent stopped evidence.
        assert job.result is None and job.resource_state == "running"
        await c.manager.request_cancel(actor, c.job_id, system_key="stop-undispatched", reason="test")
    assert (await c.manager.get_job(actor, c.state["handles"][0].job_id)).outcome == "failed"
    assert c.model.counts["BUILD"] == 1


async def test_reservation_blocks_join_even_while_admission_waits_and_caller_disconnects(ready_node, managed_workflow, monkeypatch):
    c = ready_node
    entered, release = asyncio.Event(), asyncio.Event()
    admit = c.manager.resume_workflow_node
    async def held(*args, **kwargs):
        entered.set()
        await release.wait()
        return await admit(*args, **kwargs)
    monkeypatch.setattr(c.manager, "resume_workflow_node", held)
    issuer = asyncio.create_task(c.issue())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        c.model.release["AUDIT"].set()
        await asyncio.sleep(.05)
        assert not c.state["handles"][0].task.done() and c.model.counts["REVIEW"] == 0
        issuer.cancel()
        await asyncio.gather(issuer, return_exceptions=True)
        release.set()
        await settle(managed_workflow)
        assert c.model.counts["BUILD"] == 1
        assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).generation == 1
    finally:
        release.set()
        await asyncio.gather(issuer, return_exceptions=True)


async def test_task_creation_failure_has_no_admission_and_closes_coroutine(ready_node, monkeypatch):
    c = ready_node
    create = asyncio.create_task
    async def handler():
        def fail(coro, **kwargs):
            if kwargs.get("name") == "workflow-node-resume-BUILD":
                raise RuntimeError("cannot create repair task")
            return create(coro, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "create_task", fail)
            with pytest.raises(RuntimeError, match="cannot create"):
                await runner.steer_node(c.run_id, "BUILD", c.sid, "NUDGED-43")
        return {"status": "rejected"}
    await c.issue(handler=handler)
    assert not c.engine._repairs
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).generation == 1


async def test_cancelling_resumed_execution_stops_before_workflow_settlement(ready_node, managed_workflow, monkeypatch):
    c = ready_node
    entered, release = asyncio.Event(), asyncio.Event()
    complete = c.model.complete
    async def held(request, **kwargs):
        from power_loop.core.agent_context import get_session_id
        if get_session_id() == c.sid and "NUDGED-43" in str(request.messages):
            entered.set()
            await release.wait()
        return await complete(request, **kwargs)
    monkeypatch.setattr(c.model, "complete", held)
    try:
        await c.issue()
        await asyncio.wait_for(entered.wait(), 3)
        repair = next(iter(c.engine._repairs.values()))
        await c.manager.request_cancel(c.runtime.actor(c.parent), c.state["handles"][0].job_id,
            system_key="stop-whole-workflow", reason="test")
        repair.cancel()
        await asyncio.gather(repair, return_exceptions=True)
        release.set()
        await settle(managed_workflow)
        node = await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)
        assert node.resource_state == "unknown" and node.outcome == "cancelled" and node.supervision_status == "open"
        assert c.model.counts["REVIEW"] == 0
        assert not c.engine._task_nodes and all(t.done() for t in c.engine._repairs.values())
    finally:
        release.set()


@pytest.mark.parametrize("fault", ["consumer_settled", "barrier", "executor", "wrong_sid"])
async def test_local_scope_rejection_precedes_generation_mutation(ready_node, monkeypatch, fault):
    c = ready_node
    from power_loop.runtime.node_resume import selection
    from power_loop.workflow.result import AgentResult
    node, _, _ = selection(c.engine._spec, "BUILD")
    if fault == "consumer_settled":
        c.engine._results["REVIEW"] = AgentResult(node_id="REVIEW", status="completed", text="old")
    elif fault == "barrier":
        c.engine._continuation_barriers.add(id(node))
    elif fault == "executor":
        monkeypatch.setattr(c.engine, "_executor", object())
    await c.issue(sid="wrong" if fault == "wrong_sid" else c.sid)
    assert not c.replies
    assert not c.engine._repairs and c.model.counts["BUILD"] == 1
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).generation == 1


@pytest.mark.parametrize("fault", ["provider_error", "empty_answer"])
async def test_failed_or_empty_new_generation_preserves_old_result_and_blocks_success(ready_node, managed_workflow, monkeypatch, fault):
    c = ready_node
    complete = c.model.complete
    async def broken(request, **kwargs):
        from power_loop import LLMResponse, LLMTokenUsage
        from power_loop.core.agent_context import get_session_id
        if get_session_id() == c.sid and "NUDGED-43" in str(request.messages):
            if fault == "provider_error":
                raise RuntimeError("provider unavailable")
            return LLMResponse(raw_text="", token_usage=LLMTokenUsage(total_tokens=10))
        return await complete(request, **kwargs)
    monkeypatch.setattr(c.model, "complete", broken)
    await c.issue()
    assert len(c.replies) == 1
    await settle(managed_workflow)
    actor = c.runtime.actor(c.parent)
    node = await c.manager.get_job(actor, c.job_id)
    assert node.generation == 2 and node.outcome == "failed" and node.resource_state == ("unknown" if fault == "provider_error" else "stopped")
    if fault == "provider_error":
        assert node.supervision_status == "open"
    assert (await c.manager.get_result_revision(actor, c.job_id, 1))["payload"]["result"] == c.old.result
    assert (await c.manager.get_job(actor, c.state["handles"][0].job_id)).outcome == "failed"


@pytest.mark.parametrize("key,value,code", [
    ("message", "", "invalid_nudge_message"), ("message", "bad\x00input", "invalid_nudge_message"),
    ("message", 1, "invalid_nudge_message"), ("message", "x"*64001, "invalid_nudge_message"),
    ("caller", None, "execution_lease_required"), ("workflow", None, "execution_lease_required"),
    ("executor_id", "", "invalid_executor"), ("boot_id", True, "invalid_executor"),
    ("lease_ms", True, "invalid_lease_duration"), ("lease_ms", 0, "invalid_lease_duration"),
    ("lease_ms", 300001, "invalid_lease_duration"), ("generation", True, "invalid_result_generation"),
    ("generation", 0, "invalid_result_generation"), ("generation", 2**63-1, "invalid_result_generation"),
])
async def test_invalid_port_arguments_do_not_create_a_generation(ready_node, monkeypatch, key, value, code):
    c = ready_node
    admit, errors = c.manager.resume_workflow_node, []
    async def changed(*args, **kwargs):
        if key == "generation":
            kwargs["node"] = {**kwargs["node"], "generation": value}
        else:
            kwargs[key] = value
        try:
            return await admit(*args, **kwargs)
        except JobError as exc:
            errors.append(exc.code)
            raise
    monkeypatch.setattr(c.manager, "resume_workflow_node", changed)
    await c.issue()
    assert errors == [code] and not c.replies
    assert c.model.counts["BUILD"] == 1
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).generation == 1


async def test_port_exact_replay_is_read_only_and_conflicting_replay_is_refused(ready_node, managed_workflow, monkeypatch):
    c = ready_node
    admit = c.manager.resume_workflow_node
    async def replay(*args, **kwargs):
        first, repeated = await admit(*args, **kwargs)
        second, again = await admit(*args, **kwargs)
        assert not repeated and again and first.operation_id == second.operation_id
        with pytest.raises(JobError, match="operation_conflict"):
            await admit(*args, **{**kwargs, "message": "different"})
        return first, False
    monkeypatch.setattr(c.manager, "resume_workflow_node", replay)
    await c.issue()
    assert len(c.replies) == 1
    await settle(managed_workflow)
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).outcome == "succeeded"


@pytest.mark.parametrize("fault", ["source_role", "target_role", "target_owner", "canonical_missing", "binding_closed_missing", "binding_changed"])
async def test_admission_rejects_wrong_identity_or_missing_closing_evidence(ready_node, monkeypatch, fault):
    c = ready_node
    admit, errors = c.manager.resume_workflow_node, []
    async def changed(actor, job_id, **kwargs):
        if fault == "source_role":
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET execution_role='workflow_node' WHERE job_id=?", (kwargs["caller"].job_id,))
        elif fault in ("target_role", "target_owner"):
            column = "execution_role" if fault == "target_role" else "control_owner_job_id"
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET {column}='wrong' WHERE job_id=?", (job_id,))
        elif fault == "canonical_missing":
            await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (job_id,))
        elif fault == "binding_closed_missing":
            await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='node_input_closed'", (job_id,))
        else:
            kwargs["node"] = {**kwargs["node"], "attempt": 2}
        try:
            return await admit(actor, job_id, **kwargs)
        except JobError as exc:
            errors.append(exc.code)
            raise
    monkeypatch.setattr(c.manager, "resume_workflow_node", changed)
    await c.issue()
    assert errors and not c.replies and c.model.counts["BUILD"] == 1


@pytest.mark.parametrize("fault", ["missing_dispatch", "target", "physical", "workflow", "changed_contract"])
async def test_dispatch_binding_cannot_be_borrowed_or_rewritten(ready_node, managed_workflow, monkeypatch, fault):
    c = ready_node
    bind = c.manager.bind_node_session
    async def changed(actor, proof, sid):
        if proof.generation > 1:
            if fault == "changed_contract":
                await c.store.set_runtime_state(sid, "workflow_execution_contract", {})
            elif fault == "missing_dispatch":
                await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='dispatch'", (proof.job_id,))
            elif fault == "physical":
                event_id = c.manager._node_session_event_id(proof.job_id, sid, proof.generation)
                await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
                    (json.dumps({"session_id": "other"}), event_id))
            else:
                row = await c.manager.db.fetchone(f"SELECT event_id,payload_json FROM {c.manager.events} WHERE job_id=? AND event_type='execution'", (proof.job_id,))
                payload = json.loads(row["payload_json"])
                payload["target" if fault == "target" else "workflow"]["generation"] = 999
                await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(payload), row["event_id"]))
        return await bind(actor, proof, sid)
    monkeypatch.setattr(c.manager, "bind_node_session", changed)
    await c.issue()
    assert bool(c.replies) == (fault == "changed_contract")
    await settle(managed_workflow)
    node = await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)
    assert node.outcome == "failed" and node.resource_state == "stopped" and c.model.counts["BUILD"] == 1
