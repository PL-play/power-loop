"""The original workflow really recomputes its original node sessions."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop.core.agent_context import get_session_id
from power_loop.runtime import workflow_restore_execution, workflow_results, workflow_resume
from power_loop.runtime.jobs import JobError
from power_loop.workflow import journal, runner
from power_loop.workflow.resume import resume_detached

from .test_job_registry_192 import registry_store
from .test_node_nudge_runtime_192 import client
from .test_workflow_restore_plan_192 import finished
from .test_workflow_task_runtime_192 import leaf, managed_workflow

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def native_restore(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    c.state["spec"]["root"]["steps"][-1]["inputs_from"] = ["BUILD"]
    original = c.model.complete
    async def complete(request, **kwargs):
        result = await original(request, **kwargs)
        if get_session_id() == c.parent and "NUDGED-43" in str(request.messages):
            result.raw_text = "NUDGED-43"
        return result
    monkeypatch.setattr(c.model, "complete", complete)
    c.initial = await finished(managed_workflow)
    c.actor = c.runtime.actor(c.parent)
    c.before = await c.manager.get_result_revision(c.actor, c.initial.job_id, 1)
    c.original = (await c.manager.get_job(c.actor, c.initial.job_id)).result
    c.node_jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='workflow_node'")
    async def restore(*, node="BUILD", message="NUDGED-43", repeat=False):
        handles = []
        async def handler():
            handle = await resume_detached(c.loop, c.parent, c.initial.run_id, continuation=(node, message, 3))
            handles.append(handle)
            if repeat:
                assert await resume_detached(c.loop, c.parent, c.initial.run_id, continuation=(node, message, 3)) is handle
            c.state["handles"].append(handle)
            return {"job_id": handle.job_id, "operation_id": handle.operation_id, "run_id": handle.run_id, "status": "accepted"}
        await c.issue(handler=handler)
        if handles:
            c.restored = handles[0]
            return handles[0]
        rows = await c.store.load_all_messages(c.parent)
        pytest.fail(str([(r.role, r.name, r.content) for r in rows if r.role == "tool"])[-2500:])
    c.restore = restore
    yield c


async def test_original_workflow_and_nodes_recompute_and_main_consumes_only_new_revision(native_restore):
    c = native_restore
    c.model.release["BUILD"].clear()
    c.model.entered["BUILD"].clear()
    handle = await c.restore(repeat=True)
    await asyncio.wait_for(c.model.entered["BUILD"].wait(), 4)
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert current.generation == 2 and current.status == "running" and current.result is None
    assert c.model.counts["REVIEW"] == 1
    assert handle.job_id == c.initial.job_id and handle.run_id == c.initial.run_id and handle.operation_id != c.initial.operation_id
    c.model.release["BUILD"].set()
    await asyncio.wait_for(handle.task, 10)
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert current.outcome == "succeeded" and current.result["final"]["text"] == "NUDGED-43", current
    assert await c.manager.get_result_revision(c.actor, handle.job_id, 1) == c.before
    assert current.result["usage"]["total_tokens"] == 20
    assert c.model.counts["BUILD"] == c.model.counts["REVIEW"] == 2
    assert await c.store.get_session(current.session_id) is None
    for nid, value in current.result["results"].items():
        assert value["session_id"] == c.original["results"][nid]["session_id"]
        assert (await c.store.get_session(value["session_id"])).parent_session_id is None
    assert len(await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='workflow_node'")) == 2
    consumed = await c.loop.flush_follow_ups(c.parent)
    assert consumed.final_text == "NUDGED-43" and (await c.manager.get_job(c.actor, handle.job_id)).result_status == "handled"
    [inbox] = [x for x in await c.store.inbox_items(c.parent) if x["item_id"] == "result_"+handle.job_id+"_g2"]
    envelope = json.loads(inbox["content"])
    assert envelope["operation_id"] == handle.operation_id and envelope["creation_operation_id"] == c.initial.operation_id
    assert (await journal.read(c.store, c.parent, handle.run_id))["task_generation"] == 2
    assert handle.run_id not in runner._LIVE_RUN_IDS


async def test_second_restoration_replays_unaffected_node_without_new_requests_or_usage(native_restore):
    c = native_restore
    first = await c.restore()
    await asyncio.wait_for(first.task, 10)
    await c.loop.flush_follow_ups(c.parent)
    wf2 = await c.manager.get_job(c.actor, first.job_id)
    assert wf2.outcome == "succeeded", wf2
    prior = await c.manager.get_result_revision(c.actor, first.job_id, 2)
    second = await c.restore(node="REVIEW")
    await asyncio.wait_for(second.task, 10)
    wf3 = await c.manager.get_job(c.actor, first.job_id)
    assert wf3.outcome == "succeeded" and wf3.generation == 3, wf3
    assert c.model.counts["BUILD"] == 2 and c.model.counts["REVIEW"] == 3
    assert wf3.result["usage"]["total_tokens"] == 10
    assert await c.manager.get_result_revision(c.actor, first.job_id, 2) == prior
    await c.loop.flush_follow_ups(c.parent)
    plan = await c.manager.get_workflow_restore_plan(c.actor, first.job_id, generation=3, node_id="REVIEW", message="again")
    assert plan["replay"] == ["BUILD"]


async def test_parallel_sibling_is_frozen_replay_not_a_new_execution(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    c.state["spec"]["root"] = {"type": "sequence", "steps": [
        {"type": "parallel", "branches": [leaf("BUILD"), leaf("AUDIT")]}, leaf("REVIEW", inputs_from=["BUILD"])]}
    original = await finished(managed_workflow)
    async def handler():
        handle = await resume_detached(c.loop, c.parent, original.run_id, continuation=("BUILD", "NUDGED-43", 3))
        c.state["handles"].append(handle)
        return {"operation_id": handle.operation_id}
    await c.issue(handler=handler)
    assert len(c.state["handles"]) == 2
    await asyncio.wait_for(c.state["handles"][-1].task, 10)
    result = await c.manager.get_job(c.runtime.actor(c.parent), original.job_id)
    assert result.outcome == "succeeded", result
    assert c.model.counts["AUDIT"] == 1 and result.result["usage"]["total_tokens"] == 20


async def test_successful_restore_without_provider_usage_preserves_new_estimates_and_old_history(native_restore, monkeypatch):
    c = native_restore
    original = c.model.complete
    async def no_usage(request, **kwargs):
        result = await original(request, **kwargs)
        if get_session_id() != c.parent:
            result.token_usage = None
        return result
    monkeypatch.setattr(c.model, "complete", no_usage)
    handle = await c.restore()
    await asyncio.wait_for(handle.task, 10)
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert current.outcome == "succeeded" and current.result["final"]["text"] == "NUDGED-43", current
    usage = current.result["usage"]
    # Missing supplier usage is estimated by the normal loop; it is not free.
    # The new generation keeps those actual new-request estimates rather than
    # reusing the original 20-token workflow rollup.
    assert usage["estimated_calls"] == usage["calls"] == 2 and usage["estimated_tokens"] > 0
    assert c.original["usage"]["total_tokens"] == 20
    assert await c.manager.get_result_revision(c.actor, handle.job_id, 1) == c.before
    assert await c.manager.execution_usage(c.actor, handle.job_id, generation=2) == usage
    await c.loop.flush_follow_ups(c.parent)
    assert (await c.manager.get_job(c.actor, handle.job_id)).result_status == "handled"


@pytest.mark.parametrize("port", ["admission", "dispatch", "driver", "controller", "node", "binding", "input", "settlement", "scope", "projection"])
async def test_committed_ack_loss_never_replaces_task_or_repeats_business_requests(native_restore, monkeypatch, port):
    c = native_restore
    obj, name = {"admission": (workflow_resume, "admit"), "dispatch": (workflow_resume, "claim_dispatch"),
        "driver": (workflow_restore_execution, "create_driver"), "controller": (c.manager, "bind_workflow_driver"),
        "node": (workflow_restore_execution, "claim_node"), "binding": (c.manager, "bind_node_session"),
        "input": (workflow_restore_execution, "accept_input"), "settlement": (workflow_restore_execution, "settle"),
        "scope": (c.runtime, "record_scope"), "projection": (workflow_results, "project")}[port]
    original = getattr(obj, name)
    calls = []
    async def lost(*args, **kwargs):
        value = await original(*args, **kwargs)
        hit = port != "scope" or args[0].lease.job_id == c.initial.job_id
        if hit and not calls:
            calls.append(1)
            raise RuntimeError("commit completed, ACK lost")
        return value
    monkeypatch.setattr(obj, name, lost)
    handle = await c.restore()
    await asyncio.wait_for(handle.task, 10)
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert current.outcome == "succeeded" and current.result["final"]["text"] == "NUDGED-43", current
    assert current.cancel_requested_at is None and calls == [1]
    assert c.model.counts["BUILD"] == c.model.counts["REVIEW"] == 2
    assert await c.manager.get_result_revision(c.actor, handle.job_id, 1) == c.before
    assert len(await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='workflow_node'")) == 2


async def test_stop_targets_new_generation_even_after_old_handle_settled(native_restore, monkeypatch):
    c = native_restore
    entered, release = asyncio.Event(), asyncio.Event()
    original = workflow_restore_execution.create_driver
    async def delayed(*args):
        entered.set()
        await release.wait()
        return await original(*args)
    monkeypatch.setattr(workflow_restore_execution, "create_driver", delayed)
    handle = await c.restore()
    await asyncio.wait_for(entered.wait(), 3)
    assert runner.get_run_handle(handle.run_id) is handle
    status = await runner.stop_run(handle.run_id, grace_s=0, reason="test stop restored execution")
    release.set()
    await asyncio.wait_for(handle.task, 10)
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert status in ("stopped", "forced") and current.cancel_requested_at is not None
    assert current.outcome == "cancelled" and current.resource_state == "stopped", current
    assert c.model.counts["BUILD"] == c.model.counts["REVIEW"] == 1


async def test_premodel_contract_or_binding_failure_preserves_failure_and_stops_dependents(native_restore, monkeypatch):
    c = native_restore
    original = workflow_restore_execution.claim_node
    async def corrupt(*args, **kwargs):
        lease, sid = await original(*args, **kwargs)
        await c.store.set_runtime_state(sid, "workflow_execution_contract", {"version": 1, "spec": {"wrong": "contract"}})
        return lease, sid
    monkeypatch.setattr(workflow_restore_execution, "claim_node", corrupt)
    handle = await c.restore()
    await asyncio.wait_for(handle.task, 10)
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert current.outcome == "failed", current
    assert current.result["results"]["BUILD"]["status"] == "failed"
    assert c.model.counts["BUILD"] == c.model.counts["REVIEW"] == 1
    assert await c.manager.get_result_revision(c.actor, handle.job_id, 1) == c.before


@pytest.mark.parametrize("option", ["no_original_send", "rounds", "executor", "callback"])
async def test_unsupported_or_unattributed_restore_cannot_allocate_a_new_generation(native_restore, option):
    c = native_restore
    if option == "no_original_send":
        with pytest.raises(JobError):
            await resume_detached(c.loop, c.parent, c.initial.run_id, continuation=("BUILD", "new", 3))
    else:
        async def handler():
            kwargs = {"executor": object()} if option == "executor" else {"on_complete": lambda _: None} if option == "callback" else {}
            await resume_detached(c.loop, c.parent, c.initial.run_id, continuation=("BUILD", "new", 99 if option == "rounds" else 3), **kwargs)
            raise AssertionError("unsupported call accepted")
        await c.issue(handler=handler)
    assert (await c.manager.get_job(c.actor, c.initial.job_id)).generation == 1


@pytest.mark.parametrize("field", ["kind", "execution_role", "task_owner_job_id", "workflow_run_id", "task_workflow_generation", "task_dispatch_token"])
async def test_corrupt_reserved_driver_cannot_start_node_models(native_restore, monkeypatch, field):
    c = native_restore
    original = workflow_restore_execution.create_driver
    async def corrupt(*args):
        sid = await original(*args)
        row = await c.store.get_session(sid)
        meta = {**row.metadata, field: True if field == "task_workflow_generation" else "forged"}
        await c.manager.db.execute(f"UPDATE {c.store.t.sessions} SET metadata_json=? WHERE session_id=?", (json.dumps(meta), sid))
        return sid
    monkeypatch.setattr(workflow_restore_execution, "create_driver", corrupt)
    handle = await c.restore()
    await asyncio.wait_for(handle.task, 10)
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert current.outcome == "failed" and c.model.counts["BUILD"] == c.model.counts["REVIEW"] == 1
    assert await c.manager.get_result_revision(c.actor, handle.job_id, 1) == c.before


@pytest.mark.parametrize("field", ["source_hash", "workflow", "driver_sid", "node"])
@pytest.mark.parametrize("repair_checksum", [False, True])
async def test_corrupt_node_grant_produces_failure_and_no_new_provider_dispatch(native_restore, monkeypatch, field, repair_checksum):
    c = native_restore
    original = workflow_restore_execution.claim_node
    async def corrupt(*args, **kwargs):
        lease, sid = await original(*args, **kwargs)
        key = workflow_restore_execution.identity(c.manager, "node", lease.job_id, lease.generation)
        record = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (key,))
        data = json.loads(record["payload_json"])
        data[field] = {} if field in ("workflow", "node") else "forged"
        if repair_checksum:
            from power_loop.runtime.jobs import _hash
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?", (json.dumps(data), _hash(data), key))
        else:
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), key))
        return lease, sid
    monkeypatch.setattr(workflow_restore_execution, "claim_node", corrupt)
    handle = await c.restore()
    await asyncio.wait_for(handle.task, 10)
    current = await c.manager.get_job(c.actor, handle.job_id)
    assert current.outcome == "failed" and current.supervision_status == "open"
    assert c.model.counts["BUILD"] == c.model.counts["REVIEW"] == 1


async def test_old_creator_cannot_register_new_send_under_restored_node(native_restore, monkeypatch):
    from dataclasses import replace
    c = native_restore
    original = workflow_restore_execution.claim_node
    rejected = []
    async def attack(*args, **kwargs):
        lease, sid = await original(*args, **kwargs)
        node = await c.manager.get_job(c.actor, lease.job_id)
        with pytest.raises(JobError, match="caller_is_not_executing_parent"):
            await c.manager.register_job(replace(c.actor, session_id=node.created_by_session_id), kind="send", session_id=sid,
                parent=lease, system_key="borrow-old-creator", request={"send_index": 99})
        rejected.append(1)
        return lease, sid
    monkeypatch.setattr(workflow_restore_execution, "claim_node", attack)
    handle = await c.restore()
    await asyncio.wait_for(handle.task, 10)
    assert len(rejected) == 2 and (await c.manager.get_job(c.actor, handle.job_id)).outcome == "succeeded"


async def test_shared_spec_budget_counts_prior_execution_and_stops_new_downstream(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    c.state["spec"]["budget"] = {"max_tokens": 30}
    first = await finished(managed_workflow)
    async def handler():
        handle = await resume_detached(c.loop, c.parent, first.run_id, continuation=("BUILD", "NUDGED-43", 3))
        c.state["handles"].append(handle)
        return {"operation_id": handle.operation_id}
    await c.issue(handler=handler)
    assert len(c.state["handles"]) == 2
    await asyncio.wait_for(c.state["handles"][-1].task, 10)
    result = await c.manager.get_job(c.runtime.actor(c.parent), first.job_id)
    assert result.outcome == "budget_exceeded" and result.result["usage"]["total_tokens"] == 10, result
    assert c.model.counts["BUILD"] == 2 and c.model.counts["REVIEW"] == 1


async def test_missing_settlement_receipt_cannot_be_reported_as_success(native_restore, monkeypatch):
    c = native_restore
    async def silent(*args, **kwargs):
        return None
    monkeypatch.setattr(workflow_restore_execution, "settle", silent)
    handle = await c.restore()
    await asyncio.wait_for(handle.task, 10)
    result = await c.manager.get_job(c.actor, handle.job_id)
    assert result.outcome == "failed" and result.supervision_status == "open"
    assert result.result["results"]["REVIEW"]["text"] == "NUDGED-43"
    assert c.model.counts["BUILD"] == c.model.counts["REVIEW"] == 2
    assert await c.manager.get_result_revision(c.actor, handle.job_id, 1) == c.before


async def test_accepted_but_unowned_dispatch_keeps_cancellation_responsibility(native_restore, monkeypatch):
    from power_loop.runtime import workflow_restore_runner
    c = native_restore
    async def unowned(*args, **kwargs):
        return False
    monkeypatch.setattr(workflow_resume, "claim_dispatch", unowned)
    before = set(workflow_restore_runner._TASKS)
    async def handler():
        with pytest.raises(JobError, match="dispatch_unowned"):
            await resume_detached(c.loop, c.parent, c.initial.run_id, continuation=("BUILD", "new", 3))
        return {"status": "dispatch_unresolved"}
    await c.issue(handler=handler)
    work = set(workflow_restore_runner._TASKS)-before
    if work:
        await asyncio.wait_for(asyncio.gather(*work), 5)
    current = await c.manager.get_job(c.actor, c.initial.job_id)
    assert current.generation == 2 and current.cancel_requested_at is not None and current.outcome is None
    assert c.model.counts["BUILD"] == c.model.counts["REVIEW"] == 1


async def test_restored_node_must_consume_new_required_child_before_review(managed_workflow, monkeypatch):
    from power_loop import LLMResponse, LLMTokenUsage
    c = await client(managed_workflow, monkeypatch)
    c.state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["spawn_agent"])
    c.model.spawn_from_build = True
    first = await finished(managed_workflow)
    old = await c.manager.get_result_revision(c.runtime.actor(c.parent), first.job_id, 1)
    c.model.hold.add("AUDIT")
    c.model.entered["AUDIT"].clear()
    c.model.release["AUDIT"].clear()
    original = c.model.complete
    phase = 0
    async def delegate(request, **kwargs):
        nonlocal phase
        if "ROLE-BUILD" in (request.system_prompt or "") and "NUDGED-43" in str(request.messages) and phase < 2:
            phase += 1
            name = "spawn_agent" if phase == 1 else "pass_turn"
            args = {"task": "Audit the corrected input", "name": "audit", "tools": [], "max_rounds": 2, "system_prompt": "ROLE-AUDIT"} if phase == 1 else {"reason": "wait for required child"}
            return LLMResponse(raw_text="", tool_calls=[{"id": "restored-child-"+str(phase), "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}], token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
        return await original(request, **kwargs)
    monkeypatch.setattr(c.model, "complete", delegate)
    async def handler():
        handle = await resume_detached(c.loop, c.parent, first.run_id, continuation=("BUILD", "NUDGED-43", 3))
        c.state["handles"].append(handle)
        return {"operation_id": handle.operation_id}
    await c.issue(handler=handler)
    assert len(c.state["handles"]) == 2
    handle = c.state["handles"][-1]
    await asyncio.wait_for(c.model.entered["AUDIT"].wait(), 4)
    assert not handle.task.done() and c.model.counts["REVIEW"] == 1
    c.model.release["AUDIT"].set()
    await asyncio.wait_for(handle.task, 10)
    actor = c.runtime.actor(c.parent)
    current = await c.manager.get_job(actor, first.job_id)
    assert current.outcome == "succeeded", current
    children = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='subagent'")
    assert len(children) == 2 and all(n["result_status"] == "handled" for n in children)
    assert phase == 2 and c.model.counts["REVIEW"] == 2
    assert await c.manager.get_result_revision(actor, first.job_id, 1) == old
    assert current.result["usage"] == await c.manager.execution_usage(actor, first.job_id, generation=2)
