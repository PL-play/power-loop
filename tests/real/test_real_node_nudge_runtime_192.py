"""Real main -> active workflow node correction -> review -> main receipt."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime import node_inputs, workflow_controller
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow, runner

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("fault", ["none", "lost_ack", "binding_and_closure_corruption", "controller_corruption", "empty_node_pass"])
async def test_real_main_nudges_original_workflow_node_before_review(tmp_path, record_property, monkeypatch, fault):
    store = await SessionStore.open(str(tmp_path / "real-node-nudge.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-node-nudge", principal_id="host", executor_id="executor", boot_id="boot")
    provider = BoundedProvider(limit=12)
    registry = ToolRegistry()
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_tokens=2048, max_rounds=4,
            system_prompt="Follow the requested tool sequence exactly. Acceptance is not completion. Never create replacement tasks. "
                          "When workflow completion arrives report its final number without tools or pass_turn."))
    parent = await loop.new_session()
    provider.root_sid, provider.hold_children = parent, True
    handles, replies, calls = [], [], []
    spec = {"name": "correct-active-calculation", "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "calculate", "input": "Compute 19+23.", "spec": {"name": "calculate", "tools": [], "max_rounds": 3,
            "system_prompt": "Calculate the requested arithmetic. Follow the latest correction in your inbox. Reply only with the final number. Do not call pass_turn."}},
        {"type": "agent", "id": "review", "inputs_from": ["calculate"], "input": "Report the previous calculation result.",
         "spec": {"name": "review", "tools": [], "max_rounds": 2, "system_prompt": "Read the supplied calculation result. Reply only with its number. Do not call pass_turn."}}
    ]}}
    if fault == "empty_node_pass":
        spec["root"]["steps"][0]["spec"]["system_prompt"] = (
            "Calculate the latest corrected arithmetic. On your FIRST response call pass_turn with no assistant text; "
            "put the computed number in its reason. When asked for the missing final answer in a later user message, "
            "output only the corrected number as assistant text, with no tools or pass_turn.")
        # Fault setup must actually reach the empty-pass boundary; prompt-only
        # obedience is not a reliable injector. Force only the first calculate
        # request's tool choice; arguments, text and all responses stay real.
        original_complete = provider.inner.complete
        forced = False
        async def first_pass(request, **kwargs):
            nonlocal forced
            if not forced and "Calculate the latest corrected arithmetic." in (request.system_prompt or ""):
                forced = True
                # DeepSeek thinking mode rejects named tool_choice. Disable it
                # only for this controlled fault-injection request, retaining
                # the configured real model and its unmodified response.
                # https://api-docs.deepseek.com/api/create-chat-completion/
                extra = dict(request.extra or {})
                extra["extra_body"] = {**extra.get("extra_body", {}), "thinking": {"type": "disabled"}}
                request = replace(request, tool_choice={"type": "function", "function": {"name": "pass_turn"}},
                    extra=extra,
                    system_prompt=request.system_prompt + " For this first call use pass_turn arguments only; leave assistant text empty.")
            return await original_complete(request, **kwargs)
        monkeypatch.setattr(provider.inner, "complete", first_pass)
    async def launch():
        workflow = create_workflow(spec, parent_loop=get_current_loop(), parent_session_id=get_session_id())
        handle = await workflow.start(detached=True)
        handles.append(handle)
        return json.dumps({"status": "queued", "run_id": handle.run_id, "job_id": handle.job_id, "operation_id": handle.operation_id})
    async def nudge():
        [handle] = handles
        [row] = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='node_session'")
        descriptor = json.loads(row["payload_json"])
        if fault == "controller_corruption":
            frozen = await manager.get_workflow_controller(runtime.actor(parent), handle.job_id, 1)
            key = workflow_controller.event_id(manager, handle.job_id, 1)
            [saved] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (key,))
            await manager.db.execute(f"UPDATE {manager.events} SET payload_json='[]' WHERE event_id=?", (key,))
            try:
                with pytest.raises(JobError, match="workflow_controller_mismatch"):
                    await runner.steer_node(handle.run_id, "calculate", descriptor["session_id"], "Forbidden correction")
            finally:
                await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (saved["payload_json"], key))
            await store.merge_session_metadata(frozen["session_id"], {"task_workflow_generation": True})
            try:
                with pytest.raises(JobError, match="workflow_controller_mismatch"):
                    await runner.steer_node(handle.run_id, "calculate", descriptor["session_id"], "Forbidden correction")
                assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='nudge'")
                assert (await store.inbox_counts(descriptor["session_id"]))["pending"] == 0
            finally:
                await store.merge_session_metadata(frozen["session_id"], {"task_workflow_generation": 1})
        if fault == "binding_and_closure_corruption":
            # Real main call, original held node, real stores. Neither bad
            # evidence record may enqueue input or dispatch a replacement.
            await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?",
                (json.dumps({**descriptor, "generation": 2}), row["event_id"]))
            try:
                with pytest.raises(JobError, match="node_input_binding_mismatch"):
                    await runner.steer_node(handle.run_id, "calculate", descriptor["session_id"], "Forbidden correction")
                assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='nudge'")
            finally:
                await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (row["payload_json"], row["event_id"]))
            marker = node_inputs.closed_id(manager, row["job_id"], descriptor["session_id"])
            await manager.db.execute(f"INSERT INTO {manager.events} "
                "(event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
                (marker, row["job_id"], "node_input_closed", 1, "[]", "not_applicable", manager.now()))
            try:
                with pytest.raises(JobError, match="node_input_closure_mismatch"):
                    await runner.steer_node(handle.run_id, "calculate", descriptor["session_id"], "Forbidden correction")
                assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE action='nudge'")
                assert (await store.inbox_counts(descriptor["session_id"]))["pending"] == 0
            finally:
                await manager.db.execute(f"DELETE FROM {manager.events} WHERE event_id=?", (marker,))
        reply = await runner.steer_node(handle.run_id, "calculate", descriptor["session_id"],
            "Correction: compute 20+23 instead of 19+23. Reply only with the resulting number.")
        replies.append(reply)
        return json.dumps(reply)
    registry.register(ToolDefinition(name="launch_workflow", description="Start the predefined workflow once", input_schema={"type": "object"}, pricing=Free("test_fixture")), launch)
    registry.register(ToolDefinition(name="correct_node", description="Correct the original active calculation node", input_schema={"type": "object"}, pricing=Free("test_fixture")), nudge)
    accept = manager.accept_nudge
    async def admit(*args, **kwargs):
        calls.append(1)
        result = await accept(*args, **kwargs)
        if fault == "lost_ack":
            raise RuntimeError("committed nudge; lost ACK")
        return result
    monkeypatch.setattr(manager, "accept_nudge", admit)
    try:
        async with asyncio.timeout(180):
            first = await loop.send("Call launch_workflow exactly once, then reply WAITING with no more tools or pass_turn.", parent)
            assert first.status == "completed" and len(handles) == 1
            await asyncio.wait_for(provider.child_entered.wait(), 10)
            # The acceptance receipt is historical, not a status subscription.
            # Ground the new turn in real current state: otherwise a real model
            # may reasonably refuse to correct a node it only knows as queued.
            workflow = await manager.get_job(runtime.actor(parent), handles[0].job_id)
            [active] = await manager.db.fetchall(f"SELECT job_id,status,resource_state FROM {manager.jobs} WHERE parent_job_id=? AND kind='workflow_node'",
                (workflow.job_id,))
            assert workflow.status == active["status"] == "running" and active["resource_state"] == "running"
            observation = {"workflow_job_id": workflow.job_id, "workflow_status": workflow.status,
                           "calculate_node": dict(active)}
            second = await loop.send("Current status verified by the host (the earlier queued receipt is stale): " +
                json.dumps(observation) + ". The original calculate node exists and accepts correction while running. "
                "The new requested correction is to compute 20+23 instead of 19+23. "
                "Call correct_node exactly once to send this correction to the original node. Then reply WAITING without more tools or pass_turn. Do not launch a new workflow.", parent)
            assert second.status == "completed" and calls == ([1, 1] if fault == "binding_and_closure_corruption" else [1, 1, 1] if fault == "controller_corruption" else [1]) and len(replies) == 1
            reply = replies[0]
            assert reply["status"] == "accepted" and reply["run_id"] == handles[0].run_id
            op = await manager.get_operation(runtime.actor(parent), reply["operation_id"])
            assert op.call_ref["id"] != op.receipt_ref["id"] and op.receipt_ref["tool_call_id"]
            provider.release_children.set()
            await handles[0].task
            node = await manager.get_job(runtime.actor(parent), reply["job_id"])
            assert node.outcome == "succeeded" and node.result["text"].strip() == "43", node.result
            if fault == "empty_node_pass":
                messages = await store.load_all_messages(reply["session_id"])
                assert any(call.get("function", {}).get("name") == "pass_turn" for msg in messages for call in (msg.tool_calls or []))
                assert not next(msg for msg in messages if msg.role == "assistant").content.strip()
                attempts = await manager.db.fetchall(f"SELECT e.event_id FROM {manager.events} e JOIN {manager.jobs} j ON j.job_id=e.job_id "
                    "WHERE j.kind='send' AND j.session_id=? AND e.event_type='model_attempt_finished'", (reply["session_id"],))
                assert len(attempts) >= 2  # queued input may be processed in the same original send
            [event] = await manager.db.fetchall(f"SELECT delivery_status FROM {manager.events} WHERE event_id=?", (reply["operation_id"],))
            assert event["delivery_status"] == "handled"
            wf = await manager.get_job(runtime.actor(parent), handles[0].job_id)
            assert wf.outcome == "succeeded" and wf.result["final"]["text"].strip() == "43", wf.result
            frozen = await manager.get_workflow_controller(runtime.actor(parent), wf.job_id, 1)
            assert frozen["workflow"]["generation"] == 1 and frozen["run_id"] == handles[0].run_id
            assert await store.get_session(frozen["session_id"]) is None
            assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")) == 2
            final = await loop.flush_follow_ups(parent, max_rounds=3)
            assert final.status == "completed" and "43" in final.final_text
            assert (await manager.get_job(runtime.actor(parent), wf.job_id)).result_status == "handled"
    finally:
        provider.release_children.set()
        if handles:
            await asyncio.wait_for(asyncio.gather(*(h.task for h in handles), return_exceptions=True), 15)
        # XML evidence survives pytest's temporary-directory retention cleanup,
        # including when a real model declines a requested tool invocation.
        record_property("main_transcript", json.dumps([dict(session_id=m.session_id, id=str(m.id), seq=str(m.seq),
            role=m.role, name=m.name, content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
            for m in await store.load_all_messages(parent)], ensure_ascii=False))
        provider.report(record_property)
        await provider.close()
        await store.close()
