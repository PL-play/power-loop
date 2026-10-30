"""Actual native workflow consumes a stopped node's original negative result."""

from __future__ import annotations

import asyncio
import json
from collections import Counter

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime import timer_waits as waits
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow, runner
from power_loop.workflow.admission import GATES

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_native_node_input_is_disposed_and_original_final_result_is_retained(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "native-negative-input.db"), schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(
        manager, tenant_id="native-negative-input", principal_id="host", executor_id="worker", boot_id="boot"
    )
    calls = Counter()

    class Provider(BoundedProvider):
        async def complete(self, request, **kwargs):
            calls[get_session_id()] += 1
            return await super().complete(request, **kwargs)

    provider, registry = Provider(limit=14), ToolRegistry()
    loop = StatefulAgentLoop(
        llm=provider,
        store=store,
        task_runtime=runtime,
        tool_registry=registry,
        config=AgentLoopConfig(
            delegation_policy="single_level",
            max_rounds=4,
            max_tokens=2048,
            system_prompt="Follow the requested tool sequence exactly once. A queued receipt is not completion. "
            "After the requested tool returns, reply WAITING without more tools. "
            "When the original workflow result arrives, report its actual negative status, without tools.",
        ),
    )
    parent = await loop.new_session()
    readmission_queued = asyncio.Event()

    class Capacity(asyncio.Semaphore):
        async def acquire(self):
            if self.locked():
                readmission_queued.set()
            return await super().acquire()

    gate, registered, handles = Capacity(1), [], []
    token, held, consuming = GATES.set((gate,)), False, False

    async def arm():
        sid, source = get_session_id(), _SEND.get()
        seq, call_id = current_invocation(sid)
        now = manager.now()
        reg = await waits.register_wait(
            manager,
            runtime.actor(sid),
            caller=source.lease,
            call_ref=await store.message_ref(sid, seq),
            tool_call_id=call_id,
            request=waits.WaitRequest(
                now + 300, "Original finite arithmetic review", "Continue the original arithmetic node"
            ),
            policy=waits.WaitPolicy(now + 120000),
        )
        registered.append(reg)
        return "Original wait admitted. Finish this turn without polling."

    async def launch():
        workflow = create_workflow(
            {
                "name": "native-negative-input",
                "root": {
                    "type": "agent",
                    "id": "BUILD",
                    "input": "Compute 19+23. Call arm_wait once, then return READY-42 as plain assistant text. Do not use pass_turn.",
                    "spec": {
                        "name": "BUILD",
                        "tools": ["arm_wait"],
                        "max_rounds": 4,
                        "system_prompt": "Call arm_wait exactly once, then return READY-42 as assistant text without tools or pass_turn.",
                    },
                },
            },
            parent_loop=get_current_loop(),
            parent_session_id=get_session_id(),
        )
        handle = await workflow.start(detached=True, on_complete=None)
        handles.append(handle)
        return json.dumps({"run_id": handle.run_id, "job_id": handle.job_id, "status": "queued"})

    async def nudge():
        [reg], [handle] = registered, handles
        original = await waits.read_wait(manager, runtime.actor(parent), reg.job.job_id)
        return json.dumps(
            await runner.steer_node(
                handle.run_id, "BUILD", original["binding"]["session_id"], "Continue original arithmetic"
            )
        )

    for name, fn in [("arm_wait", arm), ("launch_original", launch), ("nudge_original", nudge)]:
        registry.register(
            ToolDefinition(name=name, description=name, input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), fn
        )

    async def shape(ctx):
        if get_session_id() == parent and consuming:
            ctx.tools = []
        elif get_session_id() != parent and registered:
            ctx.tools = []

    loop.hooks.register(HookPoint.LLM_BEFORE, shape, name="test.native-negative-shape")
    try:
        async with asyncio.timeout(160):
            await loop.send("Call launch_original exactly once. Then reply WAITING.", parent)
            for _ in range(1000):
                if registered:
                    [reg] = registered
                    owner = await manager.get_job(runtime.actor(parent), reg.job.control_owner_job_id)
                    if owner.status == "waiting" and not gate.locked():
                        break
                await asyncio.sleep(0.02)
            else:
                pytest.fail("Original real node did not exit its model into a native capacity-releasing wait")
            original_wait = await waits.read_wait(manager, runtime.actor(parent), reg.job.job_id)
            child_sid = original_wait["binding"]["session_id"]
            child_calls = calls[child_sid]
            assert child_calls >= 2
            await gate.acquire()
            held = True
            await loop.send("Call nudge_original exactly once. Then reply WAITING.", parent)
            await asyncio.wait_for(readmission_queued.wait(), 10)
            before = provider.calls
            await manager.request_cancel(
                runtime.actor(parent),
                owner.job_id,
                system_key="cancel-original-real-node",
                reason="Cancel original queued continuation",
            )
            [handle] = handles
            await asyncio.wait_for(handle.task, 30)
            workflow_job = await manager.get_job(runtime.actor(parent), handle.job_id)
            result = workflow_job.result
            stopped = await manager.get_job(runtime.actor(parent), owner.job_id)
            assert stopped.outcome == "cancelled" and stopped.resource_state == "stopped"
            assert provider.calls == before and calls[child_sid] == child_calls
            assert result["final"] is not None and result["final"]["node_id"] == "BUILD"
            assert result["final"]["status"] == "cancelled" and result["final"]["text"]
            from power_loop.workflow import journal

            saved = await journal.read(store, parent, handle.run_id)
            assert saved["result"]["final"] == result["final"]
            gate.release()
            held = False
            consuming = True
            consumed = await loop.flush_follow_ups(parent)
            assert consumed is not None and consumed.status == "completed"
            [proof] = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='input_disposition'")
            assert json.loads(proof["payload_json"])["owner"]["job_id"] == owner.job_id
            rows = await manager.db.fetchall(f"SELECT kind,status,supervision_status FROM {manager.jobs}")
            assert all(r["supervision_status"] == "closed" for r in rows), rows
            record_property("native_negative_final", json.dumps(result))
            record_property("original_supervision_tree", json.dumps(rows))
    finally:
        if held:
            gate.release()
        GATES.reset(token)
        for handle in handles:
            if not handle.task.done():
                await manager.request_cancel(
                    runtime.actor(parent),
                    handle.job_id,
                    system_key="native-negative-test-cleanup",
                    reason="test cleanup",
                )
                await asyncio.wait_for(asyncio.shield(handle.task), 30)
        provider.report(record_property)
        await provider.close()
        await store.close()
