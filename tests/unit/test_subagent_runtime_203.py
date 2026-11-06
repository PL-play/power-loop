"""P3 real tool admission and independently scheduled child execution."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import JobRef
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class ParentAndChild(LLMService):
    def __init__(self):
        self.requests = []
        self.spawned = False
        self.child_gate = None
        self.child_started = asyncio.Event()

    async def complete(self, request, **kwargs):
        self.requests.append(request)
        if request.system_prompt.startswith("CHILD"):
            self.child_started.set()
            if self.child_gate:
                await self.child_gate.wait()
            return LLMResponse(raw_text="child result")
        if not self.spawned:
            self.spawned = True
            return LLMResponse(raw_text="", tool_calls=[{"id": "spawn", "type": "function",
                "function": {"name": "spawn_agent", "arguments": json.dumps({
                    "task": "compute", "system_prompt": "CHILD", "tools": []})}}])
        return LLMResponse(raw_text="parent done")

    async def close(self):
        pass


@pytest.mark.parametrize("inline", [False, True])
async def test_child_start_reference_and_parent_result(registry_store, inline):
    store, _ = registry_store
    model, registry = ParentAndChild(), ToolRegistry()
    register_spawn_agent(registry)
    if inline:
        from power_loop.core.agent_context import get_current_loop
        from power_loop.runtime.child_tasks import tool_actor
        from power_loop.runtime.delegation import start
        from power_loop.runtime.spec import AgentSpec
        from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract
        from power_loop.tools.spawn_agent import SPAWN_AGENT_DEFINITION

        async def spawn_and_wait(**kwargs):
            loop = get_current_loop()
            receipt = await start(loop, AgentSpec(name="child", system_prompt="CHILD", tools=[], lifecycle="linked"), "compute")
            _, actor = await tool_actor(loop)
            result = await loop._runtime_worker.runtime.client.wait(JobRef(**receipt["job_ref"]),
                actor=actor, deadline_ms=now_ms() + 10000)
            assert result.final_text == "child result"
            return json.dumps(receipt)

        registry.register(SPAWN_AGENT_DEFINITION, FunctionRuntimeTool(spawn_and_wait,
            ToolRuntimeContract("sdk.subagent", background_allowed=False, allowed_roles=("main", "workflow_node")),
            policy=lambda arguments: ToolDispatchPolicy(native_admitted=True)), overwrite=True)
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
    try:
        parent = await runner.send("delegate")
        assert parent.status == "completed"
        messages = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(m.content for m in messages if m.tool_call_id == "spawn"))
        ref = JobRef(**receipt["job_ref"])
        assert ref.generation == 0
        result = await runner.runtime.client.wait(ref, actor=runner.actor, deadline_ms=now_ms() + 10000)
        assert result.job_ref.generation == 1
        assert result.final_text == "child result"
        for _ in range(300):
            messages = await store.load_all_messages(runner.target.session_id)
            if any("[subagent_result]" in (m.content or "") for m in messages):
                break
            await asyncio.sleep(0.01)
        assert sum("[subagent_result]" in (m.content or "") for m in messages) == 1
        await runner.drain()
        row = await runner.runtime.worker.inputs.db.fetchone(f"SELECT result_status,supervision_status FROM {runner.runtime.worker.inputs.jobs} WHERE job_id=?", (ref.job_id,))
        assert row == {"result_status": "handled", "supervision_status": "closed"}

    finally:
        await runner.aclose()


async def test_explicit_resume_preserves_first_result_and_voids_only_new_input(registry_store):
    from power_loop import RuntimeClient
    from power_loop.agent.runtime_transport import MemoryMessageTransport
    from power_loop.agent.runtime_types import InputRef

    store, _ = registry_store
    model, registry = ParentAndChild(), ToolRegistry()
    model.child_gate = asyncio.Event()
    register_spawn_agent(registry)
    first = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
    second, client = None, None
    try:
        await first.send("delegate")
        await asyncio.wait_for(model.child_started.wait(), 5)
        messages = await store.load_all_messages(first.target.session_id)
        receipt = json.loads(next(m.content for m in messages if m.tool_call_id == "spawn"))
        ref, start = JobRef(**receipt["job_ref"]), InputRef(**receipt["input_ref"])
        await first.runtime.worker.shutdown(timeout_s=0.01)
        client = RuntimeClient(store=store, transport=MemoryMessageTransport())
        original = await client.wait(ref, actor=first.actor, deadline_ms=now_ms() + 10000)
        assert original.status == "interrupted"
        one = original.job_ref
        withdrawn = await client.resume(one, actor=first.actor, message_id="withdrawn")
        await client.cancel(withdrawn.input_ref, actor=first.actor)
        cancelled = await client.wait(withdrawn.input_ref, actor=first.actor, deadline_ms=now_ms() + 1000)
        assert cancelled.job_ref is None and cancelled.status == "cancelled"
        resume = await client.resume(one, actor=first.actor, message_id="resume")
        model.child_gate = None
        second = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
        resumed = await client.wait(resume.input_ref, actor=first.actor, deadline_ms=now_ms() + 10000)
        assert resumed.job_ref == JobRef(one.store_id, one.job_id, 2)
        assert resumed.status == "completed"
        assert await client.wait(ref, actor=first.actor, deadline_ms=now_ms() + 1000) == original
        assert (await client.wait(start, actor=first.actor, deadline_ms=now_ms() + 1000)).job_ref == one
        await second.drain()
        events = await store._db.fetchall(f"SELECT payload_json,delivery_status FROM {store.table_prefix}job_events WHERE job_id=? AND event_type='result'", (one.job_id,))
        assert {json.loads(event['payload_json'])['generation']: event['delivery_status'] for event in events} == {1:'handled', 2:'handled'}
    finally:
        if client:
            await client.aclose()
        if second:
            await second.aclose()
        await first.aclose()


@pytest.mark.parametrize("mode", ["cancel_job", "cancel_input", "missing_definition"])
async def test_real_queued_child_finishes_at_generation_zero(registry_store, mode, monkeypatch):
    from power_loop.agent.runtime_types import InputRef
    store, _ = registry_store
    model, registry = ParentAndChild(), ToolRegistry()
    register_spawn_agent(registry)
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
    acquire = runner.runtime.coordinator.acquire
    enabled = False
    async def main_only(key, **kwargs):
        if key[1] != runner.target.session_id and not enabled:
            return None
        return await acquire(key, **kwargs)
    monkeypatch.setattr(runner.runtime.coordinator, "acquire", main_only)
    try:
        await runner.send("delegate")
        messages = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(m.content for m in messages if m.tool_call_id == "spawn"))
        ref, input_ref = JobRef(**receipt["job_ref"]), InputRef(**receipt["input_ref"])
        assert (await runner.runtime.client.inspect(ref, actor=runner.actor)).state == "accepted"
        if mode.startswith("cancel"):
            await runner.runtime.client.cancel(ref if mode=="cancel_job" else input_ref, actor=runner.actor)
        else:
            build = runner.runtime.factory.build
            async def missing(descriptor, context):
                if context.session_id != runner.target.session_id:
                    raise ValueError("missing frozen child definition")
                return await build(descriptor, context)
            monkeypatch.setattr(runner.runtime.factory, "build", missing)
        enabled = True
        await runner.runtime.worker.recover()
        result = await runner.runtime.client.wait(ref, actor=runner.actor, deadline_ms=now_ms()+10000)
        assert result.job_ref.generation==0
        assert result.status == ("failed" if mode=="missing_definition" else "cancelled")
        assert (await runner.runtime.client.wait(input_ref, actor=runner.actor, deadline_ms=now_ms()+1000)).job_ref==result.job_ref
        assert not model.child_started.is_set()
        await runner.drain()
        current = await store._db.fetchone(f"SELECT result_status,supervision_status FROM {store.table_prefix}jobs WHERE job_id=?", (ref.job_id,))
        assert current == {'result_status':'handled','supervision_status':'closed'}
    finally:
        await runner.aclose()


async def test_claimed_child_rejects_zero_cancel_but_start_input_cancels_one(registry_store):
    from power_loop.agent.runtime_types import InputRef, RuntimeRequestError
    store, _ = registry_store
    model, registry = ParentAndChild(), ToolRegistry()
    model.child_gate = asyncio.Event()
    register_spawn_agent(registry)
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
    try:
        await runner.send("delegate")
        await asyncio.wait_for(model.child_started.wait(), 5)
        rows = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(m.content for m in rows if m.tool_call_id == "spawn"))
        with pytest.raises(RuntimeRequestError, match="stale_generation"):
            await runner.runtime.client.cancel(JobRef(**receipt["job_ref"]), actor=runner.actor)
        await runner.runtime.client.cancel(InputRef(**receipt["input_ref"]), actor=runner.actor)
        result = await runner.runtime.client.wait(JobRef(**receipt["job_ref"]), actor=runner.actor, deadline_ms=now_ms()+10000)
        assert result.status == "cancelled" and result.job_ref.generation == 1
    finally:
        model.child_gate.set()
        await runner.aclose()


async def test_cancelled_wait_releases_slot_without_leaking_task_reference(monkeypatch):
    from types import SimpleNamespace

    from power_loop.agent.runtime_worker import RuntimeWorker
    from power_loop.runtime.cancellation import CancellationToken
    worker=RuntimeWorker(SimpleNamespace(capacity=1,store=SimpleNamespace(table_prefix='test_',t=SimpleNamespace(follow_up_queue='queue',session_state='state'))))
    entered=asyncio.Event()
    stop=CancellationToken()
    async def wait(sid):
        ctx=SimpleNamespace(task=asyncio.current_task(),idle_users=0,stop=stop,shutdown_interrupted=False)
        async with worker.idle(ctx):
            entered.set()
            await asyncio.Event().wait()
    monkeypatch.setattr(worker,'_run_session_owned',wait)
    task=asyncio.create_task(worker._run_session('capacity-cancel'))
    await entered.wait()
    # Another execution owns the sole slot while the parent is waiting.
    await worker._capacity.acquire()
    stop.cancel('cancel parent only')
    task.cancel()
    await asyncio.wait_for(asyncio.gather(task,return_exceptions=True),1)
    assert not worker._waiting and worker._capacity.locked()
    worker._capacity.release()
    assert worker._capacity._value==1
