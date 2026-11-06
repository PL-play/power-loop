"""Background tools are independently owned jobs, not parent callbacks."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import JobRef
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.tools import create_default_tool_registry
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class BackgroundParent(LLMService):
    def __init__(self):
        self.sent = False
        self.requests = []

    async def complete(self, request, **kwargs):
        self.requests.append(request)
        if not self.sent:
            self.sent = True
            return LLMResponse(raw_text="", tool_calls=[{"id": "background", "type": "function",
                "function": {"name": "background_run", "arguments": json.dumps({"action": "tool", "tool": "slow", "args": {}})}}])
        return LLMResponse(raw_text="parent done")

    async def close(self):
        pass


@pytest.mark.parametrize("automatic", [False, True])
async def test_background_job_survives_parent_and_delivers_once(registry_store, tmp_path, automatic):
    store, _ = registry_store
    gate, began = asyncio.Event(), asyncio.Event()
    count = 0
    async def slow():
        nonlocal count
        count += 1
        began.set()
        await gate.wait()
        return "background result"
    model = BackgroundParent()
    if automatic:
        complete = model.complete
        async def direct(request, **kwargs):
            first = not model.sent
            response = await complete(request, **kwargs)
            if first:
                response.tool_calls[0]['function'] = {'name':'slow', 'arguments':'{}'}
            return response
        model.complete = direct
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    registry.register(ToolDefinition(name="slow", description="slow read", input_schema={"type": "object"},
        pricing=Free("read"), async_capable=True, interrupt="background" if automatic else "finish"), slow)
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
    try:
        if automatic:
            original = await runner.runtime.client.send(runner.target, "start background", message_id="start", actor=runner.actor)
            await asyncio.wait_for(began.wait(), 5)
            status = await runner.runtime.client.inspect(original.input_ref, actor=runner.actor)
            await runner.runtime.client.steer(status.job_ref, "respond now", message_id="steer", actor=runner.actor)
            parent = await runner.runtime.client.wait(original.input_ref, actor=runner.actor, deadline_ms=now_ms()+10000)
            await runner.finish(original.input_ref)
        else:
            parent = await runner.send("start background")
        assert parent.status == "completed"
        rows = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(m.content for m in rows if m.tool_call_id == "background"))
        assert receipt["job_ref"]["generation"] == 0
        await asyncio.wait_for(began.wait(), 5)
        ref = JobRef(**receipt["job_ref"])
        assert not (await runner.runtime.client.inspect(ref, actor=runner.actor)).result_available
        gate.set()
        result = await runner.runtime.client.wait(ref, actor=runner.actor, deadline_ms=now_ms() + 10000)
        assert result.status == "completed" and result.final_text == "background result"
        assert count == 1
        for _ in range(300):
            await asyncio.sleep(0.01)
            rows = await store.load_all_messages(runner.target.session_id)
            if any("background result" in (m.content or "") for m in rows):
                break
        await runner.drain()
        row = await runner.runtime.worker.inputs.db.fetchone(
            f"SELECT result_status,supervision_status FROM {runner.runtime.worker.inputs.jobs} WHERE job_id=?", (ref.job_id,))
        assert row == {"result_status": "handled", "supervision_status": "closed"}
        assert len(model.requests) == 3  # Parent tool call, parent return, result handling; no background model.
    finally:
        gate.set()
        await runner.aclose()


async def test_cancel_background_handler_that_does_not_poll_stop_token(registry_store,tmp_path):
    from power_loop.runtime.stop_policy import StopPolicy
    store,_=registry_store
    started,release=asyncio.Event(),asyncio.Event()
    async def slow():
        started.set()
        await release.wait()
        return 'unexpected completion'
    registry=create_default_tool_registry(workspace_dir=str(tmp_path))
    registry.register(ToolDefinition('slow','blocking coroutine',async_capable=True,interrupt='abort',pricing=Free('local')),slow)
    runner=await RunningAgent(store,BackgroundParent(),registry,
        AgentLoopConfig(compactor=None,stop_policy=StopPolicy(abort_tool_s=.01))).start()
    try:
        await runner.send('start')
        await asyncio.wait_for(started.wait(),5)
        rows=await store.load_all_messages(runner.target.session_id)
        receipt=json.loads(next(m.content for m in rows if m.tool_call_id=='background'))
        status=await runner.runtime.client.inspect(JobRef(**receipt['job_ref']),actor=runner.actor)
        await runner.runtime.client.cancel(status.job_ref,actor=runner.actor)
        result=await runner.runtime.client.wait(status.job_ref,actor=runner.actor,deadline_ms=now_ms()+10000)
        assert result.status=='cancelled' and not release.is_set()
        await runner.drain()
    finally:
        release.set()
        await runner.aclose()
