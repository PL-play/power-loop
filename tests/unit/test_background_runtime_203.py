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


async def test_background_job_survives_parent_and_delivers_once(registry_store, tmp_path):
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
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    registry.register(ToolDefinition(name="slow", description="slow read", input_schema={"type": "object"},
        pricing=Free("read"), async_capable=True, interrupt="finish"), slow)
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
    try:
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
        # The explicit call finishes after the parent has already returned.
        assert len(model.requests) == 3
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


@pytest.mark.parametrize("command", [False, True])
async def test_background_execution_obeys_capacity_one(tmp_path, command):
    import shlex
    import sys

    from power_loop import SessionStore

    store = await SessionStore.open(":memory:", schema="tasks")
    gate, began = asyncio.Event(), asyncio.Event()
    started = active = peak = 0
    async def slow():
        nonlocal started, active, peak
        started += 1
        active += 1
        peak = max(peak, active)
        began.set()
        try:
            await gate.wait()
            return "done"
        finally:
            active -= 1
    model = BackgroundParent()
    complete = model.complete
    async def three(request, **kwargs):
        first = not model.sent
        response = await complete(request, **kwargs)
        if first:
            response.tool_calls = []
            for i in range(3):
                args = {"action": "tool", "tool": "slow", "args": {}}
                if command:
                    script = (f"from pathlib import Path; import time; p=Path({str(tmp_path)!r}); "
                              f"(p/'started-{i}').touch()\n"
                              "while not (p/'release').exists(): time.sleep(.01)\n")
                    args = {"action": "run", "command": shlex.join([sys.executable, "-c", script])}
                response.tool_calls.append({"id": f"b{i}", "type": "function",
                    "function": {"name": "background_run", "arguments": json.dumps(args)}})
        return response
    model.complete = three
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    registry.register(ToolDefinition("slow", "wait", async_capable=True, pricing=Free("local")), slow)
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
    try:
        assert (await runner.send("three backgrounds")).status == "completed"
        if command:
            for _ in range(500):
                if list(tmp_path.glob("started-*")):
                    break
                await asyncio.sleep(.01)
        else:
            await asyncio.wait_for(began.wait(), 5)
        await asyncio.sleep(.1)
        assert (len(list(tmp_path.glob("started-*"))) if command else started) == 1
        gate.set()
        (tmp_path / "release").touch()
        await asyncio.wait_for(runner.drain(), 10)
        assert (len(list(tmp_path.glob("started-*"))) if command else started) == 3
        if not command:
            assert peak == 1 and active == 0
        assert runner.runtime.worker._capacity._value == 1
        assert not runner.runtime.worker._waiting
    finally:
        gate.set()
        (tmp_path / "release").touch()
        await runner.aclose()
        await store.close()
