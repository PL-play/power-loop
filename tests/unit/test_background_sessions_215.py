"""Background sessions freeze the original tool and never replay unknown outcomes."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import InputRef, TargetRef, result_message_id
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.runtime.stop_policy import StopPolicy
from power_loop.tools import create_default_tool_registry
from tests.runtime_fixture import RunningAgent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class BackgroundParent(LLMService):
    def __init__(self, command=None):
        self.sent = False
        self.command = command

    async def complete(self, request, **kwargs):
        if not self.sent:
            self.sent = True
            arguments = (
                {"action": "run", "command": self.command}
                if self.command is not None
                else {"action": "tool", "tool": "slow", "args": {}}
            )
            return LLMResponse(raw_text="", tool_calls=[{
                "id": "background", "type": "function",
                "function": {"name": "background_run", "arguments": json.dumps(arguments)},
            }])
        return LLMResponse(raw_text="parent done")

    async def close(self):
        pass


def _receipt(rows):
    return json.loads(next(m.content for m in rows if m.tool_call_id == "background"))


async def test_execute_freezes_original_tool_and_routes_result(registry_store, tmp_path):
    store, _ = registry_store
    calls = 0

    async def slow():
        nonlocal calls
        calls += 1
        return "background result"

    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    registry.register(
        ToolDefinition(
            name="slow", description="slow read", input_schema={"type": "object"},
            pricing=Free("read"), async_capable=True, interrupt="finish",
        ),
        slow,
    )
    runner = await RunningAgent(
        store, BackgroundParent(), registry, AgentLoopConfig(compactor=None),
    ).start()
    try:
        assert (await runner.send("start background")).status == "completed"
        receipt = _receipt(await store.load_all_messages(runner.target.session_id))
        start = InputRef(**receipt["input_ref"])
        target = TargetRef(**receipt["target_ref"])
        result = await runner.runtime.client.wait(
            start, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert result.status == "completed" and result.final_text == "background result"
        assert calls == 1
        parent_result = InputRef(
            store.store_id, runner.target.session_id,
            result_message_id(target, start, 1, "background_result"),
        )
        await runner.drain(after_input=parent_result)
        parent_rows = await store._db.fetchall(
            f"SELECT kind,content_json FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
            (runner.target.session_id, parent_result.message_id),
        )
        assert parent_rows[0]["kind"] == "background_result"
        assert "background result" in parent_rows[0]["content_json"]
    finally:
        await runner.aclose()


async def test_cancel_background_handler(registry_store, tmp_path):
    store, _ = registry_store
    started, release = asyncio.Event(), asyncio.Event()

    async def slow():
        started.set()
        await release.wait()
        return "unexpected completion"

    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    registry.register(
        ToolDefinition(
            "slow", "blocking coroutine", async_capable=True, interrupt="abort",
            pricing=Free("local"), input_schema={"type": "object"},
        ),
        slow,
    )
    runner = await RunningAgent(
        store, BackgroundParent(), registry,
        AgentLoopConfig(compactor=None, stop_policy=StopPolicy(abort_tool_s=0.01)),
    ).start()
    try:
        await runner.send("start")
        await asyncio.wait_for(started.wait(), 5)
        receipt = _receipt(await store.load_all_messages(runner.target.session_id))
        target = TargetRef(**receipt["target_ref"])
        await runner.runtime.client.cancel(target, actor=runner.actor, reason="stop background")
        result = await runner.runtime.client.wait(
            InputRef(**receipt["input_ref"]), actor=runner.actor,
            deadline_ms=now_ms() + 10000,
        )
        assert result.status == "cancelled" and not release.is_set()
    finally:
        release.set()
        await runner.aclose()


async def test_crash_after_execute_is_interrupted_and_not_replayed(registry_store, tmp_path):
    store, _ = registry_store
    calls = 0

    async def slow():
        nonlocal calls
        calls += 1
        return "unknown remote outcome"

    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    registry.register(
        ToolDefinition(
            "slow", "one effect", async_capable=True, interrupt="finish",
            pricing=Free("local"), input_schema={"type": "object"},
        ),
        slow,
    )
    runner = await RunningAgent(
        store, BackgroundParent(), registry, AgentLoopConfig(compactor=None),
    ).start()
    original_final = runner.runtime.worker.inputs.final
    crashed = False

    async def crash_after_execute(batch, **kwargs):
        nonlocal crashed
        if batch.target.session_id != runner.target.session_id and not crashed:
            crashed = True
            raise asyncio.CancelledError("simulated process loss after original execute")
        return await original_final(batch, **kwargs)

    runner.runtime.worker.inputs.final = crash_after_execute
    try:
        await runner.send("start")
        receipt = _receipt(await store.load_all_messages(runner.target.session_id))
        start = InputRef(**receipt["input_ref"])
        for _ in range(300):
            if crashed:
                break
            await asyncio.sleep(0.01)
        assert crashed and calls == 1
        await store._db.execute(
            f"UPDATE {store.t.session_state} SET owner_lease_until_ms=0 WHERE session_id=?",
            (start.session_id,),
        )
        recovered = await runner.runtime.worker.inputs.recover(
            TargetRef(store.store_id, start.session_id), node_id="B", boot_id="B",
        )
        assert recovered is not None and recovered.results[0].status == "interrupted"
        await runner.runtime.worker.recover()
        result = await runner.runtime.client.wait(
            start, actor=runner.actor, deadline_ms=now_ms() + 1000,
        )
        assert result.status == "interrupted" and calls == 1
    finally:
        await runner.aclose()


async def test_shell_process_tag_uses_session_and_send(registry_store, tmp_path):
    store, _ = registry_store
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    runner = await RunningAgent(
        store,
        BackgroundParent("printf '%s' \"$POWER_LOOP_BACKGROUND_SESSION\""),
        registry,
        AgentLoopConfig(compactor=None),
    ).start()
    try:
        await runner.send("start shell")
        receipt = _receipt(await store.load_all_messages(runner.target.session_id))
        start = InputRef(**receipt["input_ref"])
        result = await runner.runtime.client.wait(
            start, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert result.status == "completed", result.error
        assert f"{start.session_id}_1" in result.final_text
    finally:
        await runner.aclose()
