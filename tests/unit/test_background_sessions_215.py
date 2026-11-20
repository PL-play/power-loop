"""Background sessions freeze the original tool and never replay unknown outcomes."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import InputRef, TargetRef, result_message_id
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.background import encode_results, load_record, summarize_record
from power_loop.runtime.env import RuntimeEnv
from power_loop.runtime.pricing import Free
from power_loop.runtime.stop_policy import StopPolicy
from power_loop.runtime.tool_archive import _derive
from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine, ToolSource, active_engine
from power_loop.tools import create_default_tool_registry
from tests.runtime_fixture import RunningAgent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class BackgroundParent(LLMService):
    def __init__(self, command=None, arguments=None):
        self.sent = False
        self.command = command
        self.arguments = arguments

    async def complete(self, request, **kwargs):
        if not self.sent:
            self.sent = True
            arguments = self.arguments or (
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
        store, BackgroundParent(), registry, AgentLoopConfig(context_budget_tokens=10**9),
    ).start()
    built = []
    original_build = runner.runtime.factory.build

    async def build(descriptor, execution):
        from power_loop.runtime.session_ownership import current_ownership
        lease = current_ownership(store, execution.session_id)
        assert lease is runner.runtime.worker._active[execution.session_id].ownership
        lease.require_owned()
        assert (lease.key, lease.epoch) == (
            (execution.store_id, execution.session_id), execution.owner_epoch)
        built.append((descriptor.kind, lease))
        return await original_build(descriptor, execution)

    runner.runtime.factory.build = build
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
        assert {kind for kind, _ in built} == {"agent", "background"}
        assert all(lease.released and not lease.lost for _, lease in built)
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
        AgentLoopConfig(context_budget_tokens=10**9, stop_policy=StopPolicy(abort_tool_s=0.01)),
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
        store, BackgroundParent(), registry, AgentLoopConfig(context_budget_tokens=10**9),
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
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?",
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
        AgentLoopConfig(context_budget_tokens=10**9),
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


@pytest.mark.parametrize("keep", [True, False])
async def test_background_target_context_policy_and_original_delivery(
    registry_store, tmp_path, keep,
):
    store, _ = registry_store
    registry = create_default_tool_registry(
        include=["bash", "background_run"], workspace_dir=str(tmp_path),
    )
    bash = registry.get("bash")
    registry.register(
        replace(bash.definition, async_capable=True), bash.handler, overwrite=True,
    )
    command = "printf '%05000d' 0 | tr 0 z"
    runner = await RunningAgent(
        store,
        BackgroundParent(arguments={
            "action": "tool",
            "tool": "bash",
            "args": {
                "command": command,
                "description": "target description",
                "context_policy": "keep" if keep else "retire",
            },
        }),
        registry,
        AgentLoopConfig(context_budget_tokens=10**9),
        runtime_env=RuntimeEnv(workspace_dir=tmp_path),
    ).start()
    engine = ToolContextEngine(
        config=ToolContextConfig(overrides={
            "bash": {"insend_fuse": "truncate", "general": "archive"},
        }),
        store=store,
        owner_session_id=runner.target.session_id,
        owner_send_index=1,
        llm=None,
        model="test",
        registry=registry,
    )
    await engine.start()
    token = active_engine.set(engine)
    try:
        await runner.send("start background")
        row = next(
            row for row in await store.load_all_messages(runner.target.session_id)
            if row.tool_call_id == "background"
        )
        admitted = json.loads(row.content)
        result = await runner.runtime.client.wait(
            InputRef(**admitted["input_ref"]), actor=runner.actor,
            deadline_ms=now_ms() + 10000,
        )
        assert result.status == "completed", result
        record = await load_record(store, runner.target.session_id, admitted["task_id"])
        receipt = summarize_record(record)
        assert receipt["summary"] == record["result"]
        source = ToolSource(
            runner.target.session_id, 1, row.seq, "background", "background_run",
            {"action": "check", "task_id": admitted["task_id"]}, encode_results([receipt]),
        )
        if keep:
            assert await _derive(engine, source, "insend_fuse", 1) is None
            assert await _derive(engine, source, "insend_fuse", 2) is None
        else:
            derived = await _derive(engine, source, "insend_fuse", 2)
            assert "target description" in derived[1] and "zzzz" not in derived[1]
        engine.config = replace(engine.config, overrides={"bash": {"hard_preserve": True}})
        derived = await _derive(engine, source, "general", 2)
        assert json.loads(derived[1])["targets"][0]["result"] == record["result"]
    finally:
        active_engine.reset(token)
        await runner.aclose()
