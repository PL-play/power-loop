"""Host commands and inline children use ordinary sessions without an assistant call."""
from __future__ import annotations

import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService
from power_loop.agent.runtime_types import InputRef
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.spec import AgentSpec, run_agent_spec
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class ChildOnly(LLMService):
    def __init__(self):
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        assert request.system_prompt.startswith("CHILD")
        return LLMResponse(raw_text="inline child result")

    async def close(self):
        pass


async def test_inline_child_create_send_and_consume_without_parent_assistant(registry_store):
    store, _ = registry_store
    registry = ToolRegistry()
    model = ChildOnly()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="child", system_prompt="CHILD", tools=[], lifecycle="linked"),
            "compute",
            parent_loop=get_current_loop(),
        )

    from power_loop.contracts.tools import ToolDefinition
    from power_loop.runtime.pricing import Free
    registry.register(ToolDefinition("delegate", "inline child", pricing=Free("test")), delegate)
    runner = await RunningAgent(
        store, model, registry, AgentLoopConfig(context_budget_tokens=10**9), command=("delegate", {}),
    ).start()
    try:
        result = await runner.send("run inline child")
        assert result.status == "completed"
        payload = json.loads(result.final_text)
        assert payload["final_text"] == "inline child result"
        assert payload["input_ref"]["message_id"]
        assert model.calls == 1
        children = await runner.runtime.client.list_children(runner.target, actor=runner.actor)
        assert len(children.items) == 1
        start = InputRef(store.store_id, children.items[0].session_id, payload["input_ref"]["message_id"])
        stored = await runner.runtime.worker.inputs.get_result(start, actor=runner.actor)
        assert stored.status == "completed" and stored.final_text == "inline child result"
        rows = await store._db.fetchall(
            f"SELECT kind,status,item_id FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='agent_result'",
            (runner.target.session_id,),
        )
        assert len(rows) == 1 and rows[0]["kind"] == "agent_result" and rows[0]["status"] == "delivered"
        inspected = await runner.runtime.client.inspect(
            InputRef(store.store_id, runner.target.session_id, rows[0]["item_id"]),
            actor=runner.actor,
        )
        assert inspected.terminal_status == "completed"
    finally:
        await runner.aclose()


async def test_host_command_effective_tools_for_workflow_node_main_and_explicit_filter(registry_store):
    """Host commands share the worker's effective_tools calculation:
    - role-restricted workflow_node without options.tools publishes frozenset({'probe'}),
    - unfiltered main session publishes None,
    - explicitly filtered main session publishes the effective registered subset,
    - context resets to None after each send, and disallowed commands are rejected.
    """
    from dataclasses import replace

    from power_loop import (
        ExecutionEnvironment,
        RegistryRuntimeFactory,
        Runtime,
        RuntimeDescriptor,
        ToolDefinition,
        TrustedActor,
    )
    from power_loop.agent.runtime_store import now_ms
    from power_loop.core.agent_context import get_effective_tools
    from power_loop.runtime.pricing import Free
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract

    store, _ = registry_store
    observations: list[tuple[list[str] | None, frozenset[str] | None]] = []

    def probe():
        loop = get_current_loop()
        names = sorted(loop.tool_registry.names()) if loop and loop.tool_registry else None
        observations.append((names, get_effective_tools()))
        return "probed"

    def parent_only():
        return "parent_only"

    def make_registry():
        reg = ToolRegistry()
        reg.register(
            ToolDefinition("probe", "probe", {"type": "object", "properties": {}}, pricing=Free("test")),
            FunctionRuntimeTool(probe, ToolRuntimeContract("test.probe", allowed_roles=("main", "workflow_node"))),
        )
        reg.register(
            ToolDefinition("parent_only", "parent_only", {"type": "object", "properties": {}}, pricing=Free("test")),
            FunctionRuntimeTool(parent_only, ToolRuntimeContract("test.parent_only", allowed_roles=("main",))),
        )
        return reg

    factory = RegistryRuntimeFactory()

    async def build(descriptor, _context):
        cmd_name = descriptor.config.get("cmd_name", "probe")
        return ExecutionEnvironment(
            None,
            make_registry(),
            AgentLoopConfig(context_budget_tokens=10**9),
            command=(cmd_name, {}),
        )

    factory.register("cmd.probe@215", build)
    rt = Runtime(store=store, factory=factory, node_id="n-cmd", boot_id="b-cmd", recovery_interval_s=None)
    await rt.worker.start()
    try:
        actor = TrustedActor("t", "p", None)
        base = RuntimeDescriptor(
            2, "agent", "cmd.probe@215", None, "tools@215", "hooks@215", "res@215", {}, {},
        )

        # 1) Unfiltered main session: all registered tools available, get_effective_tools() is None.
        main_session = await rt.client.create_session(base, actor=actor)
        r_main = await rt.client.send(main_session, "run main", message_id="m-main", actor=actor)
        res_main = await rt.client.wait(r_main.input_ref, actor=actor, deadline_ms=now_ms() + 10000)
        assert res_main.status == "completed" and res_main.final_text == "probed"
        assert get_effective_tools() is None

        # 2) Role-restricted workflow_node with parent & omitted options.tools:
        #    actual registry is ['probe'] and get_effective_tools() is frozenset({'probe'}).
        leaf_desc = replace(base, config={"workflow_node": {"workflow_run_id": "wf1", "node_id": "n1"}})
        leaf_session = await rt.client.create_session(leaf_desc, parent=main_session, actor=actor)
        r_leaf = await rt.client.send(leaf_session, "run leaf", message_id="m-leaf", actor=actor)
        res_leaf = await rt.client.wait(r_leaf.input_ref, actor=actor, deadline_ms=now_ms() + 10000)
        assert res_leaf.status == "completed" and res_leaf.final_text == "probed"
        assert get_effective_tools() is None

        # 3) Explicitly filtered main session (with an unregistered tool name):
        #    publishes only the effective registered subset frozenset({'probe'}).
        filtered_desc = replace(base, config={"tools": ["probe", "not_registered"]})
        filtered_session = await rt.client.create_session(filtered_desc, actor=actor)
        r_filtered = await rt.client.send(filtered_session, "run filtered", message_id="m-filtered", actor=actor)
        res_filtered = await rt.client.wait(r_filtered.input_ref, actor=actor, deadline_ms=now_ms() + 10000)
        assert res_filtered.status == "completed" and res_filtered.final_text == "probed"
        assert get_effective_tools() is None

        # 4) Role-restricted workflow_node attempting parent_only host command is still rejected.
        denied_desc = replace(
            base,
            config={
                "workflow_node": {"workflow_run_id": "wf1", "node_id": "n2"},
                "cmd_name": "parent_only",
            },
        )
        denied_session = await rt.client.create_session(denied_desc, parent=main_session, actor=actor)
        r_denied = await rt.client.send(denied_session, "run denied", message_id="m-denied", actor=actor)
        res_denied = await rt.client.wait(r_denied.input_ref, actor=actor, deadline_ms=now_ms() + 10000)
        assert res_denied.status == "failed" and res_denied.error_code == "forbidden"
        assert get_effective_tools() is None

        assert observations == [
            (["parent_only", "probe"], None),
            (["probe"], frozenset({"probe"})),
            (["probe"], frozenset({"probe"})),
        ]
    finally:
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
