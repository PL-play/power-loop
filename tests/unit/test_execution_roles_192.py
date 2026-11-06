"""Finite host delegation: actual sessions, filtered tools and restored contracts."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, AgentSpec, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import reset_session_id, set_session_id
from power_loop.runtime import delegation
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import run_agent_spec
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow

from .test_async_task_faults_192 import ControlledModel

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def runtime(tmp_path):
    store = await SessionStore.open(str(tmp_path / "roles.db"))
    model = ControlledModel()
    for event in model.releases.values():
        event.set()
    registry = ToolRegistry()
    def unused():
        return "unused"
    for name in ("workflow", "spawn_agent", "run_agent", "create_workflow", "local_read", "schedule_wakeup", "wait_handoff"):
        allowed = ("main", "workflow_node") if name == "spawn_agent" else ("main",)
        handler = unused
        if name != "local_read":
            handler = FunctionRuntimeTool(handler, ToolRuntimeContract("test." + name, allowed_roles=allowed))
        registry.register(ToolDefinition(name=name, description=name, input_schema={"type": "object"}, pricing=Free("test_fixture")),
                          handler)
    loop = StatefulAgentLoop(llm=model, store=store, tool_registry=registry,
                             config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2))
    sid = await loop.new_session()
    token = set_session_id(sid)
    before = set(delegation._TASKS)
    try:
        yield store, loop, model, sid
    finally:
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
        reset_session_id(token)
        await store.close()


def spec(**kwargs):
    return AgentSpec(name="audit", system_prompt="ROLE-A", lifecycle="linked", max_rounds=2, **kwargs)




@pytest.mark.parametrize("operation", ["spawn", "nudge", "workflow", "fake-node"])
async def test_direct_dispatch_cannot_bypass_child_role(runtime, operation):
    store, loop, model, parent = runtime
    child = await run_agent_spec(spec(), "audit", parent_loop=loop)
    calls = len(model.requests)
    token = set_session_id(child["session_id"])
    try:
        with pytest.raises(ValueError, match="role_forbids"):
            if operation == "spawn":
                await delegation.start(loop, spec(), "forbidden")
            elif operation == "nudge":
                await delegation.nudge(loop, parent, "forbidden")
            elif operation == "fake-node":
                await run_agent_spec(spec(), "forbidden", parent_loop=loop, execution_role="workflow_node")
            else:
                workflow = create_workflow({"name": "forbidden", "root": {
                    "type": "agent", "id": "x", "spec": {"name": "x", "system_prompt": "x"}}},
                    parent_loop=loop, parent_session_id=parent)
                await workflow.start(detached=True)
        assert len(model.requests) == calls
        assert await store.get_runtime_state(parent, "workflow:index") is None
    finally:
        reset_session_id(token)




@pytest.mark.parametrize("metadata", [{}, {"execution_role": "main"}])
async def test_unknown_or_conflicting_legacy_child_is_not_promoted(runtime, metadata):
    store, loop, _, parent = runtime
    child = await store.create_session(parent_session_id=parent, metadata=metadata)
    token = set_session_id(child)
    try:
        with pytest.raises(ValueError, match="execution_role"):
            await delegation.start(loop, spec(), "forbidden")
    finally:
        reset_session_id(token)


async def test_host_config_factory_cannot_widen_child_delegation_policy(runtime):
    store, loop, _, _ = runtime
    loop.config.subagent_config_factory = lambda s, cfg: replace(cfg, delegation_policy="legacy")
    child = await run_agent_spec(spec(tools=["spawn_agent", "workflow", "local_read"]), "audit", parent_loop=loop)
    contract = await store.get_runtime_state(child["session_id"], "workflow_execution_contract")
    assert contract["tools"] == ["local_read"] and contract["delegation_policy"] == "single_level"




async def test_mutated_invalid_host_policy_cannot_silently_disable_dispatch_guard(runtime):
    _, loop, model, _ = runtime
    loop.config.delegation_policy = "typo"
    with pytest.raises(ValueError, match="invalid_delegation_policy"):
        await delegation.start(loop, spec(), "must not dispatch")
    assert not model.requests
