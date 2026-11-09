"""Design 203 workflow behavior through the public runtime."""

import pytest

from power_loop import AgentLoopConfig, LLMResponse
from power_loop.agent.runtime_store import now_ms
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_sequence(registry_store):
    store, _ = registry_store
    runner = await RunningAgent(store, Model(), config=AgentLoopConfig(compactor=None)).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target,
            {"name": "chain", "root": {"type": "sequence", "steps": [leaf("A"), leaf("B")]}},
            message_id="workflow",
            actor=runner.actor,
        )
        result = await handle.wait(deadline_ms=now_ms() + 10000)
        assert result.status == "completed", result
        assert list(result.results) == ["A", "B"]
    finally:
        await runner.aclose()


@pytest.mark.parametrize("shape,expected", [("parallel", 2), ("foreach", 4), ("loop", 2), ("branch", 2)])
async def test_graph_shapes_have_distinct_saved_nodes(registry_store, shape, expected):
    store, _ = registry_store
    runner = await RunningAgent(store, Model(), config=AgentLoopConfig(compactor=None)).start()
    if shape == "parallel":
        root = {"type": "parallel", "branches": [leaf("A"), leaf("B")]}
    elif shape == "foreach":
        root = {
            "type": "foreach",
            "id": "outer",
            "as": "region",
            "items": ["n", "s"],
            "parallel": True,
            "body": {
                "type": "foreach",
                "id": "inner",
                "as": "item",
                "items": [1, 2],
                "parallel": True,
                "body": leaf("A"),
            },
        }
    elif shape == "loop":
        root = {
            "type": "loop",
            "id": "repeat",
            "body": leaf("A"),
            "until": {"ref": "A", "op": "eq", "value": "never"},
            "max_iterations": 2,
            "on_exhausted": "continue",
        }
    else:
        root = {
            "type": "sequence",
            "steps": [leaf("A"), {"type": "branch", "on": "A", "cases": {"result:A": leaf("B")}, "default": leaf("C")}],
        }
    try:
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "shapes", "root": root}, message_id="shape", actor=runner.actor
        )
        result = await handle.wait(deadline_ms=now_ms() + 15000)
        assert result.status == "completed", result
        rows = await store._db.fetchall(f"SELECT session_id FROM {store.table_prefix}jobs WHERE kind='workflow_node'")
        assert len(rows) == expected and len({r["session_id"] for r in rows}) == expected
        assert "C" not in result.results
    finally:
        await runner.aclose()


async def test_model_submission_returns_receipt_and_delivers_to_parent(registry_store, tmp_path):
    import json

    from power_loop.agent.runtime_types import InputRef
    from power_loop.tools import create_default_tool_registry
    from power_loop.workflow import register_workflow_tool

    store, _ = registry_store
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    register_workflow_tool(registry)

    class Parent(Model):
        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("A"):
                return LLMResponse(raw_text="leaf done")
            if not any(m.get("role") == "tool" for m in request.messages):
                return LLMResponse(
                    raw_text="",
                    tool_calls=[
                        {
                            "id": "workflow-call",
                            "type": "function",
                            "function": {
                                "name": "workflow",
                                "arguments": json.dumps(
                                    {"action": "create", "spec": {"name": "from model", "root": leaf("A")}}
                                ),
                            },
                        }
                    ],
                )
            return LLMResponse(raw_text="parent finished")

    runner = await RunningAgent(store, Parent(), registry, AgentLoopConfig(compactor=None)).start()
    try:
        result = await runner.send("run")
        assert result.status == "completed"
        messages = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(m.content for m in messages if m.tool_call_id == "workflow-call"))
        assert receipt["status"] == "accepted", receipt
        result = await runner.runtime.client.wait(
            InputRef(**receipt["input_ref"]), actor=runner.actor, deadline_ms=now_ms() + 10000
        )
        assert result.status == "completed"
        await runner.drain()
        notices = await store._db.fetchall(
            f"SELECT meta_json FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='task_done'",
            (runner.target.session_id,),
        )
        assert len(notices) == 1
        assert json.loads(notices[0]["meta_json"])["result_ref"]["job_id"] == receipt["job_id"]
        assert not await store._db.fetchall(f"SELECT job_id FROM {store.table_prefix}jobs WHERE supervision_status='open'")
    finally:
        await runner.aclose()


async def test_required_child_finishes_before_downstream(registry_store, tmp_path):
    import json

    from power_loop.tools import create_default_tool_registry
    from power_loop.tools.spawn_agent import register_spawn_agent

    store, _ = registry_store
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    register_spawn_agent(registry)

    class Delegating(Model):
        child_done = False

        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("CHILD"):
                self.child_done = True
                return LLMResponse(raw_text="child result")
            if request.system_prompt.startswith("A"):
                if not any(m.get("role") == "tool" for m in request.messages):
                    return LLMResponse(
                        raw_text="",
                        tool_calls=[
                            {
                                "id": "spawn",
                                "type": "function",
                                "function": {
                                    "name": "spawn_agent",
                                    "arguments": json.dumps({"task": "work", "system_prompt": "CHILD", "tools": []}),
                                },
                            }
                        ],
                    )
                return LLMResponse(raw_text="A done")
            assert self.child_done
            return LLMResponse(raw_text="B done")

    model = Delegating()
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
    a = leaf("A")
    a["spec"].update(tools=["spawn_agent"], max_rounds=5)
    try:
        handle = await runner.runtime.client.workflow(
            runner.target,
            {"name": "required", "root": {"type": "sequence", "steps": [a, leaf("B")]}},
            message_id="children",
            actor=runner.actor,
        )
        result = await handle.wait(deadline_ms=now_ms() + 15000)
        assert result.status == "completed" and model.child_done, result
        children = await store._db.fetchall(
            f"SELECT outcome,control_owner_job_id FROM {store.table_prefix}jobs WHERE kind='subagent'"
        )
        assert len(children) == 1 and children[0]["outcome"] == "succeeded"
    finally:
        await runner.aclose()


@pytest.mark.parametrize("policy", ["retry", "fallback", "continue"])
async def test_failure_policy_uses_new_attempt_and_keeps_usage(registry_store, policy):
    store, _ = registry_store

    class Failing(Model):
        calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            # A failed declared JSON output exhausts its bounded repair.
            text = "invalid" if request.system_prompt.startswith("A") and self.calls <= 2 else '{"ok":true}'
            return LLMResponse(raw_text=text)

    model = Failing()
    runner = await RunningAgent(store, model, config=AgentLoopConfig(compactor=None)).start()
    node = leaf("A")
    node["output_schema"] = {
        "name": "answer",
        "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    }
    if policy == "retry":
        node["retry"] = {"max_attempts": 2, "on": ["failed"]}
    elif policy == "fallback":
        node["fallback"] = leaf("B")
    else:
        node["continue_on_error"] = True
    try:
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "failure", "root": node}, message_id="failure", actor=runner.actor
        )
        result = await handle.wait(deadline_ms=now_ms() + 10000)
        assert result.status == "completed", result
        assert result.usage["calls"] == model.calls
        nodes = await store._db.fetchall(f"SELECT session_id FROM {store.table_prefix}jobs WHERE kind='workflow_node'")
        assert len(nodes) == (1 if policy == "continue" else 2)
        assert len({n["session_id"] for n in nodes}) == len(nodes)
    finally:
        await runner.aclose()


async def test_loop_reuses_sessions_and_renders_dependency_output(registry_store):
    import json
    from collections import Counter

    store, _ = registry_store
    calls = Counter()

    class Reviewing(Model):
        async def complete(self, request, **kwargs):
            name = "review" if request.system_prompt.startswith("review") else "dev"
            calls[name] += 1
            if name == "dev":
                return LLMResponse(raw_text=f"revision-{calls[name]}")
            assert any(f"revision-{calls[name]}" in str(m.get("content")) for m in request.messages)
            return LLMResponse(raw_text=json.dumps({"pass": calls[name] == 2}))

    runner = await RunningAgent(store, Reviewing(), config=AgentLoopConfig(compactor=None)).start()
    dev, review = leaf("dev"), leaf("review")
    dev["spec"]["system_prompt"] = "{{input}}"
    review["inputs_from"] = ["dev"]
    review["output_schema"] = {
        "name": "Review",
        "schema": {"type": "object", "properties": {"pass": {"type": "boolean"}}, "required": ["pass"]},
    }
    try:
        handle = await runner.runtime.client.workflow(
            runner.target,
            {
                "name": "rework",
                "input": "dev",
                "root": {
                    "type": "loop",
                    "id": "rework",
                    "max_iterations": 2,
                    "continue_nodes": ["dev", "review"],
                    "body": {"type": "sequence", "steps": [dev, review]},
                    "until": {"ref": "review.pass", "op": "truthy"},
                },
            },
            message_id="rework",
            actor=runner.actor,
        )
        result = await handle.wait(deadline_ms=now_ms() + 15000)
        assert result.status == "completed", result
        assert result.usage["calls"] == 4 and calls == {"dev": 2, "review": 2}
        nodes = await store._db.fetchall(
            f"SELECT title,session_id FROM {store.table_prefix}jobs WHERE kind='workflow_node'"
        )
        assert len(nodes) == 4 and len({n["session_id"] for n in nodes}) == 2
        assert result.results["review"].payload == {"pass": True}
    finally:
        await runner.aclose()
