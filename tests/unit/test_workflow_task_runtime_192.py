"""Design 203 workflow behavior through the public runtime."""

import pytest

from power_loop import AgentLoopConfig, LLMResponse
from power_loop.agent.runtime_store import now_ms
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_sequence(registry_store):
    store, _ = registry_store
    runner = await RunningAgent(store, Model(), config=AgentLoopConfig(context_budget_tokens=10**9)).start()
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
    runner = await RunningAgent(store, Model(), config=AgentLoopConfig(context_budget_tokens=10**9)).start()
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
        rows = await store._db.fetchall(
            f"SELECT session_id,descriptor_json FROM {store.t.sessions} "
            "WHERE descriptor_json LIKE '%workflow_node%'",
        )
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
        created = False

        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("A"):
                return LLMResponse(raw_text="leaf done")
            if not self.created:
                self.created = True
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

    runner = await RunningAgent(store, Parent(), registry, AgentLoopConfig(context_budget_tokens=10**9)).start()
    try:
        result = await runner.send("run")
        assert result.status == "completed"
        messages = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(m.content for m in messages if m.tool_call_id == "workflow-call"))
        assert receipt["status"] == "accepted", receipt
        assert receipt["target_ref"]["session_id"] == receipt["input_ref"]["session_id"]
        assert "job_id" not in receipt and "job_ref" not in receipt
        result = await runner.runtime.client.wait(
            InputRef(**receipt["input_ref"]), actor=runner.actor, deadline_ms=now_ms() + 10000
        )
        assert result.status == "completed"
        await runner.runtime.worker.recover()
        await runner.drain()
        notices = await store._db.fetchall(
            f"SELECT item_id,kind,status FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result'",
            (runner.target.session_id,),
        )
        assert len(notices) == 1 and notices[0]["status"] == "delivered"
        inspected = await runner.runtime.client.inspect(
            InputRef(store.store_id, runner.target.session_id, notices[0]["item_id"]),
            actor=runner.actor,
        )
        assert inspected.terminal_status == "completed"
    finally:
        await runner.aclose()


async def test_required_child_finishes_before_downstream(registry_store, tmp_path):
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.runtime.pricing import Free
    from power_loop.tools import create_default_tool_registry
    from power_loop.tools.spawn_agent import register_spawn_agent

    store, _ = registry_store
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    register_spawn_agent(registry)

    class Delegating(Model):
        tool_done = False
        a_called_tool = False
        saw_tools_on_a: tuple[str, ...] = ()

        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("A"):
                self.saw_tools_on_a = tuple(
                    (t.get("function", {}) or {}).get("name") or t.get("name")
                    for t in (request.tools or ())
                )
                if not self.a_called_tool:
                    self.a_called_tool = True
                    return LLMResponse(
                        raw_text="",
                        tool_calls=[
                            {
                                "id": "prep",
                                "type": "function",
                                "function": {"name": "step_work", "arguments": "{}"},
                            }
                        ],
                    )
                return LLMResponse(raw_text="A done")
            if request.system_prompt.startswith("B"):
                assert self.tool_done
                return LLMResponse(raw_text="B done")
            return LLMResponse(raw_text="parent ack")

    model = Delegating()

    async def step_work():
        model.tool_done = True
        return "step complete"

    registry.register(
        ToolDefinition("step_work", "leaf work", input_schema={"type": "object"}, pricing=Free("test")),
        step_work,
    )
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(context_budget_tokens=10**9)).start()
    a = leaf("A")
    # Design 217 §5.3 / M1c single-level delegation: spawn_agent is stripped on workflow_node (depth=2)
    a["spec"].update(tools=["spawn_agent", "step_work"], max_rounds=5)
    try:
        handle = await runner.runtime.client.workflow(
            runner.target,
            {"name": "required", "root": {"type": "sequence", "steps": [a, leaf("B")]}},
            message_id="children",
            actor=runner.actor,
        )
        result = await handle.wait(deadline_ms=now_ms() + 15000)
        assert result.status == "completed" and model.tool_done, result
        assert "spawn_agent" not in model.saw_tools_on_a and "step_work" in model.saw_tools_on_a
        children = await store._db.fetchall(
            f"SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? OR depth>=3",
            (result.results["A"].session_id,),
        )
        assert children == []
    finally:
        await runner.aclose()


@pytest.mark.parametrize("policy", ["retry", "fallback", "continue"])
async def test_failure_policy_uses_new_attempt_and_keeps_usage(registry_store, policy):
    store, _ = registry_store

    class Failing(Model):
        leaf_calls = 0

        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith(("A", "B")):
                self.leaf_calls += 1
                text = "invalid" if request.system_prompt.startswith("A") and self.leaf_calls == 1 else '{"ok":true}'
                return LLMResponse(raw_text=text)
            return LLMResponse(raw_text="parent ack")

    model = Failing()
    runner = await RunningAgent(store, model, config=AgentLoopConfig(context_budget_tokens=10**9)).start()
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
        assert result.usage["calls"] == model.leaf_calls
        nodes = await store._db.fetchall(
            f"SELECT session_id FROM {store.t.sessions} WHERE descriptor_json LIKE '%workflow_node%'",
        )
        assert len(nodes) == (2 if policy == "fallback" else 1)
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

    runner = await RunningAgent(store, Reviewing(), config=AgentLoopConfig(context_budget_tokens=10**9)).start()
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
            f"SELECT session_id FROM {store.t.sessions} WHERE descriptor_json LIKE '%workflow_node%'",
        )
        entries = await store._db.fetchall(
            f"SELECT state_key FROM {store.t.session_runtime_state} WHERE state_key LIKE 'workflow:node:%'",
        )
        assert len(entries) == 4 and len(nodes) == 2 and len({n["session_id"] for n in nodes}) == 2
        assert result.results["review"].payload == {"pass": True}
    finally:
        await runner.aclose()
