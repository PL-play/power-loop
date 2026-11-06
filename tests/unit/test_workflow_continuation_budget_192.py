"""Design 203 workflow behavior through the public runtime."""

import pytest

from power_loop import AgentLoopConfig, LLMResponse
from power_loop.agent.runtime_store import now_ms
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("budget,expected_calls", [(100, 3), (2, 2)])
async def test_automatic_continuation_keeps_session_and_cumulative_usage(
    registry_store, tmp_path, budget, expected_calls
):
    from power_loop import LLMTokenUsage
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.runtime.pricing import Free
    from power_loop.tools import create_default_tool_registry

    store, _ = registry_store
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))

    async def step():
        return "saved"

    registry.register(
        ToolDefinition(name="step", description="step", input_schema={"type": "object"}, pricing=Free("local")), step
    )

    class Continuing(Model):
        calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            usage = LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            if self.calls < 3:
                return LLMResponse(
                    raw_text="",
                    token_usage=usage,
                    tool_calls=[
                        {
                            "id": f"step-{self.calls}",
                            "type": "function",
                            "function": {"name": "step", "arguments": "{}"},
                        }
                    ],
                )
            return LLMResponse(raw_text="finished", token_usage=usage)

    model = Continuing()
    runner = await RunningAgent(store, model, registry, config=AgentLoopConfig(compactor=None)).start()
    node = leaf("A")
    node["spec"].update(max_rounds=1, tools=["step"])
    node["continuation"] = {"max_continuations": 2, "extra_rounds": 1, "gate": "always"}
    try:
        handle = await runner.runtime.client.workflow(
            runner.target,
            {"name": "auto", "budget": {"max_tokens": budget}, "root": node},
            message_id="auto",
            actor=runner.actor,
        )
        result = await handle.wait(deadline_ms=now_ms() + 15000)
        assert model.calls == expected_calls, (model.calls, result)
        assert result.usage["calls"] == expected_calls
        assert result.usage["total_tokens"] == expected_calls * 2
        assert result.status == ("completed" if expected_calls == 3 else "budget_exceeded"), result
        nodes = await store._db.fetchall(
            f"SELECT session_id,generation FROM {store.table_prefix}jobs WHERE kind='workflow_node'"
        )
        assert len(nodes) == 1 and nodes[0]["generation"] == (2 if expected_calls == 3 else 1)
    finally:
        await runner.aclose()


async def test_explicit_continuations_keep_total_usage_and_exhaust_budget(registry_store):
    from power_loop import LLMTokenUsage
    from power_loop.agent.runtime_types import RuntimeRequestError

    store, _ = registry_store

    class Metered(Model):
        async def complete(self, request, **kwargs):
            return LLMResponse(raw_text="done", token_usage=LLMTokenUsage(prompt_tokens=1, total_tokens=1))

    runner = await RunningAgent(store, Metered(), config=AgentLoopConfig(compactor=None)).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target,
            {
                "name": "cumulative",
                "budget": {"max_tokens": 5, "stop_at_remaining_pct": 0},
                "root": {"type": "parallel", "branches": [leaf("A"), leaf("B")]},
            },
            message_id="cumulative",
            actor=runner.actor,
        )
        result = await handle.wait(deadline_ms=now_ms() + 15000)
        assert result.usage["calls"] == result.usage["total_tokens"] == 2
        for generation in range(1, 4):
            await runner.drain()
            handle = await handle.resume(
                generation=generation, node_id="A", message="improve", message_id=f"more-{generation}"
            )
            result = await handle.wait(deadline_ms=now_ms() + 15000)
            assert result.status == "completed", result
            assert set(result.results) == {"A", "B"}
            assert result.usage["calls"] == result.usage["total_tokens"] == generation + 2
        with pytest.raises(RuntimeRequestError, match="budget_exceeded"):
            await handle.resume(generation=4, node_id="A", message="over budget", message_id="over")
        nodes = await store._db.fetchall(
            f"SELECT title,generation FROM {store.table_prefix}jobs WHERE kind='workflow_node'"
        )
        assert {r["title"]: r["generation"] for r in nodes} == {"A": 4, "B": 1}
    finally:
        await runner.aclose()
