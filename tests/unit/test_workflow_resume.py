"""Explicit node continuation replays siblings and invalidates only dependent tails."""
import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMTokenUsage
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import InputRef, RuntimeRequestError
from power_loop.workflow import journal
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_explicit_continuation_reuses_sibling_and_recomputes_tail(registry_store):
    store, _ = registry_store
    model = Model()
    runner = await RunningAgent(store, model, config=AgentLoopConfig(compactor=None)).start()
    try:
        root = {
            "type": "sequence",
            "steps": [{"type": "parallel", "branches": [leaf("A"), leaf("B")]}, leaf("C")],
        }
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "continue", "root": root}, message_id="run", actor=runner.actor,
        )
        first = await handle.wait(deadline_ms=now_ms() + 10000)
        assert first.status == "completed"
        await runner.drain()
        continued = await handle.resume(node_id="A", message="improve A", message_id="more")
        second = await continued.wait(deadline_ms=now_ms() + 15000)
        assert second.status == "completed", second
        assert second.results["A"].session_id == first.results["A"].session_id
        assert second.results["B"].session_id == first.results["B"].session_id
        assert second.results["C"].session_id == first.results["C"].session_id
        saved = await journal.read(store, runner.target.session_id, handle.run_id)
        by_path = {}
        for entry in saved["nodes"]:
            by_path.setdefault(entry["path"], []).append(entry)
        assert {path: len(entries) for path, entries in by_path.items()} == {"A": 2, "B": 1, "C": 2}
        assert sorted(entry["attempt"] for entry in by_path["A"]) == [1, 2]
        assert sorted(entry["attempt"] for entry in by_path["C"]) == [1, 2]
        assert by_path["B"][0]["revision"] == 0
        assert (await handle.wait(deadline_ms=now_ms() + 3000)).to_dict() == first.to_dict()
    finally:
        await runner.aclose()


async def test_explicit_resume_gets_its_own_automatic_continuation(registry_store, tmp_path):
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.core.agent_context import get_session_id
    from power_loop.runtime.pricing import Free
    from power_loop.tools import create_default_tool_registry

    store, _ = registry_store
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    downstream = []

    class Continuing(Model):
        calls = 0
        completions = 0

        async def complete(self, request, **kwargs):
            usage = LLMTokenUsage(prompt_tokens=1, total_tokens=1)
            if not request.system_prompt.startswith(("A", "B")):
                return LLMResponse(raw_text="parent acknowledged", token_usage=usage)
            if request.system_prompt.startswith("B"):
                downstream.append(request.messages[-1]["content"])
                return LLMResponse(raw_text="B done", token_usage=usage)
            self.calls += 1
            latest = next((str(m.get("content")) for m in reversed(request.messages)
                           if m.get("role") == "user"), "")
            if "[continuation" in latest:
                self.completions += 1
                return LLMResponse(
                    raw_text="A old" if self.completions == 1 else "A improved", token_usage=usage,
                )
            return LLMResponse(raw_text="", token_usage=usage, tool_calls=[{
                "id": f"p{self.calls}", "type": "function",
                "function": {"name": "progress", "arguments": "{}"},
            }])

    model = Continuing()

    async def progress():
        await store.set_runtime_state(get_session_id(), "todo", {"items": [{
            "text": "finish", "status": "pending",
        }]})
        return "saved"

    registry.register(
        ToolDefinition(
            name="progress", description="progress", input_schema={"type": "object"}, pricing=Free("local"),
        ),
        progress,
    )
    runner = await RunningAgent(store, model, registry, config=AgentLoopConfig(compactor=None)).start()
    try:
        a, b = leaf("A"), leaf("B")
        a["spec"].update(max_rounds=1, tools=["progress"])
        a["continuation"] = {"max_continuations": 2, "extra_rounds": 3, "gate": "todo_remaining"}
        b["inputs_from"] = ["A"]
        handle = await runner.runtime.client.workflow(runner.target, {
            "name": "explicit-after-auto", "budget": {"max_tokens": 8, "stop_at_remaining_pct": 0},
            "root": {"type": "sequence", "steps": [a, b]},
        }, message_id="first", actor=runner.actor)
        first = await handle.wait(deadline_ms=now_ms() + 15000)
        assert first.status == "completed" and first.results["A"].text == "A old", first
        first_calls = model.calls
        assert first.usage["calls"] == first.usage["total_tokens"] == first_calls + 1
        before = await journal.read(store, runner.target.session_id, handle.run_id)
        original_a = next(
            entry for entry in before["nodes"] if entry["path"] == "A" and entry["continuation_index"] == 1
        )
        original_continuation = InputRef(**original_a["input_ref"])
        await runner.drain()
        more = await handle.resume(node_id="A", message="improve A", message_id="improve")
        second = await more.wait(deadline_ms=now_ms() + 15000)
        assert second.status == "completed", second
        assert second.results["A"].text == "A improved", (second, model.calls)
        assert model.calls > first_calls and len(downstream) == 2
        assert "A old" in downstream[0] and "A improved" in downstream[1]
        expected_usage = first_calls + 1 + (model.calls - first_calls) + 1
        assert second.usage["calls"] == second.usage["total_tokens"] == expected_usage
        saved = await journal.read(store, runner.target.session_id, handle.run_id)
        current_a = next(
            entry for entry in saved["nodes"]
            if entry["path"] == "A" and entry["attempt"] == 2 and entry["continuation_index"] == 1
        )
        assert current_a["session_id"] == original_a["session_id"]
        assert current_a["input_ref"] != original_a["input_ref"]
        old = await runner.runtime.client.wait(
            original_continuation, actor=runner.actor, deadline_ms=now_ms() + 3000,
        )
        assert old.final_text == "A old"
        assert (await handle.wait(deadline_ms=now_ms() + 3000)).to_dict() == first.to_dict()
        with pytest.raises(RuntimeRequestError, match="budget_exceeded"):
            await handle.resume(node_id="A", message="over budget", message_id="over")
    finally:
        await runner.aclose()
