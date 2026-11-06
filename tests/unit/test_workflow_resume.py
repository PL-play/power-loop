"""Design 203 workflow behavior through the public runtime."""

import asyncio

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMTokenUsage
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import InputRef, JobRef, RuntimeRequestError, plain
from power_loop.workflow import journal
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_explicit_continuation_reuses_sibling_and_recomputes_tail(registry_store):
    store, _ = registry_store
    model = Model()
    runner = await RunningAgent(store, model, config=AgentLoopConfig(compactor=None)).start()
    try:
        root = {"type": "sequence", "steps": [{"type": "parallel", "branches": [leaf("A"), leaf("B")]}, leaf("C")]}
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "continue", "root": root}, message_id="run", actor=runner.actor
        )
        first = await handle.wait(deadline_ms=now_ms() + 10000)
        assert first.status == "completed"
        await runner.drain()
        continued = await handle.resume(generation=1, node_id="A", message="improve A", message_id="more")
        second = await continued.wait(deadline_ms=now_ms() + 15000)
        assert second.status == "completed", second
        rows = await store._db.fetchall(
            f"SELECT title,generation,session_id FROM {store.table_prefix}jobs WHERE kind='workflow_node'"
        )
        assert len([r for r in rows if r["title"] == "A"]) == 1
        assert next(r for r in rows if r["title"] == "A")["generation"] == 2
        assert len([r for r in rows if r["title"] == "B"]) == 1
        assert len([r for r in rows if r["title"] == "C"]) == 2
    finally:
        await runner.aclose()


@pytest.mark.parametrize("rebuild", [False, True])
async def test_explicit_resume_gets_its_own_automatic_continuation(registry_store, tmp_path, rebuild):
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.core.agent_context import get_session_id
    from power_loop.runtime.pricing import Free
    from power_loop.tools import create_default_tool_registry

    store, _ = registry_store
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    started, release = asyncio.Event(), asyncio.Event()
    downstream = []

    class Continuing(Model):
        calls = 0

        async def complete(self, request, **kwargs):
            usage = LLMTokenUsage(prompt_tokens=1, total_tokens=1)
            if request.system_prompt.startswith("B"):
                downstream.append(request.messages[-1]["content"])
                if rebuild and len(downstream) == 2:
                    started.set()
                    await release.wait()
                return LLMResponse(raw_text="B done", token_usage=usage)
            self.calls += 1
            if self.calls % 4:
                return LLMResponse(raw_text="", token_usage=usage, tool_calls=[{
                    "id": f"p{self.calls}", "type": "function",
                    "function": {"name": "progress", "arguments": "{}"},
                }])
            return LLMResponse(raw_text="A old" if self.calls == 4 else "A improved", token_usage=usage)

    model = Continuing()

    async def progress():
        await store.set_runtime_state(get_session_id(), "todo", {"items": [{
            "text": "finish", "status": "completed" if model.calls % 4 == 3 else "pending",
        }]})
        return "saved"

    registry.register(ToolDefinition(name="progress", description="progress", input_schema={"type": "object"},
                                     pricing=Free("local")), progress)
    runner = await RunningAgent(store, model, registry, config=AgentLoopConfig(compactor=None)).start()
    try:
        a, b = leaf("A"), leaf("B")
        a["spec"].update(max_rounds=1, tools=["progress"])
        a["continuation"] = {"max_continuations": 2, "extra_rounds": 3, "gate": "todo_remaining"}
        b["inputs_from"] = ["A"]
        handle = await runner.runtime.client.workflow(runner.target, {
            "name": "explicit-after-auto", "budget": {"max_tokens": 10, "stop_at_remaining_pct": 0},
            "root": {"type": "sequence", "steps": [a, b]},
        }, message_id="first", actor=runner.actor)
        first = await handle.wait(deadline_ms=now_ms() + 15000)
        assert first.status == "completed" and first.results["A"].text == "A old" and model.calls == 4
        assert first.usage["calls"] == first.usage["total_tokens"] == 5
        before = await journal.read(store, runner.target.session_id, handle.run_id)
        original_a = next(n for n in before["nodes"] if n["path"] == "A")
        original_continuation = InputRef(**original_a["continuations"]["1"])
        await runner.drain()
        more = await handle.resume(generation=1, node_id="A", message="improve A", message_id="improve")
        generation = 2
        if rebuild:
            await asyncio.wait_for(started.wait(), 15)
            # Rebuild the driver task in this process; the separate OS-process
            # suite retains its real kill/restart coverage unchanged.
            worker = runner.runtime.worker
            ctx = worker._active[more.input_ref.session_id]
            ctx.shutdown_interrupted = True
            ctx.task.cancel()
            await asyncio.gather(ctx.task, return_exceptions=True)
            worker._activate(ctx.session_id)
            release.set()
            generation = 3
            async with asyncio.timeout(15):
                while (await store._db.fetchone(
                    f"SELECT generation FROM {store.table_prefix}jobs WHERE job_id=?", (handle.job_id,)
                ))["generation"] != generation:
                    await asyncio.sleep(0.02)
            result = await runner.runtime.client.wait(JobRef(store.store_id, handle.job_id, generation),
                                                      actor=runner.actor, deadline_ms=now_ms() + 15000)
            assert result.status == "completed", result
            second = plain(result.structured_output)
        else:
            second = (await more.wait(deadline_ms=now_ms() + 15000)).to_dict()
        assert second["status"] == "completed", second
        assert second["results"]["A"]["text"] == "A improved", (second, model.calls)
        assert model.calls == 8 and len(downstream) == 2
        assert "A old" in downstream[0] and "A improved" in downstream[1]
        assert second["usage"]["calls"] == second["usage"]["total_tokens"] == 10
        assert second["results"]["A"]["usage"]["calls"] == 4
        saved = await journal.read(store, runner.target.session_id, handle.run_id)
        latest_a = next(n for n in saved["nodes"] if n["path"] == "A")
        assert latest_a["job_id"] == original_a["job_id"] and latest_a["session_id"] == original_a["session_id"]
        assert latest_a["result_ref"]["generation"] == 4
        assert set(latest_a["continuations"]) == {"1"}
        assert latest_a["continuations"]["1"] != plain(original_continuation)
        assert len(saved["nodes"]) == 3  # One A job and the two intended B attempts.
        assert (await handle.wait(deadline_ms=now_ms() + 3000)).to_dict() == first.to_dict()
        old = await runner.runtime.client.wait(original_continuation, actor=runner.actor, deadline_ms=now_ms() + 3000)
        assert old.final_text == "A old" and old.job_ref.generation == 2
        for g in range(1, 5):
            original = await runner.runtime.client.wait(JobRef(store.store_id, original_a["job_id"], g),
                                                        actor=runner.actor, deadline_ms=now_ms() + 3000)
            assert original.usage["calls"] == original.usage["total_tokens"] == 2
        with pytest.raises(RuntimeRequestError, match="budget_exceeded"):
            await handle.resume(generation=generation, node_id="A", message="over budget", message_id="over")
    finally:
        release.set()
        await runner.aclose()
