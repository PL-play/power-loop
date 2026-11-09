"""Design 203 workflow behavior through the public runtime."""

import pytest

from power_loop import AgentLoopConfig
from power_loop.agent.runtime_store import now_ms
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_cancel_driver_stops_tree_before_new_dispatch(registry_store):
    import asyncio

    store, _ = registry_store
    started = asyncio.Event()

    class Waiting(Model):
        async def complete(self, request, **kwargs):
            started.set()
            await asyncio.Event().wait()

    runner = await RunningAgent(store, Waiting(), config=AgentLoopConfig(compactor=None)).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target,
            {"name": "cancel", "root": {"type": "sequence", "steps": [leaf("A"), leaf("B")]}},
            message_id="cancel",
            actor=runner.actor,
        )
        await asyncio.wait_for(started.wait(), 5)
        await handle.cancel()
        result = await handle.wait(deadline_ms=now_ms() + 10000)
        assert result.status == "cancelled"
        rows = await store._db.fetchall(f"SELECT title FROM {store.table_prefix}jobs WHERE kind='workflow_node'")
        assert [r["title"] for r in rows] == ["A"]
    finally:
        await runner.aclose()


async def test_cancel_one_generation_allows_other_branch_to_finish(registry_store):
    import asyncio

    from power_loop.agent.runtime_types import JobRef

    store, _ = registry_store
    entered = asyncio.Event()

    class Waiting(Model):
        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("A"):
                entered.set()
                await asyncio.Event().wait()
            return await super().complete(request, **kwargs)

    runner = await RunningAgent(store, Waiting(), config=AgentLoopConfig(compactor=None)).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target,
            {
                "name": "node stop",
                "root": {"type": "parallel", "on_error": "continue", "branches": [leaf("A"), leaf("B")]},
            },
            message_id="node-stop",
            actor=runner.actor,
        )
        await asyncio.wait_for(entered.wait(), 5)
        node = await store._db.fetchone(
            f"SELECT job_id,generation FROM {store.table_prefix}jobs WHERE kind='workflow_node' AND title='A'"
        )
        await runner.runtime.client.cancel(
            JobRef(store.store_id, node["job_id"], node["generation"]), actor=runner.actor, scope="self"
        )
        result = await handle.wait(deadline_ms=now_ms() + 15000)
        assert result.results["A"].status == "cancelled" and result.results["B"].ok
        assert result.status == "completed" and not result.ok
    finally:
        await runner.aclose()


async def test_timeout_closes_dispatch_and_cannot_be_reset_by_resume(registry_store):
    import asyncio

    from power_loop.agent.runtime_types import RuntimeRequestError

    store, _ = registry_store

    class Waiting(Model):
        async def complete(self, request, **kwargs):
            await asyncio.Event().wait()

    runner = await RunningAgent(store, Waiting(), config=AgentLoopConfig(compactor=None)).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target,
            {"name": "deadline", "timeout_s": 1, "root": {"type": "sequence", "steps": [leaf("A"), leaf("B")]}},
            message_id="deadline",
            actor=runner.actor,
        )
        result = await handle.wait(deadline_ms=now_ms() + 10000)
        assert result.status == "failed" and result.errors == ["workflow timeout"]
        with pytest.raises(RuntimeRequestError, match="workflow_timeout"):
            await handle.resume(generation=1, node_id="A", message="more", message_id="more")
        nodes = await store._db.fetchall(f"SELECT title FROM {store.table_prefix}jobs WHERE kind='workflow_node'")
        assert all(n["title"] == "A" for n in nodes)
    finally:
        await runner.aclose()


@pytest.mark.parametrize("notify", [True, False])
async def test_stopped_node_of_agent_workflow_notifies_owner_at_once(registry_store, tmp_path, notify):
    import asyncio
    import json

    from power_loop import LLMResponse
    from power_loop.agent.runtime_types import JobRef
    from power_loop.tools import create_default_tool_registry
    from power_loop.workflow import register_workflow_tool

    store, _ = registry_store
    entered, finish_b = asyncio.Event(), asyncio.Event()

    class Parent(Model):
        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("A"):
                entered.set()
                await asyncio.Event().wait()
            if request.system_prompt.startswith("B"):
                await finish_b.wait()
                return LLMResponse(raw_text="B done")
            if not any(m.get("role") == "tool" for m in request.messages):
                return LLMResponse(raw_text="", tool_calls=[{"id": "wf", "type": "function", "function": {
                    "name": "workflow", "arguments": json.dumps({"action": "create", "spec": {
                        "name": "stop", "root": {"type": "parallel", "on_error": "continue",
                                                 "branches": [leaf("A"), leaf("B")]}}})}}])
            return LLMResponse(raw_text="parent done")

    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    register_workflow_tool(registry)
    runner = await RunningAgent(store, Parent(), registry, AgentLoopConfig(compactor=None), capacity=3).start()
    try:
        await runner.send("start")
        await asyncio.wait_for(entered.wait(), 5)
        node = await store._db.fetchone(
            f"SELECT job_id,generation FROM {store.table_prefix}jobs WHERE kind='workflow_node' AND title='A'")
        await runner.runtime.client.cancel(JobRef(store.store_id, node["job_id"], node["generation"]),
                                           actor=runner.actor, reason="user stopped A", notify_parent=notify)
        async def notices():
            rows = await store.load_all_messages(runner.target.session_id)
            return [r.content for r in rows if r.role == "user" and "workflow_node_cancelled" in str(r.content)]
        if notify:
            async with asyncio.timeout(10):
                while not await notices():
                    await asyncio.sleep(.02)
            [notice] = await notices()
            assert "user stopped A" in notice and node["job_id"] in notice
        else:
            await asyncio.sleep(.3)
            assert await notices() == []
    finally:
        finish_b.set()
        await runner.aclose()
