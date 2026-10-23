"""Actual child admission, retained UTC window and original result consumption."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import completion_windows as windows
from power_loop.runtime import delegation
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import _OWNER, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("expire", [False, True])
async def test_real_original_child_keeps_admitted_utc_deadline_after_store_reopen_and_closes_tree(tmp_path, record_property, expire):
    path = str(tmp_path / "original-window.db")
    store = await SessionStore.open(path, schema="tasks")
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-window", principal_id="host", executor_id="executor", boot_id="boot")
    registry, captured = ToolRegistry(), []
    register_spawn_agent(registry)
    provider = BoundedProvider(limit=10)
    ready, release = asyncio.Event(), asyncio.Event()
    consuming = False
    child_requests = 0

    async def probe():
        owner = _OWNER.get()
        saved = await windows.read(manager, owner.actor, owner.lease.job_id)
        assert saved is not None
        captured.append(saved)
        return json.dumps({"deadline_at_ms": str(saved["deadline_at_ms"]), "status": "original-window-retained"})

    registry.register(ToolDefinition(name="deadline_probe", description="Read the original child deadline exactly once.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False}), probe)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=4, max_tokens=2048,
            system_prompt="Initially create exactly one requested child with spawn_agent, then reply WAITING as text without pass_turn. "
                "Do not create replacement tasks. Later consume the original child's result."))
    parent = await loop.new_session()

    async def hold_child(ctx):
        nonlocal child_requests
        if consuming and get_session_id() == parent:
            ctx.tools = []
        if get_session_id() != parent and captured:
            ready.set()
            await release.wait()
            ctx.tools = []
        if get_session_id() != parent:
            child_requests += 1

    loop.hooks.register(HookPoint.LLM_BEFORE, hold_child, name="test.original-window-observation")
    before = set(delegation._TASKS)
    try:
        async with asyncio.timeout(120):
            await loop.send("Use spawn_agent create once with tools=['deadline_probe'], max_rounds=3, task='Compute 7+8', "
                "system_prompt='Call deadline_probe exactly once, then answer 15 as final text with no more tools or pass_turn.'. "
                "Then reply WAITING.", parent)
            await asyncio.wait_for(ready.wait(), 30)
            [original] = captured
            assert original["deadline_at_ms"] == original["admitted_at_ms"] + windows.DEFAULT_TIMEOUT_MS
            reopened = await SessionStore.open(path, schema="tasks")
            try:
                second = JobManager(reopened)
                assert await windows.read(second, runtime.actor(parent), original["origin"]["job_id"]) == original
                assert len(await second.db.fetchall(f"SELECT event_id FROM {second.events} WHERE event_type='completion_window'")) == 1
            finally:
                await reopened.close()
            if expire:
                # Trusted fault injection advances only the runtime clock and
                # preserves short execution leases to isolate the UTC gate.
                manager.now = lambda: original["deadline_at_ms"]
                await manager.db.execute(f"UPDATE {manager.jobs} SET lease_until_ms=? WHERE resource_state='running'",
                    (manager.now() + 30000,))
            release.set()
            tasks = set(delegation._TASKS) - before
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 30)
            owner = await manager.get_job(runtime.actor(parent), original["origin"]["job_id"])
            assert owner.outcome == ("failed" if expire else "succeeded") and owner.resource_state == "stopped", owner
            if expire:
                attempts = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE event_type='model_attempt_dispatched' AND job_id IN "
                    f"(SELECT job_id FROM {manager.jobs} WHERE control_owner_job_id=?)", (owner.job_id,))
                assert len(attempts) == 1  # The probe's original actual request only.
                assert owner.result["usage"]["total_tokens"] > 0
            consuming = True
            await loop.flush_follow_ups(parent)
            rows = await manager.db.fetchall(f"SELECT kind,generation,supervision_status FROM {manager.jobs}")
            assert all(r["supervision_status"] == "closed" for r in rows), rows
            assert await windows.read(manager, runtime.actor(parent), owner.job_id) == original
            assert len(await store.list_children(parent)) == 1
            record_property("original_window", json.dumps(original))
            record_property("expiry_injected", expire)
            record_property("child_request_hooks", child_requests)
            record_property("original_owner_result", json.dumps(owner.result))
            record_property("supervision_tree", json.dumps(rows))
    finally:
        release.set()
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 20)
        for session in [parent, *(s.session_id for s in await store.list_children(parent))]:
            record_property("messages_" + session, json.dumps([asdict(m) for m in await store.load_all_messages(session)]))
        provider.report(record_property)
        await provider.close()
        await store.close()
