"""Real provider work under registered boot admission and original restoration."""
from __future__ import annotations

import asyncio

import pytest

from power_loop import AgentLoopConfig, CancellationToken, SessionStore, StatefulAgentLoop
from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.model_attempt_query import list_attempts
from power_loop.runtime.task_runtime import TaskRuntime

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_workflow_native_restore_192 import (
    test_real_native_restoration_recomputes_original_nodes_and_main_consumes_new_result as verify_restoration,
)

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def register_runtime(runtime):
    identity = registry.ExecutorIdentity(runtime.tenant_id, runtime.principal_id, runtime.executor_id, runtime.boot_id)
    await registry.register(runtime.manager, identity, contract_hash="a"*64, startup_lease_ms=300000)
    await registry.activate(runtime.manager, identity, expected_version=1, system_key="startup", lease_ms=300000)
    return identity


async def test_real_registered_boot_drain_rejects_another_request_without_spending(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path / "real-boot.db"), schema="tasks")
    provider = BoundedProvider(limit=3)
    try:
        manager = JobManager(store)
        runtime = TaskRuntime(manager, tenant_id="real-boot", principal_id="host", executor_id="executor", boot_id="boot")
        identity = await register_runtime(runtime)
        loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime,
            config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2, max_tokens=2048))
        sid = await loop.new_session()
        async with asyncio.timeout(90):
            result = await loop.send("Reply exactly 42 as plain assistant text, without any tools or pass_turn.", sid)
            assert result.status == "completed" and result.final_text.strip() == "42" and provider.calls == 1
            [job] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
            saved = await manager.get_execution_membership(runtime.actor(sid), job["job_id"], 1)
            assert saved["scope"]["version"] == 2 and job["executor_key"]
            receipt = await registry.drain(manager, identity, system_key="shutdown", reason="test host drain")
            assert not receipt["physical_stop_confirmed"]
            with pytest.raises(JobError, match="executor_not_accepting"):
                await loop.send("This request must not be sent to the provider.", sid)
            assert provider.calls == 1 and not (await registry.get(manager, identity))["admission_open"]
            assert await manager.get_execution_membership(runtime.actor(sid), job["job_id"], 1) == saved
    finally:
        provider.report(record_property)
        try:
            await provider.close()
        finally:
            await store.close()


async def test_real_inflight_request_keeps_unknown_after_durable_drain_and_local_stop(tmp_path, record_property, monkeypatch):
    store = await SessionStore.open(str(tmp_path / "real-drain.db"), schema="tasks")
    provider = BoundedProvider(limit=3)
    sending = None
    try:
        manager = JobManager(store)
        runtime = TaskRuntime(manager, tenant_id="real-drain", principal_id="host", executor_id="executor", boot_id="boot")
        identity = await register_runtime(runtime)
        loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime,
            config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2, max_tokens=2048))
        sid = await loop.new_session()
        entered, token = asyncio.Event(), CancellationToken()
        complete = provider.complete
        async def started(request, **kwargs):
            work = asyncio.create_task(complete(request, **kwargs))
            while provider.calls == 0:
                await asyncio.sleep(0)
            entered.set()
            return await work
        monkeypatch.setattr(provider, "complete", started)
        async with asyncio.timeout(90):
            sending = asyncio.create_task(loop.send("Write a detailed arithmetic explanation without tools.", sid, stop_event=token))
            await entered.wait()
            await asyncio.sleep(.05)
            assert not sending.done(), "real request completed before the stop boundary"
            receipt = await registry.drain(manager, identity, system_key="shutdown", reason="host shutdown")
            assert len(receipt["effects"]["executions"]) == 1
            token.cancel("host shutdown after durable drain")
            result = await sending
            assert result.status == "cancelled" and provider.calls == 1
            [job] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
            assert job["resource_state"] == "unknown" and job["supervision_status"] == "open"
            [attempt] = (await list_attempts(manager, runtime.actor(sid), job["job_id"]))["items"]
            assert attempt["status"] == "unknown" and attempt["estimated"] and not attempt["response_received"]
            assert attempt["observed_usage"]["prompt_tokens"] > 0
            with pytest.raises(JobError, match="executor_not_accepting"):
                await loop.send("Do not dispatch another request.", sid)
            assert provider.calls == 1
            record_property("provider_server_receipt_confirmed", False)
    finally:
        if sending is not None and not sending.done():
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)
        provider.report(record_property)
        try:
            await provider.close()
        finally:
            await store.close()


@pytest.mark.parametrize("resource_mode", ["native", "bound"])
async def test_real_registered_original_workflow_restoration_keeps_boot_binding(tmp_path, record_property, monkeypatch, resource_mode):
    claim = TaskRuntime.claim
    seen = set()
    async def registered(runtime, *args, **kwargs):
        key = runtime.manager.store.store_id, runtime.executor_id, runtime.boot_id
        if key not in seen:
            await register_runtime(runtime)
            seen.add(key)
        return await claim(runtime, *args, **kwargs)
    monkeypatch.setattr(TaskRuntime, "claim", registered)
    await verify_restoration(tmp_path, record_property, monkeypatch, "node_ack", resource_mode)
    store = await SessionStore.open(str(tmp_path / "native-restore.db"), schema="tasks")
    try:
        jobs = await store._db.fetchall(f"SELECT executor_key FROM {store.table_prefix}jobs WHERE generation>0")
        assert jobs and all(j["executor_key"] is not None for j in jobs)
        events = await store._db.fetchall(f"SELECT event_id FROM {store.table_prefix}task_executor_events WHERE action='register'")
        assert len(events) == 1
    finally:
        await store.close()
