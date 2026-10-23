"""Actual provider work under automatic boot startup and bounded shutdown."""
from __future__ import annotations

import asyncio

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.model_attempt_query import list_attempts
from power_loop.runtime.task_runtime import TaskRuntime

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_workflow_native_restore_192 import (
    test_real_native_restoration_recomputes_original_nodes_and_main_consumes_new_result as verify_restoration,
)

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_automatic_start_and_shutdown_retain_known_usage_and_block_new_provider_work(tmp_path, record_property):
    store = await SessionStore.open(str(tmp_path/"auto-boot.db"), schema="tasks")
    provider = BoundedProvider(limit=3)
    runtime = TaskRuntime(JobManager(store), tenant_id="real-auto", principal_id="host",
        executor_id="executor", boot_id="boot", boot_contract_hash="a"*64)
    try:
        loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime,
            config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2, max_tokens=2048))
        sid = await loop.new_session()
        async with asyncio.timeout(90):
            result = await loop.send("Reply exactly 42 in plain assistant text, without tools or pass_turn.", sid)
            assert result.final_text.strip() == "42" and provider.calls == 1
            [job] = await runtime.manager.db.fetchall(f"SELECT * FROM {runtime.manager.jobs}")
            [attempt] = (await list_attempts(runtime.manager, runtime.actor(sid), job["job_id"]))["items"]
            assert attempt["status"] == "returned" and not attempt["estimated"]
            stopped = await runtime.shutdown(timeout_s=5)
            assert stopped["durable_drain_confirmed"] and not stopped["pending_handles"]
            assert not stopped["physical_stop_confirmed"] and not stopped["billing_settled"]
            assert not (await registry.get(runtime.manager, runtime._lifecycle.identity))["admission_open"]
            with pytest.raises(JobError, match="executor_not_accepting"):
                await loop.send("Must not dispatch.", sid)
            assert provider.calls == 1
    finally:
        await runtime.shutdown(timeout_s=2)
        provider.report(record_property)
        await provider.close()
        await store.close()


async def test_real_automatic_shutdown_signals_inflight_default_send_without_external_stop_token(tmp_path, record_property, monkeypatch):
    store = await SessionStore.open(str(tmp_path/"auto-drain.db"), schema="tasks")
    provider = BoundedProvider(limit=3)
    runtime = TaskRuntime(JobManager(store), tenant_id="real-auto-stop", principal_id="host",
        executor_id="executor", boot_id="boot", boot_contract_hash="a"*64)
    sending = None
    try:
        loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime,
            config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2, max_tokens=2048))
        sid = await loop.new_session()
        entered = asyncio.Event()
        complete = provider.complete
        async def started(request, **kwargs):
            work = asyncio.create_task(complete(request, **kwargs))
            while provider.calls == 0:
                await asyncio.sleep(0)
            entered.set()
            return await work
        monkeypatch.setattr(provider, "complete", started)
        async with asyncio.timeout(90):
            sending = asyncio.create_task(loop.send("Write a detailed arithmetic explanation without tools.", sid))
            await entered.wait()
            await asyncio.sleep(.05)
            assert not sending.done(), "provider returned before the in-flight boundary"
            stopped = await runtime.shutdown(timeout_s=5, grace_s=1)
            result = await sending
            assert result.status == "cancelled" and provider.calls == 1
            assert stopped["durable_drain_confirmed"] and not stopped["pending_handles"]
            [job] = await runtime.manager.db.fetchall(f"SELECT * FROM {runtime.manager.jobs}")
            assert job["resource_state"] == "unknown" and job["supervision_status"] == "open"
            [attempt] = (await list_attempts(runtime.manager, runtime.actor(sid), job["job_id"]))["items"]
            assert attempt["status"] == "unknown" and attempt["estimated"] and not attempt["response_received"]
            with pytest.raises(JobError, match="executor_not_accepting"):
                await loop.send("Do not replace the original request.", sid)
            assert provider.calls == 1
    finally:
        await runtime.shutdown(timeout_s=2)
        if sending is not None and not sending.done():
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)
        provider.report(record_property)
        await provider.close()
        await store.close()


@pytest.mark.parametrize("resource_mode", ["native", "bound"])
async def test_real_original_workflow_restoration_uses_automatic_boot_lifecycle(tmp_path, record_property, monkeypatch, resource_mode):
    original_init, close_store = TaskRuntime.__init__, SessionStore.close
    runtimes = []
    async def managed_close(store):
        for runtime in runtimes:
            if runtime.manager.store is store:
                stopped = await runtime.shutdown(timeout_s=5)
                assert stopped["durable_drain_confirmed"] and not stopped["pending_handles"]
        await close_store(store)
    def configured(runtime, *args, **kwargs):
        kwargs["boot_contract_hash"] = "a"*64
        original_init(runtime, *args, **kwargs)
        runtimes.append(runtime)
    monkeypatch.setattr(TaskRuntime, "__init__", configured)
    monkeypatch.setattr(SessionStore, "close", managed_close)
    await verify_restoration(tmp_path, record_property, monkeypatch, "node_ack", resource_mode)
    assert len(runtimes) == 1
    assert runtimes[0]._lifecycle.closed
