"""Real model responses and local cancellation retain original attempt evidence."""
from __future__ import annotations

import asyncio
import secrets

import pytest

from power_loop import AgentLoopConfig, SessionStore, register_jobs_tools
from power_loop.runtime import model_attempts
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.fixture
async def model_runtime(tmp_path, record_property, runtime_agent):
    store = await SessionStore.open(str(tmp_path / "model-attempts.db"), schema="tasks")
    provider = loop = None
    try:
        manager = JobManager(store)
        runtime = TaskRuntime(manager, tenant_id="real-attempts", principal_id="host", executor_id="executor", boot_id="boot")
        provider = BoundedProvider(limit=6)
        registry = ToolRegistry()
        register_jobs_tools(registry)
        loop = await runtime_agent(model=provider, store=store, task_runtime=runtime, registry=registry,
            config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3, max_tokens=2048, retry_policy=LLMRetryPolicy(max_attempts=3, backoff_initial=0),
                system_prompt="Follow the user's requested tool sequence exactly. Query only the task ID they provide. "
                    "Do not create, stop or repeat any work. Reply in final assistant text without pass_turn."))
        sid = loop.target.session_id
        yield store, manager, runtime, provider, loop, sid
    finally:
        try:
            if provider is not None:
                provider.report(record_property)
                if loop is not None:
                    await loop.aclose()
                await provider.close()
        finally:
            await store.close()


async def test_real_response_preserves_usage(model_runtime):
    store, manager, runtime, provider, loop, sid = model_runtime
    code = "PROOF-" + secrets.token_hex(6)
    async with asyncio.timeout(120):
        result = await loop.send(f"Reply exactly {code}. Do not use any tools.")
        assert result.status == "completed" and code in result.final_text and provider.calls == 1
        assert result.usage["total_tokens"] > 0
        assert not await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs}")



async def test_real_return_with_observation_write_failure_keeps_usage_and_unknown_control(model_runtime, monkeypatch):
    _, manager, _, provider, loop, _sid = model_runtime
    original = model_attempts.finish
    observations = []
    async def unavailable(*args, **kwargs):
        observations.append((args, kwargs))
        raise RuntimeError("test: post-response evidence storage unavailable")
    monkeypatch.setattr(model_attempts, "finish", unavailable)
    async with asyncio.timeout(120):
        result = await loop.send("Return exactly 42 in final assistant text without any tools.")
        assert result.status == "completed" and "42" in result.final_text and provider.calls == 1
        assert provider.responses and result.usage["total_tokens"] == provider.responses[0].token_usage.total_tokens > 0
        assert not await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs}")
        assert observations  # All observations refer to this one physical request.
        assert len({item[0][0].attempt_id for item in observations}) == 1
        monkeypatch.setattr(model_attempts, "finish", original)
        await original(*observations[0][0], **observations[0][1])
        from power_loop.runtime.control_runtime import current
        attempt = observations[0][0][0]
        assert current(attempt.binding).get("attempt", attempt.attempt_id)["phase"] == "finished"
        # A late observation does not start another model call.
        assert provider.calls == 1


async def test_real_client_cancellation_keeps_uncertain_request_and_does_not_retry(model_runtime, monkeypatch, record_property):
    _, manager, runtime, provider, loop, sid = model_runtime
    entered = asyncio.Event()
    complete = provider.complete
    async def started(request, **kwargs):
        work = asyncio.create_task(complete(request, **kwargs))
        while provider.calls == 0:
            await asyncio.sleep(0)
        entered.set()
        return await work
    monkeypatch.setattr(provider, "complete", started)
    receipt = await loop.runtime.client.send(loop.target, "Write a long arithmetic explanation without tools.",
        message_id="cancel-original", actor=loop.actor)
    async with asyncio.timeout(120):
        await entered.wait()
        await loop.runtime.client.cancel(receipt.input_ref, actor=loop.actor, reason="user stop")
        from power_loop.agent.runtime_store import now_ms
        result = await loop.runtime.client.wait(receipt.input_ref, actor=loop.actor, deadline_ms=now_ms()+10000)
        await loop.drain()
        assert result.status == "cancelled" and provider.calls == 1
        assert not await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs}")
        record_property("provider_server_receipt_confirmed", False)
