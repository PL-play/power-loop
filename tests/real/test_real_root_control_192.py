"""Real model calls under durable original-context cancellation."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime import root_control
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.model_attempt_query import list_attempts
from power_loop.runtime.task_runtime import TaskRuntime

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]
NAMESPACE = "deeptalk.agent-run.v1"
PAYLOAD = {"origin_run_id": 101}


@pytest.mark.parametrize("inflight", [False, True])
async def test_real_context_stop_never_renews_execution_and_retains_original_usage(
    tmp_path, record_property, inflight, monkeypatch
):
    store = await SessionStore.open(str(tmp_path / "original-cancel.db"), schema="tasks")
    provider = BoundedProvider(limit=1)
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-context", principal_id="host",
        executor_id="executor", boot_id="boot", lease_ms=1000)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2, max_tokens=2048))
    sid = await loop.new_session()
    sending = None
    try:
        entered = asyncio.Event()
        complete = provider.complete
        async def started(request, **kwargs):
            work = asyncio.create_task(complete(request, **kwargs))
            try:
                while provider.calls == 0:
                    await asyncio.sleep(0)
                entered.set()
                return await work
            finally:
                if not work.done():
                    work.cancel()
                await asyncio.gather(work, return_exceptions=True)
        monkeypatch.setattr(provider, "complete", started)
        async def send():
            with runtime.root_context(sid, namespace=NAMESPACE, payload=PAYLOAD):
                return await loop.send("Reply exactly 42 without tools or pass_turn.", sid)
        async with asyncio.timeout(90):
            sending = asyncio.create_task(send())
            if inflight:
                await entered.wait()
                await asyncio.sleep(.05)
                assert not sending.done(), "provider finished before the cancellation boundary"
            else:
                result = await sending
                assert result.status == "completed" and result.final_text.strip() == "42"
                assert len(provider.responses) == 1 and result.usage["total_tokens"] > 0
            # A fresh mechanical manager neither builds a loop nor obtains a new lease.
            control = JobManager(store)
            receipt = await root_control.cancel(control, runtime.actor(sid), namespace=NAMESPACE,
                payload=PAYLOAD, system_key="original-stop", reason="user stop")
            assert len(receipt["roots"]) == 1 and receipt["resource_stop_confirmed"] is False
            assert receipt["billing_settled"] is False and receipt["execution_authority"] is False
            result = await sending
            assert result.status == ("cancelled" if inflight else "completed")
            assert provider.calls == 1
            with pytest.raises(JobError, match="root_context_cancelled"):
                await send()
            [row] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
            assert row["cancel_requested_at"] is not None
            [attempt] = (await list_attempts(manager, runtime.actor(sid), row["job_id"]))["items"]
            if inflight:
                assert row["resource_state"] == "unknown" and row["supervision_status"] == "open"
                assert attempt["status"] == "unknown" and attempt["estimated"] and not attempt["response_received"]
            else:
                assert attempt["response_received"] and attempt["observed_usage"]["total_tokens"] > 0
            assert provider.calls == 1
            record_property("original_context_stop", json.dumps(receipt))
            record_property("original_model_attempt", json.dumps(attempt))
            record_property("provider_server_receipt_confirmed", not inflight)
    finally:
        if sending is not None and not sending.done():
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)
        provider.report(record_property)
        record_property("provider_raw_responses", json.dumps([asdict(r) for r in provider.responses], default=str))
        try:
            await loop.aclose(drain_timeout_s=.1)
            await provider.close()
        finally:
            await store.close()
