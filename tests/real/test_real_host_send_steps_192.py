"""Distinct planned stages call a real model once each; repeated stages cannot."""
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop
from power_loop.runtime.host_send_steps import lookup
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.send_receipt import read
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_planned_stages_keep_original_jobs_and_results_after_fresh_process(tmp_path, record_property):
    path = tmp_path / "host-steps.db"
    store = await SessionStore.open(str(path), schema="tasks")
    provider = BoundedProvider(limit=2)
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="real-host-step", principal_id="host", executor_id="executor",
        boot_id="boot", require_host_send_steps=True)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=ToolRegistry(),
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2, max_tokens=2048,
            retry_policy=LLMRetryPolicy(max_attempts=1), system_prompt="Return only the requested integer. Never call tools."))
    async def arithmetic(ctx):
        ctx.tools = []
    loop.hooks.register(HookPoint.LLM_BEFORE, arithmetic, name="test.host-step-arithmetic")
    sid = await loop.new_session()
    actor = runtime.actor(sid)
    source = {"namespace": "test.host-original", "payload": {"run_id": 101}}
    step = {"namespace": "test.host-processing", "plan_hash": "a" * 64, "step_id": "initial:0"}
    child = None
    try:
        with runtime.root_context(sid, **source):
            with runtime.host_send_step(sid, **step):
                first = await loop.send("Compute 19+23. Reply only 42.", sid)
        assert first.status == "completed" and "42" in first.final_text and provider.calls == 1
        original = await lookup(manager, actor, **step, expected_host_context=source)
        original_receipt = await read(manager, actor, original["job_id"], expected_host_context=source)
        with runtime.root_context(sid, **source):
            with runtime.host_send_step(sid, **step):
                with pytest.raises(JobError, match="already_registered"):
                    await loop.send("This is a duplicate stage; never call a model for it.", sid)
        assert provider.calls == 1
        with runtime.root_context(sid, **source):
            with runtime.host_send_step(sid, **{**step, "step_id": "follow-up:0"}):
                second = await loop.send("Compute 20+23. Reply only 43.", sid)
        assert second.status == "completed" and "43" in second.final_text and provider.calls == 2
        assert len(await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs}")) == 2
        child = await asyncio.create_subprocess_exec(sys.executable,
            str(Path(__file__).with_name("_host_send_step_reader_192.py")), str(path),
            json.dumps(asdict(actor)), json.dumps(step), json.dumps(source),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        record_property("original_host_step_reader_pid", child.pid)
        stdout, stderr = await asyncio.wait_for(child.communicate(), 30)
        assert child.returncode == 0, stderr.decode()[-2000:]
        observed = json.loads(stdout)
        assert observed["binding"] == original and observed["receipt"] == original_receipt
        assert observed["real_provider_requests"] == 0 and provider.calls == 2
        assert original_receipt["result"]["usage"]["total_tokens"] > 0
        assert not original["execution_authority"] and not original["billing_settled"]
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        provider.report(record_property)
        try:
            await loop.aclose(drain_timeout_s=.1)
        finally:
            await provider.close()
            await store.close()
