"""A real process death must never erase an unbound tool's recovery marker."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("stage", ["before_bind", "after_bind", "clear_failed", "during_tool_insert"])
async def test_process_death_keeps_original_pending_and_does_not_repeat_effect(registry_store, tmp_path, stage):
    store, connect = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    sid = await store.create_session()
    evidence = tmp_path / "external-calls.json"
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "tests.unit._receipt_crash_worker", dsn, store.table_prefix, sid, str(evidence), stage,
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 30)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    assert process.returncode == {"before_bind": 73, "after_bind": 74, "clear_failed": 75,
                                 "during_tool_insert": 76}[stage], output.decode()
    assert json.loads(evidence.read_text()) == ["model", "effect"]

    # A fresh store/manager has none of the crashed process's memory hints.
    reader = await connect(schema="compatible")
    state = await reader.get_state(sid)
    assert state.pending is not None and state.pending["tool_call_ids"] == ["external-call"]
    rows = await reader.load_all_messages(sid)
    tools = [row for row in rows if row.role == "tool"]
    interrupted = stage == "during_tool_insert"
    assert len(tools) == int(not interrupted)
    [assistant] = [row for row in rows if row.role == "assistant"]
    assert state.pending["assistant_seq"] == assistant.seq
    operation_id = Path(str(evidence) + ".operation").read_text()
    if not interrupted:
        assert json.loads(tools[0].content)["operation_id"] == operation_id
    manager = JobManager(reader)
    actor = JobActor("crash-test", "host", sid)
    operation = await manager.get_operation(actor, operation_id)
    assert (operation.receipt_ref is None) == (stage in ("before_bind", "during_tool_insert"))
    provider_calls, effects = [], []

    class Model(LLMService):
        async def complete(self, request, **kwargs):
            provider_calls.append(request)
            return LLMResponse(raw_text="must not run")

        def stream(self, request):
            raise AssertionError("nonstream fixture")

        async def close(self):
            pass

    async def repeat():
        effects.append("repeated")
        return "must not repeat"

    registry = ToolRegistry()
    registry.register(ToolDefinition("external_effect", "Fixture external acceptance", input_schema={"type": "object"},
                                     pricing=Free("fixture")), repeat)
    runtime = TaskRuntime(manager, tenant_id=actor.tenant_id, principal_id=actor.principal_id,
                          executor_id="recovery", boot_id="after-crash", boot_contract_hash="c" * 64)
    loop = StatefulAgentLoop(llm=Model(), store=reader, task_runtime=runtime, tool_registry=registry,
                             config=AgentLoopConfig(compactor=None, delegation_policy="single_level"))
    try:
        with pytest.raises(JobError, match="adapter_unavailable:resume"):
            await loop.resume(sid)
        assert not provider_calls and not effects
        if not interrupted:
            # Cold reconciliation only attaches the already durable result, idempotently.
            ref = await reader.message_ref(sid, tools[0].seq)
            for _ in range(2):
                await manager.reconcile_receipt(actor, operation_id, ref)
            assert (await manager.get_operation(actor, operation_id)).receipt_ref == ref.to_dict()
        else:
            assert (await manager.get_operation(actor, operation_id)).receipt_ref is None
        assert (await reader.get_state(sid)).pending == state.pending
        assert len(await reader.load_all_messages(sid)) == len(rows)
        assert json.loads(evidence.read_text()) == ["model", "effect"]
    finally:
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)
