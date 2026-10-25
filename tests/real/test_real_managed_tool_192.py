"""Real model dispatch with a generic managed implementation and budget gates.

The finite effect and budget authority are test ports. Provider responses are
real; this does not claim a platform API transaction or supplier settlement.
"""

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.managed_effects import (
    ManagedTool,
    ManagedToolContract,
    ManagedToolPreparation,
    ManagedToolResult,
    read_raw_receipt,
    reconcile,
    terminate_unstarted,
)
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_budget_tree_192 import RealBudgetHost

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("pending", [False, True, "unclaimed", "receipt_lost", "completion_exit", "completion_result", "capacity"])
async def test_real_generic_base_admission_and_query_only_resolution(tmp_path, pending, record_property):
    unclaimed = pending in ("unclaimed", "receipt_lost")
    completion_failure = pending in ("completion_exit", "completion_result")
    capacity = pending == "capacity"
    store = await SessionStore.open(str(tmp_path / "managed-tool.db"), schema="tasks")
    provider, host = BoundedProvider(limit=2), RealBudgetHost()
    host.service = provider
    manager = JobManager(store)
    runtime = TaskRuntime(
        manager,
        tenant_id="generic-real",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        boot_contract_hash="d" * 64,
        budget_config=BudgetConfig("generic-real-v1", (LocalBudgetScope("authority", max_calls=2),), host),
    )
    if unclaimed:
        claim = runtime.claim

        async def fail_before_claim(actor, registration, **kwargs):
            if registration.job.kind == "background_tool":
                raise JobError("test_claim_not_started")
            return await claim(actor, registration, **kwargs)

        runtime.claim = fail_before_claim
    if pending == "receipt_lost":

        async def unavailable_receipt(*args, **kwargs):
            raise OSError("test_raw_binding_unavailable")

        manager.reconcile_receipt = unavailable_receipt

    if completion_failure:
        from tests.unit._managed_completion_faults_192 import fail_completion_once

        fail_completion_once(manager, 'exit' if pending == 'completion_exit' else 'result')

    class SetBeacon(ManagedTool):
        prepared = 0
        executed = 0
        queries = 0
        resolved = pending is not True

        @property
        def contract(self):
            return ManagedToolContract("example.beacon", "a" * 64)

        async def prepare(self, context, arguments):
            self.prepared += 1
            assert arguments == {"value": 42}
            return ManagedToolPreparation({"original_call_id": context.tool_call_id, "value": arguments["value"]})

        async def execute(self, context, original_reference):
            self.executed += 1
            assert original_reference["value"] == 42
            if pending is True:
                raise TimeoutError("Test effect committed; its immediate receipt was lost")
            return ManagedToolResult("Beacon set to 42", {"original_receipt": "beacon-42"})

        async def recover(self, context, original_reference):
            self.queries += 1
            assert original_reference["value"] == 42
            if not self.resolved:
                return None
            return ManagedToolResult("Beacon set to 42", {"original_receipt": "beacon-42"})

    implementation = SetBeacon()
    definition = ToolDefinition(
        name="set_beacon",
        description="Set the beacon to the integer value exactly once.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer", "enum": [42]}},
            "required": ["value"],
        },
    )
    registry = ToolRegistry()
    registry.register(definition, implementation.bind(runtime, definition))
    loop = StatefulAgentLoop(
        llm=provider,
        store=store,
        task_runtime=runtime,
        tool_registry=registry,
        config=AgentLoopConfig(
            max_rounds=3,
            max_tokens=2048,
            **({"tool_result_max_chars": 16} if capacity else {}),
            delegation_policy="single_level",
            retry_policy=LLMRetryPolicy(max_attempts=1),
            system_prompt="Follow the user's exact tool sequence. Never call a tool more than once.",
        ),
    )

    async def only_requested_tool(ctx):
        # Select the actual available schema without fabricating provider output.
        ctx.tools = [definition.to_openai_tool()] if provider.calls == 0 else []

    loop.hooks.register(HookPoint.LLM_BEFORE, only_requested_tool, name="test.generic-managed-tool")
    sid = await loop.new_session()
    reader = None
    try:
        async with asyncio.timeout(150):
            try:
                result = await loop.send(
                    ("Call set_beacon exactly once with value=42. It is expected to return a test error. "
                     "Read that error and acknowledge reviewing it by replying exactly REVIEWED. REVIEWED does not mean success. "
                     "Do not retry or call any other tool." if capacity or pending == "unclaimed" else
                     "Call set_beacon exactly once with value=42. After its result, reply only DONE. Do not call any other tool."),
                    sid,
                )
            except OSError as exc:
                if pending != "receipt_lost" or str(exc) != "test_raw_binding_unavailable":
                    raise
                result = None
            if unclaimed:
                assert implementation.prepared == implementation.executed == implementation.queries == 0
                [child] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='background_tool'")
                assert child["generation"] == 0
                if pending == "receipt_lost":
                    assert child["result_json"] is None
                else:
                    assert child["result_status"] == "suppressed" and child["resource_state"] == "stopped"
                if pending == "receipt_lost":
                    before = await manager.db.fetchall(f"SELECT * FROM {store.t.messages} ORDER BY seq")
                    reader = await asyncio.create_subprocess_exec(
                        sys.executable,
                        str(Path(__file__).with_name("_managed_receipt_reader_192.py")),
                        str(tmp_path / "managed-tool.db"),
                        json.dumps(asdict(runtime.actor(sid))),
                        child["job_id"],
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    record_property("original_managed_receipt_reader_pid", reader.pid)
                    stdout, stderr = await asyncio.wait_for(reader.communicate(), 30)
                    assert reader.returncode == 0, stderr.decode()[-2000:]
                    value = json.loads(stdout)
                    assert value["network_disabled"]
                    observed, negative = value["observed"], value["negative"]
                    assert await read_raw_receipt(manager, runtime.actor(sid), child["job_id"]) == observed
                    assert await manager.db.fetchall(f"SELECT * FROM {store.t.messages} ORDER BY seq") == before
                else:
                    observed = await read_raw_receipt(manager, runtime.actor(sid), child["job_id"])
                    assert result.status == "completed" and result.final_text.strip() == "REVIEWED"
                    negative = await terminate_unstarted(manager, runtime.actor(sid), child["job_id"])
                assert not observed["acknowledges_result"]
                assert negative["resolution"] == "not_dispatched" and not negative["tool_receipt_written"]
                assert (await manager.get_job(runtime.actor(sid), child["job_id"])).supervision_status == "closed"
                assert provider.calls == host.reserve_calls == len(host.applied) == (1 if pending == "receipt_lost" else 2)
                return
            if capacity:
                assert result.status == "completed" and result.final_text.strip() == "REVIEWED"
                assert implementation.prepared == implementation.executed == implementation.queries == 0
                [child] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='background_tool'")
                assert child["generation"] == 0 and child["resource_state"] == "stopped"
                assert child["result_status"] == "suppressed" and child["supervision_status"] == "closed"
                [raw] = [row for row in await store.load_all_messages(sid) if row.role == "tool"]
                assert raw.content == "Error: managed_effect_result_capacity_unavailable"
                assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type IN ('execution_scope','managed_effect_prepared','managed_effect_dispatch','managed_effect_resource_exit','managed_effect_tool_receipt')", (child["job_id"],))
                assert provider.calls == host.reserve_calls == len(host.applied) == 2
                return
            if completion_failure:
                from power_loop.runtime.managed_completion import flush_local, restore_result

                assert implementation.prepared == implementation.executed == 1 and implementation.queries == 0
                [child] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='background_tool'")
                assert child['result_json'] is None and provider.calls == host.reserve_calls == len(host.applied) == 1
                shutdown = await runtime.shutdown(timeout_s=1, grace_s=0.01)
                assert shutdown['managed_completion_pending'] == 1 and not shutdown['store_safe_to_close']
                status = await runtime.executor_status()
                assert status['managed_completion_pending'] == 1 and not status['store_safe_to_close']
                before = await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq')
                if pending == 'completion_exit':
                    with pytest.raises(JobError, match='evidence_missing'):
                        await restore_result(manager, runtime.actor(sid), child['job_id'])
                    await flush_local(runtime, runtime.actor(sid), child['job_id'])
                reader = await asyncio.create_subprocess_exec(
                    sys.executable, str(Path(__file__).with_name('_managed_completion_reader_192.py')),
                    str(tmp_path / 'managed-tool.db'), json.dumps(asdict(runtime.actor(sid))), child['job_id'],
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                record_property('original_completion_reader_pid', reader.pid)
                stdout, stderr = await asyncio.wait_for(reader.communicate(), 30)
                assert reader.returncode == 0, stderr.decode()[-2000:]
                observed = json.loads(stdout)
                assert observed['network_disabled'] and not observed['raw']['acknowledges_result']
                assert not observed['repaired']['acknowledgement_created']
                if pending == 'completion_result':
                    assert await flush_local(runtime, runtime.actor(sid), child['job_id']) == observed['repaired']
                assert runtime._managed_completions.pending_count() == 0
                assert await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq') == before
                current = await manager.get_job(runtime.actor(sid), child['job_id'])
                assert current.result_status == 'pending_delivery' and current.supervision_status == 'open'
                assert provider.calls == host.reserve_calls == len(host.applied) == implementation.executed == 1
                assert implementation.queries == 0
                return
            assert implementation.prepared == implementation.executed == 1
            [child] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='background_tool'")
            original = json.loads(child["result_json"])
            assert original["usage"] == {}
            if pending:
                assert original["managed_effect_pending"] is True
                assert child["supervision_status"] == "open"
                assert provider.calls == host.reserve_calls == len(host.applied) == 1
                assert await reconcile(runtime, runtime.actor(sid), child["job_id"], implementation) is None
                implementation.resolved = True
                recovered = await reconcile(runtime, runtime.actor(sid), child["job_id"], implementation)
                assert recovered.reference == {"original_receipt": "beacon-42"}
                await manager.finalize_supervision(runtime.actor(sid), child["job_id"], generation=1)
                assert implementation.queries == 2
                assert provider.calls == 1  # Recovery added no model or reservation.
            else:
                assert result.status == "completed" and result.final_text.strip() == "DONE"
                assert child["supervision_status"] == "closed"
                assert provider.calls == host.reserve_calls == len(host.applied) == 2
                assert implementation.queries == 0
            assert implementation.executed == 1
            assert (await manager.get_job(runtime.actor(sid), child["job_id"])).supervision_status == "closed"
    finally:
        if reader is not None and reader.returncode is None:
            reader.kill()
            await reader.wait()
        provider.report(record_property)
        record_property("effect_executions", implementation.executed)
        record_property("effect_queries", implementation.queries)
        try:
            await loop.aclose(drain_timeout_s=0.1)
        finally:
            await provider.close()
            await store.close()


async def test_real_worker_crash_after_exit_checkpoint_recovers_without_model_or_effect_replay(tmp_path, record_property):
    from power_loop.runtime.jobs import JobActor

    database, observation_file = tmp_path / 'crash.db', tmp_path / 'original-observations.json'
    children = []
    store = None
    try:
        worker = await asyncio.create_subprocess_exec(sys.executable,
            str(Path(__file__).with_name('_managed_completion_crash_worker_192.py')),
            str(database), str(observation_file), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        children.append(worker)
        record_property('original_completion_crash_worker_pid', worker.pid)
        stdout, stderr = await asyncio.wait_for(worker.communicate(), 150)
        assert worker.returncode == 75, stderr.decode()[-2000:]
        observations = json.loads(observation_file.read_text())
        for key in ('real_provider_requests', 'reported_tokens', 'local_provider_calls_refused'):
            record_property(key, observations[key])
        assert observations['real_provider_requests'] == observations['effect_executions'] == 1
        assert observations['budget_reservations'] == observations['budget_usage_observations'] == 1
        actor, job = JobActor(**observations['actor']), observations['job_id']
        store = await SessionStore.open(str(database), schema='compatible')
        manager = JobManager(store)
        before = await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq')
        assert not any(row['role'] == 'tool' for row in before)
        assert (await manager.get_job(actor, job)).result is None
        reader = await asyncio.create_subprocess_exec(sys.executable,
            str(Path(__file__).with_name('_managed_completion_reader_192.py')), str(database),
            json.dumps(asdict(actor)), job, 'allow_missing_receipt',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        children.append(reader)
        record_property('original_crash_completion_reader_pid', reader.pid)
        stdout, stderr = await asyncio.wait_for(reader.communicate(), 30)
        assert reader.returncode == 0, stderr.decode()[-2000:]
        repaired = json.loads(stdout)
        assert repaired['network_disabled'] and repaired['raw'] is None
        assert repaired['repaired']['owned_callback_exited'] and not repaired['repaired']['acknowledgement_created']
        assert not repaired['repaired']['execution_authority'] and not repaired['repaired']['billing_settled']
        assert await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq') == before
        result = await manager.get_job(actor, job)
        assert result.result['display'] == 'Original beacon set' and result.result_status == 'pending_delivery'
        assert result.supervision_status == 'open'  # no original tool ACK was fabricated
    finally:
        for child in children:
            if child.returncode is None:
                child.kill()
                await child.wait()
        if store is not None:
            await store.close()
