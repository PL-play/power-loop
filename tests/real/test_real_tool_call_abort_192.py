"""A real original response fails before admission; a new process only reads it."""

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.managed_effects import FunctionManagedTool, ManagedToolContract
from power_loop.runtime.pricing import Free
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.unit._tool_plan_faults_192 import fail_plan_once

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_budget_tree_192 import RealBudgetHost

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('stage', ['seal', 'commit_ack'])
async def test_real_response_has_positive_abort_without_effect_or_model_replay(tmp_path, record_property, stage):
    database = tmp_path/'abort.db'
    store = await SessionStore.open(str(database), schema='tasks')
    manager, provider, host = JobManager(store), BoundedProvider(limit=1), RealBudgetHost()
    host.service = provider
    runtime = TaskRuntime(manager, tenant_id='abort-real', principal_id='host', executor_id='original',
        boot_id='original-boot', boot_contract_hash='a'*64,
        budget_config=BudgetConfig('abort-v1', (LocalBudgetScope('authority', max_calls=1),), host))
    invoked = []
    async def forbidden(*args):
        invoked.append('forbidden')
        raise AssertionError('Pre-admission abort must never invoke an effect callback')
    implementation = FunctionManagedTool(ManagedToolContract('example.abort', 'b'*64), forbidden, forbidden, forbidden)
    definition = ToolDefinition('set_beacon', 'Set a beacon once with value42.', input_schema={
        'type': 'object', 'properties': {'value': {'type': 'integer', 'enum': [42]}}, 'required': ['value']}, pricing=Free("test_fixture"))
    registry = ToolRegistry()
    registry.register(definition, implementation.bind(runtime, definition))
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(max_tokens=2048, max_rounds=2, delegation_policy='single_level', retry_policy=LLMRetryPolicy(max_attempts=1)))
    async def schema(ctx):
        ctx.tools = [definition.to_openai_tool()]
    loop.hooks.register(HookPoint.LLM_BEFORE, schema, name='test.abort.real-schema')
    fault = fail_plan_once(manager, stage)
    child = None
    try:
        sid = await loop.new_session()
        with pytest.raises(OSError, match='^test_plan_'+stage+'$'):
            await asyncio.wait_for(loop.send('Call set_beacon exactly once with value=42. Do not call other tools.', sid), 90)
        assert fault.fired and not invoked
        assert provider.calls == host.reserve_calls == len(host.applied) == 1
        pending = (await store.get_state(sid)).pending
        [source] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='send'")
        assert source['resource_state'] == 'stopped' and source['status'] == 'model_closed'
        before = await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq')
        assert not any(row['role'] == 'tool' for row in before)
        child = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).with_name('_tool_abort_reader_192.py')),
            str(database), json.dumps(asdict(runtime.actor(sid))), source['job_id'], str(pending['assistant_seq']),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        record_property('original_abort_reader_pid', child.pid)
        stdout, stderr = await asyncio.wait_for(child.communicate(), 30)
        assert child.returncode == 0, stderr.decode()[-2000:]
        observed = json.loads(stdout)
        assert observed['network_disabled'] and observed['evidence_unchanged']
        proof = observed['checkpoint']
        assert proof['decision'] == 'admission_not_granted' and len(proof['entries']) == 1
        assert proof['entries'][0]['tool_name'] == 'set_beacon'
        assert not proof['execution_authority'] and not proof['acknowledgement_created'] and not proof['billing_settled']
        assert (await store.get_state(sid)).pending == pending
        assert await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq') == before
        assert not invoked and provider.calls == 1
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        provider.report(record_property)
        await loop.aclose(drain_timeout_s=.1)
        await provider.close()
        await store.close()


async def test_real_process_exit_after_abort_preserves_original_model_exit_and_pending(tmp_path, record_property):
    from power_loop.runtime.execution_membership import scope
    from power_loop.runtime.jobs import JobActor, JobError, JobLease
    from power_loop.runtime.tool_call_abort import read_abort
    database, target = tmp_path/'crash.db', tmp_path/'original-observations.json'
    child = store = None
    observations = {}
    try:
        child = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).with_name('_tool_abort_crash_worker_192.py')),
            str(database), str(target), '1', 'after_abort', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        record_property('original_abort_crash_worker_pid', child.pid)
        stdout, stderr = await asyncio.wait_for(child.communicate(), 150)
        assert child.returncode == 75, stderr.decode()[-2000:]
        observations = json.loads(target.read_text())
        assert not observations['effects'] and not observations['preparations'] and not observations['jobs']
        assert observations['real_provider_requests'] == observations['budget_reservations'] == observations['budget_usage_observations'] == 1
        store = await SessionStore.open(str(database), schema='compatible')
        manager, actor = JobManager(store), JobActor(**observations['actor'])
        source_id = observations['source_job_id']
        pending = (await store.get_state(actor.session_id)).pending
        raw = await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq')
        assert not any(row['role'] == 'tool' for row in raw)
        before = await read_abort(manager, actor, source_id, assistant_seq=pending['assistant_seq'])
        assert before == observations['checkpoint']
        async with manager._write() as tx:
            original = await scope(manager, tx, source_id, 1)
        assert original['lease']['executor_id'] == 'original' and original['lease']['boot_id'] == 'crashed-boot'
        await manager.close_model_execution(actor, JobLease(**original['lease']),
            evidence={'exit': 'interrupted', 'owned_process_exit': {'pid': child.pid, 'returncode': child.returncode}})
        source = await manager.get_job(actor, source_id)
        assert source.status == 'model_closed' and source.resource_state == 'stopped'
        closed = json.loads((await manager.db.fetchone(f'SELECT payload_json FROM {manager.events} WHERE event_id=?', ('model_closed_'+source_id,)))['payload_json'])
        assert closed['evidence']['pending_call_snapshot']['version'] == 3
        assert await read_abort(manager, actor, source_id, assistant_seq=pending['assistant_seq']) == before
        with pytest.raises(JobError):
            await manager.finalize_supervision(actor, source_id, generation=1)
        assert (await store.get_state(actor.session_id)).pending == pending
        assert await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq') == raw
        assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type IN ('processing_decision','pending_call_disposition')")
    finally:
        for key in ['real_provider_requests', 'reported_tokens', 'local_provider_calls_refused']:
            record_property(key, observations.get(key, 0))
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        if store is not None:
            await store.close()
