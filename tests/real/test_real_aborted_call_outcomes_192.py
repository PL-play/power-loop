"""Actual crashed source -> negative outcome -> original-main review and budget."""
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.execution_membership import scope
from power_loop.runtime.jobs import JobActor, JobError, JobLease, JobManager
from power_loop.runtime.managed_effects import FunctionManagedTool, ManagedToolContract
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_query import get_job
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.runtime.tool_call_plans import reconcile_pending
from power_loop.tools.jobs import JOBS_DEFINITION, JOBS_HANDLER
from power_loop.tools.registry import ToolRegistry

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_budget_tree_192 import RealBudgetHost

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('allowed_calls', [2, 1])
async def test_real_crashed_aborted_source_is_reviewed_without_replacement_or_new_quota(tmp_path, record_property, allowed_calls):
    database, target = tmp_path/'aborted.db', tmp_path/'observations.json'
    child = store = provider = loop = None
    observed, denied = {}, []
    try:
        child = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).with_name('_tool_abort_crash_worker_192.py')),
            str(database), str(target), str(allowed_calls), 'after_abort', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        record_property('original_aborted_outcome_worker_pid', child.pid)
        stdout, stderr = await asyncio.wait_for(child.communicate(), 150)
        assert child.returncode == 75, stderr.decode()[-2000:]
        observed = json.loads(target.read_text())
        assert not observed['effects'] and not observed['preparations'] and observed['real_provider_requests'] == 1
        store = await SessionStore.open(str(database), schema='compatible')
        manager, actor = JobManager(store), JobActor(**observed['actor'])
        source_id = observed['source_job_id']
        async with manager._write() as tx:
            original = await scope(manager, tx, source_id, 1)
        await manager.close_model_execution(actor, JobLease(**original['lease']),
            evidence={'exit': 'interrupted', 'owned_process_exit': {'pid': child.pid, 'returncode': child.returncode}})
        raw = await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq')
        provider, host = BoundedProvider(limit=1), RealBudgetHost()
        host.service = provider
        runtime = TaskRuntime(manager, tenant_id=actor.tenant_id, principal_id=actor.principal_id,
            executor_id='review', boot_id='review-boot', boot_contract_hash='b'*64,
            budget_config=BudgetConfig('batch-crash-v1', (LocalBudgetScope('authority', max_calls=allowed_calls),), host))
        async def forbidden(*args):
            denied.append(True)
            raise AssertionError('An original outcome review cannot execute a replacement effect')
        adapter = FunctionManagedTool(ManagedToolContract('example.batch-crash', 'a'*64), forbidden, forbidden, forbidden)
        registry = ToolRegistry()
        definition = ToolDefinition('set_first_beacon', 'Set the beacon.', input_schema={
            'type': 'object', 'properties': {'value': {'type': 'integer', 'enum': [42]}}, 'required': ['value']})
        registry.register(definition, adapter.bind(runtime, definition))
        registry.register(replace(JOBS_DEFINITION, name='inspect_saved_tasks'), JOBS_HANDLER)
        loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
            config=AgentLoopConfig(max_tokens=2048, max_rounds=2, delegation_policy='single_level',
                representation=ProjectedRepresentation(), retry_policy=LLMRetryPolicy(max_attempts=1)))
        complete = provider.complete
        async def honest_review(request, **kwargs):
            assert [t['function']['name'] for t in request.tools] == ['inspect_saved_tasks']
            assert 'ORIGINAL OUTCOME REVIEW' in request.system_prompt
            assert not any(m.get('role') == 'tool' or m.get('tool_calls') for m in request.messages)
            text = '\n'.join(str(m.get('content', '')) for m in request.messages)
            assert source_id in text and 'Original tool outcome:' in text
            return await complete(request, **kwargs)
        provider.complete = honest_review
        staged = await reconcile_pending(loop, actor, source_id)
        assert staged['mode'] == 'retired_for_abort_review' and not staged['acknowledgement_created']
        assert (await store.get_state(actor.session_id)).pending is None
        assert (await manager.get_job(actor, source_id)).result_status == 'pending_delivery'
        with pytest.raises(JobError):
            await manager.finalize_supervision(actor, source_id, generation=1)
        result = await loop.send('Read the supplied original aborted batch outcome. Do not call tools or perform actions. '
                                 'Acknowledge reviewing it by replying exactly ACKNOWLEDGED.', actor.session_id)
        source = await manager.get_job(actor, source_id)
        if allowed_calls == 2:
            assert result.status == 'completed' and result.final_text.strip() == 'ACKNOWLEDGED'
            assert provider.calls == host.reserve_calls == len(host.applied) == 1
            assert source.result_status == 'handled' and source.supervision_status == 'closed'
        else:
            assert result.status == 'degraded' and 'budget_max_calls_exhausted' in result.final_text
            assert provider.calls == host.reserve_calls == len(host.applied) == 0
            assert source.result_status == 'pending_delivery' and source.supervision_status == 'open'
        view = await get_job(manager, actor, source_id)
        assert view['node']['processing_obligation']['state'] == ('handled' if allowed_calls == 2 else 'pending')
        assert not denied and not await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind<>'send'")
        for row in raw:
            assert await manager.db.fetchone(f'SELECT * FROM {store.t.messages} WHERE id=?', (row['id'],)) == row
        assert not await manager.db.fetchall(f"SELECT id FROM {store.t.messages} WHERE role='tool'")
    finally:
        usage = {}
        if provider is not None:
            provider.report(lambda key, value: usage.__setitem__(key, value))
        for key in ['real_provider_requests', 'reported_tokens', 'local_provider_calls_refused']:
            record_property(key, int(observed.get(key, 0))+int(usage.get(key, 0)))
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        if loop is not None:
            await loop.aclose(drain_timeout_s=.1)
        if provider is not None:
            await provider.close()
        if store is not None:
            await store.close()
