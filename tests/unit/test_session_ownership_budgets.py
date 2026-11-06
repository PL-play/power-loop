"""A remote child keeps its original cumulative budget when its parent changes."""
import json
from types import SimpleNamespace

import pytest

from power_loop.runtime import model_attempts as attempts
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.control_runtime import current, flush_binding
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import Execution, TaskRuntime

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host, observed, start

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_parent_generation_boot_and_unknown_do_not_revoke_child_refill(registry_store):
    store, _ = registry_store
    host, manager = Host(), JobManager(store)
    config = BudgetConfig('tree', (LocalBudgetScope('root', max_calls=3, binding='root'),
        LocalBudgetScope('owner', max_calls=3, binding='owner', owner_kinds=('subagent',))), host, max_pending=8)
    runtime = TaskRuntime(manager, tenant_id='tenant', principal_id='host', executor_id='remote', boot_id='remote-boot',
                          boot_contract_hash='a'*64, budget_config=config)
    sid = await store.create_session()
    child_sid = await store.create_session(parent_session_id=sid)
    parent_actor, actor = runtime.actor(sid), runtime.actor(child_sid)
    await runtime.start()
    try:
        root = await manager.register_job(parent_actor, kind='send', session_id=sid, system_key='root', request={'model_attempt_protocol': 1})
        parent = await manager.claim_job(parent_actor, root.job.job_id, executor_id='remote', boot_id='remote-boot', expected_version=1)
        await attempts.enable(Execution(runtime, parent_actor, parent))
        registered = await manager.register_job(parent_actor, kind='subagent', parent=parent, session_id=child_sid, system_key='child')
        child = await runtime.claim(actor, registered)
        model = await manager.register_job(actor, kind='send', parent=child.lease, session_id=child_sid,
                                           system_key='model', request={'model_attempt_protocol': 1})
        binding = await runtime.claim(actor, model)
        await attempts.enable(binding)
        case = SimpleNamespace(store=store, manager=manager, host=host, runtime=runtime,
                               actor=actor, sid=child_sid, binding=binding)
        await observed(case, await start(case))
        allocation = binding.resource_evidence['_control_allocation']
        assert allocation.calls == allocation.used_calls == 1
        await manager.db.execute(f"UPDATE {manager.jobs} SET generation=generation+1,executor_id='replacement',"
                                 "boot_id='replacement',status='interrupted',resource_state='unknown',version=version+1 WHERE job_id=?",
                                 (parent.job_id,))
        await flush_binding(binding)
        current(binding).require_admission()
        await observed(case, await start(case))  # Must actually refill the original allocation.
        assert allocation.calls == allocation.used_calls == 2
        await flush_binding(binding, close=True)
        scopes = await manager.db.fetchall(f'SELECT limits_json,counters_json FROM {manager.prefix}task_budget_scopes')
        assert len(scopes) == 2
        for scope in scopes:
            limits, counters = json.loads(scope['limits_json']), json.loads(scope['counters_json'])
            assert limits['subject_job_id'] in (parent.job_id, child.lease.job_id)
            assert counters['used_calls'] == 2 and counters['held_calls'] == 0
        assert host.reserve_calls == 2
        await manager.request_cancel(parent_actor, parent.job_id, system_key='stop-tree', reason='stop')
        with pytest.raises(JobError, match='cancel_requested'):
            await manager.check_execution(actor, child.lease)
    finally:
        await runtime.shutdown(timeout_s=1)
