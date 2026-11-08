"""Real response captured through the SDK port survives loss of its original host."""

import pytest

from power_loop import SessionStore
from power_loop.runtime import model_budgets
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from tests.real._budget_evidence_host import RecoveryHost, config

pytestmark = pytest.mark.asyncio


@pytest.mark.real_llm
async def test_real_capture_failure_preserves_actual_observation_without_retry(tmp_path, record_property, runtime_agent):
    from power_loop import AgentLoopConfig
    from power_loop.runtime.retry import LLMRetryPolicy
    from tests.real._bounded import BoundedProvider
    service, host = BoundedProvider(limit=1), RecoveryHost()
    host.service = service
    captures = []
    async def unavailable(packet):
        captures.append(packet)
        raise OSError('independent capture endpoint unavailable')
    host.capture_response = unavailable
    store = await SessionStore.open(str(tmp_path / 'failed-capture.db'), schema='tasks')
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id='test', principal_id='host', executor_id='original',
        boot_id='original', boot_contract_hash='a' * 64, budget_config=config(host))
    loop = await runtime_agent(model=service, store=store, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy='single_level', max_rounds=1, max_tokens=2048,
                              retry_policy=LLMRetryPolicy(max_attempts=3)))
    try:
        sid = loop.target.session_id
        await loop.send('Reply only with the number 42.')
        await runtime.flush_control_checkpoints()
        [capture] = captures
        original = await model_budgets.get_budget(manager, runtime.actor(sid), capture.grant.attempt_id)
        assert original['observation']['status'] == 'returned' and not original['pending']
        assert original['observation']['usage'] == capture.usage
        assert service.calls == host.reserve_calls == host.apply_calls == 1
        assert not runtime.budget_control_tasks()
    finally:
        service.report(record_property)
        await loop.aclose()
        await service.close()
        await store.close()
