"""Bad schema ownership cannot silently reopen actual model/accounting history."""
import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.model_budgets import get_budget
from power_loop.runtime.store.schema import StoreSchemaError
from power_loop.runtime.task_runtime import TaskRuntime
from tests.real._initial_evidence_worker_192 import RecoveryHost, config
from tests.real.test_real_async_boundaries_192 import BoundedProvider
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_task_query_indexes_192 import drop_index_sql

__all__ = ['registry_store']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_original_real_history_reopens_only_with_index_on_correct_table(registry_store, record_property):
    store, connect = registry_store
    manager, service, host = JobManager(store), BoundedProvider(limit=1), RecoveryHost()
    host.service = service
    runtime = TaskRuntime(manager, tenant_id='test', principal_id='host', executor_id='original',
        boot_id='original', boot_contract_hash='a' * 64, budget_config=config(host))
    loop = StatefulAgentLoop(llm=service, store=store, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy='single_level', max_rounds=1, max_tokens=2048))
    try:
        sid = await loop.new_session()
        await loop.send('Reply only with the number 42.', sid)
        await runtime.shutdown(timeout_s=3)
        [attempt] = await manager.db.fetchall(f'SELECT attempt_id FROM {manager.prefix}task_budget_attempts')
        budget = await get_budget(manager, runtime.actor(sid), attempt['attempt_id'])
        assert not budget['pending'] and budget['observation']['status'] == 'returned'
        messages = await store.load_all_messages(sid)
        assert any(message.role == 'assistant' and '42' in str(message.content) for message in messages)
        identity = await manager.db.fetchone(f'SELECT * FROM {manager.prefix}task_store WHERE id=1')
        prefix, db = manager.prefix, manager.db
        key, target, other = prefix+'idx_events_caller_type', prefix+'job_events', prefix+'other_events'
        columns = 'caller_session_id,event_type,job_id,event_id'
        await db.execute(drop_index_sql(db.dialect.name, key, target))
        try:
            await db.execute(f'CREATE TABLE {other} ('+','.join(c+' VARCHAR(64)' for c in columns.split(','))+')')
            await db.execute(f'CREATE INDEX {key} ON {other}({columns})')
            with pytest.raises(StoreSchemaError, match='discovery indexes'):
                await connect()
        finally:
            await db.execute(f'DROP TABLE IF EXISTS {other}')
            await db.execute(f'CREATE INDEX {key} ON {target}({columns})')
        recovered = await connect()
        assert await recovered.load_all_messages(sid) == messages
        restored = JobManager(recovered)
        assert await get_budget(restored, runtime.actor(sid), attempt['attempt_id']) == budget
        assert await restored.db.fetchone(f'SELECT * FROM {prefix}task_store WHERE id=1') == identity
        assert service.calls == host.reserve_calls == host.apply_calls == 1
        record_property('verified_backend', db.dialect.name)
    finally:
        service.report(record_property)
        await runtime.shutdown(timeout_s=3)
        await service.close()
