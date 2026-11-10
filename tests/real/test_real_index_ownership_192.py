"""Bad schema ownership cannot silently reopen actual model/accounting history."""
import pytest

from power_loop import AgentLoopConfig, RuntimeClient
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.store.schema import StoreSchemaError
from power_loop.runtime.task_runtime import TaskRuntime
from tests.real._bounded import BoundedProvider
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_task_query_indexes_192 import drop_index_sql

__all__ = ['registry_store']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_original_real_history_reopens_only_with_index_on_correct_table(registry_store, record_property, runtime_agent):
    store, connect = registry_store
    manager, service = JobManager(store), BoundedProvider(limit=1)
    runtime = TaskRuntime(manager, tenant_id='test', principal_id='host', executor_id='original',
        boot_id='original', boot_contract_hash='a' * 64)
    loop = await runtime_agent(model=service, store=store, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy='single_level', max_rounds=1, max_tokens=2048))
    try:
        sid = loop.target.session_id
        result = await loop.send('Reply only with the number 42.')
        assert result.status == 'completed' and result.usage['total_tokens'] > 0
        await loop.aclose()
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
        from power_loop.agent.runtime_store import now_ms
        from power_loop.agent.runtime_transport import MemoryMessageTransport
        client = RuntimeClient(store=recovered, transport=MemoryMessageTransport())
        try:
            assert await client.wait(result.input_ref, actor=loop.actor, deadline_ms=now_ms()+5000) == result
        finally:
            await client.aclose()
            await client.transport.aclose()
        assert await restored.db.fetchone(f'SELECT * FROM {prefix}task_store WHERE id=1') == identity
        assert service.calls == 1
        record_property('verified_backend', db.dialect.name)
    finally:
        service.report(record_property)
        await loop.aclose()
        await service.close()
