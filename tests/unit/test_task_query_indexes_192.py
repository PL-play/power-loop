"""Execute recovery discovery and verify actual SQL has indexed access paths."""
import inspect
import json
import re

import pytest

from power_loop.runtime import controller_handoff, controller_recovery, managed_delivery
from power_loop.runtime.pending_calls import recover_runtime, recover_tree
from power_loop.runtime.store.task_schema import task_tables_ddl

from ._owned_pending_case_192 import owned_pending_case
from .test_controller_handoff_192 import new_main
from .test_job_registry_192 import registry_store
from .test_owned_pending_review_192 import cleanup

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
NESTED = re.compile(r'\(\s*SELECT\b', re.I)


def drop_index_sql(backend, index, table):
    return f'DROP INDEX {index}' + (f' ON {table}' if backend == 'mysql' else '')


def mysql_index_paths(rows):
    """Check eligibility, not production optimizer choice on tiny fixtures.

    MySQL may choose ALL despite possible_keys on a tiny table. Such a plan
    proves an indexed candidate exists; it does not prove the engine used it.
    Const-folded empty results do not access a relation at all.
    """
    assert rows
    paths = []
    for row in rows:
        if row.get('table') is None:
            assert any(reason in (row.get('Extra') or '').lower() for reason in
                       ('impossible', 'no matching', 'no tables used')), row
        else:
            assert row.get('key') or row.get('possible_keys') or row.get('type') in ('const', 'system'), row
        paths.append({k: row.get(k) for k in ('table', 'type', 'key', 'possible_keys', 'Extra')})
    return paths


async def test_actual_recovery_queries_are_flat_and_every_task_relation_has_an_index_path(registry_store, monkeypatch, record_property):
    c = await owned_pending_case(registry_store, 'workflow_node', effect_calls=2, native_tail='original_note')
    loop = runtime = None
    plans = {}
    try:
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        backend = c.manager.db.dialect.name
        if backend == 'sqlite':
            from power_loop.runtime.store.backends.sqlite import SqliteDatabase, _SqliteTransaction
            classes = [SqliteDatabase, _SqliteTransaction]
        elif backend == 'mysql':
            from power_loop.runtime.store.backends.mysql import MySQLDatabase, _MyTransaction
            classes = [MySQLDatabase, _MyTransaction]
        elif backend == 'postgres':
            from power_loop.runtime.store.backends.postgres import PostgresDatabase, _PgTransaction
            classes = [PostgresDatabase, _PgTransaction]
        else:
            raise AssertionError('uncovered query-plan backend: '+backend)
        names = (c.manager.jobs, c.manager.events, c.manager.prefix+'task_budget_attempts')
        async def inspect_sql(self, sql, params):
            if sql in plans or not re.match(r'\s*(SELECT|UPDATE|DELETE)\b', sql, re.I) or not any(name in sql for name in names):
                return
            assert not NESTED.search(sql), sql
            db = getattr(self, '_db', self)
            if backend == 'sqlite':
                rows = await db._fetchall('EXPLAIN QUERY PLAN '+sql, params)
                detail = [row['detail'] for row in rows]
                # Sorting temporary results is permitted; reading an unindexed
                # task relation is not. Check every joined table, not one leaf.
                assert not any(re.match(r'SCAN\s+(?:j|e|d|t|'+re.escape(c.manager.prefix)+r'\w+)\b', text)
                    and 'USING' not in text for text in detail), (sql, detail)
                plans[sql] = detail
            elif backend == 'mysql':
                async def explain(conn):
                    import aiomysql
                    async with conn.cursor(aiomysql.DictCursor) as cursor:
                        await cursor.execute('EXPLAIN '+db.dialect.translate(sql), tuple(params))
                        return mysql_index_paths(await cursor.fetchall())
                if hasattr(self, '_conn'):
                    plans[sql] = await explain(self._conn)
                else:
                    async with db._pool.acquire() as conn:
                        plans[sql] = await explain(conn)
            else:
                async def explain(conn):
                    async with conn.transaction():
                        # Tiny fault fixtures can legitimately favor seq scans.
                        # Prove eligibility without changing the real query's
                        # planner settings or claiming a production cost plan.
                        previous = await conn.fetchval('SHOW enable_seqscan')
                        assert previous in ('on', 'off')
                        await conn.execute('SET LOCAL enable_seqscan=off')
                        try:
                            raw = await conn.fetchval('EXPLAIN (FORMAT JSON) '+db.dialect.translate(sql), *params)
                        finally:
                            await conn.execute('SET LOCAL enable_seqscan='+previous)
                        tree = json.loads(raw) if isinstance(raw, str) else raw
                        leaves = []
                        def visit(node):
                            if node.get('Relation Name') in names:
                                leaves.append(node)
                            for child in node.get('Plans', []):
                                visit(child)
                        visit(tree[0]['Plan'])
                        assert leaves and all(node['Node Type'] != 'Seq Scan' for node in leaves), (sql, tree)
                        return [node['Node Type'] for node in leaves]
                if hasattr(self, '_conn'):
                    plans[sql] = await explain(self._conn)
                else:
                    async with db._pool.acquire() as conn:
                        plans[sql] = await explain(conn)
        for cls in classes:
            for name in ['execute', 'fetchone', 'fetchall']:
                original = getattr(cls, name)
                def wrap(original):
                    async def instrumented(self, sql, params=()):
                        caller = inspect.currentframe().f_back.f_code.co_filename
                        if '/power_loop/runtime/' in caller:
                            await inspect_sql(self, sql, params)
                        return await original(self, sql, params)
                    return instrumented
                monkeypatch.setattr(cls, name, wrap(original))
        result = await recover_runtime(runtime, actor, c.state.source)
        assert result['state'] == 'queued_for_review'
        source = await runtime.manager.get_job(actor, c.state.source)
        await recover_tree(runtime, actor, source.root_job_id)
        async with runtime.manager._write() as tx:
            assert await managed_delivery.pending_reviews(runtime.manager, tx, actor)
            assert await controller_recovery.candidates(runtime.manager, tx, actor) == []
        await loop.send('Handle all original outcomes without repeating actions.', c.state.root, tools=[])
        await controller_recovery.reconcile(runtime, actor, c.state.owner_binding.lease.job_id)
        assert await controller_handoff.handoff_retired_reviews(runtime, actor, c.state.owner_binding.lease.job_id) == []
        root = await runtime.manager.get_job(actor, c.state.source)
        await controller_recovery.recover_tree(runtime, actor, root.root_job_id)
        await loop.send('Handle the original interrupted controller.', c.state.root, tools=[])
        assert len(plans) >= 15
        record_property('indexed_task_sql_shapes', len(plans))
        record_property('task_sql_plan_details', json.dumps(plans, sort_keys=True))
        assert c.state.native_calls == c.state.lookups == c.state.counts['AFTER'] == 0
    finally:
        await cleanup(c, loop, runtime)


async def test_task_index_ddl_names_fit_all_supported_prefixes(registry_store):
    store, _ = registry_store
    rows = task_tables_ddl(store._db.dialect.name, 'x'*24)
    indexes = [sql for sql in rows if sql.startswith('CREATE INDEX ')]
    assert all(len(sql.split()[2]) <= 64 for sql in indexes)
    for suffix in ['idx_jobs_root_page','idx_jobs_session_kind','idx_jobs_control_owner','idx_jobs_main_page',
                   'idx_events_job_type','idx_events_caller_type','idx_events_call','idx_budget_job','idx_context_capacity']:
        assert any(sql.startswith('CREATE INDEX '+'x'*24+suffix+' ') for sql in indexes)


@pytest.mark.parametrize('fault', ['missing', 'wrong_columns'])
async def test_missing_or_wrong_indexes_require_explicit_maintenance(registry_store, fault):
    from power_loop.runtime.store.schema import StoreSchemaError
    from power_loop.runtime.store.task_schema import migrate_task_schema

    store, connect = registry_store
    sid = await store.create_session()
    await store.append_message(sid, role='user', content='Immutable original index maintenance witness')
    original = await store._db.fetchall(f'SELECT * FROM {store.t.messages} WHERE session_id=?', (sid,))
    identity = await store._db.fetchone(f'SELECT * FROM {store.table_prefix}task_store WHERE id=1')
    key = store.table_prefix+'idx_events_caller_type'
    await store._db.execute(drop_index_sql(store._db.dialect.name, key, store.table_prefix+'job_events'))
    if fault == 'wrong_columns':
        await store._db.execute(f'CREATE INDEX {key} ON {store.table_prefix}job_events(event_type,caller_session_id)')
    with pytest.raises(StoreSchemaError, match='discovery indexes'):
        await connect()
    if fault == 'wrong_columns':
        with pytest.raises(StoreSchemaError, match='discovery indexes'):
            await migrate_task_schema(store._db, store.table_prefix, maintenance=True)
        await store._db.execute(drop_index_sql(store._db.dialect.name, key, store.table_prefix+'job_events'))
    await migrate_task_schema(store._db, store.table_prefix, maintenance=True)
    reopened = await connect()
    assert await reopened._db.fetchone(f'SELECT * FROM {store.table_prefix}task_store WHERE id=1') == identity
    assert await reopened._db.fetchall(f'SELECT * FROM {store.t.messages} WHERE session_id=?', (sid,)) == original


@pytest.mark.parametrize('backend', ['sqlite', 'postgres', 'mysql'])
async def test_index_maintenance_sql_is_explicit_for_every_supported_dialect(backend):
    statement = drop_index_sql(backend, 'pl_idx_events_caller_type', 'pl_job_events')
    expected = 'DROP INDEX pl_idx_events_caller_type'
    if backend == 'mysql':
        expected += ' ON pl_job_events'
    assert statement == expected
    ddl = task_tables_ddl(backend, 'pl_')
    assert any('idx_events_caller_type' in sql for sql in ddl)
    assert any('idx_budget_job' in sql for sql in ddl)


@pytest.mark.parametrize('plan', [
    [{'table': 'j', 'type': 'ref', 'key': 'idx_jobs_root_page'}],
    [{'table': 'j', 'type': 'ALL', 'possible_keys': 'idx_jobs_root_page'}],
    [{'table': None, 'Extra': 'no matching row in const table'}],
])
async def test_mysql_plan_checker_accepts_index_eligibility_and_no_access(plan):
    assert mysql_index_paths(plan)


@pytest.mark.parametrize('plan', [[], [{'table': 'j', 'type': 'ALL'}],
    [{'table': 'j', 'key': 'PRIMARY'}, {'table': 'e', 'type': 'ALL'}],
    [{'table': None, 'Extra': 'unknown plan shape'}]])
async def test_mysql_plan_checker_rejects_each_unindexed_relation(plan):
    with pytest.raises(AssertionError):
        mysql_index_paths(plan)


async def test_same_named_index_on_wrong_table_never_satisfies_recovery_schema(registry_store):
    from power_loop.runtime.store.schema import StoreSchemaError
    from power_loop.runtime.store.task_schema import verify_task_schema
    store, connect = registry_store
    db, prefix = store._db, store.table_prefix
    key, target, unrelated = prefix+'idx_events_caller_type', prefix+'job_events', prefix+'unrelated'
    columns = 'caller_session_id,event_type,job_id,event_id'
    await db.execute(drop_index_sql(db.dialect.name, key, target))
    try:
        await db.execute(f'CREATE TABLE {unrelated} ('+','.join(c+' VARCHAR(64)' for c in columns.split(','))+')')
        await db.execute(f'CREATE INDEX {key} ON {unrelated}({columns})')
        with pytest.raises(StoreSchemaError, match='discovery indexes'):
            await verify_task_schema(db, prefix)
        with pytest.raises(StoreSchemaError, match='discovery indexes'):
            await connect()
    finally:
        await db.execute(f'DROP TABLE IF EXISTS {unrelated}')
        await db.execute(f'CREATE INDEX {key} ON {target}({columns})')
    await verify_task_schema(db, prefix)
