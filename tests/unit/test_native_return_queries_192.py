"""Execute recovery discovery and verify actual SQL has indexed access paths."""
import inspect
import json
import re

import pytest

from power_loop.runtime import controller_handoff, controller_recovery, managed_delivery
from power_loop.runtime.pending_calls import recover_runtime, recover_tree
from tests.unit._owned_pending_case_192 import owned_pending_case
from tests.unit.test_controller_handoff_192 import new_main
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_owned_pending_review_192 import cleanup

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
NESTED = re.compile(r'\(\s*SELECT\b', re.I)


async def test_native_return_recovery_queries_are_flat_and_indexed(registry_store, monkeypatch, record_property):
    c = await owned_pending_case(registry_store, 'workflow_node', effect_calls=2, native_tail='original_note', native_first=True)
    loop = runtime = None
    plans = {}
    try:
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        backend = c.manager.db.dialect.name
        if backend == 'sqlite':
            from power_loop.runtime.store.backends.sqlite import SqliteDatabase, _SqliteTransaction
            classes = [SqliteDatabase, _SqliteTransaction]
        else:
            from power_loop.runtime.store.backends.postgres import PostgresDatabase, _PgTransaction
            classes = [PostgresDatabase, _PgTransaction]
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
        assert c.state.native_calls == 1 and c.state.lookups == c.state.counts['AFTER'] == c.state.executed == 0
    finally:
        await cleanup(c, loop, runtime)
