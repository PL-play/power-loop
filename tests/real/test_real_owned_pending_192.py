"""Actual original child effects and main review, with a separate recovery process."""
import asyncio
import json
import os
import sys
from dataclasses import asdict
from urllib.parse import urlparse

import pytest

from power_loop.runtime.controller_recovery import reconcile as recover_owner
from power_loop.runtime.managed_effects import reconcile
from tests.unit._owned_pending_case_192 import owned_pending_case
from tests.unit.test_controller_handoff_192 import new_main
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_owned_pending_review_192 import cleanup

from .test_real_async_boundaries_192 import BoundedProvider

__all__ = ['registry_store']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def recover_in_process(runtime, actor, source, record_property):
    manager = runtime.manager
    if manager.db.dialect.name == 'sqlite':
        dsn = next(row['file'] for row in await manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN']
        assert urlparse(dsn).path.endswith('_test')
    code = '''import asyncio,json,sys
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.jobs import JobActor,JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.runtime.pending_calls import recover_runtime
async def main():
 d=json.load(sys.stdin)
 store=await open_store(d['dsn'],table_prefix=d['prefix'],schema='tasks')
 actor=JobActor(**d['actor'])
 runtime=TaskRuntime(JobManager(store),tenant_id=actor.tenant_id,principal_id=actor.principal_id,
  executor_id='pending-reader',boot_id='separate-process')
 try: print(json.dumps(await recover_runtime(runtime,actor,d['source'])))
 finally: await store.close()
asyncio.run(main())'''
    process = await asyncio.create_subprocess_exec(sys.executable, '-c', code,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    record_property('original_pending_recovery_pid', process.pid)
    try:
        output, error = await asyncio.wait_for(process.communicate(json.dumps({'dsn': dsn, 'prefix': manager.prefix,
            'actor': asdict(actor), 'source': source}).encode()), 30)
        assert process.returncode == 0, error[-1000:]
        return json.loads(output)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.parametrize('kind,case', [('subagent','known'), ('workflow_node','known'),
                                       ('subagent','unknown'), ('subagent','no_budget')])
async def test_real_original_child_result_is_reviewed_without_replay_or_recovery_credit(registry_store, record_property, kind, case):
    provider = BoundedProvider(limit=8)
    c = loop = runtime = None
    try:
        async with asyncio.timeout(150):
            c = await owned_pending_case(registry_store, kind, unknown=case=='unknown', provider=provider,
                                         shared_calls=3 if case=='no_budget' else 12)
            loop, runtime = await new_main(c, registry_store)
            actor = runtime.actor(c.state.root)
            original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY session_id,seq')
            usage = await c.manager.execution_usage(c.state.owner_binding.actor, c.state.owner_binding.lease.job_id, generation=1)
            canonical = await c.manager.get_result_revision(actor, c.effect['job_id'], 1)
            calls = provider.calls
            result = await recover_in_process(runtime, actor, c.state.source, record_property)
            if case == 'unknown':
                assert result['state'] == 'unconfirmed' and c.state.lookups == 0
                assert (await c.store.get_state(c.state.source_binding.actor.session_id)).pending is not None
                c.state.resolved = True
                await reconcile(runtime, actor, c.effect['job_id'], c.adapter)
                result = await recover_in_process(runtime, actor, c.state.source, record_property)
            assert result['state'] == 'queued_for_review' and provider.calls == calls
            assert not result['acknowledgement_created'] and not result['billing_settled']
            response = await loop.send('Read and acknowledge the original known tool outcome. Do not repeat any work.', c.state.root, tools=[])
            effect = await runtime.manager.get_job(actor, c.effect['job_id'])
            if case == 'no_budget':
                assert response.status == 'degraded' and 'budget_max_calls_exhausted' in response.final_text
                assert provider.calls == calls and effect.result_status == 'pending_delivery'
                assert effect.supervision_status == 'open'
            else:
                assert response.status == 'completed' and effect.result_status == 'handled'
                assert effect.supervision_status == 'closed'
                assert (await runtime.manager.get_job(actor, c.state.source)).supervision_status == 'closed'
                await recover_owner(runtime, actor, c.state.owner_binding.lease.job_id)
                await loop.send('Acknowledge the original interrupted controller; do not restart it.', c.state.root, tools=[])
                owner = await runtime.manager.get_job(actor, c.state.owner_binding.lease.job_id)
                assert owner.outcome == 'interrupted' and owner.supervision_status == 'closed' and owner.execution_role == kind
            assert await runtime.manager.execution_usage(actor, c.state.owner_binding.lease.job_id, generation=1) == usage
            assert (await runtime.manager.get_result_revision(actor, c.effect['job_id'], 1))['payload'] == canonical['payload']
            for row in original:
                assert await runtime.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (row['id'],)) == row
            assert not await runtime.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE session_id=? AND role='tool'", (c.state.source_binding.actor.session_id,))
            assert c.state.prepared == c.state.executed == c.state.counts['BUILD'] == len(c.state.effects) == 1
            assert c.state.lookups == (1 if case=='unknown' else 0) and c.state.counts['AFTER'] == 0
    finally:
        if c is not None:
            await cleanup(c, loop, runtime)
        provider.report(record_property)
        await provider.close()


@pytest.mark.parametrize('kind', ['subagent', 'workflow_node'])
async def test_real_mixed_batch_preserves_completed_effect_and_reviews_only_unstarted_subset(registry_store, record_property, kind):
    provider = BoundedProvider(limit=8)
    c = loop = runtime = None
    try:
        async with asyncio.timeout(180):
            c = await owned_pending_case(registry_store, kind, provider=provider, native_tail='inspect_original_note')
            sid = c.state.source_binding.actor.session_id
            pending = (await c.store.get_state(sid)).pending
            assert [call['function']['name'] for call in pending['tool_calls']][:2] == ['touch_once', 'inspect_original_note']
            original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY session_id,seq')
            canonical = await c.manager.get_result_revision(c.runtime.actor(c.state.root), c.effect['job_id'], 1)
            calls = provider.calls
            loop, runtime = await new_main(c, registry_store)
            actor = runtime.actor(c.state.root)
            result = await recover_in_process(runtime, actor, c.state.source, record_property)
            assert result['state'] == 'queued_for_review' and provider.calls == calls
            assert not result['acknowledgement_created'] and not result['billing_settled']
            source = await runtime.manager.get_job(actor, c.state.source)
            outcome = source.result['aborted_call_batch']
            assert outcome['version'] == 2 and outcome['outcome_scope'] == 'original_undispatched_calls'
            assert 'inspect_original_note' in [entry['tool_name'] for entry in outcome['calls']]
            assert 'touch_once' not in [entry['tool_name'] for entry in outcome['calls']]
            response = await loop.send('Acknowledge both original outcomes without repeating any action.', c.state.root, tools=[])
            assert response.status == 'completed'
            assert (await runtime.manager.get_job(actor, c.state.source)).supervision_status == 'closed'
            effect = await runtime.manager.get_job(actor, c.effect['job_id'])
            assert effect.result_status == 'handled' and effect.supervision_status == 'closed'
            assert (await runtime.manager.get_result_revision(actor, effect.job_id, 1))['payload'] == canonical['payload']
            await recover_owner(runtime, actor, c.state.owner_binding.lease.job_id)
            await loop.send('Acknowledge the interrupted original controller.', c.state.root, tools=[])
            owner = await runtime.manager.get_job(actor, c.state.owner_binding.lease.job_id)
            assert owner.outcome == 'interrupted' and owner.supervision_status == 'closed'
            for row in original:
                assert await runtime.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (row['id'],)) == row
            assert not await runtime.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE session_id=? AND role='tool'", (sid,))
            assert c.state.native_calls == c.state.lookups == c.state.counts['AFTER'] == 0
            assert c.state.prepared == c.state.executed == len(c.state.effects) == 1
    finally:
        if c is not None:
            await cleanup(c, loop, runtime)
        provider.report(record_property)
        await provider.close()
