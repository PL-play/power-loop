"""Only an original main handles a retired child's unchanged negative result."""
from dataclasses import replace

import pytest

from power_loop import StatefulAgentLoop
from power_loop.runtime import delegation
from power_loop.runtime.controller_handoff import transfer_review
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store
from .test_original_owner_review_192 import original_owner_case, wait_owned

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def new_main(c, registry_store):
    manager = JobManager(await registry_store[1]())
    runtime = TaskRuntime(manager, tenant_id=c.runtime.tenant_id, principal_id=c.runtime.principal_id,
        executor_id='original-main-recovery', boot_id='new-main-boot', lease_ms=300000,
        boot_contract_hash='c'*64, budget_config=c.runtime.budget_config)
    loop = StatefulAgentLoop(llm=c.model, store=manager.store, task_runtime=runtime,
        tool_registry=c.loop.tool_registry, config=replace(c.loop.config))
    return loop, runtime


@pytest.mark.parametrize('kind', ['subagent', 'workflow_node'])
async def test_new_host_transfers_only_negative_review_and_actual_main_must_consume_it(registry_store, kind, record_property):
    c = await original_owner_case(registry_store, kind, allowed_calls=1)
    initial = set(delegation._TASKS)
    loop = runtime = None
    try:
        await c.loop.send('Launch original work once.', c.state.root)
        await wait_owned(c, kind, initial)
        owner = c.state.owner_binding
        assert owner.stop_renewal.is_set()
        before = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        assert before.result_status == 'pending_delivery'
        usage = await c.manager.execution_usage(owner.actor, owner.lease.job_id, generation=1)
        await c.runtime.shutdown(timeout_s=3)
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        receipt = await transfer_in_process(c, record_property)
        assert receipt == await transfer_review(loop, actor, c.state.source)
        assert receipt['execution_authority'] is False and receipt['acknowledgement_created'] is False
        from power_loop.runtime.abort_outcomes import reconcile
        with pytest.raises(JobError, match='original_owner_review_transferred'):
            await reconcile(c.loop, c.state.source_binding.actor, c.state.source)
        unchanged = await runtime.manager.get_job(actor, c.state.source)
        assert unchanged == before
        assert not c.state.effects and c.state.counts['BUILD'] == 1
        await loop.send('Read and handle the queued original negative result; do not redo any work.', c.state.root, tools=[])
        after = await runtime.manager.get_job(actor, c.state.source)
        assert after.result == before.result and after.result_status == 'handled'
        assert after.supervision_status == 'closed'
        from power_loop.runtime.abort_outcomes import handled_references
        async with runtime.manager._write() as tx:
            source = await runtime.manager._authorize(tx, actor, c.state.source)
            assert await handled_references(runtime.manager, tx, source)
        assert await runtime.manager.execution_usage(owner.actor, owner.lease.job_id, generation=1) == usage
        # Result handling is not a fabricated successful child or stopped resource.
        original_owner = await runtime.manager.get_job(actor, owner.lease.job_id)
        assert original_owner.result_status == 'none' and original_owner.supervision_status == 'open'
        assert not c.state.effects and c.state.counts['BUILD'] == 1
    finally:
        c.state.release.set()
        if runtime is not None:
            await runtime.shutdown(timeout_s=3)
            await loop.aclose(drain_timeout_s=.1)
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('fault', ['no_retirement', 'unknown_model', 'claimed_notice', 'foreign_main'])
async def test_unconfirmed_or_foreign_handoff_does_not_change_any_original_result(registry_store, fault):
    c = await original_owner_case(registry_store, 'subagent', allowed_calls=1)
    initial = set(delegation._TASKS)
    try:
        await c.loop.send('Launch original work once.', c.state.root)
        await wait_owned(c, 'subagent', initial)
        actor = c.runtime.actor(c.state.root)
        owner = c.state.owner_binding
        if fault == 'no_retirement':
            await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='controller_retired'")
        elif fault == 'unknown_model':
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET resource_state='unknown' WHERE job_id=?", (c.state.source,))
        elif fault == 'claimed_notice':
            await c.manager.db.execute(f"UPDATE {c.manager.store.t.follow_up_queue} SET status='claimed' WHERE session_id=? AND kind='original_call_outcome'", (owner.actor.session_id,))
        else:
            actor = JobActor(actor.tenant_id, actor.principal_id, await c.manager.store.create_session())
        before = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        with pytest.raises(JobError):
            await transfer_review(c.loop, actor, c.state.source)
        assert await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source) == before
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='controller_review_handoff'")
        assert not c.state.effects and c.state.counts['BUILD'] == 1
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=.1)


async def transfer_in_process(c, record_property):
    import asyncio
    import json
    import os
    import sys
    from dataclasses import asdict

    if c.manager.db.dialect.name == 'sqlite':
        rows = await c.manager.db.fetchall('PRAGMA database_list')
        dsn = next(row['file'] for row in rows if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN']
        from urllib.parse import urlparse
        assert urlparse(dsn).path.endswith('_test')
    code = '''import asyncio,json,sys
from power_loop import AgentLoopConfig,LLMService,StatefulAgentLoop
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.jobs import JobManager,JobActor
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.runtime.controller_handoff import handoff_retired_reviews
class NoModel(LLMService):
 async def complete(self,*a,**k): raise AssertionError('mechanical handoff must not call model')
 def stream(self,*a,**k): raise AssertionError('no stream')
 async def close(self): pass
async def main():
 data=json.load(sys.stdin)
 store=await open_store(data['dsn'],table_prefix=data['prefix'],schema='tasks')
 actor=JobActor(**data['actor'])
 runtime=TaskRuntime(JobManager(store),tenant_id=actor.tenant_id,principal_id=actor.principal_id,
  executor_id='handoff-reconciler',boot_id='separate-process')
 loop=StatefulAgentLoop(llm=NoModel(),store=store,task_runtime=runtime,
  config=AgentLoopConfig(delegation_policy='single_level'))
 try: print(json.dumps((await handoff_retired_reviews(runtime,actor,data['owner']))[0]))
 finally:
  await loop.aclose(drain_timeout_s=0.1)
  await store.close()
asyncio.run(main())'''
    child = await asyncio.create_subprocess_exec(sys.executable, '-c', code, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    record_property('controller_handoff_pid', child.pid)
    try:
        output, error = await asyncio.wait_for(child.communicate(json.dumps({'dsn': dsn,
            'prefix': c.manager.prefix, 'actor': asdict(c.runtime.actor(c.state.root)), 'source': c.state.source, 'owner': c.state.owner_binding.lease.job_id}).encode()), 20)
        assert child.returncode == 0, error[-500:]
        return json.loads(output)
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


@pytest.mark.parametrize('fault', ['hidden', 'provider_failure', 'cancelled'])
async def test_actual_main_must_see_and_cleanly_handle_the_original_transferred_result(registry_store, monkeypatch, fault):
    from power_loop import CancellationToken, HookPoint

    c = await original_owner_case(registry_store, 'subagent', allowed_calls=1)
    initial = set(delegation._TASKS)
    loop = runtime = None
    try:
        await c.loop.send('Launch original work once.', c.state.root)
        await wait_owned(c, 'subagent', initial)
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        await transfer_review(loop, actor, c.state.source)
        before = await runtime.manager.get_job(actor, c.state.source)
        token = CancellationToken()
        if fault == 'hidden':
            async def hide(ctx):
                for message in ctx.messages:
                    if isinstance(message.get('content'), str) and 'Original tool outcome:' in message['content']:
                        message['content'] = '[original negative outcome removed]'
            loop.hooks.register(HookPoint.LLM_BEFORE, hide, name='test.handoff.hidden')
        elif fault == 'provider_failure':
            async def failed(*args, **kwargs):
                raise RuntimeError('actual main review failed')
            monkeypatch.setattr(c.model, 'complete', failed)
        else:
            token.cancel('cancel new main review')
        try:
            await loop.send('Handle the original result without repeating work.', c.state.root, tools=[], stop_event=token)
        except RuntimeError:
            assert fault == 'provider_failure'
        after = await runtime.manager.get_job(actor, c.state.source)
        assert after.result == before.result and after.result_status == 'pending_delivery'
        assert after.supervision_status == 'open' and not c.state.effects
        assert not await runtime.manager.db.fetchall(f"SELECT event_id FROM {runtime.manager.events} WHERE job_id=? AND event_type='processing_decision'", (c.state.source,))
    finally:
        c.state.release.set()
        if runtime is not None:
            await runtime.shutdown(timeout_s=3)
            await loop.aclose(drain_timeout_s=.1)
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('stage', ['rollback', 'lost_ack'])
async def test_transfer_and_queue_are_atomic_and_retry_returns_same_original_responsibility(registry_store, monkeypatch, stage):
    from contextlib import asynccontextmanager

    c = await original_owner_case(registry_store, 'subagent', allowed_calls=1)
    initial = set(delegation._TASKS)
    try:
        await c.loop.send('Launch original work once.', c.state.root)
        await wait_owned(c, 'subagent', initial)
        actor = c.runtime.actor(c.state.root)
        original, failed = c.manager._write, False

        @asynccontextmanager
        async def once():
            nonlocal failed
            if failed:
                async with original() as tx:
                    yield tx
                return
            failed = True
            async with original() as tx:
                yield tx
                if stage == 'rollback':
                    raise OSError('transfer rolled back')
            raise OSError('transfer committed; acknowledgement lost')

        monkeypatch.setattr(c.manager, '_write', once)
        with pytest.raises(OSError):
            await transfer_review(c.loop, actor, c.state.source)
        records = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='controller_review_handoff'")
        assert len(records) == (0 if stage == 'rollback' else 1)
        one = await transfer_review(c.loop, actor, c.state.source)
        assert await transfer_review(c.loop, actor, c.state.source) == one
        records = await c.manager.db.fetchall(f"SELECT item_id FROM {c.store.t.follow_up_queue} WHERE session_id=? AND kind='original_call_outcome'", (c.state.root,))
        assert len(records) == 1 and records[0]['item_id'] == one['notification_id']
        assert (await c.manager.get_job(actor, c.state.source)).result_status == 'pending_delivery'
        assert not c.state.effects and c.state.counts['BUILD'] == 1
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('corruption', ['handoff_payload', 'original_exit', 'original_notice', 'membership_seal'])
async def test_corrupt_handoff_cannot_be_used_as_model_processing_authority(registry_store, corruption):
    from power_loop.runtime.abort_outcomes import notice, result_event
    from power_loop.runtime.controller_handoff import identity

    c = await original_owner_case(registry_store, 'subagent', allowed_calls=1)
    initial = set(delegation._TASKS)
    try:
        await c.loop.send('Launch original work once.', c.state.root)
        await wait_owned(c, 'subagent', initial)
        actor = c.runtime.actor(c.state.root)
        await transfer_review(c.loop, actor, c.state.source)
        source = await c.manager._authorize(c.manager.db, actor, c.state.source)
        if corruption == 'handoff_payload':
            key = identity(c.manager, source)
        elif corruption == 'original_exit':
            key = 'model_closed_' + source['job_id']
        elif corruption == 'original_notice':
            from power_loop.runtime.abort_outcomes import notice_id
            key = notice_id(c.manager, source)
        else:
            from power_loop.runtime.execution_membership import seal_id
            key = seal_id(c.manager, c.state.owner_binding.lease.job_id, 1)
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json='{{}}' WHERE event_id=?", (key,))
        async with c.manager._write() as tx:
            with pytest.raises(JobError):
                result = await result_event(c.manager, tx, source)
                await notice(c.manager, tx, source, result)
        assert (await c.manager.get_job(actor, c.state.source)).result_status == 'pending_delivery'
        assert not c.state.effects
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=.1)


async def test_handoff_does_not_increase_exhausted_original_main_budget(registry_store):
    c = await original_owner_case(registry_store, 'subagent', allowed_calls=1, shared_calls=3)
    initial = set(delegation._TASKS)
    loop = runtime = None
    try:
        await c.loop.send('Launch original work once.', c.state.root)
        await wait_owned(c, 'subagent', initial)
        assert c.state.counts['MAIN'] == 2 and c.state.counts['BUILD'] == 1
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        await transfer_review(loop, actor, c.state.source)
        before = dict(c.state.counts)
        await loop.send('Handle original result without any new work.', c.state.root, tools=[])
        assert dict(c.state.counts) == before
        assert (await runtime.manager.get_job(actor, c.state.source)).result_status == 'pending_delivery'
        assert not await runtime.manager.db.fetchall(f"SELECT event_id FROM {runtime.manager.events} WHERE job_id=? AND event_type='processing_decision'", (c.state.source,))
    finally:
        c.state.release.set()
        if runtime is not None:
            await runtime.shutdown(timeout_s=3)
            await loop.aclose(drain_timeout_s=.1)
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=.1)
