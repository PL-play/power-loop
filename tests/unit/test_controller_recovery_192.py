"""Original controller recovery preserves results, history and paid usage."""
import asyncio

import pytest

from power_loop import HookPoint
from power_loop.runtime import delegation
from power_loop.runtime.controller_handoff import transfer_review
from power_loop.runtime.controller_recovery import handled_references, reconcile
from power_loop.runtime.finalization import closed
from power_loop.runtime.jobs import JobActor, JobError

from .test_controller_handoff_192 import new_main
from .test_job_registry_192 import registry_store
from .test_original_owner_review_192 import original_owner_case, wait_owned

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def ready(registry_store, kind, *, provider=None):
    c = await original_owner_case(registry_store, kind, allowed_calls=1, provider=provider,
                                  max_tokens=4096 if provider is not None else 2048)
    c.recovery_loop = c.recovery_runtime = None
    try:
        initial = set(delegation._TASKS)
        await c.loop.send('Launch original work once.', c.state.root)
        await wait_owned(c, kind, initial)
        await c.runtime.flush_control_checkpoints()
        loop, runtime = await new_main(c, registry_store)
        c.recovery_loop, c.recovery_runtime = loop, runtime
        c.actor = runtime.actor(c.state.root)
        await transfer_review(loop, c.actor, c.state.source)
        await loop.send('Handle the queued original negative outcome without restarting work.', c.state.root, tools=[])
    except BaseException:
        await cleanup(c)
        raise
    return c


async def test_slow_tree_cannot_permanently_starve_later_recovery_candidates(registry_store, monkeypatch):
    from power_loop.runtime import controller_recovery
    c = await ready(registry_store, 'workflow_node')
    rt, actor = c.recovery_runtime, c.actor
    try:
        owner = await rt.manager.get_job(actor, c.state.owner_binding.lease.job_id)
        original = controller_recovery.reconcile
        attempted = []
        async def slow(runtime, actor, job_id):
            attempted.append(job_id)
            if len(attempted) == 1:
                await asyncio.Event().wait()
            return await original(runtime, actor, job_id)
        monkeypatch.setattr(controller_recovery, 'reconcile', slow)
        first = await controller_recovery.recover_tree(rt, actor, owner.root_job_id, limit=1, timeout_s=.05)
        assert first['results'][0]['error'] == 'controller_recovery_timeout'
        assert first['next_cursor'] == attempted[0]
        await controller_recovery.recover_tree(rt, actor, owner.root_job_id, after_job_id=first['next_cursor'], limit=1)
        assert len(attempted) == 2 and attempted[0] != attempted[1]
        assert not c.state.effects and c.state.counts['BUILD'] == 1 and c.state.counts['AFTER'] == 0
    finally:
        await cleanup(c)


async def cleanup(c):
    c.state.release.set()
    for rt, loop in ((c.recovery_runtime, c.recovery_loop), (c.runtime, c.loop)):
        if rt is None:
            continue
        await rt.shutdown(timeout_s=3)
        await rt.flush_control_checkpoints()
        await loop.aclose(drain_timeout_s=.1)


async def recover_in_process(c, job_id, record_property):
    import json
    import os
    import sys
    from dataclasses import asdict
    from urllib.parse import urlparse
    manager = c.recovery_runtime.manager
    if manager.db.dialect.name == 'sqlite':
        dsn = next(row['file'] for row in await manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN']
        assert urlparse(dsn).path.endswith('_test')
    code = '''import asyncio,json,sys
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.jobs import JobManager,JobActor
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.runtime.controller_recovery import reconcile
async def main():
 d=json.load(sys.stdin)
 store=await open_store(d['dsn'],table_prefix=d['prefix'],schema='tasks')
 actor=JobActor(**d['actor'])
 runtime=TaskRuntime(JobManager(store),tenant_id=actor.tenant_id,principal_id=actor.principal_id,
  executor_id='controller-reconciler',boot_id='new-process')
 try: print(json.dumps(await reconcile(runtime,actor,d['job_id'])))
 finally: await store.close()
asyncio.run(main())'''
    process = await asyncio.create_subprocess_exec(sys.executable, '-c', code,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    record_property('controller_recovery_pid', process.pid)
    try:
        output, error = await asyncio.wait_for(process.communicate(json.dumps({'dsn': dsn,
            'prefix': manager.prefix, 'actor': asdict(c.actor), 'job_id': job_id}).encode()), 30)
        assert process.returncode == 0, error[-1000:]
        return json.loads(output)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.parametrize('kind', ['subagent', 'workflow_node'])
async def test_original_negative_controller_result_requires_real_main_handling_and_closes_supervision(registry_store, kind, record_property):
    c = await ready(registry_store, kind)
    rt, actor, owner = c.recovery_runtime, c.actor, c.state.owner_binding
    try:
        before = await rt.manager.execution_usage(actor, owner.lease.job_id, generation=1)
        receipt = await recover_in_process(c, owner.lease.job_id, record_property)
        assert receipt == await reconcile(rt, actor, owner.lease.job_id)
        pending = await rt.manager.get_job(actor, owner.lease.job_id)
        assert pending.outcome == 'interrupted' and pending.resource_state == 'stopped'
        assert pending.result['usage'] == before and pending.result_status == 'pending_delivery'
        assert pending.supervision_status == 'open'
        with pytest.raises(JobError):
            await rt.manager.finalize_supervision(actor, pending.job_id, generation=1)
        await c.recovery_loop.send('Read and handle the original interrupted controller result; do not repeat work.', c.state.root, tools=[])
        final = await rt.manager.get_job(actor, pending.job_id)
        assert final.result == pending.result and final.result_status == 'handled'
        assert final.supervision_status == 'closed' and final.status == 'interrupted'
        assert final.execution_role == kind and c.state.counts['BUILD'] == 1 and not c.state.effects
        if kind == 'workflow_node':
            from power_loop.workflow.result import AgentResult
            assert AgentResult(**final.result).node_id == 'BUILD'
        else:
            assert final.result['messages'] and final.result['session_id'] == owner.actor.session_id
        assert c.state.counts['AFTER'] == 0
        assert await rt.manager.execution_usage(actor, pending.job_id, generation=1) == before
        async with rt.manager._write() as tx:
            row = await rt.manager._authorize(tx, actor, pending.job_id)
            assert await handled_references(rt.manager, tx, row)
            assert await closed(rt.manager, tx, row)
        assert not await rt.manager.db.fetchall(f"SELECT event_id FROM {rt.manager.events} WHERE job_id=? AND event_type='node_result_consumed'", (pending.job_id,))
    finally:
        await cleanup(c)


async def test_cancelled_owner_remains_cancelled_when_actual_exit_is_recovered(registry_store):
    c = await ready(registry_store, 'subagent')
    rt, actor, owner = c.recovery_runtime, c.actor, c.state.owner_binding
    try:
        await rt.manager.request_cancel(actor, owner.lease.job_id, system_key='stop-original-before-recovery', reason='stop original controller')
        await reconcile(rt, actor, owner.lease.job_id)
        pending = await rt.manager.get_job(actor, owner.lease.job_id)
        assert pending.outcome == 'cancelled' and pending.result['status'] == 'cancelled'
        assert pending.result['execution_status'] == 'interrupted'
        await c.recovery_loop.send('Acknowledge the original cancelled controller result without repeating work.', c.state.root, tools=[])
        final = await rt.manager.get_job(actor, owner.lease.job_id)
        assert final.supervision_status == 'closed' and final.outcome == 'cancelled'
        assert not c.state.effects and c.state.counts['BUILD'] == 1
    finally:
        await cleanup(c)


@pytest.mark.parametrize('fault', ['missing_task_exit', 'unhandled_child', 'foreign_main'])
async def test_unconfirmed_controller_cannot_publish_a_result_or_claim_resource_stop(registry_store, fault):
    c = await ready(registry_store, 'subagent')
    rt, owner = c.recovery_runtime, c.state.owner_binding
    try:
        actor = c.actor
        if fault == 'missing_task_exit':
            await rt.manager.db.execute(f"DELETE FROM {rt.manager.events} WHERE job_id=? AND event_type='controller_task_exit'", (owner.lease.job_id,))
        elif fault == 'unhandled_child':
            await rt.manager.db.execute(f"UPDATE {rt.manager.jobs} SET supervision_status='open',result_status='pending_delivery',resource_state='unknown' WHERE job_id=?", (c.state.source,))
        else:
            actor = JobActor(actor.tenant_id, actor.principal_id, await rt.manager.store.create_session())
        before = await rt.manager.get_job(c.actor, owner.lease.job_id)
        with pytest.raises(JobError):
            await reconcile(rt, actor, owner.lease.job_id)
        assert await rt.manager.get_job(c.actor, owner.lease.job_id) == before
        assert not await rt.manager.db.fetchall(f"SELECT event_id FROM {rt.manager.events} WHERE event_type IN ('controller_stopped','controller_result_delivery')")
    finally:
        await cleanup(c)


async def test_hidden_controller_result_does_not_ack_or_close_and_can_later_be_handled(registry_store):
    c = await ready(registry_store, 'subagent')
    rt, actor, owner = c.recovery_runtime, c.actor, c.state.owner_binding
    async def hide(ctx):
        for message in ctx.messages:
            if isinstance(message.get('content'), str) and 'Original controller result:' in message['content']:
                message['content'] = '[controller result removed]'
    try:
        await reconcile(rt, actor, owner.lease.job_id)
        c.recovery_loop.hooks.register(HookPoint.LLM_BEFORE, hide, name='test.controller.hidden')
        await c.recovery_loop.send('Handle queued results.', c.state.root, tools=[])
        pending = await rt.manager.get_job(actor, owner.lease.job_id)
        assert pending.result_status == 'pending_delivery' and pending.supervision_status == 'open'
        c.recovery_loop.hooks.remove('test.controller.hidden')
        await c.recovery_loop.send('Read and handle the original controller result.', c.state.root, tools=[])
        assert (await rt.manager.get_job(actor, owner.lease.job_id)).supervision_status == 'closed'
    finally:
        await cleanup(c)


@pytest.mark.parametrize('fault', ['rollback', 'lost_ack'])
async def test_atomic_recovery_retry_preserves_the_same_original_result_and_usage(registry_store, monkeypatch, fault):
    from contextlib import asynccontextmanager

    c = await ready(registry_store, 'subagent')
    rt, actor, owner = c.recovery_runtime, c.actor, c.state.owner_binding
    original, tripped = rt.manager._write, False
    @asynccontextmanager
    async def fail_once():
        nonlocal tripped
        async with original() as tx:
            yield tx
            if not tripped:
                tripped = True
                if fault == 'rollback':
                    raise OSError('controller recovery rollback')
        if tripped and fail_once.lose_ack:
            fail_once.lose_ack = False
            raise OSError('controller recovery lost acknowledgement')
    fail_once.lose_ack = fault == 'lost_ack'
    try:
        before = await rt.manager.execution_usage(actor, owner.lease.job_id, generation=1)
        monkeypatch.setattr(rt.manager, '_write', fail_once)
        with pytest.raises(OSError):
            await reconcile(rt, actor, owner.lease.job_id)
        monkeypatch.setattr(rt.manager, '_write', original)
        if fault == 'rollback':
            assert (await rt.manager.get_job(actor, owner.lease.job_id)).result_status == 'none'
        receipts = await asyncio.gather(*(reconcile(rt, actor, owner.lease.job_id) for _ in range(4)))
        assert all(receipt == receipts[0] for receipt in receipts)
        assert len(await rt.manager.db.fetchall(f"SELECT event_id FROM {rt.manager.events} WHERE job_id=? AND event_type='result'", (owner.lease.job_id,))) == 1
        assert await rt.manager.execution_usage(actor, owner.lease.job_id, generation=1) == before
        assert not c.state.effects and c.state.counts['BUILD'] == 1
    finally:
        monkeypatch.setattr(rt.manager, '_write', original)
        await cleanup(c)


async def test_entire_original_workflow_closes_without_rewriting_its_failure_or_running_next_node(registry_store):
    c = await ready(registry_store, 'workflow_node')
    rt, actor, node = c.recovery_runtime, c.actor, c.state.owner_binding
    try:
        child = await rt.manager.get_job(actor, node.lease.job_id)
        workflow = await rt.manager.get_job(actor, child.parent_job_id)
        original = await rt.manager.get_result_revision(actor, workflow.job_id, 1)
        await reconcile(rt, actor, child.job_id)
        await c.recovery_loop.send('Handle the interrupted original node without starting more work.', c.state.root, tools=[])
        await reconcile(rt, actor, workflow.job_id)
        await c.recovery_loop.send('Handle the unchanged original workflow failure without restarting it.', c.state.root, tools=[])
        final = await rt.manager.get_job(actor, workflow.job_id)
        if final.supervision_status != 'closed':
            from power_loop.runtime.finalization import try_finalize
            pytest.fail(str((final.result_status, await try_finalize(rt.manager, actor, workflow.job_id, 1))))
        assert final.supervision_status == 'closed' and final.resource_state == 'stopped'
        assert final.result == workflow.result and final.outcome == workflow.outcome
        current = await rt.manager.get_result_revision(actor, workflow.job_id, 1)
        assert current['payload'] == original['payload'] and current['event_id'] == original['event_id']
        assert c.state.counts['AFTER'] == 0 and c.state.counts['BUILD'] == 1 and not c.state.effects
    finally:
        await cleanup(c)


async def test_missing_workflow_result_is_interrupted_and_keeps_all_recorded_node_results(registry_store, monkeypatch):
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.workflow_results import workflow_result
    original = JobManager._record_result_tx
    blocked, attempts = True, []
    async def unavailable(manager, tx, actor, lease, **kwargs):
        row = await manager._authorize(tx, actor, lease.job_id)
        if blocked and row['kind'] == 'workflow':
            attempts.append(row['job_id'])
            raise OSError('workflow canonical result unavailable before commit')
        return await original(manager, tx, actor, lease, **kwargs)
    monkeypatch.setattr(JobManager, '_record_result_tx', unavailable)
    c = await ready(registry_store, 'workflow_node')
    blocked = False
    rt, actor, node = c.recovery_runtime, c.actor, c.state.owner_binding
    try:
        child = await rt.manager.get_job(actor, node.lease.job_id)
        root = await rt.manager.get_job(actor, child.parent_job_id)
        assert attempts and root.result_status == 'none'
        await reconcile(rt, actor, child.job_id)
        await c.recovery_loop.send('Handle interrupted node; do not repeat work.', c.state.root, tools=[])
        child_result = (await rt.manager.get_job(actor, child.job_id)).result
        await reconcile(rt, actor, root.job_id)
        pending = await rt.manager.get_job(actor, root.job_id)
        projected = workflow_result(pending.result)
        assert projected.status == 'interrupted' and projected.results['BUILD'].to_dict() == child_result
        await c.recovery_loop.send('Handle original interrupted workflow without restarting it.', c.state.root, tools=[])
        assert (await rt.manager.get_job(actor, root.job_id)).supervision_status == 'closed'
        assert c.state.counts['AFTER'] == 0 and c.state.counts['BUILD'] == 1 and not c.state.effects
    finally:
        blocked = False
        await cleanup(c)


async def test_saved_closure_rechecks_exit_proof_in_a_new_transaction(registry_store):
    c = await ready(registry_store, 'subagent')
    rt, actor, owner = c.recovery_runtime, c.actor, c.state.owner_binding
    try:
        await reconcile(rt, actor, owner.lease.job_id)
        await c.recovery_loop.send('Handle the original interrupted controller result.', c.state.root, tools=[])
        async with rt.manager._write() as tx:
            row = await rt.manager._authorize(tx, actor, owner.lease.job_id)
            assert await closed(rt.manager, tx, row)
        await rt.manager.db.execute(f"DELETE FROM {rt.manager.events} WHERE job_id=? AND event_type='controller_task_exit'", (owner.lease.job_id,))
        async with rt.manager._write() as tx:
            row = await rt.manager._authorize(tx, actor, owner.lease.job_id)
            with pytest.raises(JobError):
                await closed(rt.manager, tx, row)
    finally:
        await cleanup(c)
