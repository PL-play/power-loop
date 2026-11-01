"""Dispatch permission survives crashes without reconstructing an executor."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from power_loop.runtime import control_requests, resume_dispatch, session_obligations
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_job_registry_192 import registry_store
from .test_nudge_runtime_192 import issue_nudge
from .test_subagent_resume_192 import nudge_receipt, ready
from .test_task_runtime_192 import finish, managed, receipt

__all__ = ["registry_store", "managed", "ready"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def exercise_writer(c, monkeypatch, method):
    original, write = getattr(c.manager, method), c.manager._write
    one, many = ControlTransaction.fetchone, ControlTransaction.fetchall
    footprints = []

    def note(tx, rows):
        tx.__dict__.setdefault("_resume_read_jobs", set()).update(
            row["job_id"] for row in rows if row is not None and "job_id" in row)

    async def fetchone(tx, sql, params=()):
        row = await one(tx, sql, params)
        note(tx, [row])
        return row

    async def fetchall(tx, sql, params=()):
        rows = await many(tx, sql, params)
        note(tx, rows)
        return rows

    async def observed(*args, **kwargs):
        roots = {row["job_id"]: row["root_job_id"] for row in await c.manager.db.fetchall(
            f"SELECT job_id,root_job_id FROM {c.manager.jobs}")}

        @asynccontextmanager
        async def scoped():
            async with write() as tx:
                yield tx
                jobs = tx.touched_jobs | tx.__dict__.get("_resume_read_jobs", set())
                footprints.append({roots[job] for job in jobs})

        with monkeypatch.context() as patch:
            patch.setattr(ControlTransaction, "fetchone", fetchone)
            patch.setattr(ControlTransaction, "fetchall", fetchall)
            patch.setattr(c.manager, "_write", scoped)
            return await original(*args, **kwargs)

    monkeypatch.setattr(c.manager, method, observed)
    await issue_nudge(c.fixture, monkeypatch)
    receipt = await nudge_receipt(c)
    await finish(c.fixture)
    assert receipt["status"] == "accepted" and receipt["generation"] == 2
    assert c.model.child_calls == 2
    assert footprints and all(len(roots) <= 1 for roots in footprints), footprints
    return footprints


async def test_dispatch_permission_target_and_acknowledgement_use_separate_tree_writers(ready, monkeypatch):
    footprints = await exercise_writer(ready, monkeypatch, "claim_resume_dispatch")
    assert len(footprints) == 3 and footprints[0] == footprints[2] and footprints[0] != footprints[1]


async def test_original_native_dispatch_start_is_durable_with_its_source_permission(ready, monkeypatch):
    from power_loop.runtime.managed_effects import _ref
    from power_loop.runtime.tool_call_plans import identity, start_id

    c, found = ready, []
    transaction = c.manager.db.transaction

    @asynccontextmanager
    async def inspect_commit():
        async with transaction() as tx:
            yield tx
            if not found:
                rows = await tx.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='task_control_request'")
                grant = next((row for row in rows if (lambda data:data['phase']=='accepted' and data['action']==resume_dispatch.ACTION)(
                    json.loads(row['payload_json']))), None)
                if grant is not None:
                    request = json.loads(grant['payload_json'])['request']
                    source = await c.manager._row(tx, grant['job_id'])
                    key = start_id(identity(c.manager, source, _ref(request['call_ref'])), request['tool_call_id'])
                    assert await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_id=? "
                        "AND event_type='tool_call_dispatch_started'", (key,))
                    assert not await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'")
                    found.append(key)

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, 'transaction', inspect_commit)
        await exercise_writer(c, monkeypatch, 'claim_resume_dispatch')
    assert len(found) == 1


@asynccontextmanager
async def held_dispatch(c, monkeypatch):
    entered = asyncio.Event()
    args = {}

    async def withhold(actor, operation_id, **kwargs):
        args.update(actor=actor, operation_id=operation_id, **kwargs)
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(c.manager, "claim_resume_dispatch", withhold)
    task = asyncio.create_task(issue_nudge(c.fixture, monkeypatch))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        yield SimpleNamespace(c=c, args=args, source=task)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await finish(c.fixture)


@pytest.fixture
async def pending(ready, monkeypatch):
    async with held_dispatch(ready, monkeypatch) as value:
        yield value


@pytest.fixture
async def registered_pending(managed, monkeypatch):
    from power_loop.runtime import executor_registry as registry

    store, manager, runtime, loop, model, parent, connect = managed
    actor = runtime.actor(parent)
    identity = registry.ExecutorIdentity(actor.tenant_id, actor.principal_id, runtime.executor_id, runtime.boot_id)
    await registry.register(manager, identity, contract_hash='a'*64, startup_lease_ms=300000)
    await registry.activate(manager, identity, expected_version=1, system_key='start', lease_ms=300000)
    await loop.send('delegate', parent)
    created = await receipt(managed)
    await finish(managed)
    await loop.flush_follow_ups(parent)
    old = await manager.get_job(actor, created['job_id'])
    assert old.generation == 1 and old.result_status == 'handled'
    model.release.clear()
    model.entered.clear()
    c = SimpleNamespace(store=store, manager=manager, runtime=runtime, loop=loop, model=model,
        parent=parent, connect=connect, fixture=managed, old=old)
    async with held_dispatch(c, monkeypatch) as value:
        yield value


async def records(c, manager=None):
    manager = manager or c.manager
    rows = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='task_control_request'")
    accepted = [row for row in rows if (lambda data: data['phase'] == 'accepted' and data['action'] == resume_dispatch.ACTION)(
        json.loads(row['payload_json']))]
    return accepted, rows


async def test_reconciliation_without_original_permission_cannot_create_a_dispatch(pending):
    from power_loop.runtime.managed_effects import _ref

    p, c = pending, pending.c
    op = await c.manager.get_operation(p.args['actor'], p.args['operation_id'])
    key = c.manager._origin_key(p.args['actor'], resume_dispatch.ACTION, _ref(op.call_ref), 'actual-nudge', None)
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    before = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
    with pytest.raises(JobError, match='resume_dispatch_request_original_missing'):
        await resume_dispatch.reconcile_request(cold, p.args['actor'], control_requests.identity(resume_dispatch.ACTION, key))
    assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == before
    assert c.model.child_calls == 1


async def test_independent_competing_dispatch_tokens_have_one_original_winner(pending):
    p, c = pending, pending.c
    peer = JobManager(await c.connect(), now_ms=c.manager.now)
    args = [p.args, {**p.args, 'token':'other-token'}]
    results = await asyncio.gather(resume_dispatch.claim(c.manager, **args[0]), resume_dispatch.claim(peer, **args[1]))
    assert sorted(results) == [False, True]
    [dispatch] = await peer.db.fetchall(f"SELECT payload_json FROM {peer.events} WHERE event_type='dispatch'")
    assert json.loads(dispatch['payload_json'])['token'] == args[results.index(True)]['token']
    assert not await peer.db.fetchall(f"SELECT source_id FROM {peer.prefix}session_obligations WHERE kind='control_request'")
    assert c.model.child_calls == 1
    await session_obligations.verify(peer, p.args['actor'])


async def test_a_closed_original_caller_cannot_create_a_new_dispatch_permission(pending):
    p, c = pending, pending.c
    p.source.cancel()
    await asyncio.gather(p.source, return_exceptions=True)
    with pytest.raises(JobError, match="intake_closed|not_running|cancel_requested"):
        await resume_dispatch.claim(c.manager, **p.args)
    assert not (await records(c))[0]
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'")
    assert c.model.child_calls == 1


async def test_fresh_dispatch_does_not_reread_its_admissions_raw_call_under_a_writer(pending, monkeypatch):
    p, c = pending, pending.c
    validate = c.manager._validate_call
    reads = []

    async def outside(reader, actor, ref, call_id):
        assert c.manager._control_intents.get() is None
        reads.append(ref)
        return await validate(reader, actor, ref, call_id)

    with monkeypatch.context() as patch:
        patch.setattr(c.manager, '_validate_call', outside)
        assert await resume_dispatch.claim(c.manager, **p.args)
    assert reads == [] and c.model.child_calls == 1


async def test_source_exit_after_permission_can_finish_only_the_original_dispatch_token(pending, monkeypatch):
    p, c = pending, pending.c
    publish = resume_dispatch._publish

    async def close_source(owner, actor, accepted, outcome):
        assert owner._control_intents.get() is None
        p.source.cancel()
        await asyncio.gather(p.source, return_exceptions=True)
        assert (await owner._row(owner.db, p.args['caller'].job_id))['intake'] == 'closed'
        [entry] = await owner.db.fetchall(f"SELECT source_id FROM {owner.prefix}session_obligations WHERE kind='control_request'")
        assert entry['source_id'] == accepted.event['event_id']
        return await publish(owner, actor, accepted, outcome)

    with monkeypatch.context() as patch:
        patch.setattr(resume_dispatch, '_publish', close_source)
        assert await resume_dispatch.claim(c.manager, **p.args)
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    assert await resume_dispatch.claim(cold, **p.args)
    assert not await resume_dispatch.claim(cold, **{**p.args, 'token': 'different-token'})
    assert c.model.child_calls == 1
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    await session_obligations.verify(cold, p.args['actor'])


async def test_target_stop_after_permission_records_a_refusal_and_never_claims_dispatch(pending, monkeypatch):
    p, c = pending, pending.c
    publish, stopped = resume_dispatch._publish, []

    async def stop_target(owner, actor, accepted, outcome):
        await owner.request_cancel(actor, c.old.job_id, system_key='stop-before-dispatch', reason='test')
        stopped.append(await owner.get_job(actor, c.old.job_id))
        return await publish(owner, actor, accepted, outcome)

    with monkeypatch.context() as patch:
        patch.setattr(resume_dispatch, '_publish', stop_target)
        with pytest.raises(JobError, match='cancel_requested'):
            await resume_dispatch.claim(c.manager, **p.args)
    assert await c.manager.get_job(p.args['actor'], c.old.job_id) == stopped[0]
    [accepted], _ = await records(c)
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    with pytest.raises(JobError, match='cancel_requested'):
        await resume_dispatch.reconcile_request(cold, p.args['actor'], accepted['event_id'])
    assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='dispatch'")
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    assert c.model.child_calls == 1


async def test_executor_shutdown_after_permission_still_blocks_the_target_dispatch(registered_pending, monkeypatch):
    from power_loop.runtime import executor_registry as registry

    p, c = registered_pending, registered_pending.c
    publish, stopped = resume_dispatch._publish, []

    async def shutdown(owner, actor, accepted, outcome):
        identity = registry.ExecutorIdentity(actor.tenant_id, actor.principal_id, c.runtime.executor_id, c.runtime.boot_id)
        await registry.drain(owner, identity, system_key='shutdown-before-dispatch', reason='test')
        stopped.append(await owner.get_job(actor, c.old.job_id))
        return await publish(owner, actor, accepted, outcome)

    with monkeypatch.context() as patch:
        patch.setattr(resume_dispatch, '_publish', shutdown)
        with pytest.raises(JobError, match='cancel_requested|executor_not_accepting'):
            await resume_dispatch.claim(c.manager, **p.args)
    assert len(stopped) == 1 and await c.manager.get_job(p.args['actor'], c.old.job_id) == stopped[0]
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'")
    assert not await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
    assert c.model.child_calls == 1


@pytest.mark.parametrize('phase', ['accepted', 'target', 'completed'])
@pytest.mark.parametrize('committed', [False, True])
async def test_each_dispatch_transaction_rollback_or_unknown_ack_retries_its_exact_token(pending, monkeypatch, phase, committed):
    p, c = pending, pending.c
    transaction, hits = c.manager.db.transaction, []

    @asynccontextmanager
    async def fault():
        async with transaction() as tx:
            yield tx
            rows = await tx.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='task_control_request'")
            accepted = [row for row in rows if (lambda data: data['phase']=='accepted' and data['action']==resume_dispatch.ACTION)(
                json.loads(row['payload_json']))]
            current = None
            if accepted:
                identity = accepted[0]['event_id']
                if any(row['event_id'] == control_requests.completion_id(identity) for row in rows):
                    current = 'completed'
                elif await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'"):
                    current = 'target'
                else:
                    current = 'accepted'
            hit = not hits and current == phase
            if hit:
                hits.append((phase, committed))
                if not committed:
                    raise OSError('dispatch transaction rolled back')
        if hit:
            raise OSError('dispatch committed; ACK lost')

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, 'transaction', fault)
        with pytest.raises(OSError, match='rolled back|ACK lost'):
            await resume_dispatch.claim(c.manager, **p.args)
    assert hits == [(phase, committed)]
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    assert await resume_dispatch.claim(cold, **p.args)
    assert not await resume_dispatch.claim(cold, **{**p.args, 'token':'other-token'})
    [dispatch] = await cold.db.fetchall(f"SELECT payload_json FROM {cold.events} WHERE event_type='dispatch'")
    assert json.loads(dispatch['payload_json'])['token'] == p.args['token']
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    assert c.model.child_calls == 1 and (await cold.get_job(p.args['actor'], c.old.job_id)).generation == 2


@pytest.mark.parametrize('phase', ['accepted', 'target', 'completed'])
@pytest.mark.parametrize('committed', [False, True])
async def test_real_dispatch_process_death_is_repaired_without_a_model_or_callback(pending, tmp_path, phase, committed):
    p, c = pending, pending.c
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    context = {**p.args, 'actor':asdict(p.args['actor']), 'caller':asdict(p.args['caller']), 'now':c.manager.now()}
    ready, output = tmp_path/'dispatch.ready', tmp_path/'dispatch.json'
    mode = phase+':'+str(int(committed))
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._resume_dispatch_worker',
        dsn, c.manager.prefix, json.dumps(context), mode, str(ready), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 79, stdout.decode()
        assert ready.read_text() == mode and not output.exists()
        accepted, rows = await records(c)
        if phase == 'accepted' and not committed:
            from power_loop.runtime.managed_effects import _ref
            assert not accepted
            assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'")
            op = await c.manager.get_operation(p.args['actor'], p.args['operation_id'])
            key = c.manager._origin_key(p.args['actor'], resume_dispatch.ACTION, _ref(op.call_ref), 'actual-nudge', None)
            cold = JobManager(await c.connect(), now_ms=c.manager.now)
            with pytest.raises(JobError, match='resume_dispatch_request_original_missing'):
                await resume_dispatch.reconcile_request(cold, p.args['actor'], control_requests.identity(resume_dispatch.ACTION, key))
            assert c.model.child_calls == 1
            return
        [accepted] = accepted
        completed = [row for row in rows if row['event_id'] == control_requests.completion_id(accepted['event_id'])]
        dispatches = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='dispatch'")
        assert len(dispatches) == int(phase == 'completed' or phase == 'target' and committed)
        assert len(completed) == int(phase == 'completed' and committed)
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        assert await resume_dispatch.reconcile_request(cold, p.args['actor'], accepted['event_id'])
        before = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await resume_dispatch.reconcile_request(cold, p.args['actor'], accepted['event_id'])
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == before
        assert c.model.child_calls == 1
        assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
        await session_obligations.verify(cold, p.args['actor'])
    finally:
        await stop_worker(process)
