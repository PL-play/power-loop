"""A stop is accepted by its caller tree before repeatable target application."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from power_loop.runtime import session_obligations, task_control
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker, wait_ready
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def cross_control(store, connect, *, now=None, claim_target=True):
    manager = JobManager(store, **({"now_ms": now} if now is not None else {}))
    actor = JobActor("stop-trees", "host", await store.create_session())
    old = await manager.register_job(actor, kind="send", session_id=actor.session_id,
        system_key="previous-root", request={"send_index": 1})
    old_caller = await manager.claim_job(actor, old.job.job_id, executor_id="old", boot_id="boot", expected_version=1)
    target = await manager.register_job(actor, kind="subagent", parent=old_caller, independent=True,
        system_key="previous-independent-child", session_id=actor.session_id)
    target_lease = await manager.claim_job(actor, target.job.job_id, executor_id="child", boot_id="boot", expected_version=1) if claim_target else None
    await manager.close_model_execution(actor, old_caller)
    caller_job = await manager.register_job(actor, kind="send", session_id=actor.session_id,
        system_key="current-root", request={"send_index": 2})
    caller = await manager.claim_job(actor, caller_job.job.job_id, executor_id="current", boot_id="boot", expected_version=1)
    seq = await store.append_message(actor.session_id, role="assistant", send_index=2, round_index=0,
        tool_calls=[{"id": "stop-old-child", "type": "function", "function": {"name": "jobs", "arguments":
            json.dumps({"action": "stop", "ids": [target.job.job_id]})}}])
    ref = await store.message_ref(actor.session_id, seq)

    async def stop(owner=None):
        return await task_control.request_stop(owner or manager, actor, target.job.job_id,
            caller=caller, call_ref=ref, tool_call_id="stop-old-child")

    return SimpleNamespace(store=store, manager=manager, actor=actor, old=old, caller=caller, target=target,
        target_lease=target_lease, ref=ref, stop=stop, connect=connect)


async def test_cross_tree_stop_never_reads_or_writes_two_roots_in_one_writer(registry_store, monkeypatch):
    c = await cross_control(*registry_store)
    rows = await c.manager.db.fetchall(f"SELECT job_id,root_job_id FROM {c.manager.jobs}")
    roots = {row["job_id"]: row["root_job_id"] for row in rows}
    write, one, many = c.manager._write, ControlTransaction.fetchone, ControlTransaction.fetchall
    footprints = []

    def note(tx, rows):
        tx.__dict__.setdefault("_test_read_jobs", set()).update(
            row["job_id"] for row in rows if row is not None and "job_id" in row)

    async def fetchone(tx, sql, params=()):
        row = await one(tx, sql, params)
        note(tx, [row])
        return row

    async def fetchall(tx, sql, params=()):
        rows = await many(tx, sql, params)
        note(tx, rows)
        return rows

    @asynccontextmanager
    async def scoped():
        async with write() as tx:
            yield tx
        seen = tx.touched_jobs | tx.__dict__.get("_test_read_jobs", set())
        footprint = {roots[job] for job in seen}
        assert len(footprint) <= 1, "caller acceptance and target application must be separate tree writers"
        footprints.append(footprint)

    with monkeypatch.context() as patch:
        patch.setattr(ControlTransaction, "fetchone", fetchone)
        patch.setattr(ControlTransaction, "fetchall", fetchall)
        patch.setattr(c.manager, "_write", scoped)
        operation, replayed = await c.stop()
    assert not replayed and operation.result["response"]["cancel_requested"]
    assert footprints == [{c.caller.job_id}, {c.old.job.job_id}, {c.caller.job_id}]
    assert (await c.manager.get_job(c.actor, c.target.job.job_id)).resource_state == "running"


async def test_source_close_after_acceptance_keeps_a_real_pending_request_until_target_ack(registry_store, monkeypatch):
    c = await cross_control(*registry_store)
    apply = task_control._apply_stop_request
    accepted = []

    async def closed_before_apply(owner, actor, event, data):
        assert owner._control_intents.get() is None
        accepted.append(event)
        await owner.close_model_execution(actor, c.caller, evidence={"exit": "completed"})
        pending = await owner.db.fetchall(f"SELECT * FROM {owner.prefix}session_obligations WHERE kind='control_request'")
        assert [row["source_id"] for row in pending] == [event["event_id"]]
        await owner.finalize_supervision(actor, c.caller.job_id, generation=c.caller.generation)
        assert await owner.db.fetchall(f"SELECT source_id FROM {owner.prefix}session_obligations WHERE kind='control_request'")
        return await apply(owner, actor, event, data)

    with monkeypatch.context() as patch:
        patch.setattr(task_control, "_apply_stop_request", closed_before_apply)
        operation, replayed = await c.stop()
    assert len(accepted) == 1 and not replayed and operation.result["response"]["cancel_requested"]
    assert not await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
    await c.manager.finalize_supervision(c.actor, c.caller.job_id, generation=c.caller.generation)
    assert (await c.manager.get_job(c.actor, c.target.job.job_id)).resource_state == "running"
    assert await c.stop() == (operation, True)
    await session_obligations.verify(c.manager, c.actor)


async def test_peer_completion_of_the_same_stop_updates_the_original_sends_committed_history(registry_store, monkeypatch):
    from power_loop.runtime.jobs import TRUST_OWN_WRITES

    token = TRUST_OWN_WRITES.set(True)
    try:
        c = await cross_control(*registry_store, now=lambda: 1_000_000)
        peer = JobManager(await c.connect(), now_ms=c.manager.now)
        apply, peers = task_control._apply_stop_request, []

        async def complete_in_peer(owner, actor, event, data):
            if owner is c.manager and not peers:
                peers.append(await c.stop(peer))
            return await apply(owner, actor, event, data)

        with monkeypatch.context() as patch:
            patch.setattr(task_control, "_apply_stop_request", complete_in_peer)
            operation, replayed = await c.stop()
        assert not replayed and peers == [(operation, True)]
        assert tuple(map(len, await request_records(c))) == (1, 1, 1, 0)
        await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed"})
        await c.manager.finalize_supervision(c.actor, c.caller.job_id, generation=1)
        await session_obligations.verify(c.manager, c.actor)
    finally:
        TRUST_OWN_WRITES.reset(token)


async def test_target_claim_after_acceptance_is_never_cancelled_by_the_old_request(registry_store, monkeypatch):
    clock = [1_000_000]
    c = await cross_control(*registry_store, now=lambda: clock[0], claim_target=False)
    apply = task_control._apply_stop_request
    newer = []

    async def takeover_before_apply(owner, actor, event, data):
        clock[0] += 30001
        current = await owner.get_job(actor, c.target.job.job_id)
        proof = await owner.claim_job(actor, c.target.job.job_id, executor_id="new-child", boot_id="new-boot",
            expected_version=current.version)
        newer.append(proof)
        return await apply(owner, actor, event, data)

    with monkeypatch.context() as patch:
        patch.setattr(task_control, "_apply_stop_request", takeover_before_apply)
        with pytest.raises(JobError, match="stop_generation_changed"):
            await c.stop()
    current = await c.manager.get_job(c.actor, c.target.job.job_id)
    assert len(newer) == 1 and current.generation == newer[0].generation == 1
    assert current.cancel_requested_at is None and current.resource_state == "running"
    pending = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
    assert pending == []
    rows = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='task_control_request' ORDER BY event_id")
    [accepted] = [event for event in rows if json.loads(event["payload_json"])["phase"] == "accepted"]
    [refused] = [json.loads(event["payload_json"]) for event in rows if json.loads(event["payload_json"])["phase"] == "rejected"]
    assert refused["target"]["generation"] == 0 and refused["observed"]["generation"] == 1
    assert refused["error"] == "stop_generation_changed"
    with pytest.raises(JobError, match="stop_generation_changed"):
        await task_control.reconcile_stop_request(c.manager, c.actor, accepted["event_id"])
    assert await c.manager.get_job(c.actor, c.target.job.job_id) == current
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='task_control_request' ORDER BY event_id") == rows
    await session_obligations.verify(c.manager, c.actor)




@pytest.mark.parametrize("old_protocol", [25, 26])
async def test_prior_protocol_cannot_open_with_source_control_acknowledgements(registry_store, old_protocol):
    from power_loop.runtime.store.schema import StoreSchemaError
    from power_loop.runtime.store.task_schema import TASK_PROTOCOL_VERSION

    store, connect = registry_store
    await store._db.execute(f"UPDATE {store.table_prefix}task_store SET protocol_version=? WHERE id=1", (old_protocol,))
    try:
        with pytest.raises(StoreSchemaError):
            await connect()
        assert (await store._db.fetchone(f"SELECT protocol_version FROM {store.table_prefix}task_store"))["protocol_version"] == old_protocol
    finally:
        await store._db.execute(f"UPDATE {store.table_prefix}task_store SET protocol_version=? WHERE id=1", (TASK_PROTOCOL_VERSION,))


async def request_records(c, manager=None):
    manager = manager or c.manager
    rows = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='task_control_request' ORDER BY event_id")
    accepted = [row for row in rows if json.loads(row["payload_json"])["phase"] == "accepted"]
    completed = [row for row in rows if json.loads(row["payload_json"])["phase"] == "completed"]
    stops = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE action='stop'")
    pending = await manager.db.fetchall(f"SELECT * FROM {manager.prefix}session_obligations WHERE kind='control_request'")
    return accepted, stops, completed, pending


async def spawn_stop(c, tmp_path, mode, *, label="stop", caller=None, target=None, ref=None, call_id="stop-old-child"):
    import asyncio
    import os
    import sys
    from pathlib import Path

    engine = c.store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if engine == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if engine == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    ready, release, output = (tmp_path / (label+suffix) for suffix in (".ready", ".release", ".json"))
    context = {"actor": asdict(c.actor), "caller": asdict(caller or c.caller),
        "target": target or c.target.job.job_id, "ref": (ref or c.ref).to_dict(), "call_id": call_id,
        "now": c.manager.now()}
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._stop_request_worker",
        dsn, c.manager.prefix, json.dumps(context), mode, str(ready), str(release), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    return process, ready, release, output


@pytest.mark.parametrize("phase", ["accepted", "target", "completed"])
@pytest.mark.parametrize("committed", [False, True])
async def test_actual_process_death_at_each_stop_commit_preserves_originals_and_retries_without_execution(
        registry_store, tmp_path, phase, committed):
    import asyncio

    c = await cross_control(*registry_store, now=lambda: 1_000_000)
    originals = await c.store.load_all_messages(c.actor.session_id)
    process, ready, release, output = await spawn_stop(c, tmp_path, phase if committed else "uncommitted-"+phase)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 77, stdout.decode()
        assert json.loads(ready.read_text()) == {"phase": phase, "committed": committed}
        assert not output.exists()
        accepted, stops, completed, pending = await request_records(c)
        has_acceptance = phase != "accepted" or committed
        has_outcome = phase == "completed" or phase == "target" and committed
        has_completion = phase == "completed" and committed
        assert (len(accepted), len(stops), len(completed), len(pending)) == (
            int(has_acceptance), int(has_outcome), int(has_completion), int(has_acceptance and not has_completion))
        target = await c.manager.get_job(c.actor, c.target.job.job_id)
        assert (target.cancel_requested_at is not None) == has_outcome
        assert target.generation == 1 and target.resource_state == "running" and target.outcome is None
        await session_obligations.verify(c.manager, c.actor)
        before = {row["event_id"]: row for row in [*accepted, *stops, *completed]}
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        if has_acceptance:
            # The source may have physically exited after publishing its
            # request. Recovery needs no new source lease, model or callback.
            await cold.close_model_execution(c.actor, c.caller, evidence={"exit": "completed"})
            operation = await task_control.reconcile_stop_request(cold, c.actor, accepted[0]["event_id"])
        else:
            key = cold._origin_key(c.actor, "stop", c.ref, "stop-old-child", None)
            with pytest.raises(JobError, match="stop_request_original_missing"):
                await task_control.reconcile_stop_request(cold, c.actor, task_control._request_id(key))
            assert (await cold.get_job(c.actor, c.target.job.job_id)).cancel_requested_at is None
            operation, replayed = await c.stop(cold)
            assert not replayed
        after = await request_records(c, cold)
        assert tuple(map(len, after)) == (1, 1, 1, 0)
        for row in [*after[0], *after[1], *after[2]]:
            if row["event_id"] in before:
                assert row == before[row["event_id"]]
        assert await c.stop(cold) == (operation, True)
        assert tuple(map(len, await request_records(c, cold))) == (1, 1, 1, 0)
        assert not operation.result["response"]["resource_stop_confirmed"]
        assert (await cold.get_job(c.actor, c.target.job.job_id)).resource_state == "running"
        assert await c.store.load_all_messages(c.actor.session_id) == originals
        assert (await cold.get_job(c.actor, c.caller.job_id)).generation == 1
        await session_obligations.verify(cold, c.actor)
    finally:
        release.touch()
        await stop_worker(process)


@pytest.mark.parametrize("phase", ["accepted", "target", "completed"])
@pytest.mark.parametrize("committed", [False, True])
async def test_each_stop_commit_rollback_or_ack_loss_keeps_source_barrier_and_original_receipts(
        registry_store, monkeypatch, phase, committed):
    from power_loop.runtime.jobs import TRUST_OWN_WRITES

    c = await cross_control(*registry_store, now=lambda: 1_000_000)
    transaction = c.manager.db.transaction
    fired = []

    async def current_phase(tx):
        records = await tx.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='task_control_request'")
        values = {json.loads(row["payload_json"])["phase"] for row in records}
        if "completed" in values:
            return "completed"
        if await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE action='stop'"):
            return "target"
        return "accepted" if values else None

    @asynccontextmanager
    async def fault():
        async with transaction() as tx:
            yield tx
            hit = not fired and await current_phase(tx) == phase
            if hit and not committed:
                fired.append(True)
                raise OSError("stop transaction rolled back")
        if hit and committed:
            fired.append(True)
            raise OSError("stop committed; ACK lost")

    token = TRUST_OWN_WRITES.set(True)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(c.manager.db, "transaction", fault)
            with pytest.raises(OSError, match="rolled back|ACK lost"):
                await c.stop()
        assert fired == [True]
        accepted, stops, completed, pending = await request_records(c)
        has_acceptance = phase != "accepted" or committed
        has_outcome = phase == "completed" or phase == "target" and committed
        has_completion = phase == "completed" and committed
        assert tuple(map(len, (accepted, stops, completed, pending))) == (
            int(has_acceptance), int(has_outcome), int(has_completion), int(has_acceptance and not has_completion))
        if pending:
            # Invalidate a cached absence even if commit happened but its ACK
            # never reached JobManager's in-memory commit hooks.
            await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed"})
            await c.manager.finalize_supervision(c.actor, c.caller.job_id, generation=1)
        original = {row["event_id"]: row for row in [*accepted, *stops, *completed]}
        operation, replayed = await c.stop()
        assert replayed == has_acceptance
        after = await request_records(c)
        assert tuple(map(len, after)) == (1, 1, 1, 0)
        for row in [*after[0], *after[1], *after[2]]:
            if row["event_id"] in original:
                assert row == original[row["event_id"]]
        assert await c.stop() == (operation, True)
        assert (await c.manager.get_job(c.actor, c.target.job.job_id)).resource_state == "running"
        await session_obligations.verify(c.manager, c.actor)
    finally:
        TRUST_OWN_WRITES.reset(token)


async def test_actual_opposite_direction_stop_processes_finish_from_independently_committed_source_requests(
        registry_store, tmp_path):
    import asyncio

    store, connect = registry_store
    manager = JobManager(store, now_ms=lambda: 1_000_000)
    actor = JobActor("stop-trees", "host", await store.create_session())
    roots, callers, refs = [], [], []
    for index in (1, 2):
        registration = await manager.register_job(actor, kind="send", session_id=actor.session_id,
            system_key="opposite-root-"+str(index), request={"send_index": index})
        roots.append(registration.job.job_id)
        callers.append(await manager.claim_job(actor, roots[-1], executor_id="opposite-"+str(index),
            boot_id="boot", expected_version=1))
    for index in (0, 1):
        seq = await store.append_message(actor.session_id, role="assistant", send_index=index+1,
            tool_calls=[{"id": "opposite-"+str(index), "type": "function", "function": {"name": "jobs",
                "arguments": json.dumps({"action": "stop", "job_id": roots[1-index]})}}])
        refs.append(await store.message_ref(actor.session_id, seq))
    c = SimpleNamespace(store=store, manager=manager, actor=actor, caller=callers[0], ref=refs[0],
        target=SimpleNamespace(job=SimpleNamespace(job_id=roots[1])))
    workers = []
    try:
        for index in (0, 1):
            workers.append(await spawn_stop(c, tmp_path, "hold-accepted", label="opposite-"+str(index),
                caller=callers[index], target=roots[1-index], ref=refs[index], call_id="opposite-"+str(index)))
        await asyncio.gather(*(wait_ready(process, ready) for process, ready, _, _ in workers))
        assert tuple(map(len, await request_records(c))) == (2, 0, 0, 2)
        for root in roots:
            assert (await manager.get_job(actor, root)).cancel_requested_at is None
        for _, _, release, _ in workers:
            release.touch()
        results = await asyncio.wait_for(asyncio.gather(*(process.communicate() for process, _, _, _ in workers)), 30)
        for (process, _, _, output), (stdout, _) in zip(workers, results, strict=True):
            assert process.returncode == 0, stdout.decode()
            result = json.loads(output.read_text())
            assert not result["replayed"] and result["response"]["cancel_requested"]
            assert not result["response"]["resource_stop_confirmed"]
        assert tuple(map(len, await request_records(c))) == (2, 2, 2, 0)
        cold = JobManager(await connect(), now_ms=manager.now)
        for index in (0, 1):
            result, replayed = await task_control.request_stop(cold, actor, roots[1-index], caller=callers[index],
                call_ref=refs[index], tool_call_id="opposite-"+str(index))
            assert replayed and result.result["response"]["cancel_requested"]
            row = await cold.get_job(actor, roots[index])
            assert row.cancel_requested_at is not None and row.generation == 1 and row.resource_state == "running"
        assert len(await store.load_all_messages(actor.session_id)) == 2
        await session_obligations.verify(cold, actor)
    finally:
        for process, _, release, _ in workers:
            release.touch()
            await stop_worker(process)
