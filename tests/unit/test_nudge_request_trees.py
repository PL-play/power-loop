"""Cross-tree inputs retain source permission until the target acknowledges it."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import asdict

import pytest

from power_loop.runtime import control_requests, nudges, session_obligations
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker, wait_ready
from .test_job_registry_192 import registry_store, runtime
from .test_nudge_runtime_192 import nudge_case

__all__ = ["registry_store", "runtime", "nudge_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def lose_input_ack(manager, monkeypatch, phase):
    """One genuine DB commit boundary, also used by both native input adapters."""
    from power_loop.runtime.managed_effects import _ref
    from power_loop.runtime.tool_call_plans import identity, start_id

    transaction, lost = manager.db.transaction, []

    @asynccontextmanager
    async def boundary():
        async with transaction() as tx:
            yield tx
            rows = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='task_control_request'")
            permissions = [row for row in rows if (lambda data: data["phase"] == "accepted" and data["action"] == "nudge")(
                json.loads(row["payload_json"]))]
            current, grant = None, permissions[0] if permissions else None
            if grant is not None:
                if any(row["event_id"] == control_requests.completion_id(grant["event_id"]) for row in rows):
                    current = "completed"
                elif await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE action='nudge'"):
                    current = "target"
                else:
                    current = "accepted"
            hit = not lost and current == phase
            if hit:
                lost.append(phase)
                request = json.loads(grant["payload_json"])["request"]
                source = await manager._row(tx, grant["job_id"])
                dispatched = start_id(identity(manager, source, _ref(request["call_ref"])), request["tool_call_id"])
                assert await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=? "
                    "AND event_type='tool_call_dispatch_started'", (dispatched,))
        if hit:
            raise OSError("input "+phase+" committed; ACK lost")

    monkeypatch.setattr(manager.db, "transaction", boundary)
    return lost


async def test_cross_tree_nudge_never_reads_or_writes_two_roots_in_one_writer(nudge_case, monkeypatch):
    c = nudge_case
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
        assert len(footprint) <= 1, "source permission and target input must be separate tree writers"
        footprints.append(footprint)

    with monkeypatch.context() as patch:
        patch.setattr(ControlTransaction, "fetchone", fetchone)
        patch.setattr(ControlTransaction, "fetchall", fetchall)
        patch.setattr(c.manager, "_write", scoped)
        operation, replayed = await c.accept()
    assert not replayed and operation.result["status"] == "accepted"
    assert footprints == [{roots[c.caller.job_id]}, {roots[c.target.job_id]}, {roots[c.caller.job_id]}]
    assert (await c.store.inbox_counts(c.sid))["pending"] == 1


async def records(c, manager=None):
    manager = manager or c.manager
    rows = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='task_control_request' ORDER BY event_id")
    phases = {phase: [row for row in rows if json.loads(row["payload_json"])["phase"] == phase]
        for phase in ("accepted", "completed", "rejected")}
    phases["inputs"] = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE action='nudge'")
    phases["pending"] = await manager.db.fetchall(f"SELECT * FROM {manager.prefix}session_obligations WHERE kind='control_request'")
    return phases


async def test_source_exit_after_permission_keeps_a_pending_barrier_until_the_original_input_is_acknowledged(nudge_case, monkeypatch):
    c = nudge_case
    apply = nudges._apply_nudge_request

    async def exit_source(owner, actor, event, data):
        assert owner._control_intents.get() is None
        await owner.close_model_execution(actor, c.caller, evidence={"exit": "completed"})
        found = await records(c)
        assert [row["source_id"] for row in found["pending"]] == [event["event_id"]]
        assert not found["inputs"] and not found["completed"]
        with pytest.raises(JobError, match="task_control_request_unresolved"):
            await owner.finalize_supervision(actor, c.caller.job_id, generation=1)
        return await apply(owner, actor, event, data)

    with monkeypatch.context() as patch:
        patch.setattr(nudges, "_apply_nudge_request", exit_source)
        operation, replayed = await c.accept()
    assert not replayed and operation.result["status"] == "accepted"
    found = await records(c)
    assert {key: len(value) for key, value in found.items()} == {
        "accepted": 1, "completed": 1, "rejected": 0, "inputs": 1, "pending": 0}
    await c.manager.finalize_supervision(c.actor, c.caller.job_id, generation=1)
    assert await c.accept() == (operation, True)
    assert (await c.store.inbox_counts(c.sid))["pending"] == 1
    assert (await c.manager.get_job(c.actor, c.target.job_id)).generation == 1
    await session_obligations.verify(c.manager, c.actor)


async def test_closed_source_cannot_create_a_new_permission_from_a_persisted_call(nudge_case):
    c = nudge_case
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed"})
    with pytest.raises(JobError, match="intake_closed|not_running"):
        await c.accept()
    assert not any((await records(c)).values())
    assert (await c.store.inbox_counts(c.sid))["pending"] == 0
    key = c.manager._origin_key(c.actor, "nudge", c.origin, "nudge-call", None)
    with pytest.raises(JobError, match="nudge_request_original_missing"):
        await nudges.reconcile_request(c.manager, c.actor, control_requests.identity("nudge", key))


async def test_target_stop_after_permission_is_acknowledged_as_a_refusal_without_touching_the_new_state(nudge_case, monkeypatch):
    c = nudge_case
    apply = nudges._apply_nudge_request
    stopped = []

    async def stop_before_apply(owner, actor, event, data):
        await owner.request_cancel(actor, c.target.job_id, system_key="stop-before-input", reason="stop")
        stopped.append(await owner.get_job(actor, c.target.job_id))
        return await apply(owner, actor, event, data)

    with monkeypatch.context() as patch:
        patch.setattr(nudges, "_apply_nudge_request", stop_before_apply)
        with pytest.raises(JobError, match="cancel_requested"):
            await c.accept()
    assert await c.manager.get_job(c.actor, c.target.job_id) == stopped[0]
    found = await records(c)
    assert {key: len(value) for key, value in found.items()} == {
        "accepted": 1, "completed": 1, "rejected": 1, "inputs": 0, "pending": 0}
    assert (await c.store.inbox_counts(c.sid))["pending"] == 0
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    before = await records(c, cold)
    with pytest.raises(JobError, match="cancel_requested"):
        await nudges.reconcile_request(cold, c.actor, found["accepted"][0]["event_id"])
    with pytest.raises(JobError, match="cancel_requested"):
        await c.accept()
    assert await records(c, cold) == before
    assert await cold.get_job(c.actor, c.target.job_id) == stopped[0]
    await session_obligations.verify(cold, c.actor)


async def test_peer_completion_updates_the_original_sends_committed_input_history(nudge_case, monkeypatch):
    from power_loop.runtime.jobs import TRUST_OWN_WRITES

    c = nudge_case
    token = TRUST_OWN_WRITES.set(True)
    try:
        peer = JobManager(await c.connect(), now_ms=c.manager.now)
        apply, peers = nudges._apply_nudge_request, []

        async def complete_in_peer(owner, actor, event, data):
            if owner is c.manager and not peers:
                peers.append(await peer.accept_nudge(actor, c.target, caller=c.caller, call_ref=c.origin,
                    tool_call_id="nudge-call", session_id=c.sid, message="NUDGED-42"))
            return await apply(owner, actor, event, data)

        with monkeypatch.context() as patch:
            patch.setattr(nudges, "_apply_nudge_request", complete_in_peer)
            operation, replayed = await c.accept()
        assert not replayed and peers == [(operation, True)]
        assert {key: len(value) for key, value in (await records(c)).items()} == {
            "accepted": 1, "completed": 1, "rejected": 0, "inputs": 1, "pending": 0}
        await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed"})
        await c.manager.finalize_supervision(c.actor, c.caller.job_id, generation=1)
        assert (await c.store.inbox_counts(c.sid))["pending"] == 1
        await session_obligations.verify(peer, c.actor)
    finally:
        TRUST_OWN_WRITES.reset(token)


@pytest.mark.parametrize("damage", ["missing", "request", "error"])
async def test_cold_rebuild_requires_the_original_target_refusal_before_removing_a_source_barrier(nudge_case, monkeypatch, damage):
    from power_loop.runtime.jobs import _hash, _json

    c = nudge_case
    apply = nudges._apply_nudge_request

    async def stop_before_apply(owner, actor, event, data):
        await owner.request_cancel(actor, c.target.job_id, system_key="stop-before-cold-refusal", reason="stop")
        return await apply(owner, actor, event, data)

    with monkeypatch.context() as patch:
        patch.setattr(nudges, "_apply_nudge_request", stop_before_apply)
        with pytest.raises(JobError, match="cancel_requested"):
            await c.accept()
    found = await records(c)
    [accepted], [refused] = found["accepted"], found["rejected"]
    target = await c.manager.get_job(c.actor, c.target.job_id)
    table = c.manager.prefix+"session_obligations"
    entry = (c.actor.session_id, "control_request", accepted["event_id"], c.caller.job_id, accepted["job_version"])
    await c.manager.db.execute(f"INSERT INTO {table} ({','.join(session_obligations.COLUMNS)}) VALUES (?,?,?,?,?)", entry)
    if damage == "missing":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (refused["event_id"],))
        error = "nudge_request_outcome_missing"
    else:
        value = json.loads(refused["payload_json"])
        if damage == "request":
            value["request"]["event_id"] = "a-different-permission"
        else:
            value["error"] = "a-nonexistent-target-barrier"
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?",
            (_json(value), _hash(value), refused["event_id"]))
        error = "control_request_rejection_corrupt"
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    before = await c.manager.db.fetchall(f"SELECT * FROM {table}")
    try:
        with pytest.raises(JobError, match=error):
            await session_obligations.verify(cold, c.actor)
        with pytest.raises(JobError, match=error):
            await session_obligations.rebuild(cold)
        assert await c.manager.db.fetchall(f"SELECT * FROM {table}") == before
        assert await cold.get_job(c.actor, c.target.job_id) == target
        assert (await c.store.inbox_counts(c.sid))["pending"] == 0
    finally:
        if damage == "missing":
            await c.manager.db.execute(f"INSERT INTO {c.manager.events} ({','.join(refused)}) VALUES "
                f"({','.join('?' for _ in refused)})", tuple(refused.values()))
        else:
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?",
                (refused["payload_json"], refused["request_hash"], refused["event_id"]))
        await session_obligations.rebuild(cold)
    assert await records(c) == found
    await session_obligations.verify(cold, c.actor)


@pytest.mark.parametrize("phase", ["accepted", "target", "completed"])
@pytest.mark.parametrize("committed", [False, True])
async def test_each_input_transaction_rollback_or_unknown_ack_retains_one_permission_and_one_original_input(
        nudge_case, monkeypatch, phase, committed):
    c = nudge_case
    transaction = c.manager.db.transaction
    failed = []

    @asynccontextmanager
    async def boundary():
        async with transaction() as tx:
            yield tx
            events = await tx.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='task_control_request'")
            phases = {json.loads(event["payload_json"])["phase"] for event in events}
            current = "completed" if "completed" in phases else "target" if await tx.fetchone(
                f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'") else "accepted" if phases else None
            hit = not failed and current == phase
            if hit:
                failed.append(phase)
                if not committed:
                    raise OSError("input transaction rolled back")
        if hit:
            raise OSError("input transaction committed; ACK lost")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, "transaction", boundary)
        with pytest.raises(OSError, match="rolled back|ACK lost"):
            await c.accept()
    assert failed == [phase]
    found = await records(c)
    permitted = phase != "accepted" or committed
    applied = phase == "completed" or phase == "target" and committed
    completed = phase == "completed" and committed
    assert {key: len(value) for key, value in found.items()} == {
        "accepted": int(permitted), "completed": int(completed), "rejected": 0,
        "inputs": int(applied), "pending": int(permitted and not completed)}
    assert (await c.store.inbox_counts(c.sid))["pending"] == int(applied)
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    if permitted:
        await cold.close_model_execution(c.actor, c.caller, evidence={"exit": "completed"})
        operation = await nudges.reconcile_request(cold, c.actor, found["accepted"][0]["event_id"])
    else:
        operation, replayed = await c.accept()
        assert not replayed
    after = await records(c)
    assert {key: len(value) for key, value in after.items()} == {
        "accepted": 1, "completed": 1, "rejected": 0, "inputs": 1, "pending": 0}
    for key in ("accepted", "completed", "inputs"):
        if found[key]:
            assert found[key] == after[key]
    assert await c.accept() == (operation, True)
    assert (await c.store.inbox_counts(c.sid))["pending"] == 1
    assert (await cold.get_job(c.actor, c.target.job_id)).generation == 1
    await session_obligations.verify(cold, c.actor)


async def spawn_nudge(c, tmp_path, mode, *, label="nudge", actor=None, caller=None, target=None,
        ref=None, call_id="nudge-call", sid=None, message="NUDGED-42", stop_target=False):
    import asyncio
    import os
    import sys
    from pathlib import Path

    engine = c.store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if engine == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if engine == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    ready, release, output = (tmp_path / (label+suffix) for suffix in (".ready", ".release", ".json"))
    context = {"actor": asdict(actor or c.actor), "caller": asdict(caller or c.caller), "target": asdict(target or c.target),
        "ref": (ref or c.origin).to_dict(), "call_id": call_id, "sid": sid or c.sid, "message": message,
        "now": c.manager.now(), "stop_target": stop_target}
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._nudge_request_worker",
        dsn, c.manager.prefix, json.dumps(context), mode, str(ready), str(release), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    return process, ready, release, output


@pytest.mark.parametrize("phase", ["accepted", "target", "completed"])
@pytest.mark.parametrize("committed", [False, True])
async def test_real_process_death_at_each_input_commit_recovers_without_a_model_callback_or_second_enqueue(
        nudge_case, tmp_path, phase, committed):
    import asyncio

    c = nudge_case
    raw = await c.store.load_all_messages(c.actor.session_id)
    process, ready, release, output = await spawn_nudge(c, tmp_path, phase if committed else "uncommitted-"+phase)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 78, stdout.decode()
        assert json.loads(ready.read_text()) == {"phase": phase, "committed": committed}
        assert not output.exists()
        found = await records(c)
        permitted = phase != "accepted" or committed
        applied = phase == "completed" or phase == "target" and committed
        completed = phase == "completed" and committed
        assert {key: len(value) for key, value in found.items()} == {
            "accepted": int(permitted), "completed": int(completed), "rejected": 0,
            "inputs": int(applied), "pending": int(permitted and not completed)}
        assert (await c.store.inbox_counts(c.sid))["pending"] == int(applied)
        await session_obligations.verify(c.manager, c.actor)
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        if permitted:
            await cold.close_model_execution(c.actor, c.caller, evidence={"exit": "completed"})
            operation = await nudges.reconcile_request(cold, c.actor, found["accepted"][0]["event_id"])
        else:
            key = cold._origin_key(c.actor, "nudge", c.origin, "nudge-call", None)
            with pytest.raises(JobError, match="nudge_request_original_missing"):
                await nudges.reconcile_request(cold, c.actor, control_requests.identity("nudge", key))
            assert (await c.store.inbox_counts(c.sid))["pending"] == 0
            operation, replayed = await c.accept()
            assert not replayed
        after = await records(c, cold)
        assert {key: len(value) for key, value in after.items()} == {
            "accepted": 1, "completed": 1, "rejected": 0, "inputs": 1, "pending": 0}
        for name in ("accepted", "completed", "inputs"):
            if found[name]:
                assert after[name] == found[name]
        assert await c.accept() == (operation, True)
        assert (await c.store.inbox_counts(c.sid))["pending"] == 1
        assert (await cold.get_job(c.actor, c.target.job_id)).generation == 1
        assert (await cold.get_job(c.actor, c.target.job_id)).resource_state == "running"
        assert await c.store.load_all_messages(c.actor.session_id) == raw
        assert not await c.store.load_all_messages(c.sid)
        await session_obligations.verify(cold, c.actor)
    finally:
        release.touch()
        await stop_worker(process)


@pytest.mark.parametrize("committed", [False, True])
async def test_real_target_refusal_crash_recovers_the_negative_acknowledgement_without_inventing_input(nudge_case, tmp_path, committed):
    import asyncio

    c = nudge_case
    process, ready, release, output = await spawn_nudge(c, tmp_path, "target" if committed else "uncommitted-target", stop_target=True)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 78, stdout.decode()
        assert json.loads(ready.read_text()) == {"phase": "target", "committed": committed}
        assert not output.exists()
        target = await c.manager.get_job(c.actor, c.target.job_id)
        assert target.cancel_requested_at is not None and target.resource_state == "running"
        found = await records(c)
        assert {key: len(value) for key, value in found.items()} == {
            "accepted": 1, "completed": 0, "rejected": int(committed), "inputs": 0, "pending": 1}
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        await cold.close_model_execution(c.actor, c.caller, evidence={"exit": "completed"})
        with pytest.raises(JobError, match="cancel_requested"):
            await nudges.reconcile_request(cold, c.actor, found["accepted"][0]["event_id"])
        after = await records(c, cold)
        assert {key: len(value) for key, value in after.items()} == {
            "accepted": 1, "completed": 1, "rejected": 1, "inputs": 0, "pending": 0}
        assert after["accepted"] == found["accepted"]
        if committed:
            assert after["rejected"] == found["rejected"]
        assert await cold.get_job(c.actor, c.target.job_id) == target
        assert (await c.store.inbox_counts(c.sid))["pending"] == 0
        with pytest.raises(JobError, match="cancel_requested"):
            await c.accept()
        assert await records(c, cold) == after
        await session_obligations.verify(cold, c.actor)
    finally:
        release.touch()
        await stop_worker(process)


async def test_real_opposite_direction_input_processes_never_hold_both_task_trees(registry_store, tmp_path):
    import asyncio
    from dataclasses import replace
    from types import SimpleNamespace

    store, connect = registry_store
    manager = JobManager(store, now_ms=lambda: 1_000_000)
    actor = JobActor("input-trees", "host", await store.create_session())
    sources, targets, sids, refs = [], [], [], []
    for index in (0, 1):
        source = await manager.register_job(actor, kind="send", session_id=actor.session_id,
            system_key="opposite-input-root-"+str(index), request={"send_index": index+1})
        lease = await manager.claim_job(actor, source.job.job_id, executor_id="source-"+str(index), boot_id="boot", expected_version=1)
        sources.append(lease)
        target = await manager.register_job(actor, kind="subagent", parent=lease, independent=True,
            system_key="opposite-input-target-"+str(index))
        sid = await store.create_session(parent_session_id=actor.session_id,
            metadata={"execution_role": "subagent", "standalone_subagent": True})
        bound = await manager.bind_session(actor, target.job.job_id, sid)
        targets.append(await manager.claim_job(replace(actor, session_id=sid), bound.job_id,
            executor_id="target-"+str(index), boot_id="boot", expected_version=bound.version))
        sids.append(sid)
    for index in (0, 1):
        seq = await store.append_message(actor.session_id, role="assistant", send_index=index+1, round_index=0,
            tool_calls=[{"id": "opposite-input-"+str(index), "type": "function", "function": {"name": "spawn_agent", "arguments": "{}"}}])
        refs.append(await store.message_ref(actor.session_id, seq))
    c = SimpleNamespace(store=store, manager=manager, actor=actor, caller=sources[0], target=targets[0], sid=sids[0], origin=refs[0])
    workers = []
    try:
        for index in (0, 1):
            workers.append(await spawn_nudge(c, tmp_path, "hold-accepted", label="opposite-input-"+str(index),
                caller=sources[index], target=targets[1-index], sid=sids[1-index], ref=refs[index],
                call_id="opposite-input-"+str(index), message="original-input-"+str(index)))
        await asyncio.gather(*(wait_ready(process, ready) for process, ready, _, _ in workers))
        found = await records(c)
        assert {key: len(value) for key, value in found.items()} == {
            "accepted": 2, "completed": 0, "rejected": 0, "inputs": 0, "pending": 2}
        for _, _, release, _ in workers:
            release.touch()
        outputs = await asyncio.wait_for(asyncio.gather(*(process.communicate() for process, _, _, _ in workers)), 30)
        for (process, _, _, output), (stdout, _) in zip(workers, outputs, strict=True):
            assert process.returncode == 0, stdout.decode()
            result = json.loads(output.read_text())
            assert not result["replayed"] and result["operation_id"]
        after = await records(c)
        assert {key: len(value) for key, value in after.items()} == {
            "accepted": 2, "completed": 2, "rejected": 0, "inputs": 2, "pending": 0}
        for index in (0, 1):
            [item] = await store.inbox_items(sids[1-index])
            assert item["content"] == "original-input-"+str(index) and item["status"] == "pending"
            assert (await manager.get_job(actor, targets[index].job_id)).generation == 1
        assert len(await store.load_all_messages(actor.session_id)) == 2
        cold = JobManager(await connect(), now_ms=manager.now)
        await session_obligations.verify(cold, actor)
    finally:
        for process, _, release, _ in workers:
            release.touch()
            await stop_worker(process)
