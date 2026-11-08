"""A durable executor barrier precedes independently repeatable tree stops."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobError, JobManager

from .test_executor_registry_192 import boot, root, start
from .test_job_registry_192 import registry_store

__all__ = ["boot", "registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def stop_ids(manager):
    return {row["event_id"] for row in await manager.db.fetchall(
        f"SELECT event_id FROM {manager.events} WHERE action='stop'")}


async def test_admission_barrier_ack_loss_blocks_claims_and_cold_retry_finishes(boot, monkeypatch):
    c = boot
    await start(c)
    first, _ = await root(c, key="first")
    second, _ = await root(c, key="second")
    write = c.manager._write
    lost = []

    @asynccontextmanager
    async def lose_barrier_ack():
        async with write() as tx:
            yield tx
        if not lost:
            lost.append(True)
            raise OSError("barrier commit ACK lost")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager, "_write", lose_barrier_ack)
        with pytest.raises(OSError, match="barrier commit ACK lost"):
            await registry.drain(c.manager, c.identity, system_key="stop", reason="shutdown")
    assert lost == [True]
    assert (await registry.get(c.manager, c.identity))["state"] == "draining"
    assert await stop_ids(c.manager) == set()
    for reg in (first, second):
        assert (await c.manager.get_job(c.actor, reg.job.job_id)).cancel_requested_at is None
    pending = await c.manager.register_job(c.actor, kind="send", session_id=c.actor.session_id,
                                          system_key="after-barrier")
    with pytest.raises(JobError, match="executor_not_accepting"):
        await c.manager.claim_job(c.actor, pending.job.job_id, executor_id=c.identity.executor_id,
                                 boot_id=c.identity.boot_id, expected_version=1)
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    final = await registry.drain(cold, c.identity, system_key="stop", reason="shutdown")
    assert not final["effects"]["scope_incomplete"] and not final["physical_stop_confirmed"]
    assert {j["job_id"] for j in final["effects"]["executions"]} == {first.job.job_id, second.job.job_id}
    assert len(await stop_ids(cold)) == 2
    replay = await registry.drain(cold, c.identity, system_key="stop", reason="shutdown")
    assert replay["replayed"] and replay["executor"] == final["executor"] and replay["effects"] == final["effects"]
    assert len(await stop_ids(cold)) == 2


async def test_first_tree_commit_survives_failure_and_reuses_the_frozen_scope(boot, monkeypatch):
    c = boot
    await start(c)
    original_jobs = [(await root(c, key=key))[0].job.job_id for key in ("first", "second")]
    tree = registry._drain_tree
    committed = []

    async def fail_after_first_tree(*args, **kwargs):
        result = await tree(*args, **kwargs)
        committed.append(result[0])
        raise OSError("first tree committed; caller died")

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_drain_tree", fail_after_first_tree)
        with pytest.raises(OSError, match="first tree committed"):
            await registry.drain(c.manager, c.identity, system_key="stop", reason="shutdown")
    assert len(committed) == 1 and len(committed[0]) == 1
    assert len(await stop_ids(c.manager)) == 1
    jobs = [await c.manager.get_job(c.actor, job) for job in original_jobs]
    assert sum(job.cancel_requested_at is not None for job in jobs) == 1
    assert (await registry.get(c.manager, c.identity))["state"] == "draining"
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    final = await registry.drain(cold, c.identity, system_key="stop", reason="shutdown")
    assert {job["job_id"] for job in final["effects"]["executions"]} == set(original_jobs)
    assert set(committed[0]) <= set(final["effects"]["stop_operations"])
    assert set(final["effects"]["stop_operations"]) == await stop_ids(cold)
    assert len(await stop_ids(cold)) == 2 and not final["effects"]["scope_incomplete"]
    assert not final["physical_stop_confirmed"]


async def test_unclaimed_child_is_not_owned_or_cancelled_by_creator_executor_drain(boot):
    c = boot
    await start(c)
    _, parent = await root(c)
    reg = await c.manager.register_job(c.actor, kind="subagent", parent=parent,
        system_key="queued-independent", independent=True, session_id=c.actor.session_id)
    await c.manager.close_model_execution(c.actor, parent)
    final = await registry.drain(c.manager, c.identity, system_key="stop", reason="shutdown")
    assert reg.job.job_id not in {r["job_id"] for r in final["effects"]["executions"]}
    future = JobManager(await c.connect(), now_ms=c.manager.now)
    identity = replace(c.identity, executor_id="successor", boot_id="next-boot")
    await registry.register(future, identity, contract_hash="a"*64)
    await registry.activate(future, identity, expected_version=1, system_key="startup", lease_ms=300000)
    lease = await future.claim_job(c.actor, reg.job.job_id, executor_id=identity.executor_id,
        boot_id=identity.boot_id, expected_version=1)
    await future.check_execution(c.actor,lease)
    assert (await future.get_job(c.actor,reg.job.job_id)).cancel_requested_at is None


async def spawn_worker(c, tmp_path, mode, *, sid=None, identity=None):
    import asyncio
    import os
    import sys
    from pathlib import Path

    engine = c.store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if engine == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if engine == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    ready, release, output = (tmp_path / (mode + suffix) for suffix in (".ready", ".release", ".json"))
    identity = identity or c.identity
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._executor_drain_worker",
        dsn, c.manager.prefix, identity.tenant_id, identity.principal_id,
        identity.executor_id, identity.boot_id, sid or c.actor.session_id, str(c.now[0]), mode,
        str(ready), str(release), str(output), cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    return process, ready, release, output


async def wait_ready(process, ready):
    import asyncio
    async with asyncio.timeout(30):
        while not ready.exists():
            if process.returncode is not None:
                output, _ = await process.communicate()
                raise AssertionError(output.decode())
            await asyncio.sleep(.01)


async def stop_worker(process):
    if process.returncode is None:
        process.kill()
    await process.communicate()


async def test_real_process_death_after_one_tree_commit_recovers_without_duplicate_stops(boot, tmp_path):
    import asyncio
    import json

    c = boot
    await start(c)
    roots = [(await root(c, key=key))[0].job.job_id for key in ("first", "second")]
    process, ready, release, output = await spawn_worker(c, tmp_path, "crash-after-first")
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 73, stdout.decode()
        original = json.loads(ready.read_text())
        assert original["state"] == "first-tree-committed" and len(original["stop_ids"]) == 1
        assert not output.exists()
        assert (await registry.get(c.manager, c.identity))["state"] == "draining"
        jobs = [await c.manager.get_job(c.actor, job) for job in roots]
        assert sum(job.cancel_requested_at is not None for job in jobs) == 1
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        final = await registry.drain(cold, c.identity, system_key="process-stop", reason="shutdown")
        assert {row["job_id"] for row in final["effects"]["executions"]} == set(roots)
        assert set(original["stop_ids"]) <= set(final["effects"]["stop_operations"])
        assert len(await stop_ids(cold)) == 2 and not final["effects"]["scope_incomplete"]
        assert not final["physical_stop_confirmed"]
        replay = await registry.drain(cold, c.identity, system_key="process-stop", reason="shutdown")
        assert replay["replayed"] and replay["effects"] == final["effects"]
        assert len(await stop_ids(cold)) == 2
    finally:
        release.touch()
        await stop_worker(process)






async def test_tree_stop_rollback_preserves_barrier_and_frozen_scope_for_retry(boot, monkeypatch):
    c = boot
    await start(c)
    roots = [(await root(c, key=key))[0].job.job_id for key in ("first", "second")]
    transaction = c.manager.db.transaction
    fired = []

    @asynccontextmanager
    async def reject_tree_commit():
        async with transaction() as tx:
            yield tx
            if not fired and await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE action='stop'"):
                fired.append(True)
                raise OSError("tree stop commit failed")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, "transaction", reject_tree_commit)
        with pytest.raises(OSError, match="tree stop commit failed"):
            await registry.drain(c.manager, c.identity, system_key="stop", reason="shutdown")
    assert fired == [True]
    assert (await registry.get(c.manager, c.identity))["state"] == "draining"
    assert await stop_ids(c.manager) == set()
    for job in roots:
        assert (await c.manager.get_job(c.actor, job)).cancel_requested_at is None
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    final = await registry.drain(cold, c.identity, system_key="stop", reason="shutdown")
    assert {row["job_id"] for row in final["effects"]["executions"]} == set(roots)
    assert set(final["effects"]["stop_operations"]) == await stop_ids(cold)
    assert len(await stop_ids(cold)) == 2 and not final["effects"]["scope_incomplete"]


async def test_two_callers_complete_one_original_drain_and_one_stop_per_tree(boot):
    import asyncio

    c = boot
    await start(c)
    for key in ("first", "second"):
        await root(c, key=key)
    other = JobManager(await c.connect(), now_ms=c.manager.now)
    results = await asyncio.gather(*(registry.drain(manager, c.identity, system_key="stop", reason="shutdown")
                                    for manager in (c.manager, other)))
    assert results[0]["event_id"] == results[1]["event_id"]
    assert results[0]["executor"] == results[1]["executor"]
    assert results[0]["effects"] == results[1]["effects"]
    assert not results[0]["effects"]["scope_incomplete"]
    assert len(await stop_ids(c.manager)) == 2


async def test_cold_retry_refuses_a_missing_target_instead_of_completing_a_partial_scope(boot, monkeypatch):
    c = boot
    await start(c)
    reg, _ = await root(c)
    original_job = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (reg.job.job_id,))
    scope = registry._drain_scope

    async def lose_scope_ack(*args, **kwargs):
        await scope(*args, **kwargs)
        raise OSError("original scope committed; ACK lost")

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_drain_scope", lose_scope_ack)
        with pytest.raises(OSError, match="original scope committed"):
            await registry.drain(c.manager, c.identity, system_key="stop", reason="shutdown")
    await c.manager.db.execute(f"DELETE FROM {c.manager.jobs} WHERE job_id=?", (reg.job.job_id,))
    try:
        originals = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.prefix}task_executor_events ORDER BY event_id")
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        with pytest.raises(JobError, match="executor_drain_original_scope_missing"):
            await registry.drain(cold, c.identity, system_key="stop", reason="shutdown")
        assert await cold.db.fetchall(f"SELECT * FROM {cold.prefix}task_executor_events ORDER BY event_id") == originals
        assert await stop_ids(cold) == set()
        assert (await registry.get(cold, c.identity))["state"] == "draining"
    finally:
        columns = tuple(original_job)
        await c.manager.db.execute(
            f"INSERT INTO {c.manager.jobs} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            tuple(original_job[key] for key in columns))


async def test_original_model_exit_between_scope_freeze_and_stop_keeps_its_closed_phase(boot, monkeypatch):
    c = boot
    await start(c)
    reg, lease = await root(c)
    tree = registry._drain_tree
    completed = []

    async def exit_before_stop(*args, **kwargs):
        await c.manager.close_model_execution(c.actor, lease, evidence={"exit": "completed"})
        await c.manager.finalize_supervision(c.actor, lease.job_id, generation=lease.generation)
        completed.append(await c.manager.get_job(c.actor, reg.job.job_id))
        return await tree(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_drain_tree", exit_before_stop)
        final = await registry.drain(c.manager, c.identity, system_key="stop", reason="shutdown")
    [before] = completed
    after = await c.manager.get_job(c.actor, reg.job.job_id)
    assert after == before and after.resource_state == "stopped"
    assert after.cancel_requested_at is None
    assert not final["effects"]["scope_incomplete"] and not final["physical_stop_confirmed"]




@pytest.mark.parametrize("field", ["requested_execution", "executor_drain_operation", "scope_changed", "scope_closed"])
async def test_cold_retry_rejects_damaged_original_stop_without_completing_scope(boot, monkeypatch, field):
    import json

    c = boot
    await start(c)
    for key in ("first", "second"):
        await root(c, key=key)
    tree = registry._drain_tree

    async def lose_first_tree_ack(*args, **kwargs):
        await tree(*args, **kwargs)
        raise OSError("first tree committed; ACK lost")

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_drain_tree", lose_first_tree_ack)
        with pytest.raises(OSError, match="first tree committed"):
            await registry.drain(c.manager, c.identity, system_key="stop", reason="shutdown")
    original = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE action='stop'")
    damaged = json.loads(original["payload_json"])
    damaged[field] = None
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
                              (json.dumps(damaged), original["event_id"]))
    try:
        events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
        jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} ORDER BY job_id")
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        with pytest.raises(JobError, match="executor_drain_stop_evidence_corrupt"):
            await registry.drain(cold, c.identity, system_key="stop", reason="shutdown")
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == events
        assert await cold.db.fetchall(f"SELECT * FROM {cold.jobs} ORDER BY job_id") == jobs
        assert await stop_ids(cold) == {original["event_id"]}
        assert (await registry.get(cold, c.identity))["state"] == "draining"
    finally:
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
                                  (original["payload_json"], original["event_id"]))


async def test_real_opposite_tree_drains_commit_each_tree_before_waiting_for_the_other(boot, tmp_path):
    import asyncio
    import json

    c = boot
    await start(c)
    other = replace(c.identity, executor_id="other-executor", boot_id="other-boot")
    await start(c, other)
    roots, children = [], []
    for key in ("first", "second"):
        reg, lease = await root(c, key=key)
        child = await c.manager.register_job(c.actor, kind="subagent", parent=lease,
                                             system_key="independent-" + key, independent=True, session_id=c.actor.session_id)
        await c.manager.claim_job(c.actor, child.job.job_id, executor_id=other.executor_id,
                                 boot_id=other.boot_id, expected_version=1)
        roots.append(reg.job.job_id)
        children.append(child.job.job_id)
    forward = await spawn_worker(c, tmp_path, "opposite-forward")
    reverse = await spawn_worker(c, tmp_path, "opposite-reverse", identity=other)
    try:
        for process, _ready, _release, _output in (forward, reverse):
            stdout, _ = await asyncio.wait_for(process.communicate(), 30)
            assert process.returncode == 0, stdout.decode()
        first, second = (json.loads(worker[3].read_text()) for worker in (forward, reverse))
        assert first["tree_order"] == sorted(roots) and second["tree_order"] == sorted(roots, reverse=True)
        assert json.loads(forward[1].read_text())["root"] != json.loads(reverse[1].read_text())["root"]
        for value, expected in ((first, roots), (second, children)):
            receipt = value["receipt"]
            assert {row["job_id"] for row in receipt["effects"]["executions"]} == set(expected)
            assert len(set(receipt["effects"]["stop_operations"])) == 2
            assert not receipt["effects"]["scope_incomplete"] and not receipt["physical_stop_confirmed"]
        assert len(await stop_ids(c.manager)) == 4
        for job_id in (*roots, *children):
            job = await c.manager.get_job(c.actor, job_id)
            assert job.cancel_requested_at is not None and job.resource_state == "running"
            assert job.supervision_status == "open"
    finally:
        await stop_worker(forward[0])
        await stop_worker(reverse[0])
