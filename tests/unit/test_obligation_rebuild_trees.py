"""Cold full validation precedes repeatable, conditional per-tree index repair."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop.runtime import session_obligations as obligations
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_job_registry_192 import registry_store, runtime

__all__ = ["registry_store", "runtime"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def _result(manager, actor, key, *, parent=None):
    if parent is None:
        root = await manager.register_job(actor, kind="send", session_id=actor.session_id,
            system_key="rebuild-root-"+key)
        parent = await manager.claim_job(actor, root.job.job_id,
            executor_id="rebuild-root", boot_id="boot", expected_version=1)
    child = await manager.register_job(actor, kind="subagent", parent=parent, independent=True,
        system_key="rebuild-result-"+key)
    lease = await manager.claim_job(actor, child.job.job_id,
        executor_id="child", boot_id="boot", expected_version=1)
    await manager.record_execution_result(actor, lease, outcome="succeeded", result={"value": key}, resource_state="stopped")
    return child.job.job_id


async def _rows(manager):
    return await manager.db.fetchall(f"SELECT * FROM {manager.prefix}session_obligations ORDER BY session_id,kind,source_id")


async def _originals(manager):
    return {table: await manager.db.fetchall(f"SELECT * FROM {table} ORDER BY 1")
        for table in (manager.jobs, manager.events, manager.store.t.messages)}


async def test_rebuild_computes_outside_writer_and_repairs_only_one_tree_at_a_time(runtime, monkeypatch):
    _, manager, actor, _, parent, _, _ = runtime
    await _result(manager, actor, "one", parent=parent)
    await _result(manager, actor, "two")
    expected, originals = await _rows(manager), await _originals(manager)
    await manager.db.execute(f"DELETE FROM {manager.prefix}session_obligations")
    desired, entries, write = obligations._desired, obligations._write_entries, manager._write
    footprints = []

    async def outside(owner, reader, row, **kwargs):
        assert owner._control_intents.get() is None, "original validation/calculation belongs outside writer"
        return await desired(owner, reader, row, **kwargs)

    @asynccontextmanager
    async def recorded():
        async with write() as tx:
            roots = set()
            footprints.append(roots)
            tx._test_rebuild_roots = roots
            yield tx

    async def target(owner, tx, row, pending):
        tx._test_rebuild_roots.add(row["root_job_id"])
        return await entries(owner, tx, row, pending)

    async def forbidden_read(*args, **kwargs):
        raise AssertionError("rebuild writer must only publish its prepared conditional writes")

    with monkeypatch.context() as patch:
        patch.setattr(obligations, "_desired", outside)
        patch.setattr(obligations, "_write_entries", target)
        patch.setattr(manager, "_write", recorded)
        patch.setattr(ControlTransaction, "fetchone", forbidden_read)
        patch.setattr(ControlTransaction, "fetchall", forbidden_read)
        await obligations.rebuild(manager)
    assert len(footprints) == 2 and all(len(roots) == 1 for roots in footprints)
    assert len(set.union(*footprints)) == 2
    assert await _rows(manager) == expected
    assert await _originals(manager) == originals
    await obligations.verify(manager, actor)


async def test_rebuild_does_not_erase_new_roots_or_children_admitted_after_discovery(runtime, monkeypatch):
    _, manager, actor, _, parent, connect, _ = runtime
    first = await _result(manager, actor, "old", parent=parent)
    await manager.db.execute(f"DELETE FROM {manager.prefix}session_obligations")
    other = JobManager(await connect(), now_ms=manager.now)
    publish = obligations._rebuild_tree
    attempts, added = [], []

    async def admitted_before_publish(owner, root_id, plan):
        assert owner._control_intents.get() is None
        attempts.append(root_id)
        if not added:
            added.append(await _result(other, actor, "late-child", parent=parent))
            added.append(await _result(other, actor, "late-root"))
        return await publish(owner, root_id, plan)

    with monkeypatch.context() as patch:
        patch.setattr(obligations, "_rebuild_tree", admitted_before_publish)
        await obligations.rebuild(manager)
    assert len(attempts) == 2
    assert {row["job_id"] for row in await _rows(manager)} == {first, *added}
    await obligations.verify(manager, actor)


async def test_rebuild_retries_a_real_cancel_without_replacing_its_original(runtime, monkeypatch):
    _, manager, actor, _, parent, connect, _ = runtime
    target = await _result(manager, actor, "cancelled", parent=parent)
    await manager.db.execute(f"DELETE FROM {manager.prefix}session_obligations")
    other = JobManager(await connect(), now_ms=manager.now)
    publish = obligations._rebuild_tree
    attempts, stopped = [], []

    async def stopped_before_publish(owner, root_id, plan):
        attempts.append(root_id)
        if not stopped:
            stopped.append(await other.request_cancel(actor, target,
                system_key="cancel-during-rebuild", reason="real concurrent stop"))
        return await publish(owner, root_id, plan)

    with monkeypatch.context() as patch:
        patch.setattr(obligations, "_rebuild_tree", stopped_before_publish)
        await obligations.rebuild(manager)
    assert len(attempts) == 2
    row = await manager._row(manager.db, target)
    assert row["cancel_requested_at"] is not None and row["cancel_reason"] == "real concurrent stop"
    assert row["outcome"] == "succeeded" and row["resource_state"] == "stopped"
    await obligations.verify(manager, actor)


async def test_committed_first_tree_survives_failure_of_the_second_and_retry_is_mechanical(runtime, monkeypatch):
    _, manager, actor, _, parent, _, _ = runtime
    await _result(manager, actor, "first", parent=parent)
    await _result(manager, actor, "second")
    expected, originals = await _rows(manager), await _originals(manager)
    await manager.db.execute(f"DELETE FROM {manager.prefix}session_obligations")
    entries, roots = obligations._write_entries, []

    async def fail_second(owner, tx, row, pending):
        roots.append(row["root_job_id"])
        result = await entries(owner, tx, row, pending)
        if len(roots) == 2:
            raise OSError("second root publication unavailable")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(obligations, "_write_entries", fail_second)
        with pytest.raises(OSError, match="second root publication unavailable"):
            await obligations.rebuild(manager)
    partial = await _rows(manager)
    assert len(roots) == 2 and len(partial) == 1 and partial[0] in expected
    assert await _originals(manager) == originals
    await obligations.rebuild(manager)
    await obligations.rebuild(manager)
    assert await _rows(manager) == expected
    assert await _originals(manager) == originals
    await obligations.verify(manager, actor)


async def test_new_original_notice_without_job_version_change_invalidates_the_rebuild_plan(registry_store, monkeypatch):
    from power_loop.runtime.managed_delivery import enqueue
    from power_loop.runtime.managed_effects import terminate_unstarted

    from .test_aborted_call_outcomes_192 import pending_batch
    from .test_tool_call_plans_192 import controller

    c, _ = await pending_batch(registry_store, stage="commit_ack")
    loop, model = await controller(c)
    try:
        [child, *_] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool' ORDER BY job_id")
        await terminate_unstarted(c.manager, c.actor, child["job_id"])
        before = await c.manager._row(c.manager.db, child["job_id"])
        await c.manager.db.execute(c.manager.db.dialect.insert_ignore(c.manager.prefix+"session_obligations", obligations.COLUMNS),
            (c.sid, "managed", "stale-derived-notice", child["job_id"], 1))
        publish, attempts = obligations._rebuild_tree, []

        async def delivered_before_publish(owner, root_id, plan):
            attempts.append(root_id)
            if len(attempts) == 1:
                await enqueue(loop, c.actor, child["job_id"])
                after = await c.manager._row(c.manager.db, child["job_id"])
                assert after["version"] == before["version"], "this real notice does not bump the execution version"
            return await publish(owner, root_id, plan)

        with monkeypatch.context() as patch:
            patch.setattr(obligations, "_rebuild_tree", delivered_before_publish)
            await obligations.rebuild(c.manager)
        # The concurrent producer already removed the stale derived entry and
        # published its actual notice. A stale repair must not delete that row.
        assert len(attempts) == 1
        current = [row for row in await _rows(c.manager) if row["job_id"] == child["job_id"]]
        assert len(current) == 1 and current[0]["kind"] == "managed" and current[0]["source_id"] != "stale-derived-notice"
        await obligations.verify(c.manager, c.actor)
        assert model.calls == c.state.dispatched == 0
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def test_lost_rebuild_commit_ack_keeps_the_committed_tree_and_retries_only_remaining_drift(runtime, monkeypatch):
    _, manager, actor, _, parent, _, _ = runtime
    await _result(manager, actor, "ack-first", parent=parent)
    await _result(manager, actor, "ack-second")
    expected, originals = await _rows(manager), await _originals(manager)
    await manager.db.execute(f"DELETE FROM {manager.prefix}session_obligations")
    entries, transaction, fired = obligations._write_entries, manager.db.transaction, []

    async def armed(owner, tx, row, pending):
        result = await entries(owner, tx, row, pending)
        fired.append(row["root_job_id"])
        return result

    @asynccontextmanager
    async def lost_ack():
        async with transaction() as tx:
            yield tx
        if fired:
            raise OSError("tree repair committed before lost ACK")

    with monkeypatch.context() as patch:
        patch.setattr(obligations, "_write_entries", armed)
        patch.setattr(manager.db, "transaction", lost_ack)
        with pytest.raises(OSError, match="tree repair committed before lost ACK"):
            await obligations.rebuild(manager)
    partial = await _rows(manager)
    assert len(fired) == len(partial) == 1 and partial[0] in expected
    assert await _originals(manager) == originals
    await obligations.rebuild(manager)
    await obligations.rebuild(manager)
    assert await _rows(manager) == expected
    assert await _originals(manager) == originals
    await obligations.verify(manager, actor)


async def test_rebuild_removes_orphan_derived_rows_without_inventing_jobs(runtime):
    _, manager, actor, _, _, _, _ = runtime
    originals = await _originals(manager)
    await manager.db.execute(manager.db.dialect.insert_ignore(manager.prefix+"session_obligations", obligations.COLUMNS),
        (actor.session_id, "result", "orphan-source", "missing-job", 1))
    await obligations.rebuild(manager)
    assert await _rows(manager) == []
    assert await _originals(manager) == originals
    await obligations.verify(manager, actor)


@pytest.mark.parametrize("point", ["before_first_commit", "after_first_commit", "before_second_commit"])
async def test_process_death_between_tree_repairs_leaves_only_repeatable_derived_work(registry_store, tmp_path, point):
    store, connect = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path/"registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    evidence = tmp_path/"rebuild-crash.json"
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._obligation_rebuild_worker",
        dsn, store.table_prefix, str(evidence), point, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 40)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == {"before_first_commit": 82, "after_first_commit": 83, "before_second_commit": 84}[point], output.decode()
    saved = json.loads(evidence.read_text())
    manager, actor = JobManager(await connect()), JobActor(**saved["actor"])
    partial = await _rows(manager)
    assert len(partial) == int(point != "before_first_commit")
    assert all(row in saved["pending"] for row in partial)
    assert await _originals(manager) == saved["originals"]
    for _ in range(2):
        await obligations.rebuild(manager)
        await obligations.verify(manager, actor)
        assert await _rows(manager) == saved["pending"]
        assert await _originals(manager) == saved["originals"]
