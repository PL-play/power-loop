"""A close must publish its suffix even after an earlier batch in the same TX."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from .test_execution_checkpoints import case, complete
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("outcome", ["commit", "rollback", "ack_lost"])
async def test_staged_open_batch_and_close_keep_every_record_until_transaction_outcome(registry_store, outcome):
    store, _, state, journal, _, _ = await case(registry_store)
    complete(state, 1)
    hooks, undo = [], []
    def defer(commit, rollback):
        hooks.append(commit)
        undo.append(rollback)
    try:
        async with store._db.transaction() as tx:
            await state.checkpoint(force=True, transaction=tx, defer=defer)
            assert state.cursor == 0
            await state.close(transaction=tx, defer=defer)
            head = await tx.fetchone(f"SELECT batch_no,summary_json FROM {journal.heads} WHERE job_id=? AND generation=?",
                                    (state.identity.job_id, state.identity.generation))
            assert json.loads(head["summary_json"])["closed"]
            assert head["batch_no"] == 2 and state.cursor == 0
            if outcome == "rollback":
                raise OSError("staged close rollback")
        if outcome == "ack_lost":
            raise OSError("staged close committed but ACK lost")
    except BaseException as exc:
        for hook in reversed(undo):
            hook()
        if not isinstance(exc, OSError):
            raise
        assert outcome != "commit" and state.cursor == 0
    else:
        for hook in hooks:
            hook()
        assert outcome == "commit" and state.cursor == 2
    if outcome == "ack_lost":
        from power_loop.runtime.jobs import JobError
        with pytest.raises(JobError, match="execution_checkpoint_conflict"):
            await state.close()
        saved = await journal.inspect(state.identity)
        assert saved['summary']['closed'] and saved['summary']['completed'] == 1
        return
    await state.close()
    saved = await journal.inspect(state.identity)
    assert saved["summary"]["closed"] and saved["summary"]["completed"] == 1
    assert saved["batch_no"] == state.cursor
    assert not state._dirty and state._pending is None


async def test_savepoint_rollback_retains_only_the_uncommitted_suffix_for_retry(registry_store):
    store, _, state, journal, _, _ = await case(registry_store)
    complete(state, 1)
    hooks, undo = [], []
    def defer(commit, rollback):
        hooks.append(commit)
        undo.append(rollback)
    async with store._db.transaction() as tx:
        await state.checkpoint(force=True, transaction=tx, defer=defer)
        mark = len(hooks)
        await tx.execute("SAVEPOINT test_staged_suffix")
        complete(state, 2)
        await state.checkpoint(force=True, transaction=tx, defer=defer)
        await tx.execute("ROLLBACK TO SAVEPOINT test_staged_suffix")
        for hook in reversed(undo[mark:]):
            hook()
        del hooks[mark:], undo[mark:]
    for hook in hooks:
        hook()
    assert state.cursor == 1 and state.checkpoint_failed
    saved = await journal.inspect(state.identity)
    assert saved["summary"]["completed"] == 1 and not saved["summary"]["closed"]
    await state.close()
    saved = await journal.inspect(state.identity)
    assert saved["summary"]["completed"] == 2 and saved["summary"]["closed"]
    assert state.cursor == 3 and not state._dirty and state._pending is None


@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
async def test_process_death_keeps_all_staged_original_fees_and_close_together(registry_store, tmp_path, point):
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.model_budgets import get_budget

    store, connect = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    evidence = tmp_path / "staged-close.json"
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._staged_close_crash_worker",
        dsn, store.table_prefix, str(evidence), point, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == (152 if point == "before_commit" else 153), output.decode()
    saved = json.loads(evidence.read_text())
    cold = await connect()
    manager = JobManager(cold)
    head = await manager.db.fetchone(f"SELECT summary_json FROM {manager.prefix}execution_checkpoints WHERE job_id=?",
                                    (saved["job_id"],))
    committed = point == "after_commit"
    assert json.loads(head["summary_json"])["closed"] == committed
    counters = await manager.db.fetchall(f"SELECT counters_json FROM {manager.prefix}task_budget_scopes")
    assert len(counters) == 3
    if committed:
        from power_loop.runtime.jobs import JobActor
        budget = await get_budget(manager, JobActor("budget-tenant", "host", saved["sid"]), saved["attempt_id"])
        assert budget["submission"] == saved["submission"]
        assert budget["ack"]["request_hash"] == saved["ack"]["request_hash"]
        assert budget["observation"]["usage"] == {"total_tokens": 7}
        assert all(json.loads(row["counters_json"]) == dict(used_calls=1, held_calls=0,
            used_tokens=7, held_tokens=0, inflight=0) for row in counters)
    else:
        assert all(json.loads(row["counters_json"])["held_calls"] > 0 for row in counters)
    assert saved["reserve_calls"] == saved["apply_calls"] == 1
