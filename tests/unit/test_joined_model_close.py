"""Closing and supervision publish commit together, including unknown outcomes."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop.runtime import finalization
from power_loop.runtime.jobs import JobError, JobManager

from .test_job_registry_192 import registry_store
from .test_round_change_signals import fixture_loop
from .test_send_read_scope import Model

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("stage", ["before_publish", "after_publish", "after_commit"])
async def test_process_death_never_commits_half_a_model_close(registry_store, tmp_path, stage):
    store, connect = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    evidence = tmp_path / "close.json"
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._close_crash_worker",
        dsn, store.table_prefix, str(evidence), stage, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == {"before_publish": 84, "after_publish": 85, "after_commit": 86}[stage], output.decode()
    saved = json.loads(evidence.read_text())
    reader = await connect(schema="compatible")
    manager = JobManager(reader)
    [row] = await reader._db.fetchall(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (saved["job_id"],))
    committed = stage == "after_commit"
    assert (row["status"], row["intake"], row["supervision_status"], row["version"]) == (
        ("model_closed", "closed", "closed", 4) if committed else ("running", "open", "open", 2))
    events = await reader._db.fetchall(f"SELECT event_type FROM {manager.events} WHERE job_id=? "
        "AND event_type IN ('model_closed','supervision_closed')", (saved["job_id"],))
    assert sorted(event["event_type"] for event in events) == (["model_closed", "supervision_closed"] if committed else [])
    [head] = await reader._db.fetchall(f"SELECT summary_json FROM {manager.prefix}execution_checkpoints WHERE job_id=?",
                                     (saved["job_id"],))
    assert json.loads(head["summary_json"])["closed"] == committed
    assert saved["calls"] == 1


@pytest.mark.parametrize("fault", ["rollback", "ack_loss", "finalization_rejected"])
async def test_close_retry_preserves_exact_versions_and_one_model_exit(registry_store, monkeypatch, fault):
    store, _ = registry_store
    model = Model()
    runtime, loop = fixture_loop(store, model)
    transaction, publish = store._db.transaction, finalization._publish
    closing = False
    injected = False

    async def observe(*args):
        nonlocal closing
        closing = True
        await publish(*args)
        if fault == "finalization_rejected":
            raise JobError("unfinished_supervision")

    @asynccontextmanager
    async def fail_once():
        nonlocal injected
        inject = False
        async with transaction() as tx:
            yield tx
            inject = closing and not injected
            injected |= inject
            if inject and fault == "rollback":
                raise OSError("before_close_commit")
        if inject and fault == "ack_loss":
            raise OSError("close_commit_ack_lost")

    monkeypatch.setattr(finalization, "_publish", observe)
    monkeypatch.setattr(store._db, "transaction", fail_once)
    try:
        sid = await loop.new_session()
        result = await loop.send("finish once", sid)
        assert result.status == "completed" and len(model.seen) == 1
        [row] = await runtime.manager.db.fetchall(f"SELECT * FROM {runtime.manager.jobs} WHERE session_id=?", (sid,))
        expected_closed = fault != "finalization_rejected"
        assert row["status"] == "model_closed" and row["intake"] == "closed"
        assert row["supervision_status"] == ("closed" if expected_closed else "open")
        assert row["version"] == (4 if expected_closed else 3)
        events = await runtime.manager.db.fetchall(f"SELECT event_type FROM {runtime.manager.events} WHERE job_id=? "
            "AND event_type IN ('model_closed','supervision_closed')", (row["job_id"],))
        assert sorted(event["event_type"] for event in events) == (
            ["model_closed", "supervision_closed"] if expected_closed else ["model_closed"])
    finally:
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)
