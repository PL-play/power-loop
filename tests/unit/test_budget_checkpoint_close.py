"""Full grants protect concurrent admission until cumulative close commits."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop.runtime.control_runtime import flush_binding
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from ._budget_checkpoint_crash_worker import FileHost, config
from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case, counters, observed, start

__all__ = ["registry_store", "budget_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("stage", ["open", "during_close", "closed"])
async def test_process_crash_preserves_grant_and_usage_until_atomic_close(registry_store, tmp_path, stage):
    store, connect = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    evidence = tmp_path / "budget-crash.json"
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._budget_checkpoint_crash_worker",
        dsn, store.table_prefix, str(evidence), stage, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == {"open": 81, "during_close": 82, "closed": 83}[stage], output.decode()
    saved = json.loads(evidence.read_text())
    reader = await connect(schema="compatible")
    host = FileHost(str(evidence) + ".charges")
    runtime = TaskRuntime(JobManager(reader), tenant_id="budget-tenant", principal_id="host",
        executor_id="reconciler", boot_id="new-boot", boot_contract_hash="a" * 64, budget_config=config(host))
    try:
        row = await reader._db.fetchone(f"SELECT summary_json FROM {reader.table_prefix}execution_checkpoints WHERE job_id=? AND generation=?",
                                       (saved['job_id'], saved['generation']))
        summary = json.loads(row['summary_json'])
        [row] = await reader._db.fetchall(f"SELECT counters_json FROM {reader.table_prefix}task_budget_scopes")
        actual = json.loads(row['counters_json'])
        expected = dict(used_calls=1, held_calls=0, used_tokens=12, held_tokens=0, inflight=0) if stage == "closed" else dict(
            used_calls=0, held_calls=1, used_tokens=0, held_tokens=100, inflight=1)
        assert actual == expected
        [allocation] = summary['allocations']
        assert allocation['used_calls'] == 1 and allocation['used_tokens'] == 12
        assert summary['closed'] == (stage == 'closed')
        assert (await runtime.manager.get_job(runtime.actor(saved['sid']), saved['job_id'])).job_id == saved['job_id']
        if stage == "closed":
            for _ in range(2):
                replay = await runtime.reconcile_budget_usage(saved["sid"], job_id=saved["job_id"],
                    attempt_id=saved["attempt_id"])
                assert replay["usage_acknowledged"] and not replay["execution_authority"]
        assert len(json.loads(Path(str(evidence) + ".charges").read_text())) == 1
        assert host.reserve_calls == 0 and not host.released
    finally:
        await runtime.shutdown(timeout_s=1)
        await reader.close()


@pytest.mark.parametrize("fault", ["rollback", "ack_loss"])
async def test_close_commit_failure_retries_same_usage_delta_once(budget_case, monkeypatch, fault):
    c = budget_case
    a = await start(c)
    await observed(c, a, tokens=12)
    await flush_binding(c.binding)
    original = c.manager.db.transaction
    failed = False

    @asynccontextmanager
    async def fail_once():
        nonlocal failed
        inject = not failed
        failed = True
        async with original() as tx:
            yield tx
            if inject and fault == "rollback":
                raise OSError("before_commit")
        if inject and fault == "ack_loss":
            raise OSError("committed_ack_lost")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, "transaction", fail_once)
        if fault == 'ack_loss':
            from power_loop.runtime.jobs import JobError
            with pytest.raises(JobError, match='execution_checkpoint_conflict'):
                await flush_binding(c.binding, close=True)
        else:
            await flush_binding(c.binding, close=True)
    expected = dict(used_calls=1, held_calls=0, used_tokens=12, held_tokens=0, inflight=0)
    assert await counters(c) == expected and len(c.host.applied) == 1
