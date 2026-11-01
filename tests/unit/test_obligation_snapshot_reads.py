"""Cold obligation inspection observes one snapshot without a control writer."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from power_loop.runtime import session_obligations as obligations
from power_loop.runtime.jobs import JobManager

from .test_job_registry_192 import registry_store, runtime

__all__ = ["registry_store", "runtime"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_verification_has_no_control_writer_or_original_mutation(runtime, monkeypatch):
    _, manager, actor, _, parent, _, _ = runtime
    child = await manager.register_job(actor, kind="subagent", parent=parent, independent=True,
        system_key="snapshot-child")
    lease = await manager.claim_job(actor, child.job.job_id, executor_id="child", boot_id="boot", expected_version=1)
    await manager.record_execution_result(actor, lease, outcome="succeeded", result={"value": 1}, resource_state="stopped")
    before = {table: await manager.db.fetchall(f"SELECT * FROM {table} ORDER BY 1")
        for table in (manager.jobs, manager.events, manager.prefix+"session_obligations", manager.prefix+"session_controls")}

    @asynccontextmanager
    async def forbidden():
        raise AssertionError("cold verification must not open a control writer")
        yield

    monkeypatch.setattr(manager, "_write", forbidden)
    await obligations.verify(manager, actor)
    assert {table: await manager.db.fetchall(f"SELECT * FROM {table} ORDER BY 1") for table in before} == before


async def test_verification_keeps_one_snapshot_while_a_concurrent_result_commits(runtime, monkeypatch):
    _, manager, actor, _, parent, connect, _ = runtime
    child = await manager.register_job(actor, kind="subagent", parent=parent, independent=True,
        system_key="concurrent-snapshot-child")
    lease = await manager.claim_job(actor, child.job.job_id, executor_id="child", boot_id="boot", expected_version=1)
    writer = JobManager(await connect(), now_ms=manager.now)
    entered, release = asyncio.Event(), asyncio.Event()
    desired = obligations._desired
    observed = []

    async def paused(owner, reader, row, **kwargs):
        if owner is manager and row["job_id"] == child.job.job_id:
            assert manager._control_intents.get() is None
            observed.append(row["result_status"])
            if not entered.is_set():
                entered.set()
                await release.wait()
        return await desired(owner, reader, row, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(obligations, "_desired", paused)
        inspection = asyncio.create_task(obligations.verify(manager, actor))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            result = await asyncio.wait_for(writer.record_execution_result(actor, lease,
                outcome="succeeded", result={"value": "concurrent"}, resource_state="stopped"), 5)
            assert not inspection.done()
            release.set()
            await asyncio.wait_for(inspection, 5)
        finally:
            release.set()
            await asyncio.gather(inspection, return_exceptions=True)
    assert observed == ["none"]
    assert (await writer.get_job(actor, child.job.job_id)).result_status == "pending_delivery"
    assert [row["source_id"] for row in await writer.db.fetchall(
        f"SELECT source_id FROM {writer.prefix}session_obligations WHERE job_id=?", (child.job.job_id,))] == [result]
    await obligations.verify(manager, actor)


async def test_cancelled_snapshot_releases_its_connection_without_rolling_back_another_writer(runtime):
    _, manager, actor, root, _, connect, _ = runtime
    writer = JobManager(await connect(), now_ms=manager.now)
    entered, release = asyncio.Event(), asyncio.Event()

    async def inspect():
        async with manager.db.read_snapshot() as reader:
            row = await reader.fetchone(f"SELECT cancel_requested_at FROM {manager.jobs} WHERE job_id=?", (root.job.job_id,))
            assert row["cancel_requested_at"] is None
            entered.set()
            await release.wait()

    inspection = asyncio.create_task(inspect())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        stopped = await asyncio.wait_for(writer.request_cancel(actor, root.job.job_id,
            system_key="stop-during-snapshot", reason="cancel during snapshot"), 5)
        assert stopped.result["cancel_requested"]
        inspection.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(inspection, 5)
        assert (await manager.get_job(actor, root.job.job_id)).cancel_requested_at is not None
        async with manager.db.read_snapshot() as reader:
            row = await reader.fetchone(f"SELECT cancel_requested_at FROM {manager.jobs} WHERE job_id=?", (root.job.job_id,))
            assert row["cancel_requested_at"] is not None
    finally:
        release.set()
        await asyncio.gather(inspection, return_exceptions=True)
