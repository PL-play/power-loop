"""Task inspection never opens a writer or blocks another tree's control."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from power_loop.runtime import task_query
from power_loop.runtime.jobs import JobManager

from .test_job_registry_192 import registry_store, runtime
from .test_send_sql_profile import Profiled, _counts

__all__ = ["registry_store", "runtime"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def _other_root(manager, actor):
    return await manager.register_job(actor, kind="send", session_id=actor.session_id,
        system_key="another-query-root", request={"send_index": 2})


async def test_forest_and_detail_inspection_have_no_mutation_port_or_control_transaction(runtime, monkeypatch):
    _, manager, actor, root, _, _, _ = runtime
    other = await _other_root(manager, actor)
    jobs = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} ORDER BY job_id")
    events = await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id")

    @asynccontextmanager
    async def forbidden():
        raise AssertionError("inspection must not acquire a control writer")
        yield

    monkeypatch.setattr(manager, "_write", forbidden)
    database = manager.db
    manager.db = measured = Profiled(database)
    try:
        page = await task_query.list_jobs(manager, actor, status="all")
        detail = await task_query.get_job(manager, actor, other.job.job_id)
    finally:
        manager.db = database
    assert {node["job_id"] for node in page["nodes"]} == {root.job.job_id, other.job.job_id}
    assert detail["node"]["job_id"] == other.job.job_id
    assert _counts(measured.records, measured.tx_sites)["control_tx"] == 0
    assert all(record["sql"].startswith("SELECT ") and manager.prefix+"task_store" not in record["sql"]
        for record in measured.records)
    assert await manager.db.fetchall(f"SELECT * FROM {manager.jobs} ORDER BY job_id") == jobs
    assert await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id") == events


@pytest.mark.parametrize("query", ["forest", "detail"])
async def test_slow_inspection_allows_a_concurrent_stop_to_commit(runtime, monkeypatch, query):
    _, manager, actor, root, _, connect, _ = runtime
    other = await _other_root(manager, actor)
    entered, release = asyncio.Event(), asyncio.Event()
    processing_view = task_query._processing_view

    async def paused(*args):
        assert manager._control_intents.get() is None
        if not entered.is_set():
            entered.set()
            await release.wait()
        return await processing_view(*args)

    monkeypatch.setattr(task_query, "_processing_view", paused)
    inspection = asyncio.create_task(task_query.list_jobs(manager, actor, status="all") if query == "forest"
        else task_query.get_job(manager, actor, other.job.job_id))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        writer = JobManager(await connect(), now_ms=manager.now)
        stop = await asyncio.wait_for(writer.request_cancel(actor, root.job.job_id,
            system_key="stop-during-inspection", reason="stop while another tree is inspected"), 5)
        assert stop.result["cancel_requested"]
        assert not inspection.done()
        release.set()
        result = await asyncio.wait_for(inspection, 5)
        assert result["nodes"] if query == "forest" else result["node"]["job_id"] == other.job.job_id
        assert (await writer.get_job(actor, root.job.job_id)).cancel_requested_at is not None
        assert not (await writer.get_job(actor, other.job.job_id)).cancel_requested_at
    finally:
        release.set()
        await asyncio.gather(inspection, return_exceptions=True)
