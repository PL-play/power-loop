"""Forest discovery runs outside the writer; publication fences the caller."""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop.runtime import managed_delivery
from power_loop.runtime.jobs import JobError, JobManager

from .test_job_registry_192 import registry_store, runtime
from .test_processing_decisions_192 import processing_case
from .test_send_sql_profile import Profiled, _counts

__all__ = ["registry_store", "runtime", "processing_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_other_trees_are_discovered_without_an_open_control_writer(processing_case, monkeypatch):
    c = processing_case
    target, row = await c.result()
    candidates = managed_delivery.candidates
    seen = []

    async def outside_writer(manager, reader, actor):
        assert manager._control_intents.get() is None
        seen.append(True)
        return await candidates(manager, reader, actor)

    monkeypatch.setattr(managed_delivery, "candidates", outside_writer)
    snapshot = await c.prepare([row])
    assert seen == [True] and snapshot is not None
    assert (await c.manager.get_job(c.actor, target)).result_status == "pending_delivery"
    event = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (snapshot,))
    assert event["job_id"] == c.caller.job_id and event["delivery_status"] == "prepared"


async def test_cancellation_can_commit_during_discovery_and_prevents_snapshot_publication(processing_case, monkeypatch):
    c = processing_case
    target, row = await c.result()
    candidates = managed_delivery.candidates
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold_discovery(*args, **kwargs):
        entered.set()
        await release.wait()
        return await candidates(*args, **kwargs)

    monkeypatch.setattr(managed_delivery, "candidates", hold_discovery)
    task = asyncio.create_task(c.prepare([row]))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        other = JobManager(await c.connect(), now_ms=c.manager.now)
        operation = await asyncio.wait_for(other.request_cancel(c.actor, c.caller.job_id,
            system_key="cancel-during-discovery", reason="stop original handler"), 5)
        assert operation.result["cancel_requested"]
        release.set()
        with pytest.raises(JobError, match="cancel_requested"):
            await task
        assert not await c.manager.db.fetchall(
            f"SELECT event_id FROM {c.manager.events} WHERE event_type IN ('processing_request','wait_observed')")
        assert (await c.manager.get_job(c.actor, target)).result_status == "pending_delivery"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_failed_publication_does_not_acknowledge_a_discovered_result(processing_case, monkeypatch):
    c = processing_case
    target, row = await c.result()
    transaction = c.manager.db.transaction
    failed = []

    @asynccontextmanager
    async def reject_snapshot_commit():
        async with transaction() as tx:
            yield tx
            if await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_type='processing_request'"):
                failed.append(True)
                raise OSError("snapshot commit failed")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, "transaction", reject_snapshot_commit)
        with pytest.raises(OSError, match="snapshot commit failed"):
            await c.prepare([row])
    assert failed == [True]
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='processing_request'")
    assert (await c.manager.get_job(c.actor, target)).result_status == "pending_delivery"
    snapshot = await c.prepare([row])
    assert snapshot is not None


async def test_discovery_io_is_measured_with_the_canonical_classifier(processing_case):
    c = processing_case
    target, row = await c.result()
    original = c.manager.db
    c.manager.db = measured = Profiled(original)
    try:
        assert await c.prepare([row]) is not None
    finally:
        c.manager.db = original
    assert (await c.manager.get_job(c.actor, target)).result_status == "pending_delivery"
    counts = _counts(measured.records, measured.tx_sites)
    assert counts["control_tx"] == 1
    directory = os.environ.get("POWER_LOOP_SQL_PROFILE")
    if directory:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"processing-discovery-{original.dialect.name}.json").write_text(json.dumps({
            "records": measured.records, "tx_sites": measured.tx_sites, "counts": counts}, indent=2))
