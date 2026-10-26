"""Post-commit observations cannot become authority or transaction failures."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from power_loop import AgentEvent, AgentEventBus, AgentEventType, TaskSupervisionClosedPayload
from power_loop.runtime.jobs import JobError, JobManager

from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import jobs, managed

__all__ = ["registry_store", "managed"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
TYPE = AgentEventType.TASK_SUPERVISION_CLOSED


async def completed(case):
    _, manager, runtime, loop, model, sid, _ = case
    model.root_calls = 1  # This test needs a plain send, not delegation.
    await loop.send("hello", sid)
    await runtime.flush_control_checkpoints()
    [job] = await jobs(case)
    return manager, runtime.actor(sid), job


async def test_observer_can_read_committed_proof_through_independent_connection(managed):
    manager, actor, job = await completed(managed)
    other = await managed[-1]()
    seen, bus = [], AgentEventBus()

    async def inspect(event):
        row = await other._db.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (event.data.event_id,))
        assert row and (await JobManager(other).get_job(actor, job.job_id)).supervision_status == "closed"
        seen.append(event)

    bus.subscribe(TYPE, inspect)
    observer = JobManager(manager.store, event_bus=bus)
    ref = await observer.finalize_supervision(actor, job.job_id, generation=job.generation)
    await asyncio.gather(*bus._async_tasks)
    [event] = seen
    assert isinstance(event.data, TaskSupervisionClosedPayload)
    assert event.payload == dict(store_id=manager.store.store_id, job_id=job.job_id,
        generation=job.generation, event_id=ref["event_id"], revision=ref["revision"])
    assert event.session_id == actor.session_id and event.source == "task_registry"
    assert AgentEvent.from_dict(event.to_dict()).payload == event.payload


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_bad_sync_subscriber_cannot_fail_committed_finalization(managed, error, caplog):
    manager, actor, job = await completed(managed)
    bus = AgentEventBus()

    def broken(event):
        raise error("private host data must not be logged")

    bus.subscribe(TYPE, broken)
    observer = JobManager(manager.store, event_bus=bus)
    ref = await observer.finalize_supervision(actor, job.job_id, generation=job.generation)
    assert ref == await manager.finalize_supervision(actor, job.job_id, generation=job.generation)
    assert "observer failed" in caplog.text and "private host data" not in caplog.text


async def test_payload_mutation_and_repeat_observation_do_not_mutate_durable_proof(managed):
    manager, actor, job = await completed(managed)
    before = await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id")
    bus, seen = AgentEventBus(), []

    def mutate(event):
        seen.append((event.data.store_id, event.data.event_id))
        event.data.event_id = "fake"
        event.payload["revision"] = -1

    bus.subscribe(TYPE, mutate)
    observer = JobManager(manager.store, event_bus=bus)
    first = await observer.finalize_supervision(actor, job.job_id, generation=job.generation)
    assert first == await observer.finalize_supervision(actor, job.job_id, generation=job.generation)
    assert seen == [(manager.store.store_id, first["event_id"])] * 2
    assert await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id") == before


async def test_authorization_generation_and_transaction_failure_never_emit(managed, monkeypatch):
    manager, actor, job = await completed(managed)
    bus, events = AgentEventBus(), []
    bus.subscribe(TYPE, events.append)
    observer = JobManager(manager.store, event_bus=bus)
    with pytest.raises(JobError, match="access_denied"):
        await observer.finalize_supervision(replace(actor, session_id="other"), job.job_id, generation=job.generation)
    with pytest.raises(JobError, match="generation_changed"):
        await observer.finalize_supervision(actor, job.job_id, generation=job.generation + 1)
    original = observer._write

    @asynccontextmanager
    async def rollback():
        async with original() as tx:
            yield tx
            raise OSError("commit not acknowledged")

    monkeypatch.setattr(observer, "_write", rollback)
    with pytest.raises(OSError):
        await observer.finalize_supervision(actor, job.job_id, generation=job.generation)
    assert not events


async def test_running_child_cannot_be_closed_by_an_observer(managed):
    _, manager, runtime, loop, _, sid, _ = managed
    bus, events = AgentEventBus(), []
    bus.subscribe(TYPE, events.append)
    manager.event_bus = bus
    await loop.send("delegate", sid)
    child = next(job for job in await jobs(managed) if job.kind == "subagent")
    events.clear()
    with pytest.raises(JobError):
        await manager.finalize_supervision(runtime.actor(sid), child.job_id, generation=child.generation)
    assert not events
