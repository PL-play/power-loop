"""Stop receipt inspection and local discovery have no database mutation port."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_control import recall_stop, signal_local
from power_loop.runtime.task_runtime import Execution, TaskRuntime

from .test_job_registry_192 import registry_store
from .test_send_sql_profile import Profiled
from .test_task_control_192 import control

__all__ = ["registry_store", "control"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@asynccontextmanager
async def forbidden_writer():
    raise AssertionError("stop inspection must not open a control writer")
    yield


async def test_receipt_recall_and_missing_local_handle_never_open_a_control_writer(control, monkeypatch):
    c = control
    runtime = TaskRuntime(c.manager, tenant_id="control", principal_id="host", executor_id="reader", boot_id="boot")
    operation, _ = await c.stop()
    await c.manager.close_model_execution(c.actor, c.caller)
    before = {table: await c.manager.db.fetchall(f"SELECT * FROM {table} ORDER BY 1")
        for table in (c.manager.jobs, c.manager.events, c.store.t.messages, c.manager.prefix+"session_controls")}
    database = c.manager.db
    c.manager.db = measured = Profiled(database)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(c.manager, "_write", forbidden_writer)
            recalled = await recall_stop(c.manager, c.actor, c.child.job.job_id,
                caller=c.caller, call_ref=c.ref, tool_call_id="stop-child")
            assert recalled == operation
            assert await signal_local(runtime, c.actor, operation) == 0
    finally:
        c.manager.db = database
    assert measured.transactions == 2 and measured.records
    assert all(record["sql"].startswith("SELECT ") and "FOR UPDATE" not in record["sql"]
        and c.manager.prefix+"task_store" not in record["sql"] for record in measured.records)
    assert {table: await c.manager.db.fetchall(f"SELECT * FROM {table} ORDER BY 1") for table in before} == before


async def test_paused_local_discovery_spans_two_roots_without_blocking_another_root_stop(control, monkeypatch):
    c = control
    runtime = TaskRuntime(c.manager, tenant_id="control", principal_id="host", executor_id="worker", boot_id="boot", lease_ms=300000)
    own = await c.manager.claim_job(c.actor, c.child.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
    other = await c.manager.register_job(c.actor, kind="send", session_id=c.actor.session_id,
        system_key="unrelated-local-root", request={"send_index": 2})
    other_lease = await c.manager.claim_job(c.actor, other.job.job_id,
        executor_id="worker", boot_id="boot", expected_version=1)
    third = await c.manager.register_job(c.actor, kind="send", session_id=c.actor.session_id,
        system_key="stop-during-local-discovery", request={"send_index": 3})
    finish, own_entered, other_entered, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
    discovered, release = asyncio.Event(), asyncio.Event()
    cancelled = []

    async def work(lease, entered, label):
        async with runtime.execution(Execution(runtime, c.actor, lease)):
            entered.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                cancelled.append(label)
                stopped.set()
                await finish.wait()

    tasks = [asyncio.create_task(work(own, own_entered, "own")),
        asyncio.create_task(work(other_lease, other_entered, "other"))]
    signal = None
    try:
        await asyncio.wait_for(asyncio.gather(own_entered.wait(), other_entered.wait()), 5)
        operation, _ = await c.stop(expected_generation=1)
        authorize = c.manager._authorize

        async def paused(reader, actor, job_id):
            row = await authorize(reader, actor, job_id)
            if job_id == other.job.job_id:
                assert c.manager._control_intents.get() is None
                discovered.set()
                await release.wait()
            return row

        with monkeypatch.context() as patch:
            patch.setattr(c.manager, "_authorize", paused)
            signal = asyncio.create_task(signal_local(runtime, c.actor, operation))
            await asyncio.wait_for(discovered.wait(), 5)
            writer = JobManager(await c.connect(), now_ms=c.manager.now)
            stop = await asyncio.wait_for(writer.request_cancel(c.actor, third.job.job_id,
                system_key="unblocked-independent-stop", reason="stop while two local roots are inspected"), 5)
            assert stop.result["cancel_requested"] and not signal.done()
            release.set()
            assert await asyncio.wait_for(signal, 5) == 1
            await asyncio.wait_for(stopped.wait(), 5)
        assert cancelled == ["own"] and not tasks[1].done()
        assert (await c.manager.get_job(c.actor, c.child.job.job_id)).resource_state == "running"
        assert (await c.manager.get_job(c.actor, other.job.job_id)).cancel_requested_at is None
        assert await signal_local(runtime, c.actor, operation) == 0
    finally:
        release.set()
        finish.set()
        if signal is not None:
            await asyncio.gather(signal, return_exceptions=True)
        await asyncio.gather(*tasks, return_exceptions=True)
    assert not runtime._local_executions
