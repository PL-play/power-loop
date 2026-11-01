"""Original deadlines and wait decisions are inspections, not control writers."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from power_loop.runtime import completion_windows, owner_waits, timer_waits, wait_decisions
from power_loop.runtime.completion import CompletionWindow
from power_loop.runtime.jobs import JobError, JobLease, JobManager
from power_loop.runtime.task_runtime import _OWNER, Execution, TaskRuntime

from .test_completion_windows_192 import original as window_case
from .test_control_read_ports import forbidden_writer, originals
from .test_job_registry_192 import registry_store, runtime
from .test_send_sql_profile import Profiled
from .test_timer_waits_192 import receipt, waiting
from .test_wait_effects_192 import decide
from .test_wait_handoff_192 import handoff

__all__ = ["registry_store", "runtime", "window_case", "waiting", "handoff"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def assert_inspection(measured, transactions):
    assert measured.transactions == transactions and measured.records
    assert all(record["sql"].startswith("SELECT ") and "FOR UPDATE" not in record["sql"]
        and "task_store" not in record["sql"] for record in measured.records)


async def test_window_owner_history_and_collector_binding_never_open_a_control_writer(window_case, monkeypatch):
    c = window_case
    before, database = await originals(c.manager), c.manager.db
    runtime = TaskRuntime(c.manager, tenant_id=c.actor.tenant_id, principal_id=c.actor.principal_id,
        executor_id=c.owner.executor_id, boot_id=c.owner.boot_id)
    binding = Execution(runtime, c.actor, c.owner)
    token = _OWNER.set(binding)
    c.manager.db = measured = Profiled(database)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(c.manager, "_write", forbidden_writer)
            assert await completion_windows.read(c.manager, c.main, c.owner.job_id) == c.saved
            collector = CompletionWindow()
            await completion_windows.bind_window(SimpleNamespace(task_runtime=runtime), collector, c.sid)
            assert collector.utc_deadline_at_ms == c.saved["deadline_at_ms"]
            assert await owner_waits.read(c.manager, c.main, c.owner.job_id) == {
                "job_id": c.owner.job_id, "generation": 1, "phases": []}
            await owner_waits.check_deadline(binding)
            c.now[0] += 30001
            with pytest.raises(JobError, match="lease_expired"):
                await owner_waits.check_deadline(binding)
    finally:
        c.manager.db = database
        _OWNER.reset(token)
    assert_inspection(measured, 5)
    assert await originals(c.manager) == before


async def test_wait_receipt_inspection_never_opens_a_control_writer(waiting, monkeypatch):
    c = waiting
    registration = await c.register()
    await receipt(c, registration)
    before, database = await originals(c.manager), c.manager.db
    c.manager.db = measured = Profiled(database)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(c.manager, "_write", forbidden_writer)
            saved = await timer_waits.read_wait(c.manager, c.main, registration.job.job_id)
            assert saved["receipt_ref"]["tool_call_id"] == "arm-wait"
            assert saved["request"]["request"]["continuation"] == c.request.continuation
            assert not saved["execution_authority"]
    finally:
        c.manager.db = database
    assert_inspection(measured, 1)
    assert await originals(c.manager) == before


async def test_decision_inspection_and_recall_follow_originals_across_roots_without_a_writer(handoff, monkeypatch):
    h, c = handoff, handoff.c
    await decide(h)
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_decision'")
    request = json.loads(event["payload_json"])["request"]
    request.update(caller=JobLease(**request["caller"]), call_ref=timer_waits._ref(request["call_ref"]))
    assert request["caller"].job_id != c.root.job.job_id
    before, database, calls = await originals(c.manager), c.manager.db, h.model.calls
    c.manager.db = measured = Profiled(database)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(c.manager, "_write", forbidden_writer)
            summary = await wait_decisions.inspect(c.manager, c.main, h.event_id)
            assert summary["decision_status"] == "recorded" and not summary["execution_authority"]
            assert (await wait_decisions.inspect(c.manager, c.main, h.event_id, field="note"))["text"] == c.request.note
            assert (await wait_decisions.inspect(c.manager, c.main, h.event_id, field="continuation"))["text"] == c.request.continuation
            operation = await wait_decisions.recall_decision(c.manager, c.main, **request)
            assert operation.operation_id == json.loads(event["payload_json"])["operation_id"]
    finally:
        c.manager.db = database
    assert_inspection(measured, 4)
    assert await originals(c.manager) == before and h.model.calls == calls


async def test_wait_read_joins_the_explicit_caller_transaction_and_preserves_its_rollback(waiting):
    c = waiting
    registration = await c.register()
    seq = await c.store.append_message(c.sid, role="tool", name="wait", content="original receipt",
        tool_call_id="arm-wait", send_index=1, round_index=0)
    ref = await c.store.message_ref(c.sid, seq)
    before = await originals(c.manager)
    with pytest.raises(OSError, match="caller rolled back"):
        async with c.manager.joined_write():
            await c.manager.bind_receipt(c.actor, registration.operation.operation_id, ref, caller=c.caller)
            saved = await timer_waits.read_wait(c.manager, c.main, registration.job.job_id)
            assert saved["receipt_ref"] == ref.to_dict()
            raise OSError("caller rolled back")
    assert (await timer_waits.read_wait(c.manager, c.main, registration.job.job_id))["receipt_ref"] is None
    assert await originals(c.manager) == before


async def test_paused_window_snapshot_allows_a_different_root_stop_to_commit(window_case, monkeypatch):
    c = window_case
    other = await c.manager.register_job(c.main, kind="send", session_id=c.main.session_id,
        system_key="other-window-root", request={"send_index": 2})
    await c.manager.claim_job(c.main, other.job.job_id, executor_id="other", boot_id="boot", expected_version=1)
    peer = JobManager(await c.connect(), now_ms=c.manager.now)
    creation = completion_windows._creation
    entered, release = asyncio.Event(), asyncio.Event()

    async def pause(*args):
        result = await creation(*args)
        assert c.manager._control_intents.get() is None
        entered.set()
        await release.wait()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(completion_windows, "_creation", pause)
        reader = asyncio.create_task(completion_windows.read(c.manager, c.main, c.owner.job_id))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            operation = await asyncio.wait_for(peer.request_cancel(c.main, other.job.job_id,
                system_key="stop-during-window-inspection", reason="stop other tree"), 3)
            assert operation.job_id == other.job.job_id
            assert (await peer.get_job(c.main, other.job.job_id)).cancel_requested_at is not None
            assert not reader.done()
        finally:
            release.set()
            saved = await asyncio.wait_for(reader, 3)
    assert saved == c.saved
    assert (await c.manager.get_job(c.main, c.owner.job_id)).cancel_requested_at is None
