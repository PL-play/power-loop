"""Atomic projection and callback settlement before generation takeover."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from power_loop.runtime import workflow_results as results
from power_loop.runtime.jobs import JobError, JobManager, _hash, _json
from power_loop.workflow import journal

from .test_job_registry_192 import registry_store
from .test_workflow_result_delivery_192 import delivery_case
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ["registry_store", "managed_workflow", "delivery_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def closed(c, *, consumed=False, manager=None):
    m = manager or c.manager
    async with m._write() as tx:
        return await results.require_closed(m, tx, c.actor, c.job_id, 1, consumed=consumed)


async def test_prepare_is_not_projection_and_projection_is_not_consumption(delivery_case):
    c = delivery_case
    calls = []
    async def callback(completion):
        calls.append(completion)
    await results.prepare(c.manager, c.actor, c.job_id, 1)
    with pytest.raises(JobError, match="projection_unsettled"):
        await closed(c)
    with pytest.raises(JobError, match="projection_unsettled"):
        await results.run_callback(c.loop, c.actor, c.job_id, 1, callback)
    await results.project(c.loop, c.actor, c.job_id, 1)
    await results.run_callback(c.loop, c.actor, c.job_id, 1, callback)
    assert not calls
    assert (await closed(c))["envelope"]["generation"] == 1
    with pytest.raises(JobError, match="required_result_unhandled"):
        await closed(c, consumed=True)
    before = c.model.counts.copy()
    await c.loop.flush_follow_ups(c.actor.session_id)
    assert (await closed(c, consumed=True))["envelope"]["result_event_id"] == "result_"+c.job_id
    assert c.model.counts["BUILD"] == before["BUILD"] and c.model.counts["REVIEW"] == before["REVIEW"]
    assert c.loop.pending_follow_up_count(c.actor.session_id) == 0
    for _ in range(2):
        await results.project(c.loop, c.actor, c.job_id, 1)
    assert c.loop.pending_follow_up_count(c.actor.session_id) == 0


@pytest.mark.parametrize("fault", ["journal", "inbox", "closed_receipt", "cancel", "commit_ack"])
async def test_all_projection_writes_rollback_together_or_all_commit(delivery_case, monkeypatch, fault):
    c = delivery_case
    original = c.manager._write
    canonical = await c.manager.get_result_revision(c.actor, c.job_id, 1)
    @asynccontextmanager
    async def fail():
        async with original() as tx:
            async def execute(sql, params=()):
                value = await tx.execute(sql, params)
                hit = ((fault == "journal" and sql.startswith(f"UPDATE {c.store.t.session_runtime_state} SET value_json=")) or
                    (fault == "inbox" and sql.startswith("INSERT") and c.store.t.follow_up_queue in sql) or
                    (fault in ("closed_receipt", "cancel") and "workflow_projected" in params))
                if hit:
                    if fault == "cancel":
                        raise asyncio.CancelledError()
                    raise RuntimeError("transaction statement fault")
                return value
            yield SimpleNamespace(execute=execute, fetchone=tx.fetchone, fetchall=tx.fetchall)
        if fault == "commit_ack":
            raise RuntimeError("lost commit ACK")
    with monkeypatch.context() as patch:
        patch.setattr(c.manager, "_write", fail)
        with pytest.raises(asyncio.CancelledError if fault == "cancel" else RuntimeError):
            await results.project(c.loop, c.actor, c.job_id, 1)
    committed = fault == "commit_ack"
    assert (await c.store.inbox_counts(c.actor.session_id))["pending"] == int(committed)
    j = await journal.read(c.store, c.actor.session_id, c.run_id)
    assert j["status"] == ("completed" if committed else "running")
    rows = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type IN ('workflow_delivery','workflow_projected')")
    assert len(rows) == (2 if committed else 0)
    assert await c.manager.get_result_revision(c.actor, c.job_id, 1) == canonical
    other = JobManager(await c.connect())
    if committed:
        await closed(c, manager=other)
    before = c.model.counts.copy()
    await results.project(c.loop, c.actor, c.job_id, 1)
    assert c.model.counts == before and not c.state["notifications"]
    await closed(c, manager=other)
    assert c.loop.pending_follow_up_count(c.actor.session_id) == 1


@pytest.mark.parametrize("order", ["projection_first", "takeover_first"])
async def test_projection_and_future_takeover_serialize_under_same_control_lock(delivery_case, monkeypatch, order):
    c = delivery_case
    other = JobManager(await c.connect())
    entered, release = asyncio.Event(), asyncio.Event()
    original = results._views
    if order == "takeover_first":
        await results.project(c.loop, c.actor, c.job_id, 1)
    async def paused(*args, **kwargs):
        value = await original(*args, **kwargs)
        if kwargs.get("write"):
            entered.set()
            await release.wait()
        return value
    async def takeover():
        async with other._write() as tx:
            await results.require_closed(other, tx, c.actor, c.job_id, 1)
            if order == "takeover_first":
                entered.set()
                await release.wait()
            # Storage-only fixture for the future admission transaction. No
            # controller, execution authority, provider or resume is granted.
            await tx.execute(f"UPDATE {other.jobs} SET generation=2 WHERE job_id=?", (c.job_id,))
            saved = await tx.fetchone(f"SELECT value_json FROM {c.store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
                (c.actor.session_id, journal.run_key(c.run_id)))
            j = json.loads(saved["value_json"])
            j.update(task_generation=2, status="running", result=None)
            await tx.execute(f"UPDATE {c.store.t.session_runtime_state} SET value_json=? WHERE session_id=? AND state_key=?",
                (json.dumps(j), c.actor.session_id, journal.run_key(c.run_id)))
    async def attempt():
        await results.project(c.loop, c.actor, c.job_id, 1)
    if order == "projection_first":
        monkeypatch.setattr(results, "_views", paused)
        first = asyncio.create_task(attempt())
    else:
        first = asyncio.create_task(takeover())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        second = asyncio.create_task(takeover() if order == "projection_first" else attempt())
        await asyncio.sleep(0.03)
        assert not second.done()
        release.set()
        await asyncio.wait_for(first, 5)
        if order == "projection_first":
            await asyncio.wait_for(second, 5)
        else:
            with pytest.raises(JobError, match="generation_changed"):
                await asyncio.wait_for(second, 5)
        with pytest.raises(JobError, match="generation_changed"):
            await results.project(c.loop, c.actor, c.job_id, 1)
        j = await journal.read(c.store, c.actor.session_id, c.run_id)
        assert j["task_generation"] == 2 and j["status"] == "running" and j["result"] is None
        assert (await c.store.inbox_counts(c.actor.session_id))["pending"] == 1
    finally:
        release.set()
        await asyncio.gather(first, *([second] if 'second' in locals() else []), return_exceptions=True)


async def test_lost_post_commit_hint_cannot_undo_projection_or_start_model(delivery_case, monkeypatch):
    c = delivery_case
    before = c.model.counts.copy()
    original = c.loop._note_committed_inbox
    def failed(sid):
        raise RuntimeError("local hint unavailable after commit")
    monkeypatch.setattr(c.loop, "_note_committed_inbox", failed)
    with pytest.raises(RuntimeError):
        await results.project(c.loop, c.actor, c.job_id, 1)
    await closed(c)
    assert (await c.store.inbox_counts(c.actor.session_id))["pending"] == 1 and c.model.counts == before
    monkeypatch.setattr(c.loop, "_note_committed_inbox", original)
    await results.project(c.loop, c.actor, c.job_id, 1)
    assert c.loop.pending_follow_up_count(c.actor.session_id) == 1 and c.model.counts == before


@pytest.mark.parametrize("fault", ["start_rollback", "start_ack", "exit_rollback", "exit_ack"])
async def test_callback_commit_faults_never_claim_false_completion_or_repeat_effects(managed_workflow, monkeypatch, fault):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    calls = []
    async def callback(completion):
        calls.append(completion)
    state["on_complete"] = callback
    await loop.send("run", sid)
    original = manager._write
    @asynccontextmanager
    async def fail():
        phase_hit = False
        async with original() as tx:
            async def execute(sql, params=()):
                nonlocal phase_hit
                value = await tx.execute(sql, params)
                phase_hit |= "workflow_callback_"+fault.split('_')[0] in params
                if phase_hit and fault.endswith("rollback"):
                    raise RuntimeError("callback receipt rollback")
                return value
            yield SimpleNamespace(execute=execute, fetchone=tx.fetchone, fetchall=tx.fetchall)
        if phase_hit and fault.endswith("ack"):
            raise RuntimeError("callback receipt committed before lost ACK")
    with monkeypatch.context() as patch:
        patch.setattr(manager, "_write", fail)
        model.release["BUILD"].set()
        await asyncio.wait_for(state["handles"][0].task, 6)
    c = SimpleNamespace(manager=manager, actor=runtime.actor(sid), job_id=state["handles"][0].job_id)
    assert len(calls) == int(fault.startswith("exit"))
    if fault == "exit_ack":
        await closed(c)
    else:
        with pytest.raises(JobError, match="callback_unsettled"):
            await closed(c)
    await results.project(loop, c.actor, c.job_id, 1)
    assert len(calls) == int(fault.startswith("exit"))
    await results.run_callback(loop, c.actor, c.job_id, 1, callback)
    assert len(calls) == int(fault.startswith("exit") or fault == "start_rollback")
    assert (await manager.get_job(c.actor, c.job_id)).outcome == "succeeded"


@pytest.mark.parametrize("fault", ["missing", "json", "hash", "revision", "type", "inbox", "journal", "preparation_missing"])
async def test_corrupt_closed_evidence_cannot_authorize_takeover(delivery_case, fault):
    c = delivery_case
    await results.project(c.loop, c.actor, c.job_id, 1)
    key = results._id("projected", c.job_id, 1)
    if fault in ("missing", "preparation_missing"):
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?",
            (results.delivery_event_id(c.job_id, 1) if fault == "preparation_missing" else key,))
    elif fault in ("json", "hash", "revision", "type"):
        field, value = {"json": ("payload_json", "{}"), "hash": ("request_hash", "wrong"),
                        "revision": ("job_version", 999), "type": ("event_type", "other")}[fault]
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET {field}=? WHERE event_id=?", (value, key))
    elif fault == "inbox":
        await c.manager.db.execute(f"UPDATE {c.store.t.follow_up_queue} SET content='wrong' WHERE item_id=?", ("result_"+c.job_id,))
    else:
        await journal.update(c.store, c.actor.session_id, c.run_id, allow_terminal=True, status="running", result=None)
    with pytest.raises(JobError):
        await closed(c)


async def test_silent_closed_receipt_insert_is_not_success(delivery_case, monkeypatch):
    c = delivery_case
    original = c.manager._write
    @asynccontextmanager
    async def noop():
        async with original() as tx:
            async def execute(sql, params=()):
                if "workflow_projected" in params:
                    return 0
                return await tx.execute(sql, params)
            yield SimpleNamespace(execute=execute, fetchone=tx.fetchone, fetchall=tx.fetchall)
    monkeypatch.setattr(c.manager, "_write", noop)
    with pytest.raises(JobError, match="projection_receipt_mismatch"):
        await results.project(c.loop, c.actor, c.job_id, 1)
    assert (await c.store.inbox_counts(c.actor.session_id))["pending"] == 0
    assert (await journal.read(c.store, c.actor.session_id, c.run_id))["status"] == "running"


@pytest.mark.parametrize("ending", ["complete", "raise", "cancel"])
async def test_callback_exit_is_separate_and_never_replayed(managed_workflow, ending):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def callback(completion):
        calls.append(completion)
        entered.set()
        await release.wait()
        if ending == "raise":
            raise RuntimeError("callback delivery failed")
    state["on_complete"] = callback
    await loop.send("run", sid)
    model.release["BUILD"].set()
    handle = state["handles"][0]
    c = SimpleNamespace(manager=manager, actor=runtime.actor(sid), job_id=handle.job_id)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        with pytest.raises(JobError, match="callback_unsettled"):
            await closed(c)
        await results.run_callback(loop, c.actor, c.job_id, 1, callback)
        assert len(calls) == 1
        if ending == "cancel":
            handle.task.cancel()
        release.set()
        await asyncio.wait_for(asyncio.gather(handle.task, return_exceptions=True), 5)
        if ending == "complete":
            await closed(c)
        else:
            with pytest.raises(JobError, match="callback_unsettled"):
                await closed(c)
        await results.project(loop, c.actor, c.job_id, 1)
        await results.run_callback(loop, c.actor, c.job_id, 1, callback)
        assert len(calls) == 1
        event = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (results._id("cb_exit", c.job_id, 1),))
        assert json.loads(event["payload_json"])["status"] == {"complete": "completed", "raise": "failed", "cancel": "cancelled"}[ending]
        assert (await manager.get_job(c.actor, c.job_id)).outcome == "succeeded"
    finally:
        release.set()


@pytest.mark.parametrize("fault", ["start_missing", "exit_missing", "start_json", "exit_json", "start_hash", "exit_hash",
                                  "exit_token", "start_version_bool", "exit_status", "exit_type"])
async def test_corrupt_callback_receipts_do_not_become_completion(managed_workflow, fault):
    _, manager, runtime, loop, model, sid, state, _ = managed_workflow
    await loop.send("run", sid)
    model.release["BUILD"].set()
    await state["handles"][0].task
    c = SimpleNamespace(manager=manager, actor=runtime.actor(sid), job_id=state["handles"][0].job_id)
    await closed(c)
    phase, change = fault.split('_', 1)
    key = results._id("cb_"+phase, c.job_id, 1)
    if change == "missing":
        await manager.db.execute(f"DELETE FROM {manager.events} WHERE event_id=?", (key,))
    elif change in ("json", "hash", "type"):
        field, value = {"json": ("payload_json", "[]"), "hash": ("request_hash", "bad"), "type": ("event_type", "other")}[change]
        await manager.db.execute(f"UPDATE {manager.events} SET {field}=? WHERE event_id=?", (value, key))
    else:
        row = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (key,))
        value = json.loads(row["payload_json"])
        value.update({"token": "x"*32} if change == "token" else {"version": True} if change == "version_bool" else {"status": "maybe"})
        await manager.db.execute(f"UPDATE {manager.events} SET payload_json=?,request_hash=? WHERE event_id=?", (_json(value), _hash(value), key))
    with pytest.raises(JobError):
        await closed(c)
    assert len(state["notifications"]) == 1


@pytest.mark.parametrize("legacy", ["absent", "invalid"])
async def test_legacy_callback_contract_is_never_guessed(managed_workflow, monkeypatch, legacy):
    from power_loop.runtime import workflow_tasks
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    state["on_complete"] = None
    original = workflow_tasks.admit_workflow
    async def admit(*args, **kwargs):
        reg = await original(*args, **kwargs)
        payload = dict(reg.operation.result)
        payload["request"] = dict(payload["request"])
        payload["request"].pop("completion_callback")
        if legacy == "invalid":
            payload["request"]["completion_callback"] = "maybe"
        # Emulate a historical host admission before the controller hash is
        # frozen; never change a bound controller or repair corrupt evidence.
        await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (_json(payload), reg.operation.operation_id))
        return reg
    monkeypatch.setattr(workflow_tasks, "admit_workflow", admit)
    await loop.send("run", sid)
    model.release["BUILD"].set()
    await state["handles"][0].task
    c = SimpleNamespace(manager=manager, actor=runtime.actor(sid), job_id=state["handles"][0].job_id)
    with pytest.raises(JobError, match="callback_contract_"+("unknown" if legacy == "absent" else "mismatch")):
        await closed(c)
    assert (await store.inbox_counts(sid))["pending"] == (1 if legacy == "absent" else 0)
