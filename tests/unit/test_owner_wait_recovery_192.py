"""Original waiting transitions survive a single storage ACK loss or rollback."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.core.agent_context import get_session_id
from power_loop.runtime import delegation, owner_waits
from power_loop.runtime import wait_effects as effects
from power_loop.runtime import wait_notifications as notices
from power_loop.runtime.jobs import JobError, JobLease, _hash

from .test_job_registry_192 import registry_store
from .test_owner_waits_192 import entered, original, prepare_due

__all__ = ["registry_store", "original"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def fail_transition_write(c, monkeypatch, phase, point):
    write = c.manager._write
    fired = []

    @asynccontextmanager
    async def transaction():
        changed = False
        async with write() as tx:
            before = await tx.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type=?", (phase,))
            yield tx
            after = await tx.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type=?", (phase,))
            changed = len(after) > len(before) and not fired
            if changed:
                fired.append(point)
                if point == "before_commit":
                    raise OSError("injected control transaction rollback")
        if changed and point == "after_commit":
            raise OSError("injected committed control ACK loss")

    monkeypatch.setattr(c.manager, "_write", transaction)
    return fired


async def continue_original(c, registration):
    due = await prepare_due(c.manager, c.runtime.actor(c.state.parent), registration)
    c.state.choice, c.state.due_id = "continue", due["event_id"]
    await notices.enqueue_notification(c.loop, c.runtime.actor(c.state.parent), due["event_id"])
    await c.loop.flush_follow_ups(c.state.parent)
    await effects.apply(c.manager, c.runtime.actor(c.state.parent), due["event_id"])


@pytest.mark.parametrize("phase", ["owner_waiting", "owner_wait_resumed"])
@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
async def test_original_controller_confirms_only_the_same_wait_transition(original, monkeypatch, phase, point):
    c = original
    c.state.wait_ms = 20000
    fired = fail_transition_write(c, monkeypatch, phase, point)
    registration, owner = await entered(c)
    assert c.model.calls["child"] == 2
    await continue_original(c, registration)
    await asyncio.wait_for(asyncio.gather(*set(delegation._TASKS), return_exceptions=True), 5)
    current = await c.manager.get_job(c.runtime.actor(c.state.parent), owner.job_id)
    assert fired == [point]
    assert current.outcome == "succeeded" and c.model.calls["child"] == 3, current.result
    assert len(await c.store.list_children(c.state.parent)) == 1
    for kind in ("owner_waiting", "owner_wait_resumed"):
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type=?", (owner.job_id, kind))) == 1
    await c.loop.flush_follow_ups(c.state.parent)
    assert all(row["supervision_status"] == "closed" for row in await c.manager.db.fetchall(f"SELECT supervision_status FROM {c.manager.jobs}"))


@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
async def test_expiry_confirmation_records_one_fact_and_no_extra_model_request(original, monkeypatch, point):
    c = original
    c.state.wait_ms = 2000
    fired = fail_transition_write(c, monkeypatch, "owner_wait_expired", point)
    _, owner = await entered(c)
    await asyncio.wait_for(asyncio.gather(*set(delegation._TASKS), return_exceptions=True), 5)
    current = await c.manager.get_job(c.runtime.actor(c.state.parent), owner.job_id)
    assert fired == [point]
    assert current.outcome == "timed_out" and c.model.calls["child"] == 2, current.result
    assert current.result["usage"]["total_tokens"] == 20
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_expired'")) == 1
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")
    await c.loop.flush_follow_ups(c.state.parent)


@pytest.fixture
async def paused(original, monkeypatch):
    c = original
    c.state.wait_ms = 20000
    ready, release = asyncio.Event(), asyncio.Event()
    step = owner_waits._step

    async def held(*args):
        ready.set()
        await release.wait()
        return await step(*args)

    monkeypatch.setattr(owner_waits, "_step", held)
    registration, owner = await entered(c)
    await asyncio.wait_for(ready.wait(), 2)
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='owner_waiting'", (owner.job_id,))
    saved = json.loads(event["payload_json"])
    async def transition():
        return await step(c.manager, c.runtime.actor(c.state.parent), JobLease(**saved["owner"]), event["event_id"], saved)
    try:
        yield SimpleNamespace(c=c, registration=registration, owner=owner, event=event, saved=saved,
            transition=transition, step=step, release=release)
    finally:
        release.set()


async def late_nudge(c, sid, monkeypatch):
    complete = c.model.complete
    issued = False
    async def provider(request, **kwargs):
        nonlocal issued
        result = await complete(request, **kwargs)
        if get_session_id() == c.state.parent and not issued:
            issued = True
            result.raw_text = ""
            result.tool_calls = [{"id": "late-original-nudge", "type": "function", "function": {"name": "spawn_agent",
                "arguments": json.dumps({"action": "nudge", "sub_session": sid, "message": "Also confirm this original request."})}}]
        return result
    monkeypatch.setattr(c.model, "complete", provider)
    await c.loop.send("Send one additional original input", c.state.parent)


async def test_committed_resume_recall_keeps_frozen_inputs_when_later_input_arrives(paused, monkeypatch):
    p, c = paused, paused.c
    await continue_original(c, p.registration)
    assert await p.transition() == "input"
    [before] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")
    await late_nudge(c, p.owner.session_id, monkeypatch)
    assert await p.transition() == "input"
    [after] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")
    assert after == before and c.model.calls["child"] == 2
    p.release.set()
    await asyncio.wait_for(asyncio.gather(*set(delegation._TASKS), return_exceptions=True), 5)
    assert c.model.calls["child"] == 3
    assert (await c.manager.get_job(c.runtime.actor(c.state.parent), p.owner.job_id)).outcome == "succeeded"
    inputs = await c.manager.db.fetchall(f"SELECT delivery_status FROM {c.manager.events} WHERE job_id=? AND event_type='operation' AND action='nudge'", (p.owner.job_id,))
    assert len(inputs) == 2 and all(row["delivery_status"] == "handled" for row in inputs)


async def test_expired_resume_recall_cannot_add_a_contradictory_wait_expiry(paused, monkeypatch):
    p, c = paused, paused.c
    await continue_original(c, p.registration)
    assert await p.transition() == "input"
    # Keep the actual current lease valid; test the shorter policy alone.
    deadline = p.saved["deadline_at_ms"]
    monkeypatch.setattr(c.manager, "now", lambda: deadline)
    with pytest.raises(owner_waits.WaitDeadline):
        await p.transition()
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")) == 1
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_expired'")
    p.release.set()
    await asyncio.wait_for(asyncio.gather(*set(delegation._TASKS), return_exceptions=True), 5)
    assert c.model.calls["child"] == 2
    assert (await c.manager.get_job(c.runtime.actor(c.state.parent), p.owner.job_id)).outcome == "timed_out"


async def test_read_reopened_wait_history_after_resume_result_and_policy_expiry(paused, monkeypatch):
    p, c = paused, paused.c
    reader = type(c.manager)(await c.connect())
    initial = await owner_waits.read(reader, c.runtime.actor(c.state.parent), p.owner.job_id)
    assert initial["generation"] == 1 and len(initial["phases"]) == 1
    assert initial["phases"][0]["waiting"] == p.saved and initial["phases"][0]["resumed"] is None
    assert c.model.calls["child"] == 2
    await continue_original(c, p.registration)
    await p.transition()
    resumed = await owner_waits.read(reader, c.runtime.actor(c.state.parent), p.owner.job_id, generation=1)
    assert resumed["phases"][0]["resumed"] is not None and resumed["phases"][0]["expired"] is None
    p.release.set()
    await asyncio.wait_for(asyncio.gather(*set(delegation._TASKS), return_exceptions=True), 5)
    await c.loop.flush_follow_ups(c.state.parent)
    monkeypatch.setattr(reader, "now", lambda: p.saved["deadline_at_ms"] + 1)
    assert await owner_waits.read(reader, c.runtime.actor(c.state.parent), p.owner.job_id, generation=1) == resumed
    assert c.model.calls["child"] == 3


async def test_read_cannot_accept_a_resume_identity_with_the_wrong_event_role(paused):
    p, c = paused, paused.c
    await continue_original(c, p.registration)
    await p.transition()
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")
    try:
        async with c.manager._write() as tx:
            await tx.execute(f"UPDATE {c.manager.events} SET event_type='operation' WHERE event_id=?", (event["event_id"],))
        with pytest.raises(JobError):
            await owner_waits.read(c.manager, c.runtime.actor(c.state.parent), p.owner.job_id)
    finally:
        async with c.manager._write() as tx:
            await tx.execute(f"UPDATE {c.manager.events} SET event_type=? WHERE event_id=?", (event["event_type"], event["event_id"]))


@pytest.mark.parametrize("generation", [True, 0, -1, "1", 1.5, 2])
async def test_read_does_not_invent_a_requested_generation(paused, generation):
    p, c = paused, paused.c
    with pytest.raises(JobError):
        await owner_waits.read(c.manager, c.runtime.actor(c.state.parent), p.owner.job_id, generation=generation)


@pytest.mark.parametrize("identity", ["foreign_tenant", "unrelated_session"])
async def test_read_requires_original_authorized_scope(paused, identity):
    p, c = paused, paused.c
    actor = c.runtime.actor(c.state.parent)
    actor = replace(actor, tenant_id="foreign") if identity == "foreign_tenant" else replace(actor, session_id=await c.loop.new_session())
    with pytest.raises(JobError, match="access_denied"):
        await owner_waits.read(c.manager, actor, p.owner.job_id)


@pytest.mark.parametrize("kind,allowed", [("io", True), ("timeout", True), ("domain", False), ("deadline", False),
    ("programming", False), ("sql_syntax", False), ("pg_connection", True), ("pg_serialization", True), ("pg_unique", False)])
async def test_only_known_storage_errors_allow_one_same_operation_confirmation(kind, allowed):
    if kind.startswith("pg_"):
        pg = pytest.importorskip("asyncpg")
        exc = {"pg_connection": pg.PostgresConnectionError, "pg_serialization": pg.SerializationError,
            "pg_unique": pg.UniqueViolationError}[kind]("test driver error")
    elif kind == "sql_syntax":
        db = sqlite3.connect(":memory:")
        try:
            db.execute("definitely not SQL")
        except sqlite3.OperationalError as error:
            exc = error
        finally:
            db.close()
    else:
        exc = {"io": OSError, "timeout": TimeoutError, "domain": JobError,
            "deadline": owner_waits.WaitDeadline, "programming": RuntimeError}[kind]("test error")
    calls = []
    async def operation():
        calls.append(1)
        if len(calls) == 1:
            raise exc
        return 42
    if allowed:
        assert await owner_waits._confirm(operation) == 42 and len(calls) == 2
    else:
        with pytest.raises(type(exc)):
            await owner_waits._confirm(operation)
        assert len(calls) == 1


async def test_continuing_storage_failure_is_bounded_and_cancellation_is_not_retried():
    calls = []
    async def unavailable():
        calls.append(1)
        raise OSError("still unavailable")
    with pytest.raises(OSError):
        await owner_waits._confirm(unavailable)
    assert len(calls) == 2
    calls.clear()
    async def cancelled():
        calls.append(1)
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await owner_waits._confirm(cancelled)
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["boolean_generation", "old_generation", "old_boot", "cancel", "lease_expired", "model_unknown"])
async def test_resume_confirmation_keeps_original_execution_fences(paused, monkeypatch, fault):
    p, c = paused, paused.c
    await continue_original(c, p.registration)
    await p.transition()
    lease = JobLease(**p.saved["owner"])
    restored = None
    if fault in ("boolean_generation", "old_generation", "old_boot"):
        lease = replace(lease, **({"boot_id": "old-boot"} if fault == "old_boot" else {
            "generation": True if fault == "boolean_generation" else lease.generation + 1}))
    elif fault == "cancel":
        await c.manager.request_cancel(c.runtime.actor(c.state.parent), p.owner.job_id, system_key="stop-before-confirm", reason="stop original")
    else:
        key = p.owner.job_id
        column, value = "lease_until_ms", c.manager.now()
        if fault == "model_unknown":
            [row] = await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE control_owner_job_id=? AND kind='send'", (p.owner.job_id,))
            key, column, value = row["job_id"], "resource_state", "unknown"
        [before] = await c.manager.db.fetchall(f"SELECT {column} FROM {c.manager.jobs} WHERE job_id=?", (key,))
        async with c.manager._write() as tx:
            await tx.execute(f"UPDATE {c.manager.jobs} SET {column}=? WHERE job_id=?", (value, key))
        restored = (column, before[column], key)
    try:
        with pytest.raises(JobError):
            await p.step(c.manager, c.runtime.actor(c.state.parent), lease, p.event["event_id"], p.saved)
        assert c.model.calls["child"] == 2
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")) == 1
    finally:
        if restored:
            column, value, key = restored
            async with c.manager._write() as tx:
                await tx.execute(f"UPDATE {c.manager.jobs} SET {column}=? WHERE job_id=?", (value, key))


async def test_read_rejects_rehashed_original_tracking_changes(paused):
    p, c = paused, paused.c
    [event] = await c.manager.db.fetchall(f"SELECT e.* FROM {c.manager.events} e JOIN {c.manager.jobs} j ON j.job_id=e.job_id "
        "WHERE j.control_owner_job_id=? AND e.event_type='model_attempt_tracking'", (p.owner.job_id,))
    corrupt = {**json.loads(event["payload_json"]), "session_id": "invented-session"}
    try:
        async with c.manager._write() as tx:
            await tx.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?", (json.dumps(corrupt), _hash(corrupt), event["event_id"]))
        with pytest.raises(JobError):
            await owner_waits.read(c.manager, c.runtime.actor(c.state.parent), p.owner.job_id)
        assert c.model.calls["child"] == 2
    finally:
        async with c.manager._write() as tx:
            await tx.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?", (event["payload_json"], event["request_hash"], event["event_id"]))


async def test_read_requires_the_original_ordinary_nudge_receipt(paused, monkeypatch):
    p, c = paused, paused.c
    await continue_original(c, p.registration)
    await late_nudge(c, p.owner.session_id, monkeypatch)
    await p.transition()
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='operation' AND action='nudge' AND call_ref_json IS NOT NULL", (p.owner.job_id,))
    try:
        async with c.manager._write() as tx:
            await tx.execute(f"UPDATE {c.manager.events} SET receipt_ref_json=NULL WHERE event_id=?", (event["event_id"],))
        with pytest.raises(JobError, match="receipt_missing"):
            await owner_waits.read(c.manager, c.runtime.actor(c.state.parent), p.owner.job_id)
    finally:
        async with c.manager._write() as tx:
            await tx.execute(f"UPDATE {c.manager.events} SET receipt_ref_json=? WHERE event_id=?", (event["receipt_ref_json"], event["event_id"]))


@pytest.mark.parametrize("fault,expected_calls", [("domain", 1), ("persistent_storage", 2)])
async def test_native_controller_does_not_repeat_domain_checks_or_retry_storage_forever(original, monkeypatch, fault, expected_calls):
    c = original
    calls = []
    async def unavailable(*args):
        calls.append(1)
        raise JobError("owner_wait_evidence_corrupt") if fault == "domain" else OSError("still unavailable")
    monkeypatch.setattr(owner_waits, "_enter", unavailable)
    await c.loop.send("Start original finite task", c.state.parent)
    await asyncio.wait_for(asyncio.gather(*set(delegation._TASKS), return_exceptions=True), 5)
    assert len(calls) == expected_calls and c.model.calls["child"] == 2
    [registration] = c.state.registrations
    owner = await c.manager.get_job(c.runtime.actor(c.state.parent), registration.job.control_owner_job_id)
    assert owner.outcome == "failed" and owner.result["usage"]["total_tokens"] == 20
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")
    await c.loop.flush_follow_ups(c.state.parent)


async def test_wait_identity_is_stable_when_database_returns_required_timers_in_reverse_order(original, monkeypatch):
    c = original
    c.state.wait_ms = 20000
    complete = c.model.complete
    step = owner_waits._step
    release = asyncio.Event()
    async def provider(request, **kwargs):
        result = await complete(request, **kwargs)
        if get_session_id() != c.state.parent and c.model.calls["child"] == 1:
            another = json.loads(json.dumps(result.tool_calls[0]))
            another["id"] += "-second"
            result.tool_calls.append(another)
        return result
    async def held(*args):
        await release.wait()
        return await step(*args)
    monkeypatch.setattr(c.model, "complete", provider)
    monkeypatch.setattr(owner_waits, "_step", held)
    try:
        await c.loop.send("Start original finite task", c.state.parent)
        for _ in range(300):
            if len(c.state.registrations) == 2:
                owner = await c.manager.get_job(c.runtime.actor(c.state.parent), c.state.registrations[0].job.control_owner_job_id)
                if owner.status == "waiting":
                    break
            await asyncio.sleep(.01)
        else:
            pytest.fail("The original two-wait controller did not enter waiting")
        [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='owner_waiting'")
        saved = json.loads(event["payload_json"])
        async with c.manager._write() as tx:
            transaction_type = type(tx)
        fetch = transaction_type.fetchall
        async def reverse_rows(self, sql, params=()):
            result = await fetch(self, sql, params)
            return list(reversed(result)) if "creation_json" in sql else result
        monkeypatch.setattr(transaction_type, "fetchall", reverse_rows)
        assert await owner_waits._enter(c.manager, c.runtime.actor(c.state.parent), JobLease(**saved["owner"]), saved["raw"]) == (event["event_id"], saved)
        assert [v["job_id"] for v in saved["waits"]] == sorted(v["job_id"] for v in saved["waits"])
        assert c.model.calls["child"] == 2
    finally:
        release.set()


async def test_expired_history_remains_readable_but_a_wrong_expiry_role_is_rejected(paused, monkeypatch):
    p, c = paused, paused.c
    monkeypatch.setattr(c.manager, "now", lambda: p.saved["deadline_at_ms"])
    assert await p.transition() == "expired"
    history = await owner_waits.read(c.manager, c.runtime.actor(c.state.parent), p.owner.job_id)
    assert history["phases"][0]["expired"] and history["phases"][0]["resumed"] is None
    event_id = history["phases"][0]["expired"]["event_id"]
    try:
        async with c.manager._write() as tx:
            await tx.execute(f"UPDATE {c.manager.events} SET event_type='operation' WHERE event_id=?", (event_id,))
        with pytest.raises(JobError):
            await owner_waits.read(c.manager, c.runtime.actor(c.state.parent), p.owner.job_id)
    finally:
        async with c.manager._write() as tx:
            await tx.execute(f"UPDATE {c.manager.events} SET event_type='owner_wait_expired' WHERE event_id=?", (event_id,))


async def test_malformed_input_and_transition_metadata_return_domain_errors(paused):
    p, c = paused, paused.c
    await continue_original(c, p.registration)
    await p.transition()
    [resume] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")
    [input_event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='operation' AND action='nudge'", (p.owner.job_id,))
    async with c.manager._write() as tx:
        owner = await c.manager._row(tx, p.owner.job_id)
        for request in (None, [], "wrong"):
            with pytest.raises(JobError, match="owner_wait_input_corrupt"):
                await owner_waits._input(c.manager, tx, c.runtime.actor(c.state.parent), owner, p.owner.session_id,
                    {**input_event, "payload_json": json.dumps({"request": request})})
        for timestamp in (None, "tomorrow"):
            with pytest.raises(JobError):
                await owner_waits._resume(c.manager, tx, c.runtime.actor(c.state.parent), owner, p.event["event_id"], p.saved,
                    {**resume, "created_at": timestamp})
    assert c.model.calls["child"] == 2
