"""Pending discovery is atomic and independent of completed control history."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop.runtime import session_obligations as obligations
from power_loop.runtime.abort_outcomes import pending_reviews
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.store.task_schema import TASK_PROTOCOL_VERSION, migrate_task_schema

from .test_aborted_call_outcomes_192 import pending_batch
from .test_job_registry_192 import registry_store, runtime
from .test_send_sql_profile import Profiled
from .test_tool_call_plans_192 import controller
from .test_wait_handoff_192 import handoff, waiting

__all__ = ["registry_store", "runtime", "handoff", "waiting"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def rows(manager):
    return await manager.db.fetchall(f"SELECT * FROM {manager.prefix}session_obligations ORDER BY session_id,kind,source_id")


@pytest.mark.parametrize("outcome", ["rollback", "lost_ack", "commit"])
async def test_original_notice_and_pending_index_share_the_transaction(registry_store, monkeypatch, outcome):
    from power_loop.runtime.tool_call_plans import reconcile_pending

    c, source = await pending_batch(registry_store)
    loop, model = await controller(c)
    before = await rows(c.manager)
    original = c.manager.db.transaction
    fired = []

    @asynccontextmanager
    async def transaction():
        async with original() as tx:
            yield tx
            if not fired and await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_type='source_abort_delivery'"):
                fired.append(True)
                if outcome == "rollback":
                    raise OSError("original transaction rolled back")
        if fired and outcome == "lost_ack":
            raise OSError("original commit acknowledgement lost")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(c.manager.db, "transaction", transaction)
            if outcome == "commit":
                await reconcile_pending(loop, c.actor, source["job_id"])
            else:
                with pytest.raises(OSError, match="original"):
                    await reconcile_pending(loop, c.actor, source["job_id"])
        assert fired and model.calls == 0 and c.state.dispatched == 0
        cold = JobManager(await registry_store[1]())
        if outcome == "rollback":
            assert await rows(cold) == before
            assert await pending_reviews(cold, cold.db, c.actor) == []
        else:
            assert {row["kind"] for row in await rows(cold)} == {"abort", "disposition"}
            assert len(await pending_reviews(cold, cold.db, c.actor)) == 1
        await reconcile_pending(loop, c.actor, source["job_id"])
        assert len(await pending_reviews(cold, cold.db, c.actor)) == 1
        assert (await loop.send("Review only the original outcome.", c.sid)).status == "completed"
        assert await rows(cold) == []
        assert model.calls == c.model.calls == 1 and c.state.dispatched == 0
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize("outcome", ["rollback", "lost_ack", "commit"])
async def test_current_wait_input_and_pending_index_commit_together(handoff, monkeypatch, outcome):
    from power_loop.runtime.wait_effects import apply

    from .test_wait_effects_192 import decide

    h, c = handoff, handoff.c
    await decide(h)
    before = await rows(c.manager)
    messages = await c.store.load_all_messages(c.sid)
    model_calls = h.model.calls
    original = c.manager.db.transaction
    fired = []

    @asynccontextmanager
    async def transaction():
        async with original() as tx:
            yield tx
            if not fired and await tx.fetchone(f"SELECT event_id FROM {c.manager.events} "
                    "WHERE job_id=? AND action='nudge' AND delivery_status='pending_delivery'", (c.owner.job_id,)):
                fired.append(True)
                if outcome == "rollback":
                    raise OSError("original input transaction rolled back")
        if fired and outcome == "lost_ack":
            raise OSError("original input acknowledgement lost")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(c.manager.db, "transaction", transaction)
            if outcome == "commit":
                await apply(c.manager, c.main, h.event_id)
            else:
                with pytest.raises(OSError, match="original input"):
                    await apply(c.manager, c.main, h.event_id)
        assert fired
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        if outcome == "rollback":
            assert await rows(cold) == before
            assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE action='nudge'")
        else:
            indexed = [row for row in await rows(cold) if row["kind"] == "input"]
            [event] = await cold.db.fetchall(f"SELECT * FROM {cold.events} WHERE action='nudge'")
            assert indexed == [{"session_id": c.sid, "kind": "input", "source_id": event["event_id"],
                "job_id": c.owner.job_id, "opened_version": event["job_version"]}]
        reply = await apply(c.manager, c.main, h.event_id)
        assert await apply(cold, c.main, h.event_id) == reply
        [indexed] = [row for row in await rows(cold) if row["kind"] == "input"]
        [event] = await obligations.event_rows(cold, cold.db, c.actor, "input", job_id=c.owner.job_id)
        assert indexed["source_id"] == event["event_id"] == reply["operation_id"]
        [item] = await c.store.inbox_items(c.sid)
        assert item["item_id"] == "nudge:" + event["event_id"] and item["status"] == "pending"
        await obligations.verify(cold, c.actor)
        assert await c.store.load_all_messages(c.sid) == messages and h.model.calls == model_calls
    finally:
        await h.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize("damage", [None, "handoff", "effect", "receipt"])
async def test_cold_wait_input_rebuild_uses_original_decision_proof(handoff, damage):
    from power_loop.runtime.wait_effects import apply

    from .test_wait_effects_192 import decide

    h, c = handoff, handoff.c
    try:
        await decide(h)
        reply = await apply(c.manager, c.main, h.event_id)
        expected = await rows(c.manager)
        calls = h.model.calls
        messages = await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} ORDER BY id")
        if damage is None:
            await c.manager.db.execute(f"DELETE FROM {c.manager.prefix}session_obligations "
                "WHERE job_id=? AND kind='input'", (c.owner.job_id,))
        elif damage == "handoff":
            op = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?",
                                            (reply["operation_id"],))
            data = json.loads(op["payload_json"])
            del data["request"]["handoff"]
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
                (json.dumps(data), reply["operation_id"]))
        elif damage == "effect":
            await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='wait_effect'")
        else:
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET receipt_ref_json=NULL WHERE action='wait_decision'")
        originals = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        if damage is None:
            with pytest.raises(JobError, match="session_obligation_projection_mismatch"):
                await obligations.verify(cold, c.actor)
            await obligations.rebuild(cold)
            await obligations.verify(cold, c.actor)
        else:
            with pytest.raises(JobError):
                await obligations.rebuild(cold)
        assert await rows(cold) == expected
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == originals
        assert await cold.db.fetchall(f"SELECT * FROM {c.store.t.messages} ORDER BY id") == messages
        assert h.model.calls == calls
    finally:
        await h.loop.aclose(drain_timeout_s=.1)


async def test_cold_rebuild_recovers_missing_index_without_replaying_or_changing_originals(registry_store):
    from power_loop.runtime.tool_call_plans import reconcile_pending

    c, source = await pending_batch(registry_store)
    loop, model = await controller(c)
    try:
        await reconcile_pending(loop, c.actor, source["job_id"])
        expected = await rows(c.manager)
        original_events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
        original_messages = await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} ORDER BY id")
        original_calls = c.host.reserve_calls, c.host.apply_calls, c.model.calls, c.state.dispatched
        await c.manager.db.execute(f"DELETE FROM {c.manager.prefix}session_obligations")
        from power_loop.runtime import session_controls
        from power_loop.runtime.jobs import TRUST_OWN_WRITES
        seen = await session_controls.version(c.manager, c.manager.db, c.sid)
        session_controls.mark_clean(c.manager, c.sid, seen)
        cold = JobManager(await registry_store[1]())
        with pytest.raises(JobError, match="session_obligation_projection_mismatch"):
            await obligations.verify(cold, c.actor)
        token = TRUST_OWN_WRITES.set(True)
        try:
            assert session_controls.is_clean(c.manager, c.sid, seen)
            await obligations.rebuild(c.manager)
            changed = await session_controls.version(c.manager, c.manager.db, c.sid)
            assert not session_controls.is_clean(c.manager, c.sid, changed)
        finally:
            TRUST_OWN_WRITES.reset(token)
        assert await rows(cold) == expected
        await obligations.verify(cold, c.actor)
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == original_events
        assert await cold.db.fetchall(f"SELECT * FROM {cold.store.t.messages} ORDER BY id") == original_messages
        assert (c.host.reserve_calls, c.host.apply_calls, c.model.calls, c.state.dispatched) == original_calls
        assert model.calls == 0 and len(await pending_reviews(cold, cold.db, c.actor)) == 1
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize("damage", ["notice", "result", "closure"])
async def test_cold_rebuild_refuses_damaged_originals_and_keeps_the_barrier(registry_store, damage):
    from power_loop.runtime.tool_call_plans import reconcile_pending

    c, source = await pending_batch(registry_store)
    loop, model = await controller(c)
    removed = []
    source_status = None
    try:
        await reconcile_pending(loop, c.actor, source["job_id"])
        original = await rows(c.manager)
        if damage == "closure":
            source_status = await c.manager.db.fetchone(
                f"SELECT supervision_status FROM {c.manager.jobs} WHERE job_id=?", (source["job_id"],))
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET supervision_status='closed' WHERE job_id=?", (source["job_id"],))
        else:
            removed = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type=?",
                (source["job_id"], "source_abort_delivery" if damage == "notice" else "result"))
            await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type=?",
                (source["job_id"], "source_abort_delivery" if damage == "notice" else "result"))
        damaged_events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
        cold = JobManager(await registry_store[1]())
        with pytest.raises(JobError):
            await obligations.rebuild(cold)
        assert await rows(cold) == original
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == damaged_events
        assert c.model.calls == 1 and model.calls == c.state.dispatched == 0
    finally:
        # Restore only this fault injection after all rejection/immutability
        # assertions, so the common final invariant compares valid originals.
        for event in removed:
            columns = tuple(event)
            await c.manager.db.execute(
                f"INSERT INTO {c.manager.events} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                tuple(event[key] for key in columns))
        if source_status is not None:
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET supervision_status=? WHERE job_id=?",
                (source_status["supervision_status"], source["job_id"]))
        await loop.aclose(drain_timeout_s=.1)


async def test_physically_closed_unstarted_actions_keep_their_unreviewed_notices(registry_store):
    from power_loop.runtime.managed_delivery import enqueue
    from power_loop.runtime.managed_effects import terminate_unstarted
    from power_loop.runtime.pending_calls import retire_pending

    c, source = await pending_batch(registry_store, stage="commit_ack")
    loop, model = await controller(c)
    try:
        children = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
        assert len(children) == 2
        for child in children:
            await terminate_unstarted(c.manager, c.actor, child["job_id"])
            await enqueue(loop, c.actor, child["job_id"])
            await c.manager.finalize_supervision(c.actor, child["job_id"], generation=0)
        await retire_pending(loop, c.actor, source["job_id"])
        assert {entry["job_id"] for entry in await rows(c.manager) if entry["kind"] == "managed"} == {
            child["job_id"] for child in children}
        assert len(await pending_reviews(c.manager, c.manager.db, c.actor)) == 2
        assert model.calls == c.state.dispatched == 0
        assert (await loop.send("Review the original negative results only.", c.sid)).status == "completed"
        assert await rows(c.manager) == []
        assert c.model.calls == model.calls == 1 and c.state.dispatched == 0
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def test_explicit_protocol_23_maintenance_rebuilds_the_index(registry_store):
    from power_loop.runtime.store.schema import StoreSchemaError
    from power_loop.runtime.store.task_schema import verify_task_schema
    from power_loop.runtime.tool_call_plans import reconcile_pending

    c, source = await pending_batch(registry_store)
    loop, _ = await controller(c)
    try:
        await reconcile_pending(loop, c.actor, source["job_id"])
        expected = await rows(c.manager)
        await loop.aclose(drain_timeout_s=.1)
        await c.runtime.shutdown(timeout_s=1)
        await c.manager.db.execute(f"DROP TABLE {c.manager.prefix}session_obligations")
        await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_store SET protocol_version=23 WHERE id=1")
        with pytest.raises(StoreSchemaError):
            await verify_task_schema(c.manager.db, c.manager.prefix)
        with pytest.raises(StoreSchemaError, match="maintenance_required"):
            await migrate_task_schema(c.manager.db, c.manager.prefix)
        await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
        cold = JobManager(await registry_store[1]())
        assert await rows(cold) == expected
        assert (await cold.db.fetchone(f"SELECT protocol_version FROM {cold.prefix}task_store"))["protocol_version"] == TASK_PROTOCOL_VERSION
        assert c.model.calls == 1 and c.state.dispatched == 0
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize("mode", ["produce", "consume"])
@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
async def test_process_death_keeps_originals_and_a_rebuildable_pending_index(registry_store, tmp_path, mode, point):
    store, connect = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    evidence = tmp_path / "obligation-crash.json"
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._obligation_crash_worker",
        dsn, store.table_prefix, str(evidence), mode, point, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 40)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == (152 if point == "before_commit" else 153), output.decode()
    saved = json.loads(evidence.read_text())
    cold = JobManager(await connect())
    actor = JobActor(**saved["actor"])
    pending = point == "after_commit" if mode == "produce" else point == "before_commit"
    # Production originals and their initial index still commit atomically.
    # Consumption's clean handler exit precedes the target-tree projection;
    # interrupted projection remains conservative and is rebuilt mechanically.
    assert bool(await pending_reviews(cold, cold.db, actor)) == (pending if mode == "produce" else False)
    indexed = await rows(cold)
    assert bool(indexed) == pending
    assert {row["kind"] for row in indexed} == ({"abort", "disposition"} if pending else set())
    if mode == "consume" and pending:
        originals = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        messages = await cold.db.fetchall(f"SELECT * FROM {cold.store.t.messages} ORDER BY id")
        with pytest.raises(JobError, match="session_obligation_projection_mismatch"):
            await obligations.verify(cold, actor)
        await obligations.rebuild(cold)
        await obligations.verify(cold, actor)
        assert await rows(cold) == []
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == originals
        assert await cold.db.fetchall(f"SELECT * FROM {cold.store.t.messages} ORDER BY id") == messages
    assert saved["model_calls"] == 1 and saved["dispatched"] == 0
    assert saved["review_calls"] == int(mode == "consume")
    assert saved["reserve_calls"] == saved["apply_calls"] == 1 + int(mode == "consume")


async def test_completed_control_history_does_not_enter_discovery(runtime):
    store, manager, actor, _, parent, connect, _now = runtime
    handler_count = 0

    async def handler():
        nonlocal handler_count
        handler_count += 1
        registered = await manager.register_job(actor, kind="send", session_id=actor.session_id,
            system_key=f"handler-{handler_count}", request={"send_index": handler_count})
        return await manager.claim_job(actor, registered.job.job_id, executor_id="handler", boot_id="boot", expected_version=1)

    probe = await handler()

    async def discover():
        cold = JobManager(await connect(), now_ms=manager.now)
        original = cold.db
        cold.db = measured = Profiled(original)
        try:
            assert await pending_reviews(cold, cold.db, actor) == []
            report = {}
            assert await cold.prepare_processing(actor, probe, round_index=0, send_index=1, messages=[], report=report) is None
            assert report == {"quiet": True}
            return [(record["sql"].replace(cold.prefix, "PREFIX_"), record["rows"]) for record in measured.records]
        finally:
            cold.db = original

    short = await discover()
    for batch in range(16):
        lease = await handler()
        items = []
        async with manager.joined_write():
            for index in range(64):
                registered = await manager.register_job(actor, kind="subagent", parent=parent, independent=True,
                    system_key=f"original-{batch}-{index}")
                child = await manager.claim_job(actor, registered.job.job_id, executor_id="child", boot_id="boot", expected_version=1)
                await manager.record_execution_result(actor, child, outcome="succeeded", result={"value": index}, resource_state="stopped")
                items.append({"item_id": "subagent:"+registered.operation.operation_id[3:], "kind": "task_done", "content": str(index)})
        await store.inbox_put(actor.session_id, items)
        claimed = await store.inbox_claim(actor.session_id)
        assert len(claimed) == 64
        seq = await store.append_message(actor.session_id, role="user", content="Actual batch of original results", send_index=handler_count,
            inbox={"claim_token": claimed[0]["claim_token"], "ids": [item["id"] for item in claimed]})
        snapshot = await manager.prepare_processing(actor, lease, round_index=0, send_index=handler_count,
            messages=[{"seq": seq, "role": "user", "content": "Actual batch of original results"}])
        response = await store.append_message(actor.session_id, role="assistant", content="Processed originals", send_index=handler_count, round_index=0)
        ref = await store.message_ref(actor.session_id, response)
        decisions = await manager.commit_processing_decision(actor, lease, snapshot_id=snapshot, response_ref=ref, reason="Actual processing of original result batch")
        assert len(decisions) == 64
        await manager.close_model_execution(actor, lease, evidence={"exit": "completed", "last_assistant_ref": ref.to_dict(), "processing_decisions": decisions})
    assert (await manager.db.fetchone(f"SELECT COUNT(*) AS n FROM {manager.events} WHERE event_type='processing_decision'"))["n"] == 1024
    assert await rows(manager) == []
    assert await discover() == short
