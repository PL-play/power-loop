"""Processing originals belong to the handler; target states are projections."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop.runtime import session_obligations as obligations
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.processing import require_handled

from .test_closed_obligation_trees import _another_tree_result
from .test_job_registry_192 import registry_store, runtime
from .test_nudge_runtime_192 import nudge_case
from .test_processing_decisions_192 import processing_case

__all__ = ["registry_store", "runtime", "processing_case", "nudge_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_original_decisions_for_two_other_trees_are_owned_by_the_handler(processing_case):
    c = processing_case
    first, one = await c.result()
    second, two = await _another_tree_result(c)
    snapshot, response = await c.prepare([one, two]), await c.response()
    ids = await c.commit(snapshot, response)
    decisions = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='processing_decision'")
    assert {row["event_id"] for row in decisions} == set(ids)
    assert {row["job_id"] for row in decisions} == {c.caller.job_id}
    assert {json.loads(row["payload_json"])["event"]["job_id"] for row in decisions} == {first, second}
    assert {json.loads(row["payload_json"])["response_ref"]["id"] for row in decisions} == {str(response.id)}


async def test_protocol24_cannot_open_with_handler_owned_decision_semantics(registry_store):
    from power_loop.runtime.store.schema import StoreSchemaError
    from power_loop.runtime.store.task_schema import TASK_PROTOCOL_VERSION

    store, connect = registry_store
    await store._db.execute(f"UPDATE {store.table_prefix}task_store SET protocol_version=24 WHERE id=1")
    try:
        with pytest.raises(StoreSchemaError, match="protocol|unsupported|migration"):
            await connect()
        assert (await store._db.fetchone(f"SELECT protocol_version FROM {store.table_prefix}task_store WHERE id=1"))["protocol_version"] == 24
    finally:
        await store._db.execute(f"UPDATE {store.table_prefix}task_store SET protocol_version=? WHERE id=1", (TASK_PROTOCOL_VERSION,))


@pytest.mark.parametrize("boundary", ["completed", "failed", "cancelled", "new_input", "wrong_response"])
async def test_a_logical_decision_waits_for_the_actual_matching_clean_exit(processing_case, boundary):
    c = processing_case
    target, message = await c.result()
    snapshot, response = await c.prepare([message]), await c.response()
    ids = await c.commit(snapshot, response)
    assert (await c.manager.get_job(c.actor, target)).result_status == "pending_delivery"
    child = await c.manager._row(c.manager.db, target)
    with pytest.raises(JobError, match="required_result_unhandled"):
        await require_handled(c.manager, c.manager.db, child)
    evidence = {"exit": boundary if boundary in ("failed", "cancelled") else "completed",
        "last_assistant_ref": response.to_dict(), "processing_decisions": ids}
    if boundary == "new_input":
        evidence["new_input_after_pass"] = True
    elif boundary == "wrong_response":
        evidence["last_assistant_ref"] = {**response.to_dict(), "id": str(response.id+1)}
    await c.manager.close_model_execution(c.actor, c.caller, evidence=evidence)
    current = await c.manager._row(c.manager.db, target)
    if boundary == "completed":
        assert current["result_status"] == "handled"
        await require_handled(c.manager, c.manager.db, current)
    else:
        assert current["result_status"] == "pending_delivery"
        with pytest.raises(JobError, match="required_result_unhandled"):
            await require_handled(c.manager, c.manager.db, current)


async def test_committed_clean_exit_is_not_rediscovered_while_its_projection_is_pending(processing_case, monkeypatch):
    c = processing_case
    _, message = await c.result()
    snapshot, response = await c.prepare([message]), await c.response()
    ids = await c.commit(snapshot, response)

    async def unavailable(*args):
        raise OSError("target projection unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(obligations, "_refresh_closed_tree", unavailable)
        with pytest.raises(OSError, match="target projection unavailable"):
            await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed",
                "last_assistant_ref": response.to_dict(), "processing_decisions": ids})
    reader = JobManager(await c.connect(), now_ms=c.manager.now)
    registration = await reader.register_job(c.actor, kind="send", session_id=c.actor.session_id,
        system_key="read-after-dead-handler", request={"send_index": 3})
    probe = await reader.claim_job(c.actor, registration.job.job_id,
        executor_id="new-handler", boot_id="boot", expected_version=1)
    report = {}
    assert await reader.prepare_processing(c.actor, probe, round_index=0, send_index=3,
        messages=[message], report=report) is None
    assert report == {"quiet": True}
    await obligations.reconcile_closed(reader, c.actor, c.caller.job_id)


async def test_target_stop_between_discovery_and_projection_retries_without_erasing_the_stop(processing_case, monkeypatch):
    c = processing_case
    target, message = await c.result()
    snapshot, response = await c.prepare([message]), await c.response()
    ids = await c.commit(snapshot, response)
    refresh = obligations._refresh_closed_tree
    attempts = []
    other = JobManager(await c.connect(), now_ms=c.manager.now)

    async def stopped_before_write(*args):
        assert c.manager._control_intents.get() is None
        attempts.append(True)
        if len(attempts) == 1:
            await other.request_cancel(c.actor, target, system_key="stop-before-projection", reason="stop target")
        return await refresh(*args)

    monkeypatch.setattr(obligations, "_refresh_closed_tree", stopped_before_write)
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed",
        "last_assistant_ref": response.to_dict(), "processing_decisions": ids})
    current = await c.manager._row(c.manager.db, target)
    assert len(attempts) == 2 and current["result_status"] == "handled"
    assert current["cancel_requested_at"] is not None and current["cancel_reason"] == "stop target"
    assert current["outcome"] == "succeeded" and current["resource_state"] == "stopped"


async def test_handling_one_input_does_not_remove_a_later_input_from_the_same_job(nudge_case):
    c = nudge_case
    first, _ = await c.accept()
    ids = await c.consume(first, complete=False)
    seq = await c.store.append_message(c.actor.session_id, role="assistant", send_index=1, round_index=1,
        tool_calls=[{"id": "late-input", "type": "function", "function": {"name": "spawn_agent", "arguments": "{}"}}])
    late, _ = await c.manager.accept_nudge(c.actor, c.target, caller=c.caller,
        call_ref=await c.store.message_ref(c.actor.session_id, seq), tool_call_id="late-input",
        session_id=c.sid, message="Later unobserved input")
    response = await c.store.last_assistant_ref(c.sid, 1)
    await c.manager.close_model_execution(c.child_actor, c.receiver, evidence={"exit": "completed",
        "last_assistant_ref": response.to_dict(), "processing_decisions": ids})
    first_row = await c.manager.db.fetchone(f"SELECT delivery_status FROM {c.manager.events} WHERE event_id=?", (first.operation_id,))
    late_row = await c.manager.db.fetchone(f"SELECT delivery_status FROM {c.manager.events} WHERE event_id=?", (late.operation_id,))
    assert first_row["delivery_status"] == "handled" and late_row["delivery_status"] == "pending_delivery"
    pending = await obligations.event_rows(c.manager, c.manager.db, c.child_actor, "input", job_id=c.target.job_id)
    assert [event["event_id"] for event in pending] == [late.operation_id]


async def test_handling_input_does_not_ack_a_result_published_during_its_projection(nudge_case, monkeypatch):
    c = nudge_case
    operation, _ = await c.accept()
    ids = await c.consume(operation, complete=False)
    response = await c.store.last_assistant_ref(c.sid, 1)
    refresh = obligations._refresh_closed_tree
    published = []

    async def result_before_projection(*args):
        if not published:
            await c.manager.finalize_supervision(c.child_actor, c.receiver.job_id, generation=1)
            result = await c.manager.record_execution_result(c.child_actor, c.target,
                outcome="succeeded", result={"value": "late controller result"}, resource_state="stopped")
            published.append(result)
        return await refresh(*args)

    monkeypatch.setattr(obligations, "_refresh_closed_tree", result_before_projection)
    await c.manager.close_model_execution(c.child_actor, c.receiver, evidence={"exit": "completed",
        "last_assistant_ref": response.to_dict(), "processing_decisions": ids})
    assert len(published) == 1
    assert (await c.manager.get_job(c.actor, c.target.job_id)).result_status == "pending_delivery"
    [current] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (published[0],))
    assert current["delivery_status"] == "pending_delivery"
    assert [event["event_id"] for event in await obligations.event_rows(c.manager, c.manager.db, c.actor, "result")] == published
    with pytest.raises(JobError, match="required_result_unhandled"):
        await require_handled(c.manager, c.manager.db, await c.manager._row(c.manager.db, c.target.job_id))


async def test_process_death_after_logical_commit_cannot_fabricate_a_clean_exit(registry_store, tmp_path):
    from power_loop.runtime.abort_outcomes import pending_reviews
    from power_loop.runtime.jobs import JobActor

    store, connect = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    evidence = tmp_path / "processing-decision.json"
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._processing_decision_worker",
        dsn, store.table_prefix, str(evidence), cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 40)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 77, output.decode()
    saved = json.loads(evidence.read_text())
    cold = JobManager(await connect())
    actor = JobActor(**saved["actor"])
    assert len(await pending_reviews(cold, cold.db, actor)) == 1
    assert (await cold.get_job(actor, saved['handler_job_id'])).job_id == saved['handler_job_id']
    assert not await cold.db.fetchone(f"SELECT event_id FROM {cold.events} WHERE event_id=?",
        ("model_closed_"+saved["handler_job_id"],))
    assert (await cold.get_job(actor, saved["source_job_id"])).result_status == "pending_delivery"
    assert len(await pending_reviews(cold, cold.db, actor)) == 1
    assert {row["kind"] for row in await cold.db.fetchall(f"SELECT kind FROM {cold.prefix}session_obligations")} == {"abort", "disposition"}
    assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == saved["events"]
    assert await cold.db.fetchall(f"SELECT * FROM {cold.store.t.messages} ORDER BY id") == saved["messages"]
    for table, rows in saved["fees"].items():
        assert await cold.db.fetchall(f"SELECT * FROM {table} ORDER BY 1") == rows
    assert saved["model_calls"] == saved["review_calls"] == 1 and saved["dispatched"] == 0
    assert saved["reserve_calls"] == saved["apply_calls"] == 2
