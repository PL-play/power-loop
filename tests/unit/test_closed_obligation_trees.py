"""A clean handler exit precedes repeatable target-tree index maintenance."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop.runtime import session_obligations as obligations
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store, runtime
from .test_processing_decisions_192 import processing_case
from .test_send_sql_profile import Profiled, _counts
from .test_subagent_resume_192 import nudge_receipt, ready
from .test_task_runtime_192 import finish, managed

__all__ = ["registry_store", "runtime", "processing_case", "ready", "managed"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def _another_tree_result(c):
    root = await c.manager.register_job(c.actor, kind="send", session_id=c.actor.session_id,
        system_key="another-old-root", request={"send_index": 1})
    parent = await c.manager.claim_job(c.actor, root.job.job_id,
        executor_id="another-source", boot_id="boot", expected_version=1)
    child = await c.manager.register_job(c.actor, kind="subagent", parent=parent,
        independent=True, system_key="another-old-result")
    lease = await c.manager.claim_job(c.actor, child.job.job_id,
        executor_id="another-child", boot_id="boot", expected_version=1)
    await c.manager.record_execution_result(c.actor, lease, outcome="succeeded",
        result={"value": "another tree"}, resource_state="stopped")
    content = "result from another old tree"
    await c.store.inbox_put(c.actor.session_id, [{"item_id": "subagent:"+child.operation.operation_id[3:],
        "kind": "task_done", "content": content}])
    rows = await c.store.inbox_claim(c.actor.session_id)
    seq = await c.store.append_message(c.actor.session_id, role="user", content=content, send_index=2,
        inbox={"claim_token": rows[0]["claim_token"], "ids": [row["id"] for row in rows]})
    return child.job.job_id, {"seq": seq, "role": "user", "content": content}


async def test_clean_exit_is_committed_before_refreshing_a_different_root(processing_case, monkeypatch):
    c = processing_case
    target, message = await c.result()
    other_target, other_message = await _another_tree_result(c)
    snapshot, response = await c.prepare([message, other_message]), await c.response()
    decisions = await c.commit(snapshot, response)
    reader = JobManager(await c.connect(), now_ms=c.manager.now)
    replace = obligations._write_entries
    observed = []

    async def after_original(manager, tx, row, entries):
        if row["job_id"] in (target, other_target):
            original = await reader.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?",
                ("model_closed_"+c.caller.job_id,))
            assert original is not None, "target maintenance must follow the handler's actual commit"
            assert json.loads(original["payload_json"])["evidence"]["processing_decisions"] == decisions
            observed.append(row["root_job_id"])
        return await replace(manager, tx, row, entries)

    monkeypatch.setattr(obligations, "_write_entries", after_original)
    database = c.manager.db
    c.manager.db = measured = Profiled(database)
    try:
        await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed",
            "last_assistant_ref": response.to_dict(), "processing_decisions": decisions})
    finally:
        c.manager.db = database
    assert len(observed) == len(set(observed)) == 2 and c.caller.job_id not in observed
    assert _counts(measured.records, measured.tx_sites)["control_tx"] == 3


async def test_failed_target_refresh_cannot_rollback_the_exit_and_retry_uses_its_original(processing_case, monkeypatch):
    c = processing_case
    target, message = await c.result()
    snapshot, response = await c.prepare([message]), await c.response()
    decisions = await c.commit(snapshot, response)
    replace = obligations._write_entries
    seen = []

    async def fail_once(manager, tx, row, entries):
        if row["job_id"] == target:
            seen.append((await tx.fetchone(f"SELECT result_status FROM {manager.jobs} WHERE job_id=?", (target,)))["result_status"] == "handled")
            if len(seen) == 1:
                raise OSError("target index commit unavailable")
        return await replace(manager, tx, row, entries)

    monkeypatch.setattr(obligations, "_write_entries", fail_once)
    with pytest.raises(OSError, match="target index commit unavailable"):
        await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed",
            "last_assistant_ref": response.to_dict(), "processing_decisions": decisions})
    original = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?",
        ("model_closed_"+c.caller.job_id,))
    assert original is not None
    # A retry's new argument cannot replace the already committed original.
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "failed"})
    assert seen == [True, True]
    assert await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?",
        (original["event_id"],)) == original


async def test_lost_exit_commit_ack_retries_the_projection_from_the_saved_exit(processing_case, monkeypatch):
    c = processing_case
    _, message = await c.result()
    snapshot, response = await c.prepare([message]), await c.response()
    decisions = await c.commit(snapshot, response)
    transaction = c.manager.db.transaction
    fired = []

    @asynccontextmanager
    async def lost_ack():
        hit = False
        async with transaction() as tx:
            yield tx
            hit = not fired and await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_id=?",
                ("model_closed_"+c.caller.job_id,))
        if hit:
            fired.append(True)
            raise OSError("clean exit committed before lost ACK")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, "transaction", lost_ack)
        with pytest.raises(OSError, match="clean exit committed before lost ACK"):
            await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "completed",
                "last_assistant_ref": response.to_dict(), "processing_decisions": decisions})
    assert fired == [True]
    original = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?",
        ("model_closed_"+c.caller.job_id,))
    assert original is not None
    await c.manager.close_model_execution(c.actor, c.caller, evidence={"exit": "failed"})
    assert await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?",
        (original["event_id"],)) == original


async def test_old_exit_refresh_leaves_a_real_resumed_generation_unchanged(ready, monkeypatch):
    from .test_nudge_runtime_192 import issue_nudge

    c = ready
    decisions = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='processing_decision'")
    [handler] = [json.loads(row["payload_json"])["handler_job_id"] for row in decisions
        if json.loads(row["payload_json"])["event"]["job_id"] == c.old.job_id]
    _, sent, _ = await issue_nudge(c.fixture, monkeypatch)
    assert sent.status == "completed"
    accepted = await nudge_receipt(c)
    assert accepted["generation"] == 2 and accepted["status"] == "accepted"
    await asyncio.wait_for(c.model.entered.wait(), 5)
    before = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (c.old.job_id,))
    events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
    indexed = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.prefix}session_obligations ORDER BY session_id,kind,source_id")
    assert await obligations.reconcile_closed(c.manager, c.runtime.actor(c.parent), handler)
    assert await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (c.old.job_id,)) == before
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id") == events
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.prefix}session_obligations ORDER BY session_id,kind,source_id") == indexed
    await finish(c.fixture)


async def test_process_death_between_exit_and_target_refresh_preserves_a_rebuildable_barrier(registry_store, tmp_path):
    store, connect = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    evidence = tmp_path / "closed-projection.json"
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._closed_projection_worker",
        dsn, store.table_prefix, str(evidence), cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 40)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 76, output.decode()
    saved = json.loads(evidence.read_text())
    cold = JobManager(await connect())
    actor = JobActor(**saved["actor"])
    events = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
    messages = await cold.db.fetchall(f"SELECT * FROM {cold.store.t.messages} ORDER BY id")
    assert events == saved["events"] and messages == saved["messages"]
    [closed] = [row for row in events if row["event_id"] == "model_closed_"+saved["handler_job_id"]]
    assert json.loads(closed["payload_json"])["evidence"]["exit"] == "completed"
    indexed = await cold.db.fetchall(f"SELECT kind FROM {cold.prefix}session_obligations WHERE job_id=?",
        (saved["source_job_id"],))
    assert {row["kind"] for row in indexed} == {"abort", "disposition"}
    with pytest.raises(JobError, match="session_obligation_projection_mismatch"):
        await obligations.verify(cold, actor)
    repair = TaskRuntime(cold, tenant_id=actor.tenant_id, principal_id=actor.principal_id,
        executor_id="cold-repair", boot_id="inspection")
    expected_events = [{**event, "delivery_status": "handled"}
        if event["event_id"] == "result_"+saved["source_job_id"] else event for event in events]
    for _ in range(2):
        result = await repair.repair_execution_checkpoint(actor.session_id, saved["handler_job_id"],
            generation=json.loads(closed["payload_json"])["generation"])
        assert result["repaired"] and not result["execution_authority"] and not result["replay_allowed"]
        await obligations.verify(cold, actor)
    assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == expected_events
    assert await cold.db.fetchall(f"SELECT * FROM {cold.store.t.messages} ORDER BY id") == messages
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations")
    for table, rows in saved["fees"].items():
        assert await cold.db.fetchall(f"SELECT * FROM {table} ORDER BY 1") == rows
    assert saved["model_calls"] == saved["review_calls"] == 1 and saved["dispatched"] == 0
    assert saved["reserve_calls"] == saved["apply_calls"] == 2
