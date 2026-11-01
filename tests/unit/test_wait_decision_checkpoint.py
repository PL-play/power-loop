"""Accepted wait decisions and their original model evidence commit together."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop.runtime import model_attempts, timer_waits, wait_decisions, wait_effects, wait_notifications
from power_loop.runtime.execution_state import CheckpointPolicy
from power_loop.runtime.jobs import JobError, JobLease, JobManager

from .test_wait_effects_192 import decide
from .test_wait_handoff_192 import handoff, registry_store, runtime, waiting

__all__ = ["handoff", "registry_store", "runtime", "waiting"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_wait_decision_publishes_original_return_in_its_transaction(handoff, monkeypatch):
    h, c = handoff, handoff.c
    h.loop.task_runtime.checkpoint_policy = CheckpointPolicy(interval_s=300)
    record = wait_decisions.record_decision
    seen = []
    all_attempts = model_attempts._all
    scans = []

    async def scan(*args):
        scans.append(args[-1])
        return await all_attempts(*args)

    monkeypatch.setattr(model_attempts, "_all", scan)

    async def observe(*args, **kwargs):
        caller = kwargs["caller"]
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} "
            "WHERE job_id=? AND event_type='model_attempt_finished'", (caller.job_id,))
        before_scans = len(scans)
        result = await record(*args, **kwargs)
        assert len(scans) == before_scans
        seen.append(caller.job_id)
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} "
            "WHERE job_id=? AND event_type='model_attempt_finished'", (caller.job_id,))) == 1
        return result

    monkeypatch.setattr(wait_decisions, "record_decision", observe)
    await decide(h)
    assert len(seen) == 1 and h.model.calls == 2
    reader = JobManager(await c.connect(), now_ms=c.manager.now)
    reply = await wait_effects.apply(reader, c.main, h.event_id)
    assert reply["status"] == "input_enqueued" and not reply["execution_authority"]
    assert h.model.calls == 2


@pytest.mark.parametrize("stage", ["after_checkpoint", "after_decision", "after_commit"])
async def test_process_death_never_leaves_decision_without_original_model(handoff, tmp_path, stage):
    h, c = handoff, handoff.c
    await wait_notifications.enqueue_notification(h.loop, c.main, h.event_id)
    store = c.store
    dialect = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if dialect == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if dialect == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    config, evidence = tmp_path / "config.json", tmp_path / "decision.json"
    config.write_text(json.dumps({"now": c.now[0], "tenant": c.main.tenant_id,
        "principal": c.main.principal_id, "sid": c.main.session_id, "event_id": h.event_id}))
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._wait_decision_crash_worker",
        dsn, store.table_prefix, str(evidence), str(config), stage, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == {"after_checkpoint": 91, "after_decision": 92, "after_commit": 93}[stage], output.decode()
    saved = json.loads(evidence.read_text())
    reader = JobManager(await c.connect(), now_ms=c.manager.now)
    phases = await reader.db.fetchall(f"SELECT event_type FROM {reader.events} WHERE job_id=? "
        "AND event_type IN ('model_attempt_prepared','model_attempt_dispatched','model_attempt_finished')",
        (saved["caller"]["job_id"],))
    committed = stage == "after_commit"
    assert sorted(row["event_type"] for row in phases) == (["model_attempt_dispatched", "model_attempt_finished", "model_attempt_prepared"] if committed else [])
    assert len(await reader.db.fetchall(f"SELECT event_id FROM {reader.events} WHERE event_type='wait_decision'")) == int(committed)
    request = {key: value for key, value in saved.items() if key != "model_calls"}
    request.update(caller=JobLease(**request["caller"]), call_ref=timer_waits._ref(request["call_ref"]))
    replay = await wait_decisions.recall_decision(reader, c.main, **request)
    assert (replay is not None) == committed
    if committed:
        assert replay.receipt_ref is None and replay.result["response"]["status"] == "accepted"
        with pytest.raises(JobError, match="wait_decision_receipt_required"):
            await wait_effects.apply(reader, c.main, h.event_id)
    assert saved["model_calls"] == 1 and h.model.calls == 0
    assert not await reader.db.fetchall(f"SELECT event_id FROM {reader.events} WHERE event_type='wait_effect'")


@pytest.mark.parametrize("fault", ["rollback", "ack_loss"])
async def test_failed_or_unknown_decision_commit_preserves_original_response_and_intent(handoff, monkeypatch, fault):
    h, c = handoff, handoff.c
    h.loop.task_runtime.checkpoint_policy = CheckpointPolicy(interval_s=300)
    transaction = c.store._db.transaction
    injected = False

    @asynccontextmanager
    async def fail_once():
        nonlocal injected
        inject = False
        async with transaction() as tx:
            yield tx
            row = await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'")
            if row is not None and not injected:
                injected = inject = True
                if fault == "rollback":
                    raise OSError("decision checkpoint rolled back")
        if inject:
            raise OSError("decision checkpoint ACK lost")

    monkeypatch.setattr(c.store._db, "transaction", fail_once)
    await decide(h)
    if fault == "rollback":
        # The original failed operation was rolled back; its raw response stays
        # stored. A fresh model must never be used to fabricate the old intent.
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'")
    else:
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'")) == 1
    assert injected and h.model.calls == 2
    messages = await c.store.load_all_messages(c.main.session_id)
    [tool] = [row for row in messages if row.role == "tool" and row.name == "wait_handoff"]
    reply = json.loads(tool.content)
    assert (reply.get("status") == "accepted") == (fault == "ack_loss")
    attempts = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_finished'")
    assert len(attempts) == 2
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_effect'")
