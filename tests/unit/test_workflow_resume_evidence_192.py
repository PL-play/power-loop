"""Corrupt durable receipts and concurrent responsibility cannot authorize replay."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from power_loop.runtime import workflow_resume as resume
from power_loop.runtime.jobs import JobError, JobManager, _hash, _json
from power_loop.runtime.result_revisions import execution_id

from .test_job_registry_192 import registry_store
from .test_workflow_resume_admission_192 import admitting
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ["registry_store", "managed_workflow", "admitting"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("fault", [
    "missing_execution", "bad_json", "wrong_type", "bad_checksum", "bad_version", "missing_operation",
    "op_mode", "plan_hash", "mismatched_revision", "wrong_role", "wrong_kind", "bad_caller_type", "wrong_operation",
])
async def test_damaged_execution_or_original_operation_never_returns_a_valid_replay(admitting, fault):
    c = admitting
    await c.admit()
    op = c.result[0]
    key = execution_id(c.handle.job_id, 2)
    if fault == "missing_execution":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (key,))
    elif fault in ("bad_json", "wrong_type", "bad_checksum", "mismatched_revision"):
        field, value = {"bad_json": ("payload_json", "{"), "wrong_type": ("event_type", "other"),
            "bad_checksum": ("request_hash", "0"*64), "mismatched_revision": ("job_version", 999)}[fault]
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET {field}=? WHERE event_id=?", (value, key))
    elif fault == "bad_version":
        row = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (key,))
        data = json.loads(row["payload_json"])
        data["version"] = True
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?", (_json(data), _hash(data), key))
    elif fault == "missing_operation":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (op.operation_id,))
    elif fault in ("wrong_role", "wrong_kind"):
        field, value = ("execution_role", "main") if fault == "wrong_role" else ("kind", "send")
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET {field}=? WHERE job_id=?", (value, c.handle.job_id))
    elif fault == "wrong_operation":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET action='steer' WHERE event_id=?", (op.operation_id,))
    else:
        row = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (op.operation_id,))
        body = json.loads(row["payload_json"])
        if fault == "op_mode":
            body["mode"] = "continue"
        elif fault == "plan_hash":
            body["request"]["plan"]["message"] = "a different instruction"
        else:
            # Repair the outer checksums to ensure the typed proof itself,
            # rather than just a stale hash, rejects True as generation 1.
            body["request"]["caller"]["generation"] = True
            intent = {k: body["request"][k] for k in ("caller", "target", "previous_generation", "node_id", "message", "expected_plan_hash", "lease_ms")}
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET request_hash=? WHERE event_id=?", (_hash([intent, json.loads(row["call_ref_json"])]), op.operation_id))
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (_json(body), op.operation_id))
        evidence = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (key,))
        data = json.loads(evidence["payload_json"])
        data["request_hash"] = _hash(body["request"])
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?", (_json(data), _hash(data), key))
    with pytest.raises(JobError):
        await resume.admit(c.manager, c.actor, c.handle.job_id, **c.args)
    with pytest.raises(JobError):
        await resume.claim_dispatch(c.manager, c.actor, op.operation_id, caller=c.args["caller"], token="owner-token")
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_dispatch'")


@pytest.mark.parametrize("fault", ["operation_id", "actor", "caller", "token"])
async def test_dispatch_rejects_unrelated_actor_and_identity_without_touching_original_intent(admitting, fault):
    c = admitting
    await c.admit()
    actor, op, caller, token = c.actor, c.result[0].operation_id, c.args["caller"], "owner-token"
    if fault == "operation_id":
        op = "op_missing"
    elif fault == "actor":
        actor = replace(actor, session_id="unrelated")
    elif fault == "caller":
        caller = None
    else:
        token = ""
    with pytest.raises(JobError):
        await resume.claim_dispatch(c.manager, actor, op, caller=caller, token=token)
    assert (await c.manager.get_job(c.actor, c.handle.job_id)).generation == 2


@pytest.mark.parametrize("fault", ["operation_id", "target", "driver_sid", "token"])
async def test_corrupt_claim_payload_with_repaired_checksum_still_cannot_confirm_responsibility(admitting, fault):
    c = admitting
    await c.admit(claim=True)
    key = resume.dispatch_id(c.result[0].operation_id)
    row = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (key,))
    data = json.loads(row["payload_json"])
    if fault == "target":
        data["target"]["generation"] = 99
    else:
        data[fault] = "" if fault == "token" else "forged"
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?", (_json(data), _hash(data), key))
    with pytest.raises(JobError, match="dispatch_mismatch"):
        await resume.claim_dispatch(c.manager, c.actor, c.result[0].operation_id, caller=c.args["caller"], token="owner-token")


async def test_two_managers_claim_dispatch_while_original_send_is_open_only_one_token_wins(admitting, monkeypatch):
    c, outcomes = admitting, []
    original = resume.claim_dispatch
    async def race(manager, actor, operation_id, *, caller, token):
        other = JobManager(await c.connect())
        outcomes.extend(await asyncio.gather(original(manager, actor, operation_id, caller=caller, token=token),
            original(other, actor, operation_id, caller=caller, token="competitor-token")))
        return outcomes[0]
    monkeypatch.setattr(resume, "claim_dispatch", race)
    await c.admit(claim=True)
    assert sorted(outcomes) == [False, True]
    rows = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='workflow_dispatch'")
    assert len(rows) == 1


async def test_corrupt_external_budget_class_is_not_reinterpreted_as_an_unbounded_native_budget():
    from power_loop.runtime import workflow_profile

    from .test_workflow_profile_192 import spec
    s = spec()
    data = workflow_profile.capture(s)
    data["budget"] = {"kind": "external", "class": ""}
    with pytest.raises(JobError, match="workflow_profile_mismatch"):
        workflow_profile.validate(data, spec=s, completion_callback=False)
