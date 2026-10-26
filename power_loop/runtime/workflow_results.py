"""Frozen workflow delivery revisions; replay projects evidence, never executes work.

The journal, inbox and closed projection receipt commit atomically. A frozen
preparation is not delivery or consumption evidence. Callback exits are separate.
Projection repair never invokes a callback. Its original execution owner must
separately claim and settle that callback; no new workflow generation is admitted.
"""
from __future__ import annotations

import json
import secrets

from .jobs import JobError, _hash, _json
from .result_revisions import result_id


def delivery_event_id(job_id, generation):
    if type(generation) is not int or not 0 <= generation < 2**63:
        raise JobError("invalid_result_generation")
    return "wf_delivery_" + _hash([job_id, generation])[:48]


def _id(kind, job_id, generation):
    delivery_event_id(job_id, generation)
    return "wf_" + kind + "_" + _hash([job_id, generation])[:48]


def workflow_result(data):
    from power_loop.workflow.result import AgentResult, WorkflowResult
    return WorkflowResult(name=data["name"], status=data["status"],
        results={key: AgentResult(**value) for key, value in data["results"].items()},
        final=AgentResult(**data["final"]) if data.get("final") else None,
        usage=data.get("usage") or {}, errors=data.get("errors") or [], tolerated=data.get("tolerated") or [])


async def prepare(manager, actor, job_id, generation, *, lease=None):
    """Freeze delivery evidence without publishing it or granting execution."""
    async with manager._write() as tx:
        return await _prepare(manager, tx, actor, job_id, generation, lease=lease)


async def _prepare(manager, tx, actor, job_id, generation, *, lease=None, _controller_closures=None):
    """Caller holds task control; take the normal session-state lock second."""
    key = delivery_event_id(job_id, generation)
    row = await manager._authorize(tx, actor, job_id)
    if row["kind"] != "workflow" or row["execution_role"] != "workflow" or row["created_by_session_id"] != actor.session_id:
        raise JobError("workflow_delivery_target_mismatch")
    await manager.db.dialect.lock_state(tx, manager.store.t.session_state, actor.session_id)
    if row["generation"] != generation:
        raise JobError("workflow_delivery_generation_changed")
    if lease is not None:
        manager._check_proof(row, lease)
    from .workflow_controller import creation, proof
    from .workflow_controller import read as controller
    operation_id, creation_hash = await creation(manager, tx, row)
    canonical = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result' AND job_id=?",
        (result_id(job_id, generation), job_id))
    if canonical is None or row["result_json"] is None:
        raise JobError("workflow_delivery_result_missing")
    try:
        payload = json.loads(canonical["payload_json"])
        result = json.loads(row["result_json"])
        expected = dict(payload["result"])
        if generation == 0:
            if (payload.get("unstarted") is not True or row["resource_state"] != "stopped" or
                    type(payload.get("generation", 0)) is not int or payload.get("generation", 0) != 0):
                raise JobError("workflow_delivery_result_mismatch")
        else:
            from .execution_membership import read
            membership = await read(manager, tx, row, generation)
            fields = ("kind", "execution_role", "tenant_id", "principal_id", "created_by_session_id", "parent_job_id")
            if (not membership["sealed"] or _hash(membership["scope"]["lease"]) != _hash(proof(row)) or
                    any(membership["scope"][field] != row[field] for field in fields) or
                    type(payload.get("generation", 1)) is not int or payload.get("generation", 1) != generation):
                raise JobError("workflow_delivery_result_mismatch")
            if row["session_id"] is None and payload["outcome"] == "succeeded":
                raise JobError("workflow_delivery_result_mismatch")
            if payload["outcome"] == "succeeded":
                await controller(manager, tx, row, generation)
        # The immutable execution evidence can precede the effective cancel
        # decision. Preserve both rather than reporting completed after stop.
        if row["outcome"] == "cancelled" and expected.get("status") != "cancelled":
            if generation:
                expected["execution_status"] = expected.get("status")
            expected["status"] = "cancelled"
        effective_outcome = "cancelled" if row["outcome"] == "cancelled" else payload["outcome"]
        resource_matches = row['resource_state'] == payload['resource_state']
        if (not resource_matches and row['resource_state'] == 'stopped' and payload['resource_state'] == 'unknown'
                and payload['outcome'] != 'succeeded'):
            from .controller_recovery import read as controller_stop
            resource_matches = await controller_stop(manager, tx, row, _closures=_controller_closures) is not None
        if (_hash(expected) != _hash(result) or row["outcome"] != effective_outcome or not resource_matches):
            raise JobError("workflow_delivery_result_mismatch")
        projected = workflow_result(result)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("workflow_delivery_result_mismatch") from exc
    envelope = {"event_id": canonical["event_id"], "job_id": job_id, "generation": generation,
        "result_event_id": canonical["event_id"], "result_revision": int(canonical["job_version"]),
        "result_hash": _hash(payload), "operation_id": operation_id, "run_id": operation_id[3:],
        "status": result["status"], "summary": projected.summary()[:4000]}
    if generation > 1:
        from .workflow_resume import execution
        resume_op, _ = await execution(manager, tx, row)
        envelope.update(operation_id=resume_op["event_id"], creation_operation_id=operation_id)
    from power_loop.workflow import journal
    journal_row = await tx.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
        (actor.session_id, journal.run_key(envelope["run_id"])))
    try:
        j = json.loads(journal_row["value_json"]) if journal_row else None
        if (not isinstance(j, dict) or j.get("run_id") != envelope["run_id"] or
                type(j.get("task_generation", 1)) is not int or j.get("task_generation", 1) != max(1, generation) or
                (journal._is_terminal(j.get("status")) and _hash(j.get("result")) != _hash(projected.to_dict()))):
            raise JobError("workflow_delivery_journal_mismatch")
    except (ValueError, TypeError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("workflow_delivery_journal_mismatch") from exc
    data = {"version": 1, "workflow": proof(row) if generation else None,
        "parent_session_id": actor.session_id, "creation_hash": creation_hash,
        "result": result, "envelope": envelope}
    old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
    if old:
        if (old["event_type"] != "workflow_delivery" or old["job_id"] != job_id or
                old["job_version"] != canonical["job_version"] or
                old["delivery_status"] != "not_applicable" or old["request_hash"] != _hash(data) or
                old["payload_json"] != _json(data)):
            raise JobError("workflow_delivery_revision_mismatch")
    else:
        if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (_id("projected", job_id, generation),)):
            raise JobError("workflow_delivery_revision_mismatch")
        await tx.execute(f"INSERT INTO {manager.events} "
            "(event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (key, job_id, "workflow_delivery", canonical["job_version"], _hash(data), _json(data), "not_applicable", manager.now()))
    return data


async def _event(manager, tx, key, kind, data, job_id, revision):
    old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
    if old:
        if (old["event_type"] != kind or old["job_id"] != job_id or old["job_version"] != revision or
                old["delivery_status"] != "not_applicable" or old["request_hash"] != _hash(data) or old["payload_json"] != _json(data)):
            raise JobError("workflow_projection_receipt_mismatch")
    else:
        count = await tx.execute(f"INSERT INTO {manager.events} "
            "(event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (key, job_id, kind, revision, _hash(data), _json(data), "not_applicable", manager.now()))
        if count != 1:
            raise JobError("workflow_projection_receipt_mismatch")


async def _closure(manager, tx, data):
    envelope = data["envelope"]
    op = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (envelope.get("creation_operation_id", envelope["operation_id"]),))
    request = json.loads(op["payload_json"]).get("request", {})
    required = request.get("completion_callback")
    if required is not None and type(required) is not bool:
        raise JobError("workflow_callback_contract_mismatch")
    if envelope["status"] == "cancelled":
        required = False
    return {"version": 1, "delivery_event_id": delivery_event_id(envelope["job_id"], envelope["generation"]),
        "delivery_hash": _hash(data), "callback_required": required,
        "result_event_id": envelope["result_event_id"], "result_revision": envelope["result_revision"]}


async def _views(manager, tx, actor, data, *, write):
    """Called under control -> session-state lock; never opens another transaction."""
    from power_loop.workflow import journal
    envelope = data["envelope"]
    key = journal.run_key(envelope["run_id"])
    row = await tx.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
        (actor.session_id, key))
    j = json.loads(row["value_json"]) if row else None
    projected = workflow_result(data["result"])
    if (not isinstance(j, dict) or j.get("run_id") != envelope["run_id"] or
            type(j.get("task_generation", 1)) is not int or j.get("task_generation", 1) != max(1, envelope["generation"])):
        raise JobError("workflow_delivery_journal_mismatch")
    if write and not journal._is_terminal(j.get("status")):
        now = manager.now()
        j.update(status=projected.status, result=projected.to_dict(), finished_at_ms=now, updated_at_ms=now,
                 error=projected.errors[0] if projected.errors else None)
        await tx.execute(f"UPDATE {manager.store.t.session_runtime_state} SET value_json=?,updated_at=? WHERE session_id=? AND state_key=?",
            (_json(j), now, actor.session_id, key))
        await tx.execute(f"UPDATE {manager.store.t.sessions} SET updated_at=? WHERE session_id=?", (now, actor.session_id))
        saved = await tx.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
            (actor.session_id, key))
        if saved is None or _hash(json.loads(saved["value_json"])) != _hash(j):
            raise JobError("workflow_delivery_journal_mismatch")
    if j.get("status") != projected.status or _hash(j.get("result")) != _hash(projected.to_dict()):
        raise JobError("workflow_delivery_journal_mismatch")
    content = json.dumps(envelope, ensure_ascii=False)
    if write:
        cols = ("session_id", "item_id", "kind", "mode", "status", "content", "created_at")
        await tx.execute(manager.db.dialect.insert_ignore(manager.store.t.follow_up_queue, cols),
            (actor.session_id, envelope["event_id"], "task_done", "queue", "pending", content, manager.now()))
    inbox = await tx.fetchone(f"SELECT kind,mode,status,content FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (actor.session_id, envelope["event_id"]))
    if (inbox is None or inbox["kind"] != "task_done" or inbox["mode"] != "queue" or inbox["content"] != content or
            inbox["status"] not in ("pending", "claimed", "delivered")):
        raise JobError("workflow_delivery_inbox_mismatch")
    return inbox["status"]


async def project(loop, actor, job_id, generation, *, lease=None):
    """Commit frozen revision, journal, inbox and projection receipt together."""
    manager = loop.task_runtime.manager
    async with manager._write() as tx:
        data = await _prepare(manager, tx, actor, job_id, generation, lease=lease)
        inbox_status = await _views(manager, tx, actor, data, write=True)
        closed = await _closure(manager, tx, data)
        await _event(manager, tx, _id("projected", job_id, generation), "workflow_projected", closed,
                     job_id, data["envelope"]["result_revision"])
    # A durable inbox is authoritative. A post-commit hint never dispatches a
    # model, callback or another insert. A lost ACK is repaired by retrying.
    if inbox_status == "pending":
        loop._note_committed_inbox(actor.session_id)
    return data["envelope"]


async def require_closed(manager, tx, actor, job_id, generation, *, consumed=False, _controller_closures=None):
    """Transaction-local takeover prerequisite, not workflow resume permission."""
    data = await _prepare(manager, tx, actor, job_id, generation, _controller_closures=_controller_closures)
    expected = await _closure(manager, tx, data)
    key = _id("projected", job_id, generation)
    event = await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (key,))
    if event is None:
        raise JobError("workflow_projection_unsettled")
    await _event(manager, tx, key, "workflow_projected", expected, job_id, data["envelope"]["result_revision"])
    await _views(manager, tx, actor, data, write=False)
    if expected["callback_required"] is None:
        raise JobError("workflow_callback_contract_unknown")
    if expected["callback_required"]:
        start, exit = await _callback_records(manager, tx, data)
        if start is None or exit is None or exit["status"] != "completed":
            raise JobError("workflow_callback_unsettled")
    if consumed:
        from .processing import require_handled
        await require_handled(manager, tx, await manager._row(tx, job_id))
    return data


async def _callback_records(manager, tx, data):
    envelope = data["envelope"]
    records = []
    for phase in ("start", "exit"):
        key = _id("cb_"+phase, envelope["job_id"], envelope["generation"])
        row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
        if row is None:
            records.append(None)
            continue
        try:
            value = json.loads(row["payload_json"])
            if (type(value.get("version")) is not int or value.get("version") != 1 or value.get("delivery_hash") != _hash(data) or
                    not isinstance(value.get("token"), str) or len(value["token"]) != 32 or
                    (phase == "exit" and value.get("status") not in ("completed", "failed", "cancelled"))):
                raise ValueError("callback evidence")
            await _event(manager, tx, key, "workflow_callback_"+phase, value, envelope["job_id"], envelope["result_revision"])
        except (ValueError, TypeError, AttributeError) as exc:
            if isinstance(exc, JobError):
                raise
            raise JobError("workflow_callback_evidence_mismatch") from exc
        records.append(value)
    start, exit = records
    if exit is not None and (start is None or exit["token"] != start["token"]):
        raise JobError("workflow_callback_evidence_mismatch")
    return start, exit


async def run_callback(loop, actor, job_id, generation, callback):
    """A separately fenced, at-most-once callback; never replayed by project()."""
    from power_loop.workflow import runner
    manager, token = loop.task_runtime.manager, secrets.token_hex(16)
    async with manager._write() as tx:
        data = await _prepare(manager, tx, actor, job_id, generation)
        expected = await _closure(manager, tx, data)
        closed = await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (_id("projected", job_id, generation),))
        if closed is None:
            raise JobError("workflow_projection_unsettled")
        await _event(manager, tx, closed["event_id"], "workflow_projected", expected, job_id, data["envelope"]["result_revision"])
        await _views(manager, tx, actor, data, write=False)
        if expected["callback_required"] is not True or (await manager._row(tx, job_id))["cancel_requested_at"] is not None:
            return
        start, _ = await _callback_records(manager, tx, data)
        if start is not None:
            return
        value = {"version": 1, "delivery_hash": _hash(data), "token": token}
        await _event(manager, tx, _id("cb_start", job_id, generation), "workflow_callback_start", value,
                     job_id, data["envelope"]["result_revision"])
    status = "cancelled"
    try:
        envelope = data["envelope"]
        ok = await runner._run_on_complete(callback, actor.session_id, envelope["run_id"], envelope["status"],
                                            runner._wake_note(envelope["run_id"], envelope["status"]),
                                            task_generation=generation, result_event_id=envelope["result_event_id"])
        status = "completed" if ok else "failed"
    finally:
        async with manager._write() as tx:
            current = await _prepare(manager, tx, actor, job_id, generation)
            start, _ = await _callback_records(manager, tx, current)
            if start is None or start["token"] != token:
                raise JobError("workflow_callback_evidence_mismatch")
            await _event(manager, tx, _id("cb_exit", job_id, generation), "workflow_callback_exit",
                {**value, "status": status}, job_id, data["envelope"]["result_revision"])
