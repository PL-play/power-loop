"""Host-only original-context cancellation, without creating an execution.

The barrier and stops share the task-control lock with root registration.
An accepted stop never asserts physical exit or releases financial holds.
"""
import json
from dataclasses import asdict

from .jobs import JobActor, JobError, _hash, _json, _operation
from .root_context import _document, _read


def _barrier_key(actor, document):
    return "power-loop.root-stop.v1:" + _hash([asdict(actor), document["context_hash"]])


async def _state(manager, tx, actor, key):
    row = await tx.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} "
        "WHERE session_id=? AND state_key=?", (actor.session_id, key))
    if row is None:
        return None
    try:
        value = json.loads(row["value_json"])
        shapes = ({"version", "actor", "document", "first_request_hash", "evidence_hash"},
            {"version", "actor", "request", "response", "evidence_hash"})
        if (type(value) is not dict or set(value) not in shapes
                or type(value.get("version")) is not int or value["version"] != 1 or value.get("actor") != asdict(actor)
                or value.get("evidence_hash") != _hash({k: v for k, v in value.items() if k != "evidence_hash"})):
            raise ValueError()
        return value
    except (ValueError, TypeError):
        raise JobError("root_control_evidence_corrupt") from None


async def _save(manager, tx, actor, key, value):
    value = {"version": 1, "actor": asdict(actor), **value}
    value["evidence_hash"] = _hash(value)
    sql = manager.db.dialect.upsert(manager.store.t.session_runtime_state,
        ("session_id", "state_key"), ("value_json", "updated_at"))
    await tx.execute(sql, (actor.session_id, key, _json(value), manager.now()))
    return value


async def _admit(manager, tx, actor, document):
    if (type(document) is not dict or set(document) != {"version", "namespace", "payload", "context_hash"}
            or type(document.get("version")) is not int or document["version"] != 1
            or _document(document.get("namespace"), document.get("payload")) != document):
        raise JobError("root_context_evidence_corrupt")
    value = await _state(manager, tx, actor, _barrier_key(actor, document))
    if value is not None:
        if value.get("document") != document:
            raise JobError("root_control_evidence_corrupt")
        raise JobError("root_context_cancelled")


async def _cancel_job(manager, tx, actor, job_id, *, key, fingerprint, reason):
    old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE operation_key=?", (key,))
    if old:
        await manager._authorize(tx, actor, old["job_id"])
        if old["request_hash"] != fingerprint:
            raise JobError("operation_conflict")
        return _operation(old)
    row = await manager._authorize(tx, actor, job_id)
    if row["cancel_requested_at"] is None:
        # Cancellation retains an already closed model phase. Rewriting it as
        # running cancellation would strand its original pending/result proof.
        status = row["status"] if row["supervision_status"] == "closed" or row["status"] == "model_closed" else (
            "cancelled" if row["resource_state"] == "not_started" else "cancelling")
        await tx.execute(f"UPDATE {manager.jobs} SET status=?,cancel_requested_at=?,cancel_reason=?,"
            "intake='closed',version=version+1,updated_at=? WHERE job_id=?",
            (status, manager.now(), reason, manager.now(), job_id))
        row = await manager._row(tx, job_id)
    return await manager._insert_operation(tx, job_id=job_id, version=row["version"], actor=actor,
        action="stop", key=key, fingerprint=fingerprint, payload={"job_id": job_id,
        "status": row["status"], "cancel_requested": True, "resource_state": row["resource_state"]})


async def cancel(manager, actor, *, namespace, payload, system_key, reason):
    """Freeze this exact context and stop its roots, including future attempts.

    No model, runtime, executor, execution lease or current policy is required.
    Different contexts in the same main session remain independent. The saved
    receipt is historical; inspect the jobs for current resource/usage state.
    """
    document = _document(namespace, payload)
    if type(actor) is not JobActor:
        raise JobError("trusted_root_control_actor_required")
    if not isinstance(reason, str) or not reason or len(reason) > 1000:
        raise JobError("invalid_cancel_reason")
    request_key = "power-loop.root-stop-request.v1:" + manager._origin_key(actor, "stop_context", None, None, system_key)
    request = {"document": document, "system_key": system_key, "reason": reason}
    session = await manager.store.get_session(actor.session_id)
    if (session is None or session.kind.value != "root" or session.parent_session_id is not None
            or session.metadata.get("execution_role", "main") != "main"):
        raise JobError("root_control_main_session_required")
    async with manager._write() as tx:
        original = await _state(manager, tx, actor, request_key)
        if original is not None:
            if original.get("request") != request:
                raise JobError("operation_conflict")
            barrier = await _state(manager, tx, actor, _barrier_key(actor, document))
            if barrier is None or barrier.get("document") != document:
                raise JobError("root_control_evidence_corrupt")
            response = original.get("response", {})
            if (type(response) is not dict or response.get("request_hash") != _hash(request)
                    or response.get("context_hash") != document["context_hash"]
                    or response.get("state") != "accepted" or type(response.get("roots")) is not list
                    or any(response.get(k) is not False for k in (
                        "execution_authority", "resource_stop_confirmed", "billing_settled"))):
                raise JobError("root_control_evidence_corrupt")
            for item in response["roots"]:
                try:
                    operation = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?",
                        (item["operation"]["operation_id"],))
                    if (operation is None or operation["action"] != "stop"
                            or operation["job_id"] != item["source"]["root_job_id"]
                            or asdict(_operation(operation)) != item["operation"]):
                        raise ValueError()
                    await manager._authorize(tx, actor, operation["job_id"])
                except (ValueError, TypeError, KeyError):
                    raise JobError("root_control_evidence_corrupt") from None
            return response
        candidates = await tx.fetchall(f"SELECT j.job_id,e.payload_json FROM {manager.jobs} j "
            f"LEFT JOIN {manager.events} e ON e.job_id=j.job_id AND e.event_type='operation' AND e.action='create' "
            "WHERE j.tenant_id=? AND j.principal_id=? AND j.session_id=? "
            "AND j.kind='send' AND j.parent_job_id IS NULL ORDER BY j.job_id LIMIT 10001",
            (actor.tenant_id, actor.principal_id, actor.session_id))
        if len(candidates) > 10000:
            raise JobError("root_control_history_requires_compaction")
        roots = []
        for candidate in candidates:
            try:
                creation = json.loads(candidate["payload_json"])
                if type(creation) is not dict or type(creation.get("request")) is not dict:
                    raise ValueError()
                saved = creation.get("request", {}).get("host_context")
                if saved is None:
                    continue
                source = await _read(manager, tx, actor, candidate["job_id"])
                if source["context_hash"] != document["context_hash"]:
                    continue
                if any(source[k] != document[k] for k in document):
                    raise ValueError()
            except (ValueError, TypeError, KeyError):
                raise JobError("root_control_evidence_corrupt") from None
            key = manager._origin_key(actor, "stop", None, None,
                "context-stop:" + _hash([system_key, document["context_hash"], candidate["job_id"]])[:48])
            operation = await _cancel_job(manager, tx, actor, candidate["job_id"], key=key,
                fingerprint=_hash([candidate["job_id"], reason]), reason=reason)
            roots.append({"source": source, "operation": asdict(operation)})
        barrier_key = _barrier_key(actor, document)
        barrier = await _state(manager, tx, actor, barrier_key)
        if barrier is None:
            await _save(manager, tx, actor, barrier_key, {"document": document, "first_request_hash": _hash(request)})
        elif barrier.get("document") != document:
            raise JobError("root_control_evidence_corrupt")
        response = {"version": 1, "state": "accepted", "request_hash": _hash(request),
            "context_hash": document["context_hash"], "roots": roots, "execution_authority": False,
            "resource_stop_confirmed": False, "billing_settled": False}
        if len(_json(response).encode()) > 1_000_000:
            raise JobError("root_control_history_requires_compaction")
        await _save(manager, tx, actor, request_key, {"request": request, "response": response})
        return response
