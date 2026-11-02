"""Host-only original-context cancellation, without creating an execution.

Admission closes before discovery, followed by one repeatable stop per tree.
The session's derived control-version row serializes admission and context
original publication without acquiring any task tree. Originals remain authority.
An accepted stop never asserts physical exit or releases financial holds.
"""
import json
from dataclasses import asdict

from .jobs import JobActor, JobError, _hash, _json, _operation
from .root_context import _document, _read


def _barrier_key(actor, document):
    return "power-loop.root-stop.v1:" + _hash([asdict(actor), document["context_hash"]])


async def _state(manager, tx, actor, key):
    row = await tx.fetchone(f"SELECT payload_json FROM {manager.prefix}task_context_events "
        "WHERE session_id=? AND state_key=?", (actor.session_id, key))
    if row is None:
        return None
    return _decode(actor, row['payload_json'])


def _decode(actor, payload):
    try:
        value = json.loads(payload)
        shapes = ({"version", "actor", "document", "first_request_hash", "evidence_hash"},
            {"version", "actor", "request", "response", "evidence_hash"})
        if (type(value) is not dict or set(value) not in shapes
                or type(value.get("version")) is not int or value["version"] != 1 or value.get("actor") != asdict(actor)
                or value.get("evidence_hash") != _hash({k: v for k, v in value.items() if k != "evidence_hash"})):
            raise ValueError()
        return value
    except (ValueError, TypeError):
        raise JobError("root_control_evidence_corrupt") from None


async def _lock_session(manager, tx, actor):
    table = manager.prefix+'session_controls'
    suffix = '' if manager.db.dialect.name == 'sqlite' else ' FOR UPDATE'
    sql = f'SELECT version FROM {table} WHERE session_id=?'+suffix
    if await tx.fetchone(sql, (actor.session_id,)) is None:
        # A cold session or rebuilt derived table has no admission row yet.
        await tx.execute(manager.db.dialect.insert_ignore(table, ('session_id', 'version')), (actor.session_id, 0))
        await tx.fetchone(sql, (actor.session_id,))


async def _admission_state(manager, tx, actor, key):
    controls, events = manager.prefix+'session_controls', manager.prefix+'task_context_events'
    suffix = '' if manager.db.dialect.name == 'sqlite' else ' FOR UPDATE'
    sql = (f'SELECT c.version AS current_version,'
        f'(SELECT version FROM {controls} WHERE session_id=?) AS observed_version,'
        f'(SELECT payload_json FROM {events} WHERE session_id=? AND state_key=?) AS payload_json '
        f'FROM {controls} c WHERE c.session_id=?'+suffix)
    params = (actor.session_id, actor.session_id, key, actor.session_id)
    row = await tx.fetchone(sql, params)
    if row is None:
        await tx.execute(manager.db.dialect.insert_ignore(controls, ('session_id', 'version')), (actor.session_id, 0))
        row = await tx.fetchone(sql, params)
    # PostgreSQL can return a newly committed locked row while its original
    # subquery still uses the snapshot from before the wait. Cancellation bumps
    # this same version atomically,so only that race needs a fresh original read.
    if row['current_version'] != row['observed_version']:
        return await _state(manager, tx, actor, key)
    return None if row['payload_json'] is None else _decode(actor, row['payload_json'])


async def _save(manager, tx, actor, key, value):
    value = {"version": 1, "actor": asdict(actor), **value}
    value["evidence_hash"] = _hash(value)
    columns = ('event_id', 'session_id', 'state_key', 'payload_json', 'created_at')
    identity = 'rootctl_' + _hash([asdict(actor), key])[:48]
    await tx.append_context_event(columns, (identity, actor.session_id, key, _json(value), manager.now()))
    return value


async def _admit(manager, tx, actor, document):
    if (type(document) is not dict or set(document) != {"version", "namespace", "payload", "context_hash"}
            or type(document.get("version")) is not int or document["version"] != 1
            or _document(document.get("namespace"), document.get("payload")) != document):
        raise JobError("root_context_evidence_corrupt")
    value = await _admission_state(manager, tx, actor, _barrier_key(actor, document))
    if value is not None:
        if value.get("document") != document:
            raise JobError("root_control_evidence_corrupt")
        raise JobError("root_context_cancelled")


async def _cancel_job(manager, tx, actor, job_id, *, key, fingerprint, reason):
    row = await manager._authorize(tx, actor, job_id)
    old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE operation_key=?", (key,))
    if old:
        if old["request_hash"] != fingerprint or old["job_id"] != job_id or old["action"] != "stop":
            raise JobError("operation_conflict")
        manager._control_notifications.get().add(job_id)
        return _operation(old)
    if row["cancel_requested_at"] is None:
        # Cancellation retains an already closed model phase. Rewriting it as
        # running cancellation would strand its original pending/result proof.
        status = row["status"] if row["supervision_status"] == "closed" or row["status"] == "model_closed" else (
            "cancelled" if row["resource_state"] == "not_started" else "cancelling")
        now = manager.now()
        count = await tx.write_job(job_id, f"UPDATE {manager.jobs} SET status=?,cancel_requested_at=?,cancel_reason=?,"
            "intake='closed',version=version+1,updated_at=? WHERE job_id=? AND generation=?",
            (status, now, reason, now, job_id, row["generation"]))
        if count != 1:
            raise JobError("stale_execution")
        row = {**row, "status": status, "cancel_requested_at": now, "cancel_reason": reason,
               "intake": "closed", "version": row["version"]+1, "updated_at": now}
        tx.remember_row(manager.jobs, "job_id", job_id, row)
    return await manager._insert_operation(tx, job_id=job_id, version=row["version"], actor=actor,
        action="stop", key=key, fingerprint=fingerprint, payload={"job_id": job_id,
        "status": row["status"], "cancel_requested": True, "resource_state": row["resource_state"]})


async def _request(manager, tx, actor, key, request):
    original = await _state(manager, tx, actor, key)
    if original is not None and original.get("request") != request:
        raise JobError("operation_conflict")
    return original


async def _begin(manager, actor, request_key, request):
    """Reserve this exact request and close admission without writing a root."""
    document = request["document"]
    begin_key = "power-loop.root-stop-begin.v1:" + _hash(request_key)
    response = {"state": "admission_closed", "request_hash": _hash(request)}
    async with manager._write() as tx:
        await _lock_session(manager, tx, actor)
        original = await _request(manager, tx, actor, request_key, request)
        barrier_key = _barrier_key(actor, document)
        barrier = await _state(manager, tx, actor, barrier_key)
        if original is not None:
            if barrier is None or barrier.get("document") != document:
                raise JobError("root_control_evidence_corrupt")
            return original
        begin = await _request(manager, tx, actor, begin_key, request)
        if begin is not None:
            if begin.get("response") != response or barrier is None or barrier.get("document") != document:
                raise JobError("root_control_evidence_corrupt")
            return None
        if barrier is None:
            await _save(manager, tx, actor, barrier_key, {"document": document, "first_request_hash": _hash(request)})
        elif barrier.get("document") != document:
            raise JobError("root_control_evidence_corrupt")
        await _save(manager, tx, actor, begin_key, {"request": request, "response": response})
    return None


async def _discover(manager, actor, document):
    """Read immutable provenance outside the control writer's lock."""
    candidates = await manager.db.fetchall(f"SELECT j.job_id,e.payload_json FROM {manager.jobs} j "
        f"LEFT JOIN {manager.events} e ON e.job_id=j.job_id AND e.event_type='operation' AND e.action='create' "
        "WHERE j.tenant_id=? AND j.principal_id=? AND j.session_id=? "
        "AND j.kind='send' AND j.parent_job_id IS NULL ORDER BY j.job_id LIMIT 10001",
        (actor.tenant_id, actor.principal_id, actor.session_id))
    if len(candidates) > 10000:
        raise JobError("root_control_history_requires_compaction")
    sources = []
    for candidate in candidates:
        try:
            creation = json.loads(candidate["payload_json"])
            if type(creation) is not dict or type(creation.get("request")) is not dict:
                raise ValueError()
            if creation["request"].get("host_context") is None:
                continue
            source = await _read(manager, manager.db, actor, candidate["job_id"])
            if source["context_hash"] != document["context_hash"]:
                continue
            if any(source[k] != document[k] for k in document):
                raise ValueError()
            sources.append(source)
        except (ValueError, TypeError, KeyError):
            raise JobError("root_control_evidence_corrupt") from None
    return sources


def _scope(original, request):
    try:
        response = original["response"]
        sources = response["sources"]
        if (response["state"] != "scoped" or response["request_hash"] != _hash(request)
                or type(sources) is not list or len(sources) > 10000
                or len({s["root_job_id"] for s in sources}) != len(sources)):
            raise ValueError()
        return sources
    except (ValueError, TypeError, KeyError):
        raise JobError("root_control_evidence_corrupt") from None


async def _freeze_scope(manager, actor, request_key, request):
    key = "power-loop.root-stop-scope.v1:" + _hash(request_key)
    original = await _request(manager, manager.db, actor, key, request)
    if original is not None:
        return _scope(original, request)
    sources = await _discover(manager, actor, request["document"])
    value = {"request": request, "response": {"state": "scoped", "request_hash": _hash(request), "sources": sources}}
    if len(_json(value).encode()) > 1_000_000:
        raise JobError("root_control_history_requires_compaction")
    async with manager._write() as tx:
        await _lock_session(manager, tx, actor)
        original = await _request(manager, tx, actor, key, request)
        if original is not None:
            return _scope(original, request)
        await _save(manager, tx, actor, key, value)
    return sources


async def _verify_source(manager, actor, source, document):
    try:
        current = await _read(manager, manager.db, actor, source["root_job_id"])
        # Context cancellation covers future attempts too. The generation saved
        # in the scope is historical; the write fences the current generation.
        if (source.keys() != current.keys() or type(source["root_generation"]) is not int
                or not 0 <= source["root_generation"] <= current["root_generation"]
                or any(source[k] != document[k] for k in document)
                or {k: v for k, v in current.items() if k != "root_generation"}
                != {k: v for k, v in source.items() if k != "root_generation"}):
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise JobError("root_control_evidence_corrupt") from None


async def _cancel_tree(manager, actor, source, request):
    await _verify_source(manager, actor, source, request["document"])
    job_id = source["root_job_id"]
    key = manager._origin_key(actor, "stop", None, None,
        "context-stop:" + _hash([request["system_key"], request["document"]["context_hash"], job_id])[:48])
    async with manager._write() as tx:
        operation = await _cancel_job(manager, tx, actor, job_id, key=key,
            fingerprint=_hash([job_id, request["reason"]]), reason=request["reason"])
    return {"source": source, "operation": asdict(operation)}


async def _receipt(manager, actor, original, request):
    response = original.get("response", {})
    if (type(response) is not dict or response.get("request_hash") != _hash(request)
            or response.get("context_hash") != request["document"]["context_hash"]
            or response.get("state") != "accepted" or type(response.get("roots")) is not list
            or any(response.get(k) is not False for k in (
                "execution_authority", "resource_stop_confirmed", "billing_settled"))):
        raise JobError("root_control_evidence_corrupt")
    for item in response["roots"]:
        try:
            operation = await manager.db.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?",
                (item["operation"]["operation_id"],))
            if (operation is None or operation["action"] != "stop"
                    or operation["job_id"] != item["source"]["root_job_id"]
                    or asdict(_operation(operation)) != item["operation"]):
                raise ValueError()
            await _verify_source(manager, actor, item["source"], request["document"])
        except (ValueError, TypeError, KeyError):
            raise JobError("root_control_evidence_corrupt") from None
    return response


async def cancel(manager, actor, *, namespace, payload, system_key, reason):
    """Close exact-context admission, then stop each original root independently.

    Progress survives process death or an unknown commit. The final receipt is
    historical and grants no execution, resource-stop or billing authority.
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
    original = await _begin(manager, actor, request_key, request)
    if original is not None:
        return await _receipt(manager, actor, original, request)
    sources = await _freeze_scope(manager, actor, request_key, request)
    roots = [await _cancel_tree(manager, actor, source, request) for source in sources]
    response = {"version": 1, "state": "accepted", "request_hash": _hash(request),
        "context_hash": document["context_hash"], "roots": roots, "execution_authority": False,
        "resource_stop_confirmed": False, "billing_settled": False}
    if len(_json(response).encode()) > 1_000_000:
        raise JobError("root_control_history_requires_compaction")
    async with manager._write() as tx:
        await _lock_session(manager, tx, actor)
        original = await _request(manager, tx, actor, request_key, request)
        if original is None:
            await _save(manager, tx, actor, request_key, {"request": request, "response": response})
    return response if original is None else original["response"]
