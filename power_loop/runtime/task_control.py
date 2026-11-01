"""Durable model-origin cancellation, separate from physical stop confirmation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .jobs import JobActor, JobError, JobLease, _hash, _json, _operation, verify_stored
from .message_ref import MessageRef


def _receipt(manager, event, request, request_ref):
    try:
        data = json.loads(event["payload_json"])
        if (event["event_type"] != "operation" or event["action"] != "stop" or event["job_id"] != request["job_id"] or
                event["caller_session_id"] != request["call_ref"]["session_id"] or event["tool_call_id"] != request["tool_call_id"] or
                json.loads(event["call_ref_json"]) != request["call_ref"] or event["request_hash"] != _hash(request) or
                not isinstance(data, dict) or set(data) != {"version", "request", "request_ref", "target", "response", "evidence_hash"} or
                type(data["version"]) is not int or data["version"] != 1 or data["request"] != request or
                data["request_ref"] != request_ref or
                data["evidence_hash"] != _hash({k: v for k, v in data.items() if k != "evidence_hash"})):
            raise JobError("stop_receipt_mismatch")
        target, response = data["target"], data["response"]
        if (set(target) != {"job_id", "generation"} or target["job_id"] != request["job_id"] or
                type(target["generation"]) is not int or not 0 <= target["generation"] < 2**63 or
                not isinstance(response, dict) or set(response) != {"job_id", "generation", "status", "cancel_requested", "control_pending", "resource_state", "resource_stop_confirmed"} or
                response.get("job_id") != target["job_id"] or type(response.get("generation")) is not int or response.get("generation") != target["generation"] or
                response.get("status") not in ("accepted", "already_closed") or response.get("resource_state") not in ("not_started", "running", "stopped", "unknown") or
                type(response.get("cancel_requested")) is not bool or type(response.get("control_pending")) is not bool or
                response.get("resource_stop_confirmed") is not False):
            raise JobError("stop_receipt_mismatch")
        return _operation(event)
    except (ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("stop_receipt_mismatch") from exc


async def _source(manager, tx, actor, caller, call_ref, tool_call_id):
    source = await manager._authorize(tx, actor, caller.job_id)
    manager._check_proof(source, caller)
    if source["kind"] != "send" or source["session_id"] != actor.session_id:
        raise JobError("stop_caller_mismatch")
    # Cancellation is a safe control action even when this actual send's lease
    # expired. A closed or superseded execution still cannot issue new actions.
    if source["intake"] != "open" or source["status"] not in ("running", "waiting", "settling"):
        raise JobError("stop_caller_closed")
    seen, owner = {source["job_id"]}, source
    for _ in range(64):
        if owner["control_owner_job_id"] is None:
            break
        if owner["control_owner_job_id"] in seen:
            raise JobError("invalid_control_owner")
        seen.add(owner["control_owner_job_id"])
        parent = await manager._row(tx, owner["control_owner_job_id"])
        if parent is None or (parent["tenant_id"], parent["principal_id"]) != (actor.tenant_id, actor.principal_id):
            raise JobError("invalid_control_owner")
        manager._check_proof(parent, JobLease(parent["job_id"], owner["control_owner_executor_id"],
            owner["control_owner_boot_id"], owner["control_owner_generation"]))
        if parent["intake"] != "open":
            raise JobError("stop_caller_closed")
        owner = parent
    else:
        raise JobError("job_depth_exceeded")
    await manager._validate_call(tx, actor, call_ref, tool_call_id)
    creation = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create' AND event_type='operation'", (caller.job_id,))
    try:
        index = json.loads(creation["payload_json"])["request"]["send_index"] if creation else None
        if type(index) is not int or call_ref.send_index != index:
            raise JobError("stop_call_send_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("stop_call_send_mismatch") from exc
    return source


def _intent(job_id, caller, call_ref, tool_call_id, reason, expected_generation):
    if not isinstance(job_id, str) or not 0 < len(job_id) <= 64:
        raise JobError("invalid_stop_target")
    if (not isinstance(caller, JobLease) or not isinstance(call_ref, MessageRef) or
            not isinstance(tool_call_id, str) or not 0 < len(tool_call_id) <= 255):
        raise JobError("persisted_tool_call_required")
    from .execution_membership import _lease
    try:
        _lease(asdict(caller))
    except JobError as exc:
        raise JobError("invalid_stop_caller") from exc
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise JobError("invalid_cancel_reason")
    if expected_generation is not None and (type(expected_generation) is not int or not 0 <= expected_generation < 2**63):
        raise JobError("invalid_stop_generation")
    return {"job_id": job_id, "caller": asdict(caller), "call_ref": call_ref.to_dict(),
        "tool_call_id": tool_call_id, "reason": reason, "expected_generation": expected_generation}


def _request_id(key):
    return "stop_req_"+key[:48]


def _completion_id(request_id):
    return "stop_done_"+_hash(request_id)[:48]


def _reference(event):
    from .model_exit import content_reference
    return content_reference(event)


@dataclass(frozen=True)
class _AcceptedStop:
    event: dict
    data: dict
    source: dict
    completion: dict | None = None


@verify_stored
async def _stored_request(manager, reader, actor, event, request, completion=None):
    """Cold retry validates the original permission, never recreates it."""
    from .execution_membership import scope
    from .managed_effects import _ref

    try:
        data = json.loads(event["payload_json"])
        key = manager._origin_key(actor, "stop", _ref(request["call_ref"]), request["tool_call_id"], None)
        if (event["event_id"] != _request_id(key) or event["event_type"] != "task_control_request"
                or event["job_id"] != request["caller"]["job_id"] or event["delivery_status"] != "not_applicable"
                or event["caller_session_id"] != actor.session_id or event["tool_call_id"] != request["tool_call_id"]
                or json.loads(event["call_ref_json"]) != request["call_ref"]
                or set(data) != {"version", "phase", "action", "actor", "source", "target", "request"}
                or data["version"] != 1 or type(data["version"]) is not int or data["phase"] != "accepted"
                or data["action"] != "stop" or data["actor"] != asdict(actor)
                or event["request_hash"] != _hash(data)):
            raise JobError("stop_request_corrupt")
        if data["request"] != request:
            raise JobError("operation_conflict")
        source = await manager._authorize(reader, actor, event["job_id"])
        frozen = await scope(manager, reader, event["job_id"], request["caller"]["generation"])
        if (data["source"] != {**request["caller"], "root_job_id": source["root_job_id"]}
                or frozen["lease"] != request["caller"] or frozen["kind"] != "send"
                or (frozen["tenant_id"], frozen["principal_id"], frozen["session_id"]) !=
                    (actor.tenant_id, actor.principal_id, actor.session_id)
                or set(data["target"]) != {"job_id", "root_job_id", "generation"}
                or data["target"]["job_id"] != request["job_id"]
                or type(data["target"]["generation"]) is not int or data["target"]["generation"] < 0
                or request["expected_generation"] is not None and data["target"]["generation"] != request["expected_generation"]):
            raise JobError("stop_request_corrupt")
        await manager._validate_call(reader, actor, _ref(request["call_ref"]), request["tool_call_id"])
        return _AcceptedStop(dict(event), data, dict(source), completion)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("stop_request_corrupt") from exc


def _completed(accepted, event, outcome):
    expected = {"version": 1, "phase": "completed", "request": _reference(accepted.event), "outcome": _reference(outcome)}
    try:
        if (event["event_id"] != _completion_id(accepted.event["event_id"])
                or event["event_type"] != "task_control_request" or event["job_id"] != accepted.event["job_id"]
                or event["caller_session_id"] != accepted.data["actor"]["session_id"]
                or event["delivery_status"] != "not_applicable" or event["job_version"] <= accepted.event["job_version"]
                or event["request_hash"] != _hash(expected) or json.loads(event["payload_json"]) != expected):
            raise JobError("stop_request_completion_corrupt")
    except (ValueError, TypeError, KeyError) as exc:
        raise JobError("stop_request_completion_corrupt") from exc


async def _accept_stop_request(manager, actor, request, target):
    from .managed_effects import _ref

    ref, caller = _ref(request["call_ref"]), JobLease(**request["caller"])
    key = manager._origin_key(actor, "stop", ref, request["tool_call_id"], None)
    event_id = _request_id(key)
    old = None
    async with manager._write() as tx:
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
        if old is None:
            source = await _source(manager, tx, actor, caller, ref, request["tool_call_id"])
            ancestor = source
            for _ in range(64):
                if ancestor["job_id"] == request["job_id"]:
                    raise JobError("cannot_stop_current_execution")
                if ancestor["parent_job_id"] is None:
                    break
                ancestor = await manager._row(tx, ancestor["parent_job_id"])
            else:
                raise JobError("job_depth_exceeded")
            # A normal native dispatch start may still be buffered. It must
            # commit with acceptance before another tree can apply the request.
            from .task_runtime import _SEND
            binding = _SEND.get()
            if binding is not None and binding.runtime.manager is manager and binding.lease == caller:
                from .control_runtime import flush_binding
                await flush_binding(binding, transaction=tx)
            version = source["version"]+1
            changed = await tx.write_job(caller.job_id, f"UPDATE {manager.jobs} SET version=version+1,updated_at=? "
                "WHERE job_id=? AND root_job_id=? AND generation=? AND executor_id=? AND boot_id=? AND version=? "
                "AND intake='open' AND status IN ('running','waiting','settling')",
                (manager.now(), caller.job_id, source["root_job_id"], caller.generation, caller.executor_id, caller.boot_id, source["version"]))
            if changed != 1:
                raise JobError("stop_caller_changed")
            data = {"version": 1, "phase": "accepted", "action": "stop", "actor": asdict(actor),
                "source": {**asdict(caller), "root_job_id": source["root_job_id"]}, "target": target, "request": request}
            event = {"event_id": event_id, "job_id": caller.job_id, "event_type": "task_control_request", "job_version": version,
                "caller_session_id": actor.session_id, "tool_call_id": request["tool_call_id"], "call_ref_json": _json(ref.to_dict()),
                "request_hash": _hash(data), "payload_json": _json(data), "delivery_status": "not_applicable", "created_at": manager.now()}
            await tx.append_event(tuple(event), tuple(event.values()))
            _remember_request_event(manager, event)
            tx.open_obligation(session_id=actor.session_id, kind="control_request", source_id=event_id,
                job_id=caller.job_id, opened_version=version)
    if old is not None:
        async with manager._read() as reader:
            completion = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (_completion_id(event_id),))
            return await _stored_request(manager, reader, actor, old, request, completion), True
    return _AcceptedStop(event, data, {**source, "version": version}), False


async def _apply_stop_request(manager, actor, event, data):
    """Only target-tree reads/writes; source acceptance has already committed."""
    request, target, request_ref = data["request"], data["target"], _reference(event)
    from .managed_effects import _ref
    key = manager._origin_key(actor, "stop", _ref(request["call_ref"]), request["tool_call_id"], None)
    job_id = request["job_id"]
    async with manager._write() as tx:
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE operation_key=?", (key,))
        if old is not None:
            await manager._authorize(tx, actor, old["job_id"])
            if old["request_hash"] != _hash(request):
                raise JobError("operation_conflict")
            return _receipt(manager, old, request, request_ref), old
        row = await manager._authorize(tx, actor, job_id)
        if {key: row[key] for key in ("job_id", "root_job_id", "generation")} != target:
            raise JobError("stop_generation_changed")
        if row["supervision_status"] != "closed" and row["cancel_requested_at"] is None:
            now = manager.now()
            status = "cancelled" if row["resource_state"] == "not_started" else "cancelling"
            if await tx.write_job(job_id, f"UPDATE {manager.jobs} SET status=?,cancel_requested_at=?,cancel_reason=?,intake='closed',"
                    "version=version+1,updated_at=? WHERE job_id=? AND root_job_id=? AND version=? AND generation=?",
                    (status, now, request["reason"], now, job_id, target["root_job_id"], row["version"], target["generation"])) != 1:
                raise JobError("stop_write_mismatch")
            row = {**row, "version": row["version"]+1, "updated_at": now, "status": status,
                "cancel_requested_at": now, "cancel_reason": request["reason"], "intake": "closed"}
        payload = {"version": 1, "request": request, "request_ref": request_ref,
            "target": {"job_id": job_id, "generation": row["generation"]},
            "response": {"job_id": job_id, "generation": row["generation"],
                "status": "accepted" if row["cancel_requested_at"] is not None else "already_closed",
                "cancel_requested": row["cancel_requested_at"] is not None, "control_pending": row["supervision_status"] != "closed",
                "resource_state": row["resource_state"], "resource_stop_confirmed": False}}
        payload["evidence_hash"] = _hash(payload)
        operation = await manager._insert_operation(tx, job_id=job_id, version=row["version"], actor=actor, action="stop", key=key,
            fingerprint=_hash(request), call_ref=_ref(request["call_ref"]), tool_call_id=request["tool_call_id"], payload=payload)
        outcome = {"event_id": operation.operation_id, "job_id": job_id, "event_type": "operation", "job_version": row["version"],
            "request_hash": _hash(request), "payload_json": _json(payload)}
    return operation, outcome


class _RequestProjectionConflict(Exception):
    pass


async def _complete_stop_request(manager, actor, accepted, outcome):
    """A source-owned completion receipt and its pending-index removal are atomic."""
    if accepted.completion is not None:
        _completed(accepted, accepted.completion, outcome)
        _remember_request_event(manager, accepted.completion)
        return
    source = accepted.source
    event_id = _completion_id(accepted.event["event_id"])
    value = {"version": 1, "phase": "completed", "request": _reference(accepted.event), "outcome": _reference(outcome)}
    while True:
        try:
            async with manager._write() as tx:
                changed = await tx.write_job(source["job_id"], f"UPDATE {manager.jobs} SET version=version+1,updated_at=? "
                    "WHERE job_id=? AND root_job_id=? AND generation=? AND version=?",
                    (manager.now(), source["job_id"], accepted.data["source"]["root_job_id"], source["generation"], source["version"]))
                if changed != 1:
                    raise _RequestProjectionConflict()
                event = {"event_id": event_id, "job_id": source["job_id"], "event_type": "task_control_request",
                    "job_version": source["version"]+1, "caller_session_id": actor.session_id,
                    "request_hash": _hash(value), "payload_json": _json(value), "delivery_status": "not_applicable", "created_at": manager.now()}
                await tx.append_event(tuple(event), tuple(event.values()))
                _remember_request_event(manager, event)
                tx.close_obligation(session_id=actor.session_id, kind="control_request",
                    source_id=accepted.event["event_id"], job_id=source["job_id"])
            return
        except _RequestProjectionConflict:
            async with manager._read() as reader:
                completed = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
                if completed is not None:
                    _completed(accepted, completed, outcome)
                    _remember_request_event(manager, completed)
                    return
                source = await manager._authorize(reader, actor, source["job_id"])


async def _discover_stop_request(manager, reader, actor, request):
    from .managed_effects import _ref
    key = manager._origin_key(actor, "stop", _ref(request["call_ref"]), request["tool_call_id"], None)
    identity = _request_id(key)
    rows = await reader.fetchall(f"SELECT * FROM {manager.events} WHERE event_id IN (?,?) OR operation_key=?",
        (identity, _completion_id(identity), key))
    original = next((row for row in rows if row["event_id"] == identity), None)
    completion = next((row for row in rows if row["event_id"] == _completion_id(identity)), None)
    outcome = next((row for row in rows if row["operation_key"] == key), None)
    if original is None:
        if outcome is not None or completion is not None:
            raise JobError("stop_request_original_missing")
        return None, None
    accepted = await _stored_request(manager, reader, actor, original, request, completion)
    if outcome is not None:
        await manager._authorize(reader, actor, request["job_id"])
        _receipt(manager, outcome, request, _reference(original))
    elif completion is not None:
        raise JobError("stop_request_outcome_missing")
    return accepted, outcome


async def recall_stop(manager, actor, job_id, *, caller, call_ref, tool_call_id, reason="Stopped by request", expected_generation=None):
    """Read back one accepted intent after ACK loss, without creating an action."""
    request = _intent(job_id, caller, call_ref, tool_call_id, reason, expected_generation)
    async with manager._read() as tx:
        accepted, outcome = await _discover_stop_request(manager, tx, actor, request)
        if outcome is None:
            return None
        if accepted.completion is not None:
            _completed(accepted, accepted.completion, outcome)
        return _receipt(manager, outcome, request, _reference(accepted.event))


async def request_stop(manager, actor, job_id, *, caller, call_ref, tool_call_id, reason="Stopped by request", expected_generation=None):
    """Accept in the caller tree, apply in the target, then acknowledge in caller.

    This never runs an executor, claims physical stop, releases billing or closes
    supervision. Trusted host shutdown continues to use its mechanical port.
    """
    request = _intent(job_id, caller, call_ref, tool_call_id, reason, expected_generation)
    async with manager._read() as reader:
        accepted, outcome = await _discover_stop_request(manager, reader, actor, request)
        if accepted is None:
            row = await manager._authorize(reader, actor, job_id)
            if expected_generation is not None and row["generation"] != expected_generation:
                raise JobError("stop_generation_changed")
            target = {key: row[key] for key in ("job_id", "root_job_id", "generation")}
    replayed = accepted is not None
    if accepted is None:
        accepted, replayed = await _accept_stop_request(manager, actor, request, target)
    return await _finish_accepted_stop(manager, actor, accepted, outcome), replayed


async def _finish_accepted_stop(manager, actor, accepted, outcome):
    """Resume only the exact committed request, without another source grant."""
    _remember_request_event(manager, accepted.event)
    request = accepted.data["request"]
    if outcome is None:
        operation, outcome = await _apply_stop_request(manager, actor, accepted.event, accepted.data)
    else:
        operation = _receipt(manager, outcome, request, _reference(accepted.event))
    await _complete_stop_request(manager, actor, accepted, outcome)
    return operation


@verify_stored
async def _recover_stop(manager, actor, request):
    async with manager._read() as reader:
        accepted, outcome = await _discover_stop_request(manager, reader, actor, request)
    if accepted is None:
        raise JobError("stop_control_unconfirmed")
    return await _finish_accepted_stop(manager, actor, accepted, outcome)


@verify_stored
async def reconcile_stop_request(manager, actor, request_id):
    """Host-only completion of an already accepted stop; no callback or model."""
    from .managed_completion import _host_only
    _host_only()
    async with manager._read() as reader:
        event = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='task_control_request'", (request_id,))
        if event is None:
            raise JobError("stop_request_original_missing")
        request = json.loads(event["payload_json"])["request"]
        accepted, outcome = await _discover_stop_request(manager, reader, actor, request)
    return await _finish_accepted_stop(manager, actor, accepted, outcome)


def _request_records(events):
    accepted, completed = {}, {}
    for event in events:
        if event["event_type"] != "task_control_request":
            continue
        data = json.loads(event["payload_json"])
        if data["phase"] == "accepted":
            accepted[event["event_id"]] = (event, data)
        elif data["phase"] == "completed":
            completed[data["request"]["event_id"]] = event
        else:
            raise JobError("stop_request_corrupt")
    return accepted, completed


def note_fresh_source(manager, job_id):
    from .execution_membership import _remember_after_commit
    _remember_after_commit(manager, None, "_control_request_events", job_id, {})


def _remember_request_event(manager, event):
    from .execution_membership import _cache, remember_merge
    name, job_id = "_control_request_events", event["job_id"]
    cache, overlay = _cache(manager, name), manager._control_overlay.get()
    if job_id in cache or overlay is not None and (name, job_id) in overlay:
        remember_merge(manager, name, job_id, {event["event_id"]: event})
        # An unknown commit must not leave a cached *absence* of a request.
        manager.after_outcome(lambda: None, lambda: cache.pop(job_id, None))


async def request_events(manager, reader, job_id):
    from .execution_membership import _trusted
    known = _trusted(manager, "_control_request_events", job_id)
    if known is not None:
        return list(known.values())
    return await reader.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='task_control_request'", (job_id,))


async def request_obligations(manager, reader, row, events, *, verify=False):
    """Source-owned receipts determine pending state; cold checks follow proofs."""
    accepted, completed = _request_records(events)
    if verify and completed.keys()-accepted.keys():
        raise JobError("stop_request_original_missing")
    entries = set()
    for identity, (event, data) in accepted.items():
        if verify:
            actor = JobActor(**data["actor"])
            original = await _stored_request(manager, reader, actor, event, data["request"])
            target = await manager._authorize(reader, actor, data["target"]["job_id"])
            if target["root_job_id"] != data["target"]["root_job_id"]:
                raise JobError("stop_request_target_changed")
            if identity in completed:
                value = json.loads(completed[identity]["payload_json"])
                outcome = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (value["outcome"]["event_id"],))
                if outcome is None:
                    raise JobError("stop_request_outcome_missing")
                _receipt(manager, outcome, data["request"], _reference(event))
                _completed(original, completed[identity], outcome)
        if identity not in completed:
            entries.add((data["actor"]["session_id"], "control_request", identity, row["job_id"], event["job_version"]))
    return entries


def completed_request_events(events):
    """An execution may exit before its accepted cross-tree stop is acknowledged."""
    accepted, completed = _request_records(events)
    if accepted.keys()-completed.keys():
        raise JobError("task_control_request_unresolved")
    return [event for identity, (event, _) in sorted(accepted.items()) for event in (event, completed[identity])]


async def signal_local(runtime, actor, operation):
    """Signal retained current tasks once; missing handles leave durable control pending."""
    manager = runtime.manager
    snapshots = list(runtime._local_executions.values())
    tasks = set()
    selected = []
    async with manager._read() as tx:
        target = await manager._authorize(tx, actor, operation.job_id)
        event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (operation.operation_id,))
        op = _receipt(manager, event, operation.result["request"], operation.result["request_ref"]) if event else None
        if op is not None and op.job_id != target["job_id"]:
            raise JobError("stop_receipt_mismatch")
        if op is None or target["generation"] != op.result["target"]["generation"] or target["cancel_requested_at"] is None:
            return 0
        for local in snapshots:
            binding = local["binding"]
            if binding.runtime is not runtime or local["signalled"] or binding.stop_renewal.is_set():
                continue
            try:
                row = await manager._authorize(tx, actor, binding.lease.job_id)
            except JobError as exc:
                if exc.code == "access_denied":
                    continue
                raise
            try:
                manager._check_proof(row, binding.lease)
            except JobError:
                continue
            if row["resource_state"] not in ("running", "unknown"):
                continue
            current = row
            for _ in range(64):
                if current["job_id"] == target["job_id"]:
                    tasks.add(local["task"])
                    selected.append(local)
                    break
                if current["parent_job_id"] is None:
                    break
                current = await manager._row(tx, current["parent_job_id"])
            else:
                raise JobError("job_depth_exceeded")
    import asyncio
    count = 0
    current = list(runtime._local_executions.values())
    live_ids = {id(local) for local in current}
    selected = [local for local in selected if id(local) in live_ids and not local["binding"].stop_renewal.is_set()]
    selected_ids = {id(local) for local in selected}
    signalled_tokens = set()
    from .executor_lifecycle import signal_token
    for local in selected:
        token = local.get("cancel_token")
        if token is not None and id(token) not in signalled_tokens:
            signalled_tokens.add(id(token))
            try:
                was_cancelled = token.is_cancelled()
            except Exception:
                was_cancelled = False
            if signal_token(token, "durable task cancellation requested"):
                count += not was_cancelled
                local["signalled"] = True
    for task in tasks:
        if task is asyncio.current_task() or task.done() or task.cancelling():
            continue
        members = [local for local in current if local["task"] is task]
        if any(local["signalled"] for local in selected if local["task"] is task):
            # The operation's cooperative token also covers its provider/tool
            # race. Task.cancel would bypass node-result bookkeeping and make
            # a containing parallel gather treat this as whole-run failure.
            for local in selected:
                if local["task"] is task:
                    local["signalled"] = True
            continue
        if not members or any(id(local) not in selected_ids for local in members):
            # Never cancel a task now hosting an ancestor, another generation,
            # or exit bookkeeping outside the requested stop scope.
            for local in selected:
                if local["task"] is task:
                    local["signalled"] = True
            continue
        # Mark all nested executions before signalling once. Repeated requests
        # cannot cancel the same task again while it records actual exit usage.
        for local in current:
            if local["task"] is task:
                local["signalled"] = True
        count += bool(task.cancel("durable task cancellation requested"))
    return count


async def stop_from_current_call(loop, session_id, job_id, *, reason="Stopped by request", expected_generation=None):
    """Shared model-control adapter for jobs/workflow, not another model tool.

    Capture actual invocation identity, reconcile uncertain committed acceptance,
    and signal only after the persistent stop intent has committed.
    """
    from .invocation import current_invocation
    from .task_runtime import _SEND

    runtime = getattr(loop, "task_runtime", None)
    source = _SEND.get()
    if runtime is None or source is None or source.runtime is not runtime or source.actor.session_id != session_id:
        raise JobError("persisted_tool_call_required")
    manager, actor = runtime.manager, runtime.actor(session_id)
    seq, tool_call_id = current_invocation(session_id)
    if seq is None or not tool_call_id:
        raise JobError("persisted_tool_call_required")
    arguments = dict(caller=source.lease, call_ref=await manager.store.message_ref(session_id, seq),
        tool_call_id=tool_call_id, reason=reason, expected_generation=expected_generation)
    try:
        operation, replayed = await request_stop(manager, actor, job_id, **arguments)
    except JobError:
        raise
    except Exception:
        operation = await _recover_stop(manager, actor, _intent(job_id, **arguments))
        replayed = True
    signals, signal_error = 0, None
    try:
        signals = await signal_local(runtime, actor, operation)
    except Exception:
        signal_error = "local_signal_requires_reconciliation"
    return {"version": 1, "backend": "task_registry", "operation_id": operation.operation_id, "replayed": replayed,
        **operation.result["response"], "local_signals": signals, "signal_error": signal_error}
