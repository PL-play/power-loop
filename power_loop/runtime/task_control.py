"""Durable model-origin cancellation, separate from physical stop confirmation."""
from __future__ import annotations

import json
from dataclasses import asdict

from . import control_requests as requests
from .jobs import JobError, JobLease, _hash, _json, _operation, verify_stored
from .message_ref import MessageRef

TARGET_BARRIERS = frozenset({"stop_generation_changed"})


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
    if source["intake"] != "open" or source["status"] not in ("running", "settling"):
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
    return requests.identity("stop", key)


def _completion_id(request_id):
    return requests.completion_id(request_id)


def _reference(event):
    return requests.reference(event)


@verify_stored
async def _stored_request(manager, reader, actor, event, request, completion=None):
    accepted = await requests.stored(manager, reader, actor, event, request, action="stop", completion=completion)
    if request["expected_generation"] is not None and accepted.data["target"]["generation"] != request["expected_generation"]:
        raise JobError("stop_request_corrupt")
    return accepted


def _completed(accepted, event, outcome):
    return requests.completed(accepted, event, outcome)


async def _accept_stop_request(manager, actor, request, target):
    from .managed_effects import _ref

    async def authorize(tx):
        source = await _source(manager, tx, actor, JobLease(**request["caller"]), _ref(request["call_ref"]), request["tool_call_id"])
        ancestor = source
        for _ in range(64):
            if ancestor["job_id"] == request["job_id"]:
                raise JobError("cannot_stop_current_execution")
            if ancestor["parent_job_id"] is None:
                return source
            ancestor = await manager._row(tx, ancestor["parent_job_id"])
        raise JobError("job_depth_exceeded")

    return await requests.accept(manager, actor, request, target, action="stop", authorize=authorize)


async def _apply_stop_request(manager, actor, event, data):
    """Only target-tree reads/writes; source acceptance has already committed."""
    request, target, request_ref = data["request"], data["target"], _reference(event)
    from .managed_effects import _ref
    key = manager._origin_key(actor, "stop", _ref(request["call_ref"]), request["tool_call_id"], None)
    job_id = request["job_id"]
    async with manager.tree_write(job_id) as tx:
        found = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE operation_key=? OR event_id=?",
            (key, requests.rejection_id(event["event_id"])))
        if found:
            old = found[0]
            await manager._authorize(tx, actor, old["job_id"])
            return requests.read_outcome(manager, requests.AcceptedRequest(event, data, {}), old), old
        row = await manager._authorize(tx, actor, job_id)
        if {key: row[key] for key in ("job_id", "root_job_id", "generation")} != target:
            return None, await requests.reject(manager, tx, actor, event, data, row, "stop_generation_changed")
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


async def _complete_stop_request(manager, actor, accepted, outcome):
    await requests.complete(manager, actor, accepted, outcome)


async def _discover_stop_request(manager, reader, actor, request):
    return await requests.discover(manager, reader, actor, request, action="stop")


async def recall_stop(manager, actor, job_id, *, caller, call_ref, tool_call_id, reason="Stopped by request", expected_generation=None):
    """Read back one accepted intent after ACK loss, without creating an action."""
    request = _intent(job_id, caller, call_ref, tool_call_id, reason, expected_generation)
    async with manager._read() as tx:
        accepted, outcome = await _discover_stop_request(manager, tx, actor, request)
        if outcome is None:
            return None
        if accepted.completion is not None:
            _completed(accepted, accepted.completion, outcome)
        return requests.read_outcome(manager, accepted, outcome)


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
    requests.remember_event(manager, accepted.event)
    if outcome is None:
        operation, outcome = await _apply_stop_request(manager, actor, accepted.event, accepted.data)
    else:
        operation = requests.read_outcome(manager, accepted, outcome)
    await _complete_stop_request(manager, actor, accepted, outcome)
    if operation is None:
        raise JobError(json.loads(outcome["payload_json"])["error"])
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
    """Stop the original logical task through the shared control channel."""
    from power_loop.agent.runtime_types import JobRef, plain

    from .child_tasks import tool_actor
    binding, actor = await tool_actor(loop)
    port = loop._runtime_worker.inputs
    async with binding.runtime.manager._read() as reader:
        await binding.runtime.manager._fence(reader, binding.actor, binding.lease)
    row = await port.resolve_job(port.db, JobRef(port.store.store_id, job_id), actor)
    generation = row["generation"] if expected_generation is None else expected_generation
    receipt = await loop._runtime_worker.runtime.client.cancel(
        JobRef(port.store.store_id, job_id, generation), actor=actor, scope="subtree", reason=reason)
    return {"version": 1, "backend": "task_registry", "job_id": job_id,
            "generation": generation, "status": "accepted", "receipt": plain(receipt)}
