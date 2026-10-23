"""Durable model-origin cancellation, separate from physical stop confirmation."""
from __future__ import annotations

import json
from dataclasses import asdict

from .jobs import JobError, JobLease, _hash, _operation
from .message_ref import MessageRef


def _receipt(manager, event, request):
    try:
        data = json.loads(event["payload_json"])
        if (event["event_type"] != "operation" or event["action"] != "stop" or event["job_id"] != request["job_id"] or
                event["caller_session_id"] != request["call_ref"]["session_id"] or event["tool_call_id"] != request["tool_call_id"] or
                json.loads(event["call_ref_json"]) != request["call_ref"] or event["request_hash"] != _hash(request) or
                not isinstance(data, dict) or set(data) != {"version", "request", "target", "response", "evidence_hash"} or
                type(data["version"]) is not int or data["version"] != 1 or data["request"] != request or
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


async def recall_stop(manager, actor, job_id, *, caller, call_ref, tool_call_id, reason="Stopped by request", expected_generation=None):
    """Read back one accepted intent after ACK loss, without creating an action."""
    request = _intent(job_id, caller, call_ref, tool_call_id, reason, expected_generation)
    key = manager._origin_key(actor, "stop", call_ref, tool_call_id, None)
    async with manager._write() as tx:
        row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE operation_key=?", (key,))
        if row is None:
            return None
        await manager._authorize(tx, actor, row["job_id"])
        if row["request_hash"] != _hash(request):
            raise JobError("operation_conflict")
        return _receipt(manager, row, request)


async def request_stop(manager, actor, job_id, *, caller, call_ref, tool_call_id, reason="Stopped by request", expected_generation=None):
    """Atomically accept one raw model call; replay returns that exact decision.

    This never runs an executor, claims physical stop, releases billing or closes
    supervision. Trusted host shutdown continues to use its mechanical port.
    """
    request = _intent(job_id, caller, call_ref, tool_call_id, reason, expected_generation)
    key = manager._origin_key(actor, "stop", call_ref, tool_call_id, None)
    async with manager._write() as tx:
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE operation_key=?", (key,))
        if old is not None:
            await manager._authorize(tx, actor, old["job_id"])
            if old["request_hash"] != _hash(request):
                raise JobError("operation_conflict")
            return _receipt(manager, old, request), True
        row = await manager._authorize(tx, actor, job_id)
        source = await _source(manager, tx, actor, caller, call_ref, tool_call_id)
        ancestor = source
        for _ in range(64):
            if ancestor["job_id"] == job_id:
                raise JobError("cannot_stop_current_execution")
            if ancestor["parent_job_id"] is None:
                break
            ancestor = await manager._row(tx, ancestor["parent_job_id"])
        else:
            raise JobError("job_depth_exceeded")
        if expected_generation is not None and row["generation"] != expected_generation:
            raise JobError("stop_generation_changed")
        if row["supervision_status"] != "closed" and row["cancel_requested_at"] is None:
            now = manager.now()
            status = "cancelled" if row["resource_state"] == "not_started" else "cancelling"
            if await tx.execute(f"UPDATE {manager.jobs} SET status=?,cancel_requested_at=?,cancel_reason=?,intake='closed',"
                    "version=version+1,updated_at=? WHERE job_id=? AND version=? AND generation=?",
                    (status, now, reason, now, job_id, row["version"], row["generation"])) != 1:
                raise JobError("stop_write_mismatch")
            saved = await manager._row(tx, job_id)
            if (saved["version"] != row["version"] + 1 or saved["generation"] != row["generation"] or
                    saved["cancel_requested_at"] != now or saved["cancel_reason"] != reason or saved["intake"] != "closed" or saved["status"] != status):
                raise JobError("stop_write_mismatch")
            row = saved
        payload = {"version": 1, "request": request, "target": {"job_id": job_id, "generation": row["generation"]},
            "response": {"job_id": job_id, "generation": row["generation"], "status": "accepted" if row["cancel_requested_at"] is not None else "already_closed",
                "cancel_requested": row["cancel_requested_at"] is not None, "control_pending": row["supervision_status"] != "closed",
                "resource_state": row["resource_state"], "resource_stop_confirmed": False}}
        payload["evidence_hash"] = _hash(payload)
        op = await manager._insert_operation(tx, job_id=job_id, version=row["version"], actor=actor, action="stop", key=key,
            fingerprint=_hash(request), call_ref=call_ref, tool_call_id=tool_call_id, payload=payload)
        saved = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (op.operation_id,))
        if saved is None:
            raise JobError("stop_write_mismatch")
        return _receipt(manager, saved, request), False


async def signal_local(runtime, actor, operation):
    """Signal retained current tasks once; missing handles leave durable control pending."""
    manager = runtime.manager
    snapshots = list(runtime._local_executions.values())
    tasks = set()
    selected = []
    async with manager._write() as tx:
        target = await manager._authorize(tx, actor, operation.job_id)
        event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (operation.operation_id,))
        op = _receipt(manager, event, operation.result["request"]) if event else None
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
    for local in selected:
        token = local.get("cancel_token")
        if token is not None and id(token) not in signalled_tokens:
            signalled_tokens.add(id(token))
            count += not token.is_cancelled()
            local["signalled"] = True
            token.cancel("durable task cancellation requested")
    for task in tasks:
        if task is asyncio.current_task() or task.done() or task.cancelling():
            continue
        members = [local for local in current if local["task"] is task]
        if any(local.get("cancel_token") is not None for local in selected if local["task"] is task):
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

    Capture actual invocation identity, reconcile uncertain acceptance read-only,
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
        operation = await recall_stop(manager, actor, job_id, **arguments)
        if operation is None:
            raise JobError("stop_control_unconfirmed") from None
        replayed = True
    signals, signal_error = 0, None
    try:
        signals = await signal_local(runtime, actor, operation)
    except Exception:
        signal_error = "local_signal_requires_reconciliation"
    return {"version": 1, "backend": "task_registry", "operation_id": operation.operation_id, "replayed": replayed,
        **operation.result["response"], "local_signals": signals, "signal_error": signal_error}
