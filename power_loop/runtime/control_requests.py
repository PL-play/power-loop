"""Source-owned permissions and receipts for repeatable, single-tree control steps.

Only accepted task events grant permission. A target acknowledgement, including
an explicit refusal, closes the source's pending obligation in another tree TX.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .jobs import JobActor, JobError, JobLease, _hash, _json, verify_stored
from .model_exit import content_reference as reference


@dataclass(frozen=True)
class AcceptedRequest:
    event: dict
    data: dict
    source: dict
    completion: dict | None = None


def identity(action, key):
    return action+"_req_"+key[:48]


def completion_id(request_id):
    return "control_done_"+_hash(request_id)[:48]


def rejection_id(request_id):
    return "control_no_"+_hash(request_id)[:48]


def _call_ref(action, request):
    if action == 'record':
        from .operation_records import _call_ref as record_ref
        return record_ref(request['call_ref'])
    from .managed_effects import _ref
    return _ref(request['call_ref'])


def _origin(manager, actor, request, action):
    operation = (request['operation_action'] if action == 'record' else
        'nudge' if action in ('resume', 'resume_node', 'restore') else 'wait_decision' if action=='decide' else action)
    return manager._origin_key(actor, operation,
        _call_ref(action, request), request["tool_call_id"], None)


def _domain(action):
    if action == "stop":
        from . import task_control
        return task_control
    if action == "nudge":
        from . import nudges
        return nudges
    if action == "dispatch":
        from . import resume_dispatch
        return resume_dispatch
    if action == "resume":
        from . import subagent_admission
        return subagent_admission
    if action == "resume_node":
        from . import node_admission
        return node_admission
    if action == "restore":
        from . import workflow_admission
        return workflow_admission
    if action == 'decide':
        from . import wait_decision_admission
        return wait_decision_admission
    if action == 'record':
        from . import operation_records
        return operation_records
    raise JobError("control_request_action_unknown")


@verify_stored
async def stored(manager, reader, actor, event, request, *, action, completion=None,
                 source_kinds=('send',), source_created_by=False):
    """Verify an original source permission; a raw plan alone never supplies it."""
    from .execution_membership import scope

    error = action+"_request_corrupt"
    try:
        data = json.loads(event["payload_json"])
        key = _origin(manager, actor, request, action)
        if (event["event_id"] != identity(action, key) or event["event_type"] != "task_control_request"
                or event["job_id"] != request["caller"]["job_id"] or event["delivery_status"] != "not_applicable"
                or event["caller_session_id"] != actor.session_id or event["tool_call_id"] != request["tool_call_id"]
                or json.loads(event["call_ref_json"]) != request["call_ref"]
                or set(data) != {"version", "phase", "action", "actor", "source", "target", "request"}
                or type(data["version"]) is not int or data["version"] != 1 or data["phase"] != "accepted"
                or data["action"] != action or data["actor"] != asdict(actor)
                or event["request_hash"] != _hash(data)):
            raise JobError(error)
        if data["request"] != request:
            raise JobError("operation_conflict")
        source = await manager._authorize(reader, actor, event["job_id"])
        frozen = await scope(manager, reader, event["job_id"], request["caller"]["generation"])
        if (data["source"] != {**request["caller"], "root_job_id": source["root_job_id"],
                    "execution_role": frozen["execution_role"]}
                or frozen["lease"] != request["caller"] or frozen["kind"] not in source_kinds
                or (frozen["tenant_id"], frozen["principal_id"], frozen["session_id"] or
                    (frozen['created_by_session_id'] if source_created_by else None)) !=
                    (actor.tenant_id, actor.principal_id, actor.session_id)
                or set(data["target"]) != {"job_id", "root_job_id", "generation"}
                or data["target"]["job_id"] != request["job_id"]
                or type(data["target"]["generation"]) is not int or data["target"]["generation"] < 0):
            raise JobError(error)
        await manager._validate_call(reader, actor, _call_ref(action, request), request["tool_call_id"])
        return AcceptedRequest(dict(event), data, dict(source), completion)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError(error) from exc


async def accept(manager, actor, request, target, *, action, authorize):
    """Commit a source-only permission plus its pending index, never a target."""
    ref, caller = _call_ref(action, request), JobLease(**request["caller"])
    key = _origin(manager, actor, request, action)
    event_id = identity(action, key)
    async with manager.tree_write(caller.job_id) as tx:
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
        if old is None:
            source = await authorize(tx)
            from .task_runtime import _SEND
            binding = _SEND.get()
            if binding is not None and binding.runtime.manager is manager and binding.lease == caller:
                from .control_runtime import flush_binding
                await flush_binding(binding, transaction=tx)
            version, now = source["version"]+1, manager.now()
            params = (now, caller.job_id, source["root_job_id"], caller.generation, caller.executor_id,
                caller.boot_id, source["version"])
            changed = await tx.write_job(caller.job_id, f"UPDATE {manager.jobs} SET version=version+1,updated_at=? "
                "WHERE job_id=? AND root_job_id=? AND generation=? AND executor_id=? AND boot_id=? AND version=? "
                "AND intake='open' AND status IN ('running','waiting','settling')", params)
            if changed != 1:
                raise JobError(action+"_caller_changed")
            data = {"version": 1, "phase": "accepted", "action": action, "actor": asdict(actor),
                "source": {**asdict(caller), "root_job_id": source["root_job_id"], "execution_role": source["execution_role"]},
                "target": target, "request": request}
            event = {"event_id": event_id, "job_id": caller.job_id, "event_type": "task_control_request", "job_version": version,
                "caller_session_id": actor.session_id, "tool_call_id": request["tool_call_id"], "call_ref_json": _json(ref.to_dict()),
                "request_hash": _hash(data), "payload_json": _json(data), "delivery_status": "not_applicable", "created_at": now}
            await tx.append_event(tuple(event), tuple(event.values()))
            remember_event(manager, event)
            tx.open_obligation(session_id=actor.session_id, kind="control_request", source_id=event_id,
                job_id=caller.job_id, opened_version=version)
    if old is not None:
        async with manager._read() as reader:
            done = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (completion_id(event_id),))
            return await _domain(action)._stored_request(manager, reader, actor, old, request, done), True
    return AcceptedRequest(event, data, {**source, "version": version}), False


def completed(accepted, event, outcome):
    expected = {"version": 1, "phase": "completed", "request": reference(accepted.event), "outcome": reference(outcome)}
    error = accepted.data["action"]+"_request_completion_corrupt"
    try:
        if (event["event_id"] != completion_id(accepted.event["event_id"])
                or event["event_type"] != "task_control_request" or event["job_id"] != accepted.event["job_id"]
                or event["caller_session_id"] != accepted.data["actor"]["session_id"]
                or event["delivery_status"] != "not_applicable" or event["job_version"] <= accepted.event["job_version"]
                or event["request_hash"] != _hash(expected) or json.loads(event["payload_json"]) != expected):
            raise JobError(error)
    except (ValueError, TypeError, KeyError) as exc:
        raise JobError(error) from exc


class _ProjectionConflict(Exception):
    pass


async def complete(manager, actor, accepted, outcome):
    """Source completion and pending removal commit together, even after exit."""
    if accepted.completion is not None:
        completed(accepted, accepted.completion, outcome)
        remember_event(manager, accepted.completion)
        return
    source = accepted.source
    event_id = completion_id(accepted.event["event_id"])
    value = {"version": 1, "phase": "completed", "request": reference(accepted.event), "outcome": reference(outcome)}
    while True:
        try:
            async with manager._write() as tx:
                changed = await tx.write_job(source["job_id"], f"UPDATE {manager.jobs} SET version=version+1,updated_at=? "
                    "WHERE job_id=? AND root_job_id=? AND generation=? AND version=?",
                    (manager.now(), source["job_id"], accepted.data["source"]["root_job_id"], source["generation"], source["version"]))
                if changed != 1:
                    raise _ProjectionConflict()
                event = {"event_id": event_id, "job_id": source["job_id"], "event_type": "task_control_request",
                    "job_version": source["version"]+1, "caller_session_id": actor.session_id,
                    "request_hash": _hash(value), "payload_json": _json(value), "delivery_status": "not_applicable", "created_at": manager.now()}
                await tx.append_event(tuple(event), tuple(event.values()))
                remember_event(manager, event)
                tx.close_obligation(session_id=actor.session_id, kind="control_request",
                    source_id=accepted.event["event_id"], job_id=source["job_id"])
            return
        except _ProjectionConflict:
            async with manager._read() as reader:
                done = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
                if done is not None:
                    completed(accepted, done, outcome)
                    remember_event(manager, done)
                    return
                source = await manager._authorize(reader, actor, source["job_id"])


_OBSERVED = ("job_id", "root_job_id", "generation", "executor_id", "boot_id", "status", "intake",
    "resource_state", "cancel_requested_at")


async def reject(manager, tx, actor, event, data, row, code):
    """A target refusal is an original outcome, never new execution or input.

    Conditional insertion leaves a superseding owner's row unchanged. Recovery
    can finish the source acknowledgement without retrying an external action.
    """
    value = {"version": 1, "phase": "rejected", "action": data["action"], "request": reference(event),
        "target": data["target"], "observed": {key: row[key] for key in _OBSERVED}, "error": code}
    refused = {"event_id": rejection_id(event["event_id"]), "job_id": row["job_id"], "event_type": "task_control_request",
        "job_version": row["version"], "caller_session_id": actor.session_id, "tool_call_id": data["request"]["tool_call_id"],
        "call_ref_json": _json(data["request"]["call_ref"]), "request_hash": _hash(value), "payload_json": _json(value),
        "delivery_status": "not_applicable", "created_at": manager.now()}
    if await tx.append_event_if_current(refused, row) != 1:
        raise JobError("control_request_target_changed")
    return refused


def rejection(accepted, event):
    try:
        value = json.loads(event["payload_json"])
        if (event["event_id"] != rejection_id(accepted.event["event_id"])
                or event["event_type"] != "task_control_request" or event["job_id"] != accepted.data["target"]["job_id"]
                or event["caller_session_id"] != accepted.data["actor"]["session_id"]
                or event["delivery_status"] != "not_applicable" or event["request_hash"] != _hash(value)
                or set(value) != {"version", "phase", "action", "request", "target", "observed", "error"}
                or type(value["version"]) is not int or value["version"] != 1 or value["phase"] != "rejected"
                or value["action"] != accepted.data["action"] or value["request"] != reference(accepted.event)
                or value["target"] != accepted.data["target"] or set(value["observed"]) != set(_OBSERVED)
                or value["observed"]["job_id"] != event["job_id"]
                or value["observed"]["root_job_id"] != accepted.data["target"]["root_job_id"]
                or value["error"] not in _domain(value["action"]).TARGET_BARRIERS):
            raise JobError("control_request_rejection_corrupt")
        return value["error"]
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("control_request_rejection_corrupt") from exc


def read_outcome(manager, accepted, event):
    if event["event_type"] == "task_control_request":
        rejection(accepted, event)
        return None
    request = accepted.data["request"]
    key = _origin(manager, JobActor(**accepted.data["actor"]), request, accepted.data["action"])
    if event["operation_key"] != key:
        raise JobError("control_request_outcome_mismatch")
    if accepted.data["action"] == "stop" and json.loads(event["payload_json"])["target"] != {
            "job_id": accepted.data["target"]["job_id"], "generation": accepted.data["target"]["generation"]}:
        raise JobError("stop_receipt_mismatch")
    return _domain(accepted.data["action"])._receipt(manager, event, accepted.data["request"], reference(accepted.event))


async def discover(manager, reader, actor, request, *, action):
    key = _origin(manager, actor, request, action)
    request_id = identity(action, key)
    rows = await reader.fetchall(f"SELECT * FROM {manager.events} WHERE event_id IN (?,?,?) OR operation_key=?",
        (request_id, completion_id(request_id), rejection_id(request_id), key))
    original = next((row for row in rows if row["event_id"] == request_id), None)
    done = next((row for row in rows if row["event_id"] == completion_id(request_id)), None)
    outcomes = [row for row in rows if row["operation_key"] == key or row["event_id"] == rejection_id(request_id)]
    error = action+"_request_"
    if original is None:
        if outcomes or done is not None:
            raise JobError(error+"original_missing")
        return None, None
    accepted = await _domain(action)._stored_request(manager, reader, actor, original, request, done)
    if len(outcomes) > 1:
        raise JobError(error+"outcome_conflict")
    outcome = outcomes[0] if outcomes else None
    if outcome is not None:
        await manager._authorize(reader, actor, request["job_id"])
        read_outcome(manager, accepted, outcome)
    elif done is not None:
        raise JobError(error+"outcome_missing")
    return accepted, outcome


def note_fresh_source(manager, job_id):
    from .execution_membership import _remember_after_commit
    _remember_after_commit(manager, None, "_control_request_events", job_id, {})


def remember_event(manager, event):
    from .execution_membership import _cache, remember_merge
    name, job_id = "_control_request_events", event["job_id"]
    cache, overlay = _cache(manager, name), manager._control_overlay.get()
    if job_id in cache or overlay is not None and (name, job_id) in overlay:
        remember_merge(manager, name, job_id, {event["event_id"]: event})
        manager.after_outcome(lambda: None, lambda: cache.pop(job_id, None))


async def request_events(manager, reader, job_id):
    from .execution_membership import _trusted
    known = _trusted(manager, "_control_request_events", job_id)
    if known is not None:
        return list(known.values())
    return await reader.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='task_control_request'", (job_id,))


def _records(events):
    accepted, done = {}, {}
    for event in events:
        if event["event_type"] != "task_control_request":
            continue
        value = json.loads(event["payload_json"])
        if value["phase"] == "accepted":
            accepted[event["event_id"]] = (event, value)
        elif value["phase"] == "completed":
            done[value["request"]["event_id"]] = event
        elif value["phase"] != "rejected":
            raise JobError("control_request_corrupt")
    return accepted, done


async def request_obligations(manager, reader, row, events, *, verify=False):
    accepted, done = _records(events)
    if verify and done.keys()-accepted.keys():
        raise JobError("control_request_original_missing")
    entries = set()
    for request_id, (event, data) in accepted.items():
        if verify:
            actor = JobActor(**data["actor"])
            original = await _domain(data["action"])._stored_request(manager, reader, actor, event, data["request"])
            target = await manager._authorize(reader, actor, data["target"]["job_id"])
            if target["root_job_id"] != data["target"]["root_job_id"]:
                raise JobError(data["action"]+"_request_target_changed")
            if request_id in done:
                value = json.loads(done[request_id]["payload_json"])
                outcome = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (value["outcome"]["event_id"],))
                if outcome is None:
                    raise JobError(data["action"]+"_request_outcome_missing")
                read_outcome(manager, original, outcome)
                completed(original, done[request_id], outcome)
        if request_id not in done:
            entries.add((data["actor"]["session_id"], "control_request", request_id, row["job_id"], event["job_version"]))
    return entries


def completed_request_events(events):
    accepted, done = _records(events)
    if accepted.keys()-done.keys():
        raise JobError("task_control_request_unresolved")
    return [event for request_id, (event, _) in sorted(accepted.items()) for event in (event, done[request_id])]
