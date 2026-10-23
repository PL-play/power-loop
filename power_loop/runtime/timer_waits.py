"""Durable finite-wait admission and due-event evidence, before tool enablement.

These experimental SDK ports do not dispatch a model, deliver a notification,
decide a wait, or close its owner. A queued timer is a required child. The public
legacy timer tool remains unavailable to finite children until the owning
executor's wait/decision/resume adapters are installed together.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .jobs import JobActor, JobError, JobLease, _hash, _json
from .message_ref import MessageRef


def _number(value, name, *, maximum=2**63 - 1):
    if type(value) is not int or not 1 <= value <= maximum:
        raise JobError("invalid_wait_" + name)
    return value


def _text(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\x00" in value:
        raise JobError("invalid_wait_" + name)
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise JobError("invalid_wait_" + name) from exc
    return value  # Preserve the original text, including meaningful whitespace.


@dataclass(frozen=True)
class WaitRequest:
    due_at_ms: int
    note: str
    continuation: str

    def __post_init__(self):
        _number(self.due_at_ms, "due_at_ms")
        _text(self.note, "note", 4000)
        _text(self.continuation, "continuation", 64000)


@dataclass(frozen=True)
class WaitPolicy:
    """Trusted owner deadline; never populated from child-model tool arguments."""

    deadline_at_ms: int
    max_wait_ms: int = 1800000

    def __post_init__(self):
        _number(self.deadline_at_ms, "deadline_at_ms")
        _number(self.max_wait_ms, "max_wait_ms", maximum=30 * 86400000)


def _request(raw):
    try:
        if (not isinstance(raw, dict) or set(raw) != {"managed_wait", "request", "policy"}
                or type(raw["managed_wait"]) is not int or raw["managed_wait"] != 1):
            raise ValueError()
        return WaitRequest(**raw["request"]), WaitPolicy(**raw["policy"])
    except (TypeError, ValueError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("invalid_wait_request") from exc


def _proof(row):
    return asdict(JobLease(row["job_id"], row["executor_id"], row["boot_id"], row["generation"]))


async def prepare_binding(manager, tx, actor, source, request, *, caller, session_id, independent,
                          call_ref, tool_call_id):
    """Called inside the original generic registration transaction, before INSERT."""
    desired, policy = _request(request)
    from .execution_membership import _lease

    if not isinstance(caller, JobLease):
        raise JobError("wait_source_required")
    _lease(asdict(caller))
    now = manager.now()
    if not now < desired.due_at_ms <= policy.deadline_at_ms <= now + policy.max_wait_ms:
        raise JobError("wait_deadline_out_of_bounds")
    if (source is None or source["kind"] != "send" or source["session_id"] != actor.session_id
            or session_id != actor.session_id or independent or call_ref is None):
        raise JobError("wait_source_required")
    if _hash(asdict(caller)) != _hash(_proof(source)):
        raise JobError("wait_source_required")
    from .task_control import _source

    await _source(manager, tx, actor, JobLease(**_proof(source)), call_ref, tool_call_id)
    owner = await manager._row(tx, source["control_owner_job_id"])
    if (owner is None or owner["kind"] not in ("subagent", "workflow_node")
            or source["execution_role"] != owner["kind"]):
        raise JobError("wait_owner_required")
    physical = await tx.fetchone(f"SELECT parent_session_id,metadata_json FROM {manager.store.t.sessions} WHERE session_id=?",
                                (session_id,))
    meta = _object(physical["metadata_json"] or "{}") if physical else {}
    if not isinstance(meta, dict) or meta.get("execution_role") != owner["kind"]:
        raise JobError("wait_session_role_mismatch")
    if owner["kind"] == "subagent":
        if owner["session_id"] != session_id or meta.get("standalone_subagent") is not True:
            raise JobError("wait_session_role_mismatch")
        route = {"kind": "subagent", "session_id": session_id}
    else:
        from .node_inputs import descriptor, is_closed
        from .workflow_controller import for_node

        if await is_closed(manager, tx, owner, session_id):
            raise JobError("wait_owner_closed")
        route = {"kind": "workflow_node", "node": await descriptor(manager, tx, owner, session_id),
                 "workflow": (await for_node(manager, tx, owner))["workflow"]}
    root = await manager._row(tx, source["root_job_id"])
    main = await tx.fetchone(f"SELECT parent_session_id,metadata_json FROM {manager.store.t.sessions} WHERE session_id=?",
                            (root["session_id"],)) if root else None
    main_meta = _object(main["metadata_json"] or "{}") if main else {}
    if (root is None or root["kind"] != "send" or root["execution_role"] != "main" or main is None
            or main["parent_session_id"] is not None or not isinstance(main_meta, dict)
            or main_meta.get("execution_role") not in (None, "main") or main_meta.get("kind") == "wf_driver"
            or main_meta.get("workflow_run_id") or main_meta.get("standalone_subagent")):
        raise JobError("wait_main_session_unavailable")
    binding = {"version": 1, "store_id": manager.store.store_id,
        "source": _proof(source), "owner": _proof(owner), "session_id": session_id,
        "main_session_id": root["session_id"], "root_job_id": root["job_id"],
        "completion_scope_job_id": owner["job_id"], "route": route,
        "call_ref": call_ref.to_dict(), "tool_call_id": tool_call_id, "request_hash": _hash(request)}
    binding["evidence_hash"] = _hash(binding)
    return binding


async def register_wait(manager, actor, *, caller, call_ref, tool_call_id, request, policy):
    """Accept one persisted source call; replay never recomputes due time or owner."""
    if type(request) is not WaitRequest or type(policy) is not WaitPolicy:
        raise JobError("invalid_wait_request")
    registration = await manager.register_job(actor, kind="timer", title="Deferred task continuation",
        session_id=actor.session_id, parent=caller, call_ref=call_ref, tool_call_id=tool_call_id,
        request={"managed_wait": 1, "request": asdict(request), "policy": asdict(policy)})
    # Also verify accepted evidence on ACK replay, including after source closure.
    await read_wait(manager, actor, registration.job.job_id)
    return registration


def _object(raw):
    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError()
        _hash(result)
        return result
    except (TypeError, ValueError) as exc:
        raise JobError("wait_evidence_corrupt") from exc


async def _read(manager, tx, actor, job_id):
    job = await manager._authorize(tx, actor, job_id)
    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'",
                              (job_id,))
    if job["kind"] != "timer" or len(events) != 1:
        raise JobError("wait_evidence_missing")
    [creation] = events
    data = _object(creation["payload_json"])
    binding = data.get("wait_binding")
    if (set(data) != {"job_id", "status", "request", "wait_binding"} or data["job_id"] != job_id
            or data["status"] != "queued" or not isinstance(binding, dict)):
        raise JobError("wait_evidence_corrupt")
    _request(data["request"])
    from .execution_membership import _lease, scope
    from .task_query import _reference

    keys = {"version", "store_id", "source", "owner", "session_id", "main_session_id", "root_job_id",
            "completion_scope_job_id", "route", "call_ref", "tool_call_id", "request_hash", "evidence_hash"}
    if (set(binding) != keys or type(binding["version"]) is not int or binding["version"] != 1
            or binding["evidence_hash"] != _hash({k: v for k, v in binding.items() if k != "evidence_hash"})
            or binding["store_id"] != manager.store.store_id or binding["request_hash"] != _hash(data["request"])):
        raise JobError("wait_evidence_corrupt")
    _lease(binding["source"])
    owner = _lease(binding["owner"])
    call = _reference(manager, job, creation["call_ref_json"], "assistant")
    receipt = _reference(manager, job, creation["receipt_ref_json"], "tool")
    if (call is None or _hash(call) != _hash(binding["call_ref"]) or creation["tool_call_id"] != binding["tool_call_id"]
            or job["session_id"] != binding["session_id"] or creation["caller_session_id"] != binding["session_id"]
            or job["parent_job_id"] != binding["source"]["job_id"] or not job["blocks_completion"]
            or job["root_job_id"] != binding["root_job_id"] or job["completion_scope_job_id"] != owner["job_id"]
            or binding["completion_scope_job_id"] != owner["job_id"]
            or (job["control_owner_job_id"], job["control_owner_generation"], job["control_owner_executor_id"], job["control_owner_boot_id"])
                != (owner["job_id"], owner["generation"], owner["executor_id"], owner["boot_id"])
            or (job["parent_generation"], job["parent_executor_id"], job["parent_boot_id"])
                != (binding["source"]["generation"], binding["source"]["executor_id"], binding["source"]["boot_id"])):
        raise JobError("wait_evidence_corrupt")
    fingerprint = _hash(["timer", job["title"], job["session_id"], job["parent_job_id"], False,
                         data["request"], call])
    if creation["request_hash"] != fingerprint:
        raise JobError("wait_evidence_corrupt")
    source_scope = await scope(manager, tx, binding["source"]["job_id"], binding["source"]["generation"])
    owner_scope = await scope(manager, tx, owner["job_id"], owner["generation"])
    root = await manager._row(tx, binding["root_job_id"])
    route = binding["route"]
    if (source_scope["lease"] != binding["source"] or source_scope["owner"] != owner
            or source_scope["kind"] != "send" or source_scope["session_id"] != binding["session_id"]
            or owner_scope["lease"] != owner or owner_scope["kind"] not in ("workflow_node", "subagent")
            or not isinstance(route, dict) or route.get("kind") != owner_scope["kind"]
            or root is None or root["kind"] != "send" or root["execution_role"] != "main"
            or root["session_id"] != binding["main_session_id"]):
        raise JobError("wait_evidence_corrupt")
    if route["kind"] == "subagent":
        if (set(route) != {"kind", "session_id"} or route["session_id"] != binding["session_id"]
                or owner_scope["session_id"] != binding["session_id"]):
            raise JobError("wait_evidence_corrupt")
    else:
        if (set(route) != {"kind", "node", "workflow"} or _hash(_lease(route["workflow"])) != _hash(owner_scope["owner"])):
            raise JobError("wait_evidence_corrupt")
        descriptor = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND job_id=? AND event_type='node_session'",
            (manager._node_session_event_id(owner["job_id"], binding["session_id"], owner["generation"]), owner["job_id"]))
        if descriptor is None or _hash(_object(descriptor["payload_json"])) != _hash(route["node"]):
            raise JobError("wait_evidence_corrupt")
    if receipt is not None and (receipt["tool_call_id"] != binding["tool_call_id"]
                                or receipt["send_index"] != call["send_index"]):
        raise JobError("wait_receipt_mismatch")
    return job, creation, {"job_id": job_id, "creation_operation_id": creation["event_id"],
        "request": data["request"], "binding": binding, "call_ref": call, "receipt_ref": receipt,
        "execution_authority": False}


async def read_wait(manager, actor, job_id):
    """Read frozen evidence; history is never permission to resume an execution."""
    async with manager._write() as tx:
        return (await _read(manager, tx, actor, job_id))[2]


async def has_required_wait(manager, tx, actor, owner):
    """Managed waits cannot be bypassed by the legacy optional success check."""
    timers = await tx.fetchall(f"SELECT j.job_id,e.payload_json FROM {manager.jobs} j LEFT JOIN {manager.events} e ON e.job_id=j.job_id "
        "AND e.event_type='operation' AND e.action='create' "
        "WHERE j.control_owner_job_id=? AND j.control_owner_generation=? AND j.kind='timer'",
        (owner["job_id"], owner["generation"]))
    for timer in timers:
        data = _object(timer["payload_json"])
        if data.get("job_id") != timer["job_id"]:
            raise JobError("wait_evidence_corrupt")
        request = data.get("request", {})
        if "wait_binding" in data or isinstance(request, dict) and "managed_wait" in request:
            await _read(manager, tx, actor, timer["job_id"])
    # The registered dependency itself cannot disappear when its optional
    # projection/marker is erased. Legacy required timer jobs also block success.
    return bool(timers)


def _ref(data):
    return MessageRef(data["store_id"], data["session_id"], int(data["id"]), int(data["seq"]), data["role"],
                      data["tool_call_id"], int(data["send_index"]) if data["send_index"] is not None else None,
                      data["round_index"])


def due_event_id(manager, job_id, creation_id):
    return "wait_due_" + _hash([manager.store.store_id, job_id, creation_id])[:48]


async def _prepare_due(manager, tx, actor, job_id):
    """Persist one at-least-once main-decision event, without delivering it.

    The event is evidence only. A later notification claim and decision adapter
    must recheck the original owner, deadline, cancellation, pause and budget.
    """
    job, creation, saved = await _read(manager, tx, actor, job_id)
    desired, policy = _request(saved["request"])
    binding = saved["binding"]
    if manager.now() > policy.deadline_at_ms:
        raise JobError("wait_deadline_exceeded")
    if (job["generation"] != 0 or job["status"] != "queued" or job["resource_state"] != "not_started"
            or job["intake"] != "open" or job["supervision_status"] != "open" or job["outcome"] is not None):
        raise JobError("wait_state_changed")
    await manager._supervision_barriers(tx, job)
    await manager._control_owners(tx, job)
    owner = await manager._row(tx, binding["owner"]["job_id"])
    if owner is None:
        raise JobError("wait_owner_required")
    manager._check_proof(owner, JobLease(**binding["owner"]))
    main = await tx.fetchone(f"SELECT parent_session_id,metadata_json FROM {manager.store.t.sessions} WHERE session_id=?",
                            (binding["main_session_id"],))
    main_meta = _object(main["metadata_json"] or "{}") if main else {}
    if (main is None or main["parent_session_id"] is not None
            or main_meta.get("execution_role") not in (None, "main") or main_meta.get("kind") == "wf_driver"
            or main_meta.get("workflow_run_id") or main_meta.get("standalone_subagent")):
        raise JobError("wait_main_session_unavailable")
    if binding["route"]["kind"] == "workflow_node":
        from .node_inputs import descriptor, is_closed
        if await is_closed(manager, tx, owner, binding["session_id"]):
            raise JobError("wait_owner_closed")
        if _hash(await descriptor(manager, tx, owner, binding["session_id"])) != _hash(binding["route"]["node"]):
            raise JobError("wait_route_changed")
    else:
        physical = await tx.fetchone(f"SELECT metadata_json FROM {manager.store.t.sessions} WHERE session_id=?",
                                    (binding["session_id"],))
        meta = _object(physical["metadata_json"] or "{}") if physical else {}
        if meta.get("execution_role") != "subagent" or meta.get("standalone_subagent") is not True:
            raise JobError("wait_session_role_mismatch")
    if manager.now() < desired.due_at_ms:
        return {"status": "armed", "job_id": job_id, "due_at_ms": desired.due_at_ms, "execution_authority": False}
    if saved["receipt_ref"] is None:
        raise JobError("wait_receipt_missing")
    source_actor = JobActor(actor.tenant_id, actor.principal_id, binding["session_id"])
    await manager._validate_call(tx, source_actor, _ref(saved["call_ref"]), binding["tool_call_id"])
    await manager._validate_ref(tx, source_actor, _ref(saved["receipt_ref"]), role="tool")
    if (saved["receipt_ref"]["tool_call_id"] != binding["tool_call_id"]
            or saved["receipt_ref"]["send_index"] != saved["call_ref"]["send_index"]):
        raise JobError("wait_receipt_mismatch")
    event_id = due_event_id(manager, job_id, creation["event_id"])
    payload = {"version": 1, "wait": saved}
    old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
    if old is not None:
        if (old["job_id"] != job_id or old["event_type"] != "wait_due"
                or old["request_hash"] != _hash(payload) or _hash(_object(old["payload_json"])) != _hash(payload)
                or old["delivery_status"] not in ("pending_delivery", "delivered", "handled")):
            raise JobError("wait_due_evidence_corrupt")
    else:
        version = job["version"] + 1
        await tx.execute(f"UPDATE {manager.jobs} SET version=?,updated_at=? WHERE job_id=?", (version, manager.now(), job_id))
        await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                         (event_id, job_id, "wait_due", version, _hash(payload), _json(payload), "pending_delivery", manager.now()))
    return {"status": "decision_pending", "event_id": event_id, "payload": payload, "execution_authority": False}


async def prepare_due(manager, actor, job_id):
    """Persist due evidence using one original control transaction."""
    async with manager._write() as tx:
        return await _prepare_due(manager, tx, actor, job_id)
