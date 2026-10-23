"""Original finite-owner deadlines, independent of process monotonic clocks.

Only a trusted adapter's admission marker selects this protocol. No read
refreshes a deadline, claims a lease, dispatches work or releases money.
"""
from __future__ import annotations

from .jobs import JobError, _hash, _json

PROTOCOL = "completion_window_protocol"
DEFAULT_TIMEOUT_MS = 1800000


def event_id(manager, job_id):
    return "completion_window_" + _hash([manager.store.store_id, job_id])[:48]


async def _creation(manager, tx, row):
    import json

    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='create'", (row["job_id"],))
    if len(events) != 1:
        raise JobError("completion_window_creation_missing")
    [creation] = events
    try:
        data = json.loads(creation["payload_json"])
        request = data.get("request", {})
        if not isinstance(request, dict):
            raise ValueError()
        protocol = request.get(PROTOCOL)
        if PROTOCOL in request and (type(protocol) is not int or protocol != 1):
            raise ValueError()
    except (AttributeError, TypeError, ValueError) as exc:
        raise JobError("completion_window_evidence_corrupt") from exc
    return creation, protocol


async def read_row(manager, tx, row):
    from .execution_membership import _lease, scope
    from .timer_waits import _object

    creation, protocol = await _creation(manager, tx, row)
    event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id(manager, row["job_id"]),))
    if protocol is None:
        if event is not None:
            raise JobError("completion_window_evidence_corrupt")
        return None  # Legacy is explicit, never an invented original deadline.
    if event is None:
        raise JobError("completion_window_evidence_missing")
    data = _object(event["payload_json"])
    keys = {"version", "origin", "creation_id", "creation_hash", "timeout_ms", "admitted_at_ms", "deadline_at_ms"}
    if (set(data) != keys or type(data["version"]) is not int or data["version"] != 1
            or type(data["timeout_ms"]) is not int or data["timeout_ms"] != DEFAULT_TIMEOUT_MS
            or type(data["admitted_at_ms"]) is not int or not 0 < data["admitted_at_ms"] < 2**63 - DEFAULT_TIMEOUT_MS
            or type(data["deadline_at_ms"]) is not int or data["deadline_at_ms"] != data["admitted_at_ms"] + data["timeout_ms"]
            or data["creation_id"] != creation["event_id"] or data["creation_hash"] != creation["request_hash"]
            or event["event_type"] != "completion_window" or event["job_id"] != row["job_id"]
            or event["delivery_status"] != "not_applicable" or event["request_hash"] != _hash(data)
            or event["created_at"] != data["admitted_at_ms"] or row["kind"] not in ("subagent", "workflow_node")):
        raise JobError("completion_window_evidence_corrupt")
    origin = _lease(data["origin"])
    if origin["job_id"] != row["job_id"] or origin["generation"] != 1 or row["generation"] < 1:
        raise JobError("completion_window_evidence_corrupt")
    original = await scope(manager, tx, row["job_id"], origin["generation"])
    if (_hash(original["lease"]) != _hash(origin)
            or any(original[k] != row[k] for k in ("tenant_id", "principal_id", "kind", "execution_role", "parent_job_id", "created_by_session_id"))):
        raise JobError("completion_window_evidence_corrupt")
    return data


async def admit(manager, tx, row):
    """Part of the first original owner claim transaction; no standalone grant."""
    from .workflow_controller import proof

    _, protocol = await _creation(manager, tx, row)
    if protocol is None:
        return
    if row["generation"] != 1 or row["kind"] not in ("subagent", "workflow_node"):
        raise JobError("completion_window_admission_required")
    now = manager.now()
    if type(now) is not int or not 0 < now < 2**63 - DEFAULT_TIMEOUT_MS:
        raise JobError("invalid_completion_window_time")
    creation, _ = await _creation(manager, tx, row)
    data = {"version": 1, "origin": proof(row), "creation_id": creation["event_id"], "creation_hash": creation["request_hash"],
        "timeout_ms": DEFAULT_TIMEOUT_MS, "admitted_at_ms": now, "deadline_at_ms": now + DEFAULT_TIMEOUT_MS}
    await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
        (event_id(manager, row["job_id"]), row["job_id"], "completion_window", row["version"], _hash(data), _json(data), "not_applicable", now))
    await read_row(manager, tx, row)


async def read(manager, actor, job_id):
    async with manager._write() as tx:
        return await read_row(manager, tx, await manager._authorize(tx, actor, job_id))


async def require_time(manager, tx, row):
    if row["kind"] in ("subagent", "workflow_node"):
        saved = await read_row(manager, tx, row)
        if saved is not None and manager.now() >= saved["deadline_at_ms"]:
            raise JobError("owner_completion_deadline_exceeded")


async def bind_window(loop, window, sid):
    """Bind a process-local collector to its original persisted UTC limit."""
    from .task_runtime import _OWNER

    runtime = getattr(loop, "task_runtime", None)
    owner = _OWNER.get()
    if runtime is None or owner is None or owner.runtime is not runtime:
        window.bind(sid)
        return
    async with runtime.manager._write() as tx:
        row = await runtime.manager._authorize(tx, owner.actor, owner.lease.job_id)
        runtime.manager._check_proof(row, owner.lease)
        saved = await read_row(runtime.manager, tx, row)
    if saved is None:
        window.bind(sid)
    else:
        window.bind(sid, deadline_at_ms=saved["deadline_at_ms"], now_ms=runtime.manager.now())
