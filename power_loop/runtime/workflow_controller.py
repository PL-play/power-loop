"""Generation-qualified workflow drivers; historical reads grant no execution."""
from __future__ import annotations

import json
from dataclasses import asdict

from .jobs import JobError, JobLease, _hash, _json


def event_id(manager, job_id, generation):
    if isinstance(generation, bool) or not isinstance(generation, int) or not 1 <= generation < 2**63:
        raise JobError("invalid_execution_generation")
    return "controller_" + _hash([manager.store.store_id, job_id, generation])[:48]


def _object(raw):
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("object required")
        _hash(data)  # Reject non-finite JSON before comparisons or persistence.
    except (TypeError, ValueError) as exc:
        raise JobError("workflow_controller_mismatch") from exc
    return data


def proof(row):
    return asdict(JobLease(row["job_id"], row["executor_id"], row["boot_id"], row["generation"]))


async def creation(manager, reader, row):
    events = await reader.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND action='create'", (row["job_id"],))
    if len(events) != 1:
        raise JobError("workflow_controller_creation_mismatch")
    op = events[0]
    payload = _object(op["payload_json"])
    if (op["event_type"] != "operation" or op["caller_session_id"] != row["created_by_session_id"] or
            payload.get("job_id") != row["job_id"] or not op["event_id"].startswith("op_") or
            not isinstance(op["request_hash"], str) or len(op["request_hash"]) != 64):
        raise JobError("workflow_controller_creation_mismatch")
    frozen = {k: op[k] for k in ("event_id", "request_hash", "caller_session_id", "tool_call_id", "call_ref_json")}
    return op["event_id"], _hash({**frozen, "payload": payload})


async def expected(manager, reader, row, generation, sid):
    from .execution_membership import scope
    anchor = await scope(manager, reader, row["job_id"], generation)
    if (row["kind"] != "workflow" or row["execution_role"] != "workflow" or anchor["kind"] != "workflow" or anchor["execution_role"] != "workflow" or
            (anchor["tenant_id"], anchor["principal_id"], anchor["created_by_session_id"]) !=
            (row["tenant_id"], row["principal_id"], row["created_by_session_id"])):
        raise JobError("workflow_controller_mismatch")
    op_id, creation_hash = await creation(manager, reader, row)
    data = {"version": 1, "workflow": anchor["lease"], "session_id": sid,
        "run_id": op_id[3:], "creation_operation_id": op_id, "creation_hash": creation_hash,
        "created_by_session_id": anchor["created_by_session_id"]}
    if generation > 1:
        from .workflow_restore_execution import driver_proof
        op, request, claim = await driver_proof(manager, reader, row, generation)
        if sid != request["driver_sid"]:
            raise JobError("workflow_controller_mismatch")
        data.update(resume_operation_id=op["event_id"], dispatch_hash=_hash(claim))
    return data


async def read(manager, reader, row, generation):
    event = await reader.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id(manager, row["job_id"], generation),))
    if event is None:
        raise JobError("workflow_controller_missing")
    data = _object(event["payload_json"])
    sid = data.get("session_id")
    if (event["job_id"] != row["job_id"] or event["event_type"] != "workflow_controller" or
            event["delivery_status"] != "not_applicable" or not isinstance(sid, str) or not 0 < len(sid) <= 64 or
            event["request_hash"] != _hash(data) or
            _hash(data) != _hash(await expected(manager, reader, row, generation, sid))):
        raise JobError("workflow_controller_mismatch")
    return data


async def physical(manager, reader, row, sid, run_id):
    session = await reader.fetchone(f"SELECT parent_session_id,metadata_json FROM {manager.store.t.sessions} WHERE session_id=?", (sid,))
    meta = _object(session["metadata_json"] or "{}") if session else {}
    if (session is None or session["parent_session_id"] is not None or meta.get("kind") != "wf_driver" or
            meta.get("execution_role") != "workflow" or meta.get("task_owner_job_id") != row["job_id"]):
        raise JobError("session_ownership_mismatch")
    if meta.get("workflow_run_id") != run_id:
        raise JobError("workflow_run_mismatch")
    generation = meta.get("task_workflow_generation", 1)
    if type(generation) is not int or generation != row["generation"]:
        raise JobError("workflow_controller_mismatch")
    if generation > 1:
        from .workflow_restore_execution import driver_proof
        op, request, claim = await driver_proof(manager, reader, row)
        if (sid != request["driver_sid"] or meta.get("task_restore_operation_id") != op["event_id"] or
                meta.get("task_dispatch_token") != claim["token"]):
            raise JobError("workflow_controller_mismatch")


async def current(manager, reader, row):
    data = await read(manager, reader, row, row["generation"])
    if _hash(data["workflow"]) != _hash(proof(row)) or row["session_id"] != data["session_id"]:
        raise JobError("workflow_controller_mismatch")
    await physical(manager, reader, row, data["session_id"], data["run_id"])
    return data


async def has_binding(manager, reader, row):
    return row["session_id"] is not None or await reader.fetchone(
        f"SELECT event_id FROM {manager.events} WHERE event_id=?", (event_id(manager, row["job_id"], row["generation"]),)) is not None


async def for_node(manager, reader, row):
    workflow = await manager._row(reader, row["parent_job_id"])
    if (row["kind"] != "workflow_node" or row["execution_role"] != "workflow_node" or
            workflow is None or workflow["kind"] != "workflow" or row["control_owner_job_id"] != workflow["job_id"]):
        raise JobError("workflow_controller_mismatch")
    data = await current(manager, reader, workflow)
    owner = {"job_id": row["control_owner_job_id"], "generation": row["control_owner_generation"],
        "executor_id": row["control_owner_executor_id"], "boot_id": row["control_owner_boot_id"]}
    if (_hash(owner) != _hash(data["workflow"]) or
            (row["created_by_session_id"] != data["session_id"] and workflow["generation"] == 1) or
            (row["tenant_id"], row["principal_id"]) != (workflow["tenant_id"], workflow["principal_id"])):
        raise JobError("workflow_controller_mismatch")
    if workflow["generation"] > 1:
        from .workflow_restore_execution import authority
        from .workflow_resume import execution
        _, request = await execution(manager, reader, workflow)
        matches = [n for n in request["plan"]["nodes"].values() if n["lease"]["job_id"] == row["job_id"]]
        if len(matches) != 1:
            raise JobError("workflow_node_authority_mismatch")
        await authority(manager, reader, row, matches[0]["binding"]["session_id"], workflow=data)
    if row["generation"] > 0:
        from .execution_membership import scope
        anchor = await scope(manager, reader, row["job_id"], row["generation"])
        fields = ("kind", "execution_role", "tenant_id", "principal_id", "created_by_session_id", "parent_job_id")
        if (_hash(anchor["owner"]) != _hash(owner) or _hash(anchor["lease"]) != _hash(proof(row)) or
                any(anchor[k] != row[k] for k in fields)):
            raise JobError("workflow_controller_mismatch")
    return data


async def validate_dispatch(manager, reader, row):
    # _fence has already checked the complete live control-owner lineage. A
    # required subagent's next request also needs its workflow driver intact.
    physical_execution = row["session_id"] is not None and row["kind"] in ("send", "subagent")
    for _ in range(64):
        if row["kind"] == "workflow":
            if await has_binding(manager, reader, row):
                await current(manager, reader, row)
            return
        if row["kind"] == "workflow_node":
            workflow = await manager._row(reader, row["parent_job_id"])
            if (workflow is not None and (physical_execution or await has_binding(manager, reader, workflow) or
                    row["created_by_session_id"] != workflow["created_by_session_id"])):
                await for_node(manager, reader, row)
            return
        if row["control_owner_job_id"] is None:
            return
        row = await manager._row(reader, row["control_owner_job_id"])
    raise JobError("invalid_control_owner")


async def bind(manager, tx, row, sid):
    if row["generation"] != 1:
        from .workflow_restore_execution import dispatch
        try:
            _, request, _ = await dispatch(manager, tx, row)
        except JobError as exc:
            raise JobError("task_adapter_unavailable:workflow_resume") from exc
        if sid != request["driver_sid"]:
            raise JobError("workflow_controller_mismatch")
    if not isinstance(sid, str) or not 0 < len(sid) <= 64:
        raise JobError("session_ownership_mismatch")
    if row["session_id"] is not None:
        if row["session_id"] != sid:
            raise JobError("session_already_bound")
        await current(manager, tx, row)
        return
    data = await expected(manager, tx, row, row["generation"], sid)
    if _hash(data["workflow"]) != _hash(proof(row)):
        raise JobError("workflow_controller_mismatch")
    await physical(manager, tx, row, sid, data["run_id"])
    if await tx.fetchone(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow' AND session_id=? AND job_id<>?", (sid, row["job_id"])):
        raise JobError("workflow_controller_already_bound")
    key = event_id(manager, row["job_id"], row["generation"])
    if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (key,)):
        raise JobError("workflow_controller_mismatch")
    await tx.execute(f"UPDATE {manager.jobs} SET session_id=?,version=version+1,updated_at=? WHERE job_id=?", (sid, manager.now(), row["job_id"]))
    await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
        (key, row["job_id"], "workflow_controller", row["version"]+1, _hash(data), _json(data), "not_applicable", manager.now()))


async def get(manager, actor, job_id, generation):
    event_id(manager, job_id, generation)
    async with manager._write() as tx:
        row = await manager._authorize(tx, actor, job_id)
        return await read(manager, tx, row, generation)
