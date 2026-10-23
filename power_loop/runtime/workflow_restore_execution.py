"""Native restored drivers/nodes. Historical parents never become permission."""
from __future__ import annotations

import json
from dataclasses import asdict

from .jobs import JobError, JobLease, _hash, _json
from .result_revisions import result_id
from .workflow_resume import dispatch_id, execution, reservation_id


def identity(manager, kind, job_id, generation):
    return "wfr_"+kind+"_"+_hash([manager.store.store_id, job_id, generation])[:48]


async def put(manager, tx, key, job_id, kind, version, data, *, versioned=True):
    if await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (key, job_id, kind, version, _hash(data), _json(data), "not_applicable", manager.now())) != 1:
        raise JobError("workflow_restore_write_mismatch")
    actual = await get(manager, tx, key, job_id, kind, versioned=versioned)
    if _hash(actual) != _hash(data):
        raise JobError("workflow_restore_write_mismatch")


async def get(manager, tx, key, job_id, kind, *, versioned=True):
    row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
    try:
        data = json.loads(row["payload_json"]) if row else {}
        if (not row or row["job_id"] != job_id or row["event_type"] != kind or row["delivery_status"] != "not_applicable" or
                not isinstance(data, dict) or (versioned and (type(data.get("version")) is not int or data["version"] != 1)) or row["request_hash"] != _hash(data)):
            raise JobError("workflow_restore_evidence_mismatch")
        if kind == "workflow_node_authority":
            from .execution_membership import _lease
            if set(data) != {"version", "operation_id", "workflow", "node", "node_id", "driver_sid", "session_id", "source_hash", "token"}:
                raise JobError("workflow_node_authority_mismatch")
            _lease(data["workflow"])
            _lease(data["node"])
            if (any(not isinstance(data[k], str) or not 0 < len(data[k]) <= 64 for k in ("operation_id", "driver_sid", "session_id", "token")) or
                    not isinstance(data["node_id"], str) or not data["node_id"] or not isinstance(data["source_hash"], str) or len(data["source_hash"]) != 64):
                raise JobError("workflow_node_authority_mismatch")
        return data
    except (ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("workflow_restore_evidence_mismatch") from exc


async def dispatch(manager, tx, row, generation=None):
    op, request = await execution(manager, tx, row, generation)
    record = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (dispatch_id(op["event_id"]),))
    try:
        data = json.loads(record["payload_json"]) if record else {}
        expected = {"version": 1, "operation_id": op["event_id"], "target": request["target"], "driver_sid": request["driver_sid"], "token": data.get("token")}
        if (not record or record["event_type"] != "workflow_dispatch" or record["job_id"] != row["job_id"] or
                record["delivery_status"] != "claimed" or record["request_hash"] != _hash(data) or _hash(expected) != _hash(data) or
                not isinstance(data.get("token"), str) or not 0 < len(data["token"]) <= 64):
            raise JobError("workflow_restore_dispatch_required")
        return op, request, data
    except (ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, JobError):
            raise
        raise JobError("workflow_restore_dispatch_required") from exc


async def driver_proof(manager, tx, row, generation=None):
    generation = row["generation"] if generation is None else generation
    op, request, claim = await dispatch(manager, tx, row, generation)
    data = await get(manager, tx, identity(manager, "driver", row["job_id"], generation), row["job_id"], "workflow_restore_driver")
    expected = {"version": 1, "operation_id": op["event_id"], "workflow": request["target"],
        "session_id": request["driver_sid"], "dispatch_hash": _hash(claim)}
    if _hash(data) != _hash(expected):
        raise JobError("workflow_restore_driver_mismatch")
    return op, request, claim


async def create_driver(runtime, binding, operation_id):
    manager = runtime.manager
    async with manager._write() as tx:
        row = await manager._fence(tx, binding.actor, binding.lease)
        op, request, claim = await dispatch(manager, tx, row)
        if op["event_id"] != operation_id:
            raise JobError("workflow_restore_driver_mismatch")
        sid = request["driver_sid"]
        meta = {"kind": "wf_driver", "execution_role": "workflow", "task_owner_job_id": row["job_id"],
            "workflow_run_id": request["plan"]["run_id"], "task_workflow_generation": row["generation"],
            "task_restore_operation_id": operation_id, "task_dispatch_token": claim["token"]}
        key = identity(manager, "driver", row["job_id"], row["generation"])
        old = await tx.fetchone(f"SELECT session_id,metadata_json FROM {manager.store.t.sessions} WHERE session_id=?", (sid,))
        if old:
            await driver_proof(manager, tx, row)
            if _hash(json.loads(old["metadata_json"])) != _hash(meta):
                raise JobError("workflow_restore_driver_mismatch")
            return sid
        if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (key,)):
            raise JobError("workflow_restore_driver_mismatch")
        now = manager.now()
        if await tx.execute(f"INSERT INTO {manager.store.t.sessions} (session_id,created_at,updated_at,config_json,status,kind,spawn_depth,lifecycle,metadata_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (sid, now, now, "{}", "active", "root", 0, "ephemeral", _json(meta))) != 1:
            raise JobError("workflow_restore_driver_mismatch")
        await tx.execute(f"INSERT INTO {manager.store.t.session_state} (session_id) VALUES (?)", (sid,))
        await put(manager, tx, key, row["job_id"], "workflow_restore_driver", row["version"],
            {"version": 1, "operation_id": operation_id, "workflow": request["target"], "session_id": sid, "dispatch_hash": _hash(claim)})
        return sid


async def has_authority(manager, tx, row):
    return await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?",
        (identity(manager, "node", row["job_id"], row["generation"]),)) is not None


async def controls_node(manager, tx, row, actor, sid=None):
    if row["kind"] != "workflow_node" or not await has_authority(manager, tx, row):
        return False
    data = await get(manager, tx, identity(manager, "node", row["job_id"], row["generation"]), row["job_id"], "workflow_node_authority")
    await authority(manager, tx, row, data["session_id"])
    return data["driver_sid"] == actor.session_id and (sid is None or sid == data["session_id"])


async def authority(manager, tx, row, sid, *, workflow=None, historical=False):
    from .node_inputs import descriptor
    from .workflow_controller import current, proof
    from .workflow_controller import read as controller_read
    data = await get(manager, tx, identity(manager, "node", row["job_id"], row["generation"]), row["job_id"], "workflow_node_authority")
    wf = await manager._row(tx, row["parent_job_id"])
    generation = data["workflow"]["generation"]
    op, request, _ = await driver_proof(manager, tx, wf, generation)
    control = workflow or (await controller_read(manager, tx, wf, generation) if historical else await current(manager, tx, wf))
    source = request["plan"]["nodes"].get(data["node_id"])
    if (source is None or data["node_id"] not in request["affected"] or data["session_id"] != sid or
            data["operation_id"] != op["event_id"] or data["driver_sid"] != request["driver_sid"] or
            control["session_id"] != data["driver_sid"] or _hash(data["workflow"]) != _hash(request["target"]) or
            _hash(data["node"]) != _hash(proof(row)) or data["source_hash"] != _hash(source) or
            data["node"]["generation"] != source["lease"]["generation"]+1 or source["lease"]["job_id"] != row["job_id"] or
            source["binding"]["session_id"] != sid or not isinstance(data.get("token"), str) or not data["token"]):
        raise JobError("workflow_node_authority_mismatch")
    if not historical:
        owner = {"job_id": row["control_owner_job_id"], "generation": row["control_owner_generation"],
            "executor_id": row["control_owner_executor_id"], "boot_id": row["control_owner_boot_id"]}
        if _hash(owner) != _hash(request["target"]) or _hash(proof(wf)) != _hash(request["target"]):
            raise JobError("workflow_node_authority_mismatch")
    old = await descriptor(manager, tx, {**row, "generation": source["lease"]["generation"]}, sid, historical=True)
    saved = await tx.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key='workflow_execution_contract'", (sid,))
    if _hash(old) != source["binding_hash"] or saved is None or _hash(json.loads(saved["value_json"])) != source["contract_hash"]:
        raise JobError("resume_contract_mismatch")
    binding = {**old, "generation": row["generation"]}
    actual = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND event_type='node_session' AND job_id=?",
        (manager._node_session_event_id(row["job_id"], sid, row["generation"]), row["job_id"]))
    if actual is None or _hash(json.loads(actual["payload_json"])) != _hash(binding):
        raise JobError("workflow_node_authority_mismatch")
    return data, request


async def claim_node(runtime, workflow_binding, node_id, driver_sid, token):
    from .execution_membership import require_current, start
    from .node_inputs import descriptor
    from .workflow_controller import current, proof
    manager = runtime.manager
    actor = runtime.actor(driver_sid)
    async with manager._write() as tx:
        wf = await manager._fence(tx, actor, workflow_binding.lease)
        control = await current(manager, tx, wf)
        op, request = await execution(manager, tx, wf)
        if driver_sid != request["driver_sid"] or node_id not in request["affected"]:
            raise JobError("workflow_node_authority_mismatch")
        source = request["plan"]["nodes"][node_id]
        reservation = await get(manager, tx, reservation_id(manager, wf["job_id"], wf["generation"], node_id), wf["job_id"], "workflow_node_reservation")
        expected = {"version": 1, "workflow": request["target"], "operation_id": op["event_id"], "node_id": node_id,
            "previous": source["lease"], "generation": source["lease"]["generation"]+1, "source_hash": _hash(source)}
        if _hash(reservation) != _hash(expected):
            raise JobError("workflow_node_authority_mismatch")
        row = await manager._row(tx, source["lease"]["job_id"])
        sid = source["binding"]["session_id"]
        if row["generation"] == reservation["generation"]:
            data, _ = await authority(manager, tx, row, sid, workflow=control)
            if data["token"] != token:
                raise JobError("workflow_node_already_dispatched")
            return JobLease(**data["node"]), sid
        if (_hash(proof(row)) != _hash(source["lease"]) or row["resource_state"] != "stopped" or row["intake"] != "closed"):
            raise JobError("workflow_node_authority_mismatch")
        await require_current(manager, tx, row)
        old = await descriptor(manager, tx, row, sid, historical=True)
        if _hash(old) != source["binding_hash"]:
            raise JobError("workflow_node_authority_mismatch")
        lease = JobLease(row["job_id"], wf["executor_id"], wf["boot_id"], reservation["generation"])
        from .executor_registry import ExecutorIdentity, require_accepting
        executor = await require_accepting(manager, tx, ExecutorIdentity(actor.tenant_id, actor.principal_id, lease.executor_id, lease.boot_id),
            required=row["executor_key"] is not None or wf["executor_key"] is not None)
        await tx.execute(f"UPDATE {manager.jobs} SET generation=?,executor_id=?,boot_id=?,executor_key=?,lease_until_ms=?,status='running',supervision_status='open',intake='open',resource_state='running',"
            "outcome=NULL,result_json=NULL,result_status='none',control_owner_generation=?,control_owner_executor_id=?,control_owner_boot_id=?,version=version+1,updated_at=? WHERE job_id=?",
            (lease.generation, lease.executor_id, lease.boot_id, executor["executor_key"] if executor else None,
                manager.now()+runtime.lease_ms, wf["generation"], wf["executor_id"], wf["boot_id"], manager.now(), row["job_id"]))
        await put(manager, tx, manager._node_session_event_id(row["job_id"], sid, lease.generation), row["job_id"], "node_session", row["version"]+1,
            {**old, "generation": lease.generation}, versioned=False)
        data = {"version": 1, "operation_id": op["event_id"], "workflow": request["target"], "node": asdict(lease),
            "node_id": node_id, "driver_sid": driver_sid, "session_id": sid, "source_hash": _hash(source), "token": token}
        await put(manager, tx, identity(manager, "node", row["job_id"], lease.generation), row["job_id"], "workflow_node_authority", row["version"]+1, data)
        await start(manager, tx, await manager._row(tx, row["job_id"]))
        from .finalization import reopen
        await reopen(manager, tx, row, asdict(lease), op["event_id"])
        await authority(manager, tx, await manager._row(tx, row["job_id"]), sid, workflow=control)
        return lease, sid


async def accept_input(runtime, binding, sid, message):
    from .node_inputs import descriptor
    manager = runtime.manager
    async with manager._write() as tx:
        row = await manager._fence(tx, binding.actor, binding.lease)
        data, request = await authority(manager, tx, row, sid)
        node = await descriptor(manager, tx, row, sid)
        payload = {"target": asdict(binding.lease), "session_id": sid, "message": message, "node": node,
            "restore_operation_id": data["operation_id"], "workflow": request["target"]}
        key = manager._origin_key(binding.actor, "nudge", None, None, "restore-input:"+data["operation_id"]+":"+data["node_id"])
        if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE operation_key=?", (key,)):
            raise JobError("workflow_node_input_already_dispatched")
        op = await manager._insert_operation(tx, job_id=row["job_id"], version=row["version"], actor=binding.actor, action="nudge",
            key=key, fingerprint=_hash(payload), payload={"job_id": row["job_id"], "mode": "restore_input", "request": payload})
        await tx.execute(f"UPDATE {manager.events} SET delivery_status='pending_delivery' WHERE event_id=?", (op.operation_id,))
        cols = ("session_id", "item_id", "kind", "mode", "status", "content", "created_at")
        if await tx.execute(manager.db.dialect.insert_ignore(manager.store.t.follow_up_queue, cols),
                (sid, "nudge:"+op.operation_id, "relay", "steer", "pending", message, manager.now())) != 1:
            raise JobError("nudge_inbox_conflict")


async def confirm_input(runtime, binding, sid, message):
    """A lost enqueue ACK is read back, never reinserted or re-dispatched."""
    manager = runtime.manager
    async with manager._write() as tx:
        row = await manager._fence(tx, binding.actor, binding.lease)
        data, _ = await authority(manager, tx, row, sid)
        key = manager._origin_key(binding.actor, "nudge", None, None, "restore-input:"+data["operation_id"]+":"+data["node_id"])
        op = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE operation_key=?", (key,))
        request = json.loads(op["payload_json"]).get("request", {}) if op else {}
        inbox = await tx.fetchone(f"SELECT * FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND item_id=?", (sid, "nudge:"+op["event_id"])) if op else None
        if (op is None or op["job_id"] != row["job_id"] or op["event_type"] != "operation" or op["action"] != "nudge" or
                op["request_hash"] != _hash(request) or request.get("message") != message or request.get("session_id") != sid or
                _hash(request.get("target")) != _hash(asdict(binding.lease)) or request.get("restore_operation_id") != data["operation_id"] or
                inbox is None or inbox["content"] != message or inbox["kind"] != "relay" or inbox["mode"] != "steer" or inbox["status"] != "pending"):
            raise JobError("workflow_node_input_unsettled")


async def settle(runtime, binding, status):
    from .processing import require_inputs_handled
    manager = runtime.manager
    async with manager._write() as tx:
        row = await manager._authorize(tx, binding.actor, binding.lease.job_id)
        manager._check_proof(row, binding.lease)
        op, request = await execution(manager, tx, row)
        nodes = {}
        for nid in request["affected"]:
            source = request["plan"]["nodes"][nid]
            node = await manager._row(tx, source["lease"]["job_id"])
            if node["generation"] == source["lease"]["generation"] and node["resource_state"] == "stopped" and status != "completed":
                nodes[nid] = {"void": True, "generation": node["generation"]+1}
                continue
            if node["generation"] != source["lease"]["generation"]+1 or node["resource_state"] != "stopped":
                raise JobError("workflow_restore_reservation_unsettled")
            if status == "completed":
                grant, _ = await authority(manager, tx, node, source["binding"]["session_id"], historical=True)
            else:
                # A known failed attempt may have failed precisely because its
                # mutable contract changed. Its immutable claim/exit remain
                # evidence of failure; they do not authorize another request.
                grant = await get(manager, tx, identity(manager, "node", node["job_id"], node["generation"]), node["job_id"], "workflow_node_authority")
            from .workflow_controller import proof
            if (_hash(grant["workflow"]) != _hash(request["target"]) or _hash(grant["node"]) != _hash(proof(node)) or
                    grant["operation_id"] != op["event_id"] or grant["source_hash"] != _hash(source)):
                raise JobError("workflow_restore_reservation_unsettled")
            result = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND event_type='result'", (result_id(node["job_id"], node["generation"]),))
            if result is None or (status == "completed" and node["outcome"] != "succeeded"):
                raise JobError("workflow_restore_reservation_unsettled")
            if status == "completed":
                await require_inputs_handled(manager, tx, node)
            nodes[nid] = {"event_id": result_id(node["job_id"], node["generation"]), "hash": _hash(json.loads(result["payload_json"]))}
        data = {"version": 1, "operation_id": op["event_id"], "workflow": request["target"], "status": status, "nodes": nodes}
        await put(manager, tx, identity(manager, "settled", row["job_id"], row["generation"]), row["job_id"], "workflow_restore_settled", row["version"], data)
        await tx.execute(f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=?", (op["event_id"],))


async def require_settled(manager, tx, row, event, *, success=False):
    payload = json.loads(event["payload_json"])
    generation = payload["generation"]
    op, request = await execution(manager, tx, row, generation, expected_operation=event["event_id"])
    data = await get(manager, tx, identity(manager, "settled", row["job_id"], generation), row["job_id"], "workflow_restore_settled")
    if (op["delivery_status"] != "handled" or data["operation_id"] != op["event_id"] or
            _hash(data["workflow"]) != _hash(request["target"]) or set(data["nodes"]) != set(request["affected"]) or
            (success and data["status"] != "completed")):
        raise JobError("workflow_restore_reservation_unsettled")
    for nid, saved in data["nodes"].items():
        if saved.get("void"):
            if data["status"] == "completed":
                raise JobError("workflow_restore_reservation_unsettled")
            continue
        source = request["plan"]["nodes"][nid]
        if saved["event_id"] != result_id(source["lease"]["job_id"], source["lease"]["generation"]+1):
            raise JobError("workflow_restore_reservation_unsettled")
        result = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (saved["event_id"],))
        if result is None or _hash(json.loads(result["payload_json"])) != saved["hash"]:
            raise JobError("workflow_restore_reservation_unsettled")


async def validate_engine(runtime, binding, engine, spec):
    manager = runtime.manager
    async with manager._write() as tx:
        row = await manager._fence(tx, binding.actor, binding.lease)
        op, request = await execution(manager, tx, row)
        replay = {nid: request["plan"]["nodes"][nid]["result"] for nid in request["plan"]["replay"]}
        sessions = {nid: request["plan"]["nodes"][nid]["binding"]["session_id"] for nid in request["affected"]}
        allowed = sorted(engine._allowed_tools) if engine._allowed_tools is not None else None
        if (op["event_id"] != engine._restore_operation_id or _hash(request) != _hash(engine._restore_request) or
                _hash(spec.to_dict()) != _hash(request["plan"]["spec"]) or _hash(allowed) != _hash(request["plan"]["allowed_tools"]) or
                _hash({k: v.to_dict() for k, v in engine._replay.items()}) != _hash(replay) or
                _hash(engine._resume_sessions) != _hash(sessions) or engine._replay_no_charge != set(replay)):
            raise JobError("workflow_restore_engine_mismatch")
        from .workflow_restore_resources import admission
        resources = getattr(engine, "_restore_resources", None)
        admission(resources, manager, runtime.actor(row["created_by_session_id"]), request["plan"]["profile"],
            prior_usage=request["prior_usage"], previous=request.get("resource_evidence"))
        if resources is not None:
            if (engine._executor is not resources.resources.executor or engine._file_io is not resources.resources.file_io or
                    resources.resources.budget is not None and engine._budget is not resources.resources.budget):
                raise JobError("workflow_restore_engine_mismatch")
            if resources.resources.budget is not None:
                return
        if spec.budget is not None:
            if engine._budget is None or engine._budget.max_tokens != spec.budget.max_tokens or engine._budget.spent != request["prior_usage"].get("total_tokens", 0):
                raise JobError("workflow_restore_engine_mismatch")
        elif engine._budget is not None:
            raise JobError("workflow_restore_engine_mismatch")
