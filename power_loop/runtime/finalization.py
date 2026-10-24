"""Mechanical supervision closure; closed execution is not billing settlement.

Every closure retains immutable evidence and child closure references. Resume
adapters explicitly reopen historical supervision without reviving old sends.
No model call, host callback, or execution dispatch occurs in this module.
"""
from __future__ import annotations

import json

from .jobs import JobError, _hash, _json
from .result_revisions import result_id


def _identity(manager, kind, values):
    return "sup_" + kind + "_" + _hash([manager.store.store_id, *values])[:48]


def _lease(row):
    from .workflow_controller import proof
    return proof(row)


def _object(raw):
    def invalid(value):
        raise ValueError("non-finite JSON")
    try:
        data = json.loads(raw, parse_constant=invalid)
    except (ValueError, TypeError) as exc:
        raise JobError("finalization_evidence_corrupt") from exc
    if not isinstance(data, dict):
        raise JobError("finalization_evidence_corrupt")
    return data


async def _reference(manager, tx, event_id):
    event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
    if event is None:
        raise JobError("finalization_evidence_missing")
    return {"event_id": event_id, "job_id": event["job_id"], "event_type": event["event_type"],
        "revision": event["job_version"], "request_hash": event["request_hash"],
        "delivery_status": event["delivery_status"], "payload_hash": _hash(_object(event["payload_json"]))}


async def _closed_event(manager, tx, event, state=None, depth=0):
    state = {"visiting": set(), "done": {}} if state is None else state
    event_id = event["event_id"]
    if event_id in state["visiting"] or depth >= 64 or len(state["done"]) + len(state["visiting"]) >= 10000:
        raise JobError("finalization_evidence_corrupt")
    if event_id in state["done"]:
        return state["done"][event_id]
    state["visiting"].add(event_id)
    data = _object(event["payload_json"])
    if (event["event_type"] != "supervision_closed" or event["delivery_status"] != "not_applicable" or
            set(data) != {"version", "scope", "proof", "proof_hash"} or type(data["version"]) is not int or data["version"] not in (1, 2) or
            not isinstance(data["scope"], dict) or not isinstance(data["proof"], dict) or
            data["scope"]["job_id"] != event["job_id"] or data["proof_hash"] != _hash(data["proof"]) or
            event["request_hash"] != _hash(data) or event["event_id"] != _identity(manager, "closed",
                [event["job_id"], data["scope"]["generation"], event["job_version"], data["proof_hash"]])):
        raise JobError("finalization_evidence_corrupt")
    from .execution_membership import _lease as validate_lease
    if data["version"] == 1:
        validate_lease(data["scope"])
    else:
        row = await manager._row(tx, event["job_id"])
        if (row is None or row["kind"] != "timer" or data["proof"].get("kind") != "timer" or data["proof"].get("outcome") != row["outcome"]
                or type(data["scope"].get("generation")) is not int
                or data["scope"]["generation"] != 0 or _hash(data["scope"]) != _hash(_lease(row))):
            raise JobError("finalization_evidence_corrupt")
        from .wait_terminal import references
        if _hash(data["proof"].get("references")) != _hash(await references(manager, tx, row)):
            raise JobError("finalization_evidence_changed")
    from .jobs import KINDS
    proof = data["proof"]
    if (not isinstance(proof, dict) or set(proof) != {"kind", "outcome", "references"} or
            proof["kind"] not in KINDS or
            (proof["outcome"] is not None and proof["outcome"] not in
                ("succeeded", "failed", "cancelled", "timed_out", "budget_exceeded", "interrupted"))):
        raise JobError("finalization_evidence_corrupt")
    refs = data["proof"]["references"]
    if not isinstance(refs, list) or not refs or len(refs) > 10000:
        raise JobError("finalization_evidence_corrupt")
    for ref in refs:
        if await _reference(manager, tx, ref["event_id"]) != ref:
            raise JobError("finalization_evidence_changed")
        if ref["event_type"] == "supervision_closed":
            child = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (ref["event_id"],))
            await _closed_event(manager, tx, child, state, depth + 1)
        elif ref["event_type"] == "input_disposition":
            from .input_dispositions import read_tx
            saved = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (ref["event_id"],))
            data_input = _object(saved["payload_json"])
            original = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (data_input["operation"]["event_id"],))
            if original is None or (await read_tx(manager, tx, original, required=True))["event_id"] != ref["event_id"]:
                raise JobError("finalization_evidence_changed")
    state["visiting"].remove(event_id)
    state["done"][event_id] = data
    return data


async def closed(manager, tx, row, generation=None):
    """Validate saved immutable closure, including historical generations."""
    generation = row["generation"] if generation is None else generation
    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='supervision_closed' ORDER BY job_version DESC,event_id", (row["job_id"],))
    for event in events:
        data = _object(event["payload_json"])
        if data.get("scope", {}).get("generation") != generation:
            continue
        data = await _closed_event(manager, tx, event)
        if generation == row["generation"] and _hash(data["scope"]) != _hash(_lease(row)):
            raise JobError("finalization_execution_changed")
        if generation == row["generation"] and row["supervision_status"] == "closed":
            expected_status = "model_closed" if row["kind"] == "send" else row["outcome"]
            if (row["status"] != expected_status or row["resource_state"] != "stopped" or row["intake"] != "closed" or
                    data["proof"]["kind"] != row["kind"] or data["proof"]["outcome"] != row["outcome"]):
                raise JobError("finalization_execution_changed")
            if row["kind"] != "send" and data["version"] == 1:
                result = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (result_id(row["job_id"], generation),))
                if result is None or _hash(_object(result["payload_json"])["result"]) != _hash(_object(row["result_json"])):
                    raise JobError("finalization_execution_changed")
        return event, data
    raise JobError("finalization_evidence_missing")


async def _inputs(manager, tx, row, generation):
    from .processing import require_event_handled
    refs = []
    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='nudge'", (row["job_id"],))
    for event in events:
        payload = json.loads(event["payload_json"])
        request = payload.get("request", {})
        target = request.get("target", {})
        if type(target.get("generation")) is not int:
            raise JobError("finalization_input_unknown")
        if target["generation"] != generation:
            continue
        if row["kind"] == "workflow" and payload.get("mode") == "workflow_continue":
            from .workflow_restore_execution import identity, require_settled
            await require_settled(manager, tx, {**row, "generation": generation}, event)
            refs.append(await _reference(manager, tx, identity(manager, "settled", row["job_id"], generation)))
        else:
            await require_event_handled(manager, tx, event, session_id=request.get("session_id"), owner_job_id=row["job_id"])
            refs.extend(await _processing_references(manager, tx, event))
        refs.append(await _reference(manager, tx, event["event_id"]))
    return refs


async def _processing_references(manager, tx, event):
    if event.get("action") == "nudge":
        from .input_dispositions import event_ids, read_tx
        disposed = await read_tx(manager, tx, event)
        if disposed is not None:
            return [await _reference(manager, tx, key) for key in event_ids(disposed)]
    decision_id = "decision_" + _hash([event["event_id"], event["job_version"], _hash(_object(event["payload_json"]))])[:48]
    decision = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='processing_decision'", (decision_id,))
    if decision is None:
        raise JobError("finalization_evidence_missing")
    data = _object(decision["payload_json"])
    return [await _reference(manager, tx, decision_id),
        await _reference(manager, tx, "model_closed_" + data["handler_job_id"])]


async def require_node_consumed(manager, tx, row):
    from .execution_membership import read, scope
    membership = await read(manager, tx, row, row["generation"])
    ack = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='node_result_consumed'", (_identity(manager, "node", [row["job_id"], row["generation"]]),))
    data = _object(ack["payload_json"]) if ack else {}
    canonical = await _reference(manager, tx, result_id(row["job_id"], row["generation"]))
    if (ack is None or ack["job_id"] != row["job_id"] or ack["delivery_status"] != "not_applicable" or
            set(data) != {"version", "workflow", "node", "result"} or type(data["version"]) is not int or data["version"] != 1 or
            ack["request_hash"] != _hash(data) or data["node"] != _lease(row) or data["result"] != canonical or
            data["workflow"] != membership["scope"]["owner"]):
        raise JobError("finalization_node_result_unconsumed")
    workflow = data["workflow"]
    parent = await scope(manager, tx, workflow["job_id"], workflow["generation"])
    if parent["kind"] != "workflow" or parent["lease"] != workflow or row["parent_job_id"] != workflow["job_id"]:
        raise JobError("finalization_node_result_unconsumed")
    return await _reference(manager, tx, ack["event_id"])


async def _prepare(manager, tx, actor, row, generation, state, depth=0):
    from .execution_membership import read, scope_id, seal_id
    key = (row["job_id"], generation)
    if key in state["done"]:
        return state["done"][key]
    if key in state["visiting"] or depth >= 64 or len(state["done"]) + len(state["visiting"]) >= 10000:
        raise JobError("invalid_finalization_tree")
    if generation < row["generation"] or row["supervision_status"] == "closed":
        event, _ = await closed(manager, tx, row, generation)
        ref = await _reference(manager, tx, event["event_id"])
        state["done"][key] = ref
        return ref
    if generation != row["generation"]:
        raise JobError("finalization_generation_changed")
    if generation == 0 and row["kind"] == "timer" and (row["resource_state"] != "stopped" or row["outcome"] != "succeeded"):
        from .wait_terminal import terminate_tx
        await terminate_tx(manager, tx, actor, row["job_id"])
        row = await manager._row(tx, row["job_id"])
    if row["kind"] == "send":
        exit = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", ("model_closed_" + row["job_id"],))
        if exit and _object(exit["payload_json"]).get("evidence", {}).get("model_attempt_snapshot_error"):
            from .model_exit import capture
            await capture(manager, tx, row)  # Keep the original precise corruption error.
            raise JobError("model_exit_snapshot_incomplete")
    if row["resource_state"] != "stopped" or row["intake"] != "closed":
        raise JobError("finalization_execution_unsettled")
    state["visiting"].add(key)
    if generation == 0:
        if row["kind"] != "timer":
            raise JobError("finalization_adapter_required")
        from .wait_terminal import references
        refs = await references(manager, tx, row)
        proof = {"kind": "timer", "outcome": row["outcome"], "references": refs}
        data = {"version": 2, "scope": _lease(row), "proof": proof, "proof_hash": _hash(proof)}
        version = row["version"] + 1
        event_id = _identity(manager, "closed", [row["job_id"], 0, version, data["proof_hash"]])
        ref = {"event_id": event_id, "job_id": row["job_id"], "event_type": "supervision_closed", "revision": version,
            "request_hash": _hash(data), "delivery_status": "not_applicable", "payload_hash": _hash(data)}
        state["plans"].append((row, version, data, ref))
        state["visiting"].remove(key)
        state["done"][key] = ref
        return ref
    membership = await read(manager, tx, row, generation)
    if not membership["sealed"] or _hash(membership["scope"]["lease"]) != _hash(_lease(row)):
        raise JobError("finalization_execution_unsettled")
    refs = [await _reference(manager, tx, scope_id(manager, row["job_id"], generation)),
        await _reference(manager, tx, seal_id(manager, row["job_id"], generation))]
    if row["kind"] in ("subagent", "workflow_node"):
        if row["outcome"] in ("failed", "cancelled", "timed_out", "budget_exceeded", "interrupted"):
            from .input_dispositions import reconcile_owner_tx
            await reconcile_owner_tx(manager, tx, actor, row, generation)
            row = await manager._row(tx, row["job_id"])
        from .completion_windows import event_id, read_row
        if await read_row(manager, tx, row) is not None:
            refs.append(await _reference(manager, tx, event_id(manager, row["job_id"])))
        from .owner_waits import references
        refs.extend([await _reference(manager, tx, ref["event_id"]) for ref in
            await references(manager, tx, actor, row, success=row["outcome"] == "succeeded")])
    if row["kind"] == "send":
        model = membership["model"]
        if (row["status"] != "model_closed" or model is None or model["payload"].get("resource_state") != "stopped"):
            raise JobError("finalization_model_exit_unconfirmed")
        from .model_exit import SNAPSHOT, require_stopped
        evidence = model["payload"].get("evidence", {})
        if evidence.get("exit") != "completed" and evidence.get("exit") not in ("failed", "cancelled", "interrupted", "hit_round_limit", "timed_out", "budget_exceeded", "degraded"):
            raise JobError("finalization_model_exit_unconfirmed")
        if evidence.get("exit") != "completed" and SNAPSHOT not in evidence:
            raise JobError("finalization_model_exit_unconfirmed")
        if SNAPSHOT in evidence:
            await require_stopped(manager, tx, row, model["payload"])
        refs.append(await _reference(manager, tx, model["event_id"]))
        from .model_attempts import require_quiescent
        refs.extend([await _reference(manager, tx, key) for key in await require_quiescent(manager, tx, row)])
    else:
        result = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result'", (result_id(row["job_id"], generation),))
        payload = json.loads(result["payload_json"]) if result else {}
        if (result is None or payload.get("resource_state") != "stopped" or payload.get("outcome") != row["outcome"] or
                _hash(payload.get("result")) != _hash(json.loads(row["result_json"] or "null"))):
            raise JobError("finalization_result_unsettled")
        refs.append(await _reference(manager, tx, result["event_id"]))
        if row["kind"] == "workflow_node":
            refs.append(await require_node_consumed(manager, tx, row))
        elif row["kind"] in ("workflow", "subagent"):
            from .processing import require_handled
            await require_handled(manager, tx, row)
            refs.extend(await _processing_references(manager, tx, result))
            if row["kind"] == "workflow":
                from .workflow_profile import read as profile
                from .workflow_results import _id, delivery_event_id, require_closed
                if (await profile(manager, tx, row))["notification"]["kind"] != "durable_inbox":
                    raise JobError("finalization_notification_receipt_required")
                await require_closed(manager, tx, actor, row["job_id"], generation, consumed=True)
                refs.append(await _reference(manager, tx, delivery_event_id(row["job_id"], generation)))
                refs.append(await _reference(manager, tx, _id("projected", row["job_id"], generation)))
        else:
            raise JobError("finalization_adapter_required")
        refs.extend(await _inputs(manager, tx, row, generation))
    children = {}
    owned = set()
    for edge in membership["members"]:
        member = edge["payload"]["member"]
        children[(member["job_id"], member["generation"])] = member
        owned.add((member["job_id"], member["generation"]))
        refs.append(await _reference(manager, tx, edge["event_id"]))
    # Historical provenance children also retain supervision, including
    # independent workflows that outlive their creator's model send.
    rows = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE parent_job_id=? OR (control_owner_job_id=? AND control_owner_generation=?)", (row["job_id"], row["job_id"], generation))
    for child in rows:
        children[(child["job_id"], child["generation"])] = _lease(child)
    for (job_id, child_generation), lease in sorted(children.items()):
        child = await manager._row(tx, job_id)
        if child is None or (child["tenant_id"], child["principal_id"]) != (row["tenant_id"], row["principal_id"]):
            raise JobError("invalid_finalization_tree")
        if child_generation == 0:
            if child["kind"] != "timer" or _hash(lease) != _hash(_lease(child)) or (job_id, 0) in owned:
                raise JobError("invalid_finalization_tree")
        else:
            child_scope = await read(manager, tx, child, child_generation)
            if (_hash(child_scope["scope"]["lease"]) != _hash(lease) or
                    (job_id, child_generation) in owned and _hash(child_scope["scope"]["owner"]) != _hash(_lease(row))):
                raise JobError("invalid_finalization_tree")
        refs.append(await _prepare(manager, tx, actor, child, child_generation, state, depth+1))
    refs = list({ref["event_id"]: ref for ref in refs}.values())
    proof = {"kind": row["kind"], "outcome": row["outcome"], "references": refs}
    data = {"version": 1, "scope": _lease(row), "proof": proof, "proof_hash": _hash(proof)}
    version = row["version"] + 1
    event_id = _identity(manager, "closed", [row["job_id"], generation, version, data["proof_hash"]])
    ref = {"event_id": event_id, "job_id": row["job_id"], "event_type": "supervision_closed", "revision": version,
        "request_hash": _hash(data), "delivery_status": "not_applicable", "payload_hash": _hash(data)}
    state["plans"].append((row, version, data, ref))
    state["visiting"].remove(key)
    state["done"][key] = ref
    return ref


async def _publish(manager, tx, plans):
    for row, version, data, ref in plans:
        status = "model_closed" if row["kind"] == "send" else row["outcome"]
        if await tx.execute(f"UPDATE {manager.jobs} SET status=?,supervision_status='closed',version=?,updated_at=? WHERE job_id=? AND version=? AND generation=? AND supervision_status='open'",
                (status, version, manager.now(), row["job_id"], row["version"], row["generation"])) != 1:
            raise JobError("finalization_execution_changed")
        await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (ref["event_id"], row["job_id"], "supervision_closed", version, _hash(data), _json(data), "not_applicable", manager.now()))


async def finalize(manager, actor, job_id, *, generation):
    if type(generation) is not int or not 0 <= generation < 2**63:
        raise JobError("invalid_result_generation")
    async with manager._write() as tx:
        row = await manager._authorize(tx, actor, job_id)
        if generation == 0 and row["kind"] != "timer":
            raise JobError("invalid_result_generation")
        if row["generation"] != generation:
            raise JobError("finalization_generation_changed")
        state = {"done": {}, "visiting": set(), "plans": []}
        ref = await _prepare(manager, tx, actor, row, generation, state)
        await _publish(manager, tx, state["plans"])
        return ref


async def consume_node(runtime, workflow, node, result):
    """Original engine consumes this exact canonical node result, without dispatch."""
    manager = runtime.manager
    async with manager._write() as tx:
        parent = await manager._authorize(tx, workflow.actor, workflow.lease.job_id)
        manager._check_proof(parent, workflow.lease)
        row = await manager._authorize(tx, workflow.actor, node.lease.job_id)
        manager._check_proof(row, node.lease)
        from .execution_membership import read
        membership = await read(manager, tx, row, row["generation"])
        if _hash(membership["scope"]["owner"]) != _hash(_lease(parent)) or parent["kind"] != "workflow":
            raise JobError("node_result_owner_mismatch")
        saved = json.loads(row["result_json"])
        if _hash({key: saved.get(key) for key in result}) != _hash(result) or set(saved) - set(result) - {"execution_status"}:
            raise JobError("node_result_consumption_mismatch")
        # Actual engine consumption is independent of supervision closure:
        # an unknown child may retain supervision after its node was consumed.
        # Freeze the handled canonical marker, never a pending delivery marker.
        if row["result_status"] != "handled":
            await tx.execute(f"UPDATE {manager.jobs} SET result_status='handled',version=version+1,updated_at=? WHERE job_id=?", (manager.now(), row["job_id"]))
            row = await manager._row(tx, row["job_id"])
        await tx.execute(f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=?", (result_id(row["job_id"], row["generation"]),))
        canonical = await _reference(manager, tx, result_id(row["job_id"], row["generation"]))
        data = {"version": 1, "workflow": _lease(parent), "node": _lease(row), "result": canonical}
        key = _identity(manager, "node", [row["job_id"], row["generation"]])
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
        if old:
            if old["request_hash"] != _hash(data) or json.loads(old["payload_json"]) != data:
                raise JobError("node_result_consumption_conflict")
        else:
            await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (key, row["job_id"], "node_result_consumed", row["version"], _hash(data), _json(data), "not_applicable", manager.now()))


async def reopen(manager, tx, previous, target, operation_id):
    """Same admission transaction; preserve old closure and reopen provenance."""
    current = previous
    visited = set()
    while current is not None:
        if current["job_id"] in visited or len(visited) >= 64:
            raise JobError("invalid_job_lineage")
        visited.add(current["job_id"])
        if current["supervision_status"] == "closed":
            event, _ = await closed(manager, tx, current)
            data = {"version": 1, "scope": _lease(current), "previous_closure": await _reference(manager, tx, event["event_id"]),
                "operation_id": operation_id, "target": target}
            row = await manager._row(tx, current["job_id"])
            if current["job_id"] != previous["job_id"]:
                await tx.execute(f"UPDATE {manager.jobs} SET supervision_status='open',version=version+1,updated_at=? WHERE job_id=? AND supervision_status='closed'", (manager.now(), current["job_id"]))
                row = await manager._row(tx, current["job_id"])
            elif row["supervision_status"] != "open":
                raise JobError("finalization_reopen_missing")
            key = _identity(manager, "reopened", [current["job_id"], operation_id])
            await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (key, current["job_id"], "supervision_reopened", row["version"], _hash(data), _json(data), "not_applicable", manager.now()))
        current = await manager._row(tx, current["parent_job_id"]) if current["parent_job_id"] else None


async def require_resumable(manager, tx, row):
    if row["outcome"] != "succeeded" or row["resource_state"] != "stopped" or row["intake"] != "closed":
        raise JobError("resume_target_unsettled")
    if row["supervision_status"] == "open" and row["status"] == "settling":
        return
    if row["supervision_status"] == "closed" and row["status"] == row["outcome"]:
        from .execution_membership import require_current
        await require_current(manager, tx, row)
        await closed(manager, tx, row)
        return
    raise JobError("resume_target_unsettled")


async def try_finalize(manager, actor, job_id, generation):
    """Unready/corrupt evidence retains durable open responsibility, no dispatch."""
    try:
        return {"status": "closed", "closure": await manager.finalize_supervision(actor, job_id, generation=generation)}
    except JobError as exc:
        return {"status": "pending", "reason": exc.code}


async def after_send(manager, actor, lease):
    """Read the original successful consumer exits, then close eligible scopes."""
    results = []
    exit = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND event_type='model_closed'", ("model_closed_" + lease.job_id,))
    decision_ids = json.loads(exit["payload_json"]).get("evidence", {}).get("processing_decisions", []) if exit else []
    if not isinstance(decision_ids, list) or len(decision_ids) > 10000:
        raise JobError("finalization_consumer_evidence_corrupt")
    targets = {lease.job_id}
    for decision_id in decision_ids:
        event = await manager.db.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='processing_decision'", (decision_id,))
        if event is None or event["caller_session_id"] != actor.session_id or json.loads(event["payload_json"]).get("handler_job_id") != lease.job_id:
            raise JobError("finalization_consumer_evidence_corrupt")
        targets.add(event["job_id"])
    # Older creator sends may now have all their independent work settled.
    for job_id in list(targets):
        try:
            row = await manager._authorize(manager.db, actor, job_id)
        except JobError as exc:
            # Physical node sessions may consume their own input while the
            # logical node is authorized only to its owning engine. Do not
            # enlarge that session's authority or fail its completed send.
            results.append({"status": "pending", "reason": exc.code})
            continue
        if row["parent_job_id"]:
            targets.add(row["parent_job_id"])
    for job_id in sorted(targets):
        try:
            row = await manager._authorize(manager.db, actor, job_id)
            results.append(await try_finalize(manager, actor, job_id, row["generation"]))
        except JobError as exc:
            results.append({"status": "pending", "reason": exc.code})
    return results
