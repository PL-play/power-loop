"""Mechanical supervision closure; closed execution is not billing settlement.

Every closure retains immutable evidence and child closure references. Resume
adapters explicitly reopen historical supervision without reviving old sends.
No model call, host callback, or execution dispatch occurs in this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .finalization_plan import PlanChanged as _PlanChanged
from .finalization_plan import Preparation
from .jobs import JobError, _hash, _json
from .result_revisions import result_id


@dataclass(frozen=True)
class _TreeRevision:
    root_id: str
    generation: int
    count: int
    version_sum: int
    preparation: Preparation


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
    prepared = manager._finalization_preparation.get()
    if prepared is not None and event_id in prepared.events:
        return _event_reference(prepared.project_event(prepared.events[event_id]))
    from .execution_membership import trusted_ref
    known = trusted_ref(manager, event_id)
    if known is not None:
        return known
    event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
    if event is None:
        raise JobError("finalization_evidence_missing")
    return _event_reference(event if prepared is None else prepared.project_event(event))


def _event_reference(event):
    return {"event_id": event["event_id"], "job_id": event["job_id"], "event_type": event["event_type"],
        "revision": event["job_version"], "request_hash": event["request_hash"],
        "delivery_status": event["delivery_status"], "payload_hash": _hash(_object(event["payload_json"]))}


async def _references(manager, tx, event_ids, known=None):
    """Read the exact referenced events in bounded batches, preserving order.

    ``known`` holds rows already read in this transaction.
    """
    event_ids = list(event_ids)
    found = {key: known[key] for key in event_ids if known and key in known}
    prepared = manager._finalization_preparation.get()
    if prepared is not None:
        found.update({key: prepared.events[key] for key in event_ids if key in prepared.events})
    missing = [key for key in event_ids if key not in found]
    for offset in range(0, len(missing), 100):
        part = missing[offset:offset + 100]
        rows = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE event_id IN (" +
                                 ",".join("?" for _ in part) + ")", tuple(part))
        found.update({row["event_id"]: row for row in rows})
    if any(key not in found for key in event_ids):
        raise JobError("finalization_evidence_missing")
    return [_event_reference(found[key] if prepared is None else prepared.project_event(found[key])) for key in event_ids]


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
            set(data) != {"version", "scope", "proof", "proof_hash"} or type(data["version"]) is not int or data["version"] not in (1, 2, 3) or
            not isinstance(data["scope"], dict) or not isinstance(data["proof"], dict) or
            data["scope"]["job_id"] != event["job_id"] or data["proof_hash"] != _hash(data["proof"]) or
            event["request_hash"] != _hash(data) or event["event_id"] != _identity(manager, "closed",
                [event["job_id"], data["scope"]["generation"], event["job_version"], data["proof_hash"]])):
        raise JobError("finalization_evidence_corrupt")
    from .execution_membership import _lease as validate_lease
    if data["version"] == 1:
        validate_lease(data["scope"])
    elif data["version"] == 2:
        row = await manager._row(tx, event["job_id"])
        if (row is None or row["kind"] != "timer" or data["proof"].get("kind") != "timer" or data["proof"].get("outcome") != row["outcome"]
                or type(data["scope"].get("generation")) is not int
                or data["scope"]["generation"] != 0 or _hash(data["scope"]) != _hash(_lease(row))):
            raise JobError("finalization_evidence_corrupt")
        from .wait_terminal import references
        if _hash(data["proof"].get("references")) != _hash(await references(manager, tx, row)):
            raise JobError("finalization_evidence_changed")
    else:
        row = await manager._row(tx, event["job_id"])
        if (row is None or row["kind"] != "background_tool" or data["scope"].get("generation") != 0
                or type(data["scope"].get("generation")) is not int
                or _hash(data["scope"]) != _hash(_lease(row)) or data["proof"].get("kind") != row["kind"]
                or data["proof"].get("outcome") != row["outcome"]):
            raise JobError("finalization_evidence_corrupt")
        from .managed_effects import unstarted_references
        if _hash(data["proof"].get("references")) != _hash(await unstarted_references(manager, tx, row)):
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
    actual_refs = await _references(manager, tx, (ref["event_id"] for ref in refs))
    for ref, actual in zip(refs, actual_refs, strict=True):
        if actual != ref:
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
        elif ref['event_type'] == 'controller_stopped':
            from .controller_recovery import handled_references
            from .controller_recovery import read as controller_stop
            recovered = await manager._row(tx, ref['job_id'])
            stopped = await controller_stop(manager, tx, recovered, optional=False, _closures=state)
            if stopped[0]['event_id'] != ref['event_id']:
                raise JobError('finalization_evidence_changed')
            await handled_references(manager, tx, recovered, _closures=state)
    if data["version"] == 1 and proof["kind"] == "background_tool":
        current = await manager._row(tx, event["job_id"])
        if current is None or current["generation"] != data["scope"]["generation"]:
            raise JobError("finalization_evidence_changed")
        from .managed_effects import require_closed
        result = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (result_id(current["job_id"], current["generation"]),))
        if result is None:
            raise JobError("finalization_evidence_missing")
        # The delivery and actual handling transcript remain part of retained
        # proof, including when this closed child is read through its parent.
        await require_closed(manager, tx, current, result)
    if data["version"] == 1 and proof["kind"] == "send":
        from .pending_calls import negative_review_references, require_disposition
        source = await manager._row(tx, event["job_id"])
        await require_disposition(manager, tx, source)
        await negative_review_references(manager, tx, source)
    state["visiting"].remove(event_id)
    state["done"][event_id] = data
    return data


async def closed(manager, tx, row, generation=None, *, _state=None):
    """Validate saved immutable closure, including historical generations."""
    generation = row["generation"] if generation is None else generation
    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='supervision_closed' ORDER BY job_version DESC,event_id", (row["job_id"],))
    for event in events:
        data = _object(event["payload_json"])
        if data.get("scope", {}).get("generation") != generation:
            continue
        data = await _closed_event(manager, tx, event, state=_state)
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
    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND "
        "(event_type='task_control_request' OR (event_type='operation' AND action='nudge'))", (row["job_id"],))
    from .control_requests import completed_request_events
    refs.extend(_event_reference(event) for event in completed_request_events(events))
    for event in events:
        if event["event_type"] == "task_control_request":
            continue
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
    validated = state.setdefault('validated_closures', {'visiting': set(), 'done': {}})
    if key in state["done"]:
        return state["done"][key]
    if key in state["visiting"] or depth >= 64 or len(state["done"]) + len(state["visiting"]) >= 10000:
        raise JobError("invalid_finalization_tree")
    if generation < row["generation"] or row["supervision_status"] == "closed":
        event, _ = await closed(manager, tx, row, generation, _state=validated)
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
        from .execution_membership import _trusted
        cached = _trusted(manager, "_membership_reads", key)
        if cached is not None:
            exit_evidence = (json.loads(cached)["model"] or {}).get("payload", {}).get("evidence", {})
        else:
            exit = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", ("model_closed_" + row["job_id"],))
            exit_evidence = _object(exit["payload_json"]).get("evidence", {}) if exit else {}
        if exit_evidence.get("model_attempt_snapshot_error"):
            from .model_exit import capture
            await capture(manager, tx, row)  # Keep the original precise corruption error.
            raise JobError("model_exit_snapshot_incomplete")
    if row["resource_state"] != "stopped" or row["intake"] != "closed":
        raise JobError("finalization_execution_unsettled")
    state["visiting"].add(key)
    if generation == 0:
        if row["kind"] == "timer":
            from .wait_terminal import references
            refs = await references(manager, tx, row)
            version_code = 2
        elif row["kind"] == "background_tool":
            from .managed_effects import unstarted_references
            refs = await unstarted_references(manager, tx, row)
            version_code = 3
        else:
            raise JobError("finalization_adapter_required")
        proof = {"kind": row["kind"], "outcome": row["outcome"], "references": refs}
        data = {"version": version_code, "scope": _lease(row), "proof": proof, "proof_hash": _hash(proof)}
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
        known = {}
        if SNAPSHOT in evidence:
            attempt_refs = await require_stopped(manager, tx, row, model["payload"], known=known)
            from .model_budgets import require_quiescent as require_budget_quiescent
            attempt_refs.extend(await require_budget_quiescent(manager, tx, row))
        else:
            from .model_attempts import require_quiescent
            attempt_refs = await require_quiescent(manager, tx, row)
        refs.append(await _reference(manager, tx, model["event_id"]))
        from .pending_calls import require_disposition
        pending_ref = await require_disposition(manager, tx, row, model["payload"])
        if pending_ref is not None:
            refs.append(pending_ref)
            from .pending_calls import negative_review_references
            refs.extend(await negative_review_references(manager, tx, row))
        refs.extend(await _references(manager, tx, attempt_refs, known))
        from .control_requests import completed_request_events, request_events
        refs.extend(_event_reference(event) for event in completed_request_events(await request_events(manager, tx, row["job_id"])))
    else:
        result = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result'", (result_id(row["job_id"], generation),))
        payload = json.loads(result["payload_json"]) if result else {}
        recovered = None
        if row['kind'] in ('subagent', 'workflow_node', 'workflow'):
            from .controller_recovery import read as controller_stop
            recovered = await controller_stop(manager, tx, row, _closures=validated)
        if (result is None or (payload.get("resource_state") != "stopped" and recovered is None) or payload.get("outcome") != row["outcome"] or
                _hash(payload.get("result")) != _hash(json.loads(row["result_json"] or "null"))):
            raise JobError("finalization_result_unsettled")
        refs.append(await _reference(manager, tx, result["event_id"]))
        if recovered is not None:
            from .controller_recovery import handled_references
            refs.extend(await handled_references(manager, tx, row, _closures=validated))
        elif row["kind"] == "workflow_node":
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
        elif row["kind"] == "background_tool":
            from .managed_effects import require_closed
            refs.extend(await require_closed(manager, tx, row, result))
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
    from .execution_membership import childless
    rows = [] if childless(manager, row["job_id"]) else await tx.fetchall(
        f"SELECT * FROM {manager.jobs} WHERE parent_job_id=? OR (control_owner_job_id=? AND control_owner_generation=?)",
        (row["job_id"], row["job_id"], generation))
    for child in rows:
        children[(child["job_id"], child["generation"])] = _lease(child)
    for (job_id, child_generation), lease in sorted(children.items()):
        child = await manager._row(tx, job_id)
        if child is None or (child["tenant_id"], child["principal_id"]) != (row["tenant_id"], row["principal_id"]):
            raise JobError("invalid_finalization_tree")
        if child_generation == 0:
            if child["kind"] not in ("timer", "background_tool") or _hash(lease) != _hash(_lease(child)) or (job_id, 0) in owned:
                raise JobError("invalid_finalization_tree")
            if child["kind"] == "background_tool":
                from .managed_effects import descriptor
                if await descriptor(manager, tx, child) is None:
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


def _closure_event(manager, row, version, data, ref):
    return {"event_id": ref["event_id"], "job_id": row["job_id"], "event_type": "supervision_closed",
        "job_version": version, "request_hash": _hash(data), "payload_json": _json(data),
        "delivery_status": "not_applicable", "created_at": manager.now()}


async def _publish(manager, tx, plans, *, tree=None):
    # A closed descendant may have resumed without being in `plans`. Guard
    # the complete tree once, before changing any of its planned projections.
    if tree is not None:
        row, version, data, ref = plans[0]
        if await tx.append_event_if_tree_current(_closure_event(manager, row, version, data, ref), tree) != 1:
            raise _PlanChanged()
    prepared = tree.preparation if tree is not None else manager._finalization_preparation.get()
    if prepared is not None:
        await prepared.apply(tx)
    for row, version, data, ref in plans:
        status = "model_closed" if row["kind"] == "send" else row["outcome"]
        deferred = manager._deferred_model_updates.get()
        close = deferred.get(row["job_id"]) if deferred is not None else None
        if close is not None:
            resource, _now, previous_version, generation = close
            changed = await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET status=?,intake='closed',resource_state=?,"
                "lease_until_ms=NULL,supervision_status='closed',version=?,updated_at=? "
                "WHERE job_id=? AND version=? AND generation=? AND supervision_status='open'", (status, resource, version, manager.now(), row["job_id"], previous_version, generation))
        else:
            changed = await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET status=?,supervision_status='closed',version=?,updated_at=? WHERE job_id=? AND root_job_id=? AND version=? AND generation=? AND supervision_status='open'", (status, version, manager.now(), row["job_id"], row["root_job_id"], row["version"], row["generation"]))
        if changed != 1:
            raise JobError("finalization_execution_changed")
        if tree is None or ref["event_id"] != plans[0][3]["event_id"]:
            event = _closure_event(manager, row, version, data, ref)
            await tx.append_event(tuple(event), tuple(event.values()))


async def _validate_request(manager, reader, actor, job_id, generation):
    row = await manager._authorize(reader, actor, job_id)
    if generation == 0 and row["kind"] not in ("timer", "background_tool"):
        raise JobError("invalid_result_generation")
    if generation == 0 and row["kind"] == "background_tool":
        from .managed_effects import descriptor
        if await descriptor(manager, reader, row) is None:
            raise JobError("invalid_result_generation")
    if row["generation"] != generation:
        raise JobError("finalization_generation_changed")
    return row


async def _prepare_snapshot(manager, actor, job_id, generation):
    async with manager._read() as reader:
        prepared = Preparation(manager, reader)
        token = manager._finalization_preparation.set(prepared)
        try:
            row = await _validate_request(manager, reader, actor, job_id, generation)
            state = {"done": {}, "visiting": set(), "plans": []}
            ref = await _prepare(manager, reader, actor, row, generation, state)
            if not state["plans"]:
                return ref, [], None
            root = row["root_job_id"]
            summary = await reader.fetchone(f"SELECT COUNT(*) AS job_count,COALESCE(SUM(version),0) AS version_sum,"
                f"MAX(CASE WHEN job_id=? THEN generation END) AS root_generation FROM {manager.jobs} WHERE root_job_id=?",
                (root, root))
            tree = _TreeRevision(root, summary["root_generation"], summary["job_count"], summary["version_sum"], prepared)
            return ref, state["plans"], tree
        finally:
            manager._finalization_preparation.reset(token)


async def _publish_plan(manager, plans, tree):
    async with manager._write() as tx:
        await _publish(manager, tx, plans, tree=tree)


async def finalize(manager, actor, job_id, *, generation):
    if type(generation) is not int or not 0 <= generation < 2**63:
        raise JobError("invalid_result_generation")
    joined = manager._joined_write.get()
    if joined is not None:
        # The live childless send closes using its own transaction-local facts.
        # A tree with descendants is deferred until after its model exit commits.
        prepared = Preparation(manager, joined)
        token = manager._finalization_preparation.set(prepared)
        try:
            row = await _validate_request(manager, joined, actor, job_id, generation)
            state = {"done": {}, "visiting": set(), "plans": []}
            ref = await _prepare(manager, joined, actor, row, generation, state)
            await _publish(manager, joined, state["plans"])
            return ref
        finally:
            manager._finalization_preparation.reset(token)
    while True:
        try:
            ref, plans, tree = await _prepare_snapshot(manager, actor, job_id, generation)
            if plans:
                await _publish_plan(manager, plans, tree)
            return ref
        except _PlanChanged:
            continue


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
            await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET result_status='handled',version=version+1,updated_at=? WHERE job_id=?", (manager.now(), row["job_id"]), obligations=True)
            row = await manager._row(tx, row["job_id"])
        await tx.update_event(row["job_id"], f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=?", (result_id(row["job_id"], row["generation"]),))
        canonical = await _reference(manager, tx, result_id(row["job_id"], row["generation"]))
        data = {"version": 1, "workflow": _lease(parent), "node": _lease(row), "result": canonical}
        key = _identity(manager, "node", [row["job_id"], row["generation"]])
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
        if old:
            if old["request_hash"] != _hash(data) or json.loads(old["payload_json"]) != data:
                raise JobError("node_result_consumption_conflict")
        else:
            await tx.append_event(('event_id', 'job_id', 'event_type', 'job_version', 'request_hash', 'payload_json', 'delivery_status', 'created_at'), (key, row["job_id"], "node_result_consumed", row["version"], _hash(data), _json(data), "not_applicable", manager.now()))


async def prepare_reopen(manager, tx, previous):
    """Freeze retained ancestor closures before a later tree publication."""
    current = previous
    visited = set()
    prepared = []
    while current is not None:
        if current["job_id"] in visited or len(visited) >= 64:
            raise JobError("invalid_job_lineage")
        visited.add(current["job_id"])
        if current["supervision_status"] == "closed":
            event, _ = await closed(manager, tx, current)
            prepared.append((current, await _reference(manager, tx, event["event_id"])))
        current = await manager._row(tx, current["parent_job_id"]) if current["parent_job_id"] else None
    return prepared


async def publish_reopen(manager, tx, previous, target, operation_id, prepared):
    for current, reference in prepared:
        data = {"version": 1, "scope": _lease(current), "previous_closure": reference,
            "operation_id": operation_id, "target": target}
        row = await manager._row(tx, current["job_id"])
        if current["job_id"] != previous["job_id"]:
            if await tx.write_job(current["job_id"], f"UPDATE {manager.jobs} SET supervision_status='open',version=version+1,updated_at=? "
                    "WHERE job_id=? AND root_job_id=? AND generation=? AND version=? AND supervision_status='closed'",
                    (manager.now(), current["job_id"], current["root_job_id"], current["generation"], current["version"])) != 1:
                raise JobError("finalization_execution_changed")
            row = await manager._row(tx, current["job_id"])
        elif row["supervision_status"] != "open":
            raise JobError("finalization_reopen_missing")
        key = _identity(manager, "reopened", [current["job_id"], operation_id])
        await tx.append_event(('event_id', 'job_id', 'event_type', 'job_version', 'request_hash', 'payload_json', 'delivery_status', 'created_at'),
            (key, current["job_id"], "supervision_reopened", row["version"], _hash(data), _json(data), "not_applicable", manager.now()))


async def reopen(manager, tx, previous, target, operation_id):
    """Same admission transaction; preserve old closure and reopen provenance."""
    await publish_reopen(manager, tx, previous, target, operation_id, await prepare_reopen(manager, tx, previous))


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
    from .execution_membership import _trusted
    cached = _trusted(manager, "_membership_reads", (lease.job_id, lease.generation))
    if cached is not None and json.loads(cached)["model"] is not None:
        evidence = json.loads(cached)["model"]["payload"].get("evidence", {})
    else:
        exit = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND event_type='model_closed'", ("model_closed_" + lease.job_id,))
        evidence = json.loads(exit["payload_json"]).get("evidence", {}) if exit else {}
    decision_ids = evidence.get("processing_decisions", [])
    if not isinstance(decision_ids, list) or len(decision_ids) > 10000:
        raise JobError("finalization_consumer_evidence_corrupt")
    targets, rows = {lease.job_id}, {}
    for decision_id in decision_ids:
        event = await manager.db.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='processing_decision'", (decision_id,))
        if event is None or event["caller_session_id"] != actor.session_id or json.loads(event["payload_json"]).get("handler_job_id") != lease.job_id:
            raise JobError("finalization_consumer_evidence_corrupt")
        targets.add(json.loads(event["payload_json"])["event"]["job_id"])
    # Older creator sends may now have all their independent work settled.
    for job_id in list(targets):
        try:
            row = rows[job_id] = await manager._authorize(manager.db, actor, job_id)
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
            row = rows.get(job_id) or await manager._authorize(manager.db, actor, job_id)
            results.append(await try_finalize(manager, actor, job_id, row["generation"]))
        except JobError as exc:
            results.append({"status": "pending", "reason": exc.code})
    return results
