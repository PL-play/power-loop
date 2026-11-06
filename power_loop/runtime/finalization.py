"""Close supervision from real tasks, pending inputs, results and unsettled costs.

No ancestor closure proofs, callbacks or local handle counts participate.
"""
from __future__ import annotations

import hashlib
import json

from .jobs import JobError, _hash
from .result_revisions import result_id


def _identity(manager, kind, values):
    return "sup_" + kind + "_" + _hash([manager.store.store_id, *values])[:48]


def _lease(row):
    from .execution_membership import proof
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
        return _event_reference(prepared.events[event_id])
    from .execution_membership import trusted_ref
    known = trusted_ref(manager, event_id)
    if known is not None:
        return known
    event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (event_id,))
    if event is None:
        raise JobError("finalization_evidence_missing")
    return _event_reference(event)


def _event_reference(event):
    return {"event_id": event["event_id"], "job_id": event["job_id"], "event_type": event["event_type"],
        "revision": event["job_version"], "request_hash": event["request_hash"],
        "payload_hash": _hash(_object(event["payload_json"]))}


async def _reference_rows(manager, tx, event_ids, known=None):
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
    return found


async def _references(manager, tx, event_ids, known=None):
    event_ids = list(event_ids)
    found = await _reference_rows(manager, tx, event_ids, known)
    return [_event_reference(found[key]) for key in event_ids]


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
        await _reference(manager, tx, result_id(data["handler_job_id"], 1))]



async def closed(manager, tx, row, generation=None, *, _state=None):
    generation = row["generation"] if generation is None else generation
    event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result'",
                             (result_id(row["job_id"], generation),))
    if event is None or (generation == row["generation"] and row["supervision_status"] != "closed"):
        raise JobError("finalization_execution_unsettled")
    return event, json.loads(event["payload_json"])


async def _finish(manager, tx, row, seen):
    if row["job_id"] in seen or len(seen) >= 10000:
        raise JobError("invalid_finalization_tree")
    seen.add(row["job_id"])
    if row["resource_state"] != "stopped" or row["intake"] != "closed":
        raise JobError("finalization_execution_unsettled")
    event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result'",
                             (result_id(row["job_id"], row["generation"]),))
    if event is None:
        raise JobError("finalization_result_unsettled")
    if row["supervision_status"] == "closed":
        return _event_reference(event)
    from .model_budgets import require_quiescent
    await require_quiescent(manager, tx, row)
    if row["kind"] == "send":
        from .pending_calls import SNAPSHOT, require_disposition
        payload = json.loads(event["payload_json"])
        if SNAPSHOT in payload.get("evidence", {}):
            await require_disposition(manager, tx, row, payload)
    item_id = hashlib.sha256(("result:" + event["event_id"]).encode()).hexdigest()
    parent = await manager._row(tx, row["parent_job_id"]) if row["parent_job_id"] else None
    if parent:
        pending = await tx.fetchone(f"SELECT item_id FROM {manager.store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=? AND status IN ('pending','claimed')", (parent["session_id"], item_id))
        if pending:
            raise JobError("finalization_result_unhandled")
    # Query unfinished children only. Their own result and cost identities are
    # unaffected by a parent/driver changing execution generation.
    children = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE parent_job_id=? "
        "AND supervision_status='open' ORDER BY job_id LIMIT 10001", (row["job_id"],))
    if len(children) > 10000:
        raise JobError("job_count_exceeded")
    for child in children:
        await _finish(manager, tx, child, seen)
    changed = await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET supervision_status='closed',"
        "version=version+1,updated_at=? WHERE job_id=? AND generation=? AND version=? AND supervision_status='open'",
        (manager.now(), row["job_id"], row["generation"], row["version"]), obligations=True)
    if changed != 1:
        raise JobError("finalization_execution_changed")
    for table in ("execution_checkpoints", "execution_allocations", "task_budget_attempts"):
        await tx.execute(f"DELETE FROM {manager.prefix}{table} WHERE job_id=?", (row["job_id"],))
    return _event_reference(event)


async def finalize(manager, actor, job_id, *, generation):
    async with manager.joined_write() as tx:
        row = await manager._authorize(tx, actor, job_id)
        if row["generation"] != generation:
            raise JobError("finalization_generation_changed")
        return await _finish(manager, tx, row, set())


async def try_finalize(manager, actor, job_id, generation):
    try:
        return {"status": "closed", "closure": await manager.finalize_supervision(actor, job_id, generation=generation)}
    except JobError as exc:
        return {"status": "pending", "reason": exc.code}

async def after_send(manager, actor, lease):
    """Read the original successful consumer exits, then close eligible scopes."""
    results = []
    exit = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND event_type='result'", (result_id(lease.job_id, lease.generation),))
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


async def reopen_ancestors(manager, tx, row):
    """A real explicit continuation reopens its ancestors' tree projection."""
    parent = row["parent_job_id"]
    while parent:
        ancestor = await manager._row(tx, parent)
        if ancestor["supervision_status"] == "closed":
            await tx.write_job(parent, f"UPDATE {manager.jobs} SET supervision_status='open',version=version+1 "
                "WHERE job_id=? AND generation=? AND version=?",
                (parent, ancestor["generation"], ancestor["version"]), obligations=True)
        parent = ancestor["parent_job_id"]
