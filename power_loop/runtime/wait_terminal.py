"""Mechanical terminal evidence for an admitted wait which never had an executor.

No model, stop signal, new execution generation or payment mutation lives here.
Source/owner history is evidence, not authority to affect a later generation.
"""
from __future__ import annotations

from . import timer_waits as waits
from . import wait_effects as effects
from .jobs import JobActor, JobError, _hash, _json
from .model_exit import content_reference, historical_row, require_stopped

_NEGATIVE = frozenset({"failed", "cancelled", "timed_out", "budget_exceeded", "interrupted"})
_EVENTS = frozenset({"operation", "wait_due", "wait_notification", "wait_decision", "wait_effect",
    "wait_review", "wait_disposition", "wait_resolved", "wait_terminated", "result", "supervision_closed"})


def terminal_id(manager, job_id, creation_id):
    return "wait_ended_" + _hash([manager.store.store_id, job_id, creation_id])[:48]


async def admission(manager, tx, actor, job_id):
    job, creation, saved = await waits._read(manager, tx, actor, job_id)
    if (type(job["generation"]) is not int or job["generation"] != 0
            or any(job[k] is not None for k in ("executor_id", "boot_id", "executor_key", "lease_until_ms"))):
        raise JobError("wait_unstarted_evidence_required")
    events = await tx.fetchall(f"SELECT event_type,action FROM {manager.events} WHERE job_id=?", (job_id,))
    if any(e["event_type"] not in _EVENTS or e["event_type"] == "operation" and e["action"] not in ("create", "wait_decision", "stop") for e in events):
        raise JobError("wait_execution_evidence_present")
    if await tx.fetchone(f"SELECT job_id FROM {manager.jobs} WHERE parent_job_id=? OR control_owner_job_id=? LIMIT 1", (job_id, job_id)):
        raise JobError("wait_execution_evidence_present")
    return job, creation, saved


async def _owner_exit(manager, tx, actor, saved):
    from .execution_membership import read, scope_id, seal_id
    from .result_revisions import result_id

    original = saved["binding"]["owner"]
    queue, seen, refs, outcome = [(original, None)], set(), {}, None
    for _ in range(64):
        if not queue:
            return outcome, [refs[k] for k in sorted(refs)]
        following = []
        for lease, parent in queue:
            key = (lease["job_id"], lease["generation"])
            if key in seen or len(seen) >= 10000:
                raise JobError("wait_owner_exit_evidence_corrupt")
            seen.add(key)
            current = await manager._row(tx, lease["job_id"])
            if current is None or (current["tenant_id"], current["principal_id"]) != (actor.tenant_id, actor.principal_id):
                raise JobError("wait_owner_exit_evidence_missing")
            member = await read(manager, tx, current, lease["generation"])
            scope = member["scope"]
            if (not member["sealed"] or _hash(scope["lease"]) != _hash(lease)
                    or parent is not None and _hash(scope["owner"]) != _hash(parent)):
                raise JobError("wait_owner_exit_unconfirmed")
            row = historical_row(current, scope)
            if current["generation"] == lease["generation"] and (current["resource_state"] != "stopped" or current["intake"] != "closed"):
                raise JobError("wait_owner_exit_unconfirmed")
            ids = [scope_id(manager, lease["job_id"], lease["generation"]), seal_id(manager, lease["job_id"], lease["generation"])]
            if scope["kind"] == "send":
                model = member["model"]
                if model is None:
                    raise JobError("wait_owner_exit_unconfirmed")
                ids.extend(await require_stopped(manager, tx, row, model["payload"]))
                ids.append(model["event_id"])
            else:
                result = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result'", (result_id(lease["job_id"], lease["generation"]),))
                data = waits._object(result["payload_json"]) if result else {}
                if (result is None or result["job_id"] != lease["job_id"] or data.get("resource_state") != "stopped"
                        or type(data.get("generation", 1)) is not int or data.get("generation", 1) != lease["generation"]
                        or data.get("outcome") not in _NEGATIVE | {"succeeded"}):
                    raise JobError("wait_owner_exit_unconfirmed")
                if current["generation"] == lease["generation"] and (current["outcome"] != data["outcome"] or _hash(waits._object(current["result_json"])) != _hash(data.get("result"))):
                    raise JobError("wait_owner_exit_evidence_corrupt")
                if parent is None:
                    if data["outcome"] not in _NEGATIVE or scope["kind"] != saved["binding"]["route"]["kind"]:
                        raise JobError("wait_owner_not_negative")
                    outcome = data["outcome"]
                ids.append(result["event_id"])
            claimed = set()
            for edge in member["members"]:
                child = edge["payload"]["member"]
                claimed.add((child["job_id"], child["generation"]))
                ids.append(edge["event_id"])
                following.append((child, scope["lease"]))
            owned = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE control_owner_job_id=? AND control_owner_generation=?", key)
            for child in owned:
                if child["generation"] == 0 and child["kind"] == "timer":
                    # Another logical wait has no executing resource either.
                    await admission(manager, tx, actor, child["job_id"])
                elif (child["job_id"], child["generation"]) not in claimed:
                    raise JobError("wait_owner_exit_unconfirmed")
            for identity in ids:
                event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (identity,))
                if event is None:
                    raise JobError("wait_owner_exit_evidence_missing")
                refs[identity] = content_reference(event)
        queue = following
    raise JobError("wait_owner_exit_depth_exceeded")


def _result_projection(job, canonical):
    if set(canonical) == {"outcome", "result", "resource_state", "unstarted", "requested"}:
        from .result_payloads import require_replay
        require_replay(canonical, canonical["requested"])
        expected, legacy = canonical["result"], True
    elif set(canonical) == {"outcome", "result", "resource_state", "unstarted"}:
        expected = {**canonical["result"], "status": job["outcome"]}
        legacy = True
    else:
        expected, legacy = canonical.get("result"), False
    if (canonical.get("unstarted") is not True or canonical.get("resource_state") != "stopped"
            or canonical.get("outcome") not in _NEGATIVE or job["outcome"] not in _NEGATIVE
            or canonical["outcome"] != job["outcome"]
            or job["resource_state"] != "stopped" or job["intake"] != "closed"
            or _hash(expected) != _hash(waits._object(job["result_json"]))):
        raise JobError("wait_terminal_result_mismatch")
    return legacy


async def _cancel_effect(manager, tx, saved, *, handled=False, _state=None):
    from .wait_notifications import decision_id
    # Authority comes from the admitted task row, never from a model field.
    job = await manager._row(tx, saved["job_id"])
    main = JobActor(job["tenant_id"], job["principal_id"], saved["binding"]["main_session_id"])
    from .wait_reviews import history
    _, creation, _ = await waits._read(manager, tx, main, job["job_id"])
    due = (_state if _state is not None else await history(manager, tx, main, job, creation, saved))["event_id"]
    decision = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (decision_id(manager, due),))
    if decision is None or waits._object(decision["payload_json"]).get("request", {}).get("decision") != "cancel":
        return None
    effect = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (effects.effect_id(manager, decision["event_id"]),))
    if effect is None:
        return None
    _, _, _, decision, data = await effects._decision(manager, tx, main, due)
    effect = await effects._effect(manager, tx, main, job, saved, decision, data)
    if effect is None or handled and effect["delivery_status"] != "handled":
        raise JobError("wait_cancel_effect_unsettled")
    return {"effect_id": effect["event_id"], "effect_hash": effect["request_hash"], "decision_id": decision["event_id"],
        "decision_hash": decision["request_hash"], "operation_id": waits._object(effect["payload_json"])["operation_id"]}


async def _terminated(manager, tx, actor, job, creation, saved, *, _state=None):
    identity = terminal_id(manager, job["job_id"], creation["event_id"])
    row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (identity,))
    if row is None:
        raise JobError("wait_terminal_evidence_missing")
    outcome, refs = await _owner_exit(manager, tx, actor, saved)
    result = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", ("result_" + job["job_id"],))
    if result is None or result["job_id"] != job["job_id"] or result["event_type"] != "result" or result["delivery_status"] != "handled":
        raise JobError("wait_terminal_result_mismatch")
    canonical = waits._object(result["payload_json"])
    legacy = _result_projection(job, canonical)
    expected = {"version": 1, "creation_id": creation["event_id"], "creation_hash": creation["request_hash"],
        "owner": saved["binding"]["owner"], "owner_outcome": outcome, "owner_exit": refs,
        "result": content_reference(result), "legacy_result": legacy, "cancel_effect": await _cancel_effect(manager, tx, saved, handled=True, _state=_state)}
    if not legacy:
        target = "timed_out" if outcome == "timed_out" else "cancelled"
        payload = {"outcome": target, "result": {"status": target, "wait_terminated_event_id": identity, "owner_outcome": outcome},
            "resource_state": "stopped", "unstarted": True, "logical_wait": True}
        if _hash(canonical) != _hash(payload) or result["request_hash"] != _hash(payload) or job["outcome"] != target:
            raise JobError("wait_terminal_result_mismatch")
    if (row["event_type"] != "wait_terminated" or row["job_id"] != job["job_id"] or row["request_hash"] != _hash(expected)
            or row["delivery_status"] != "not_applicable" or _hash(waits._object(row["payload_json"])) != _hash(expected)):
        raise JobError("wait_terminal_evidence_corrupt")
    return row


async def terminate(manager, actor, job_id):
    """End one unexecuted wait only after the original owner's stopped exit."""
    async with manager._write() as tx:
        return await terminate_tx(manager, tx, actor, job_id)


async def terminate_tx(manager, tx, actor, job_id):
    """The same control transaction also supports atomic tree finalization."""
    job, creation, saved = await admission(manager, tx, actor, job_id)
    identity = terminal_id(manager, job_id, creation["event_id"])
    old = await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (identity,))
    if old:
        await _terminated(manager, tx, actor, job, creation, saved)
        from .wait_dispositions import reconcile_tx
        await reconcile_tx(manager, tx, actor, job, creation, saved)
        return {"status": "terminated", "job_id": job_id, "termination_event_id": identity, "execution_authority": False}
    outcome, refs = await _owner_exit(manager, tx, actor, saved)
    cancelled = await _cancel_effect(manager, tx, saved)
    result_id = "result_" + job_id
    result = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (result_id,))
    if result is None:
        if job["outcome"] is not None or job["resource_state"] != "not_started" or job["supervision_status"] != "open":
            raise JobError("wait_state_changed")
        target = "timed_out" if outcome == "timed_out" else "cancelled"
        projection = {"status": target, "wait_terminated_event_id": identity, "owner_outcome": outcome}
        canonical = {"outcome": target, "result": projection, "resource_state": "stopped", "unstarted": True, "logical_wait": True}
        await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (result_id, job_id, "result", job["version"]+1, _hash(canonical), _json(canonical), "handled", manager.now()))
        await tx.execute(f"UPDATE {manager.jobs} SET status='settling',outcome=?,resource_state='stopped',intake='closed',result_status='handled',"
            "result_json=?,version=version+1,updated_at=? WHERE job_id=?", (target, _json(projection), manager.now(), job_id))
        job = await manager._row(tx, job_id)
        result = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (result_id,))
        legacy = False
    else:
        legacy = _result_projection(job, waits._object(result["payload_json"]))
        if not legacy:
            raise JobError("wait_terminal_evidence_missing")
        await tx.execute(f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=?", (result_id,))
        await tx.execute(f"UPDATE {manager.jobs} SET result_status='handled',version=version+1,updated_at=? WHERE job_id=?", (manager.now(), job_id))
    data = {"version": 1, "creation_id": creation["event_id"], "creation_hash": creation["request_hash"],
        "owner": saved["binding"]["owner"], "owner_outcome": outcome, "owner_exit": refs,
        "result": content_reference(result), "legacy_result": legacy, "cancel_effect": cancelled}
    await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
        (identity, job_id, "wait_terminated", job["version"]+1, _hash(data), _json(data), "not_applicable", manager.now()))
    if cancelled:
        await tx.execute(f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id IN (?,?)", (cancelled["effect_id"], cancelled["decision_id"]))
    await _terminated(manager, tx, actor, job, creation, saved)
    from .wait_dispositions import reconcile_tx
    await reconcile_tx(manager, tx, actor, job, creation, saved)
    return {"status": "terminated", "job_id": job_id, "termination_event_id": identity, "execution_authority": False}

async def references(manager, tx, job):
    """Strict generation-zero adapter for the common supervision tree."""
    from .execution_membership import scope_id
    from .finalization import _reference

    actor = JobActor(job["tenant_id"], job["principal_id"], job["created_by_session_id"])
    job, creation, saved = await admission(manager, tx, actor, job["job_id"])
    if job["resource_state"] != "stopped" or job["intake"] != "closed" or job["result_status"] != "handled":
        raise JobError("wait_resolution_unsettled")
    ids = [creation["event_id"], "result_" + job["job_id"], scope_id(manager, saved["binding"]["source"]["job_id"], saved["binding"]["source"]["generation"]),
        scope_id(manager, saved["binding"]["owner"]["job_id"], saved["binding"]["owner"]["generation"])]
    if job["outcome"] == "succeeded":
        projection = waits._object(job["result_json"])
        resolved = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (projection.get("wait_resolved_event_id"),))
        if resolved is None:
            raise JobError("wait_resolution_evidence_missing")
        body = waits._object(resolved["payload_json"])
        effect = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (body.get("effect_id"),))
        if effect is None:
            raise JobError("wait_effect_evidence_missing")
        event_id = waits._object(effect["payload_json"])["binding"]["event_id"]
        main = JobActor(job["tenant_id"], job["principal_id"], saved["binding"]["main_session_id"])
        _, _, _, decision, data = await effects._decision(manager, tx, main, event_id)
        effect = await effects._effect(manager, tx, main, job, saved, decision, data)
        if effect is None or effect["delivery_status"] != "handled" or data["request"]["decision"] != "continue":
            raise JobError("wait_resolution_unsettled")
        await effects._resolved(manager, tx, job, effect)
        ids.extend([event_id, decision["event_id"], data["operation_id"], effect["event_id"], projection["wait_resolved_event_id"], body["input_id"],
            *(r["event_id"] for r in body["proofs"])])
    elif job["outcome"] in _NEGATIVE:
        ended = await _terminated(manager, tx, actor, job, creation, saved)
        ids.append(ended["event_id"])
        cancelled = waits._object(ended["payload_json"])["cancel_effect"]
        if cancelled:
            ids.extend(cancelled[k] for k in ("effect_id", "decision_id", "operation_id"))
    else:
        raise JobError("wait_resolution_unsettled")
    from .wait_reviews import history
    state = await history(manager, tx, actor, job, creation, saved)
    if job["outcome"] in _NEGATIVE:
        from .wait_dispositions import event_ids, read_tx
        disposed = await read_tx(manager, tx, actor, job, creation, saved, state, required=True)
        if disposed is not None:
            ids.extend(event_ids(disposed))
    ids.extend(state["references"])
    return [await _reference(manager, tx, identity) for identity in sorted(set(ids))]
