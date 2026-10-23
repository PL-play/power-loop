"""Retained original controllers wait after their model phase has stopped.

Waiting facts never claim physical stop, dispatch a model or settle money.
Only an accepted original input can move the controller back to running.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from . import timer_waits as waits
from .jobs import JobActor, JobError, _hash, _json
from .model_exit import content_reference, historical_row, require_stopped


class WaitDeadline(TimeoutError):
    pass


class _WaitCancelled(Exception):
    pass


def _id(manager, prefix, data):
    return prefix + _hash([manager.store.store_id, data])[:48]


def _fact(event, owner, kind, identity, data):
    if (event is None or event["event_id"] != identity or event["job_id"] != owner["job_id"]
            or event["event_type"] != kind or event["delivery_status"] != "not_applicable"
            or type(event["job_version"]) is not int or not 0 < event["job_version"] <= owner["version"]
            or event["request_hash"] != _hash(data) or _hash(waits._object(event["payload_json"])) != _hash(data)):
        raise JobError("owner_wait_evidence_corrupt")


async def _validate_entry(manager, tx, actor, owner, event, data):
    from .execution_membership import _lease

    if (not isinstance(data, dict) or set(data) != {"version", "owner", "session_id", "raw", "waits", "deadline_at_ms", "model_exits"}
            or type(data["version"]) is not int or data["version"] != 1
            or _hash(_lease(data["owner"])) != _hash(waits._proof(owner))
            or not isinstance(data["session_id"], str) or not data["session_id"]
            or not isinstance(data["raw"], dict) or data["raw"].get("status") != "completed"
            or data["raw"].get("session_id") != data["session_id"]
            or not isinstance(data["waits"], list) or not data["waits"]
            or not isinstance(data["model_exits"], list) or not data["model_exits"]
            or type(data["deadline_at_ms"]) is not int):
        raise JobError("owner_wait_evidence_corrupt")
    _fact(event, owner, "owner_waiting", _id(manager, "owner_wait_", data), data)
    deadlines, seen = [], set()
    for item in data["waits"]:
        if (not isinstance(item, dict) or set(item) != {"job_id", "creation_id", "creation_hash"}
                or not isinstance(item["job_id"], str) or item["job_id"] in seen):
            raise JobError("owner_wait_evidence_corrupt")
        seen.add(item["job_id"])
        _, creation, saved = await waits._read(manager, tx, actor, item["job_id"])
        if (creation["event_id"] != item["creation_id"] or creation["request_hash"] != item["creation_hash"]
                or _hash(saved["binding"]["owner"]) != _hash(data["owner"])
                or saved["binding"]["session_id"] != data["session_id"] or saved["receipt_ref"] is None):
            raise JobError("owner_wait_evidence_corrupt")
        deadlines.append(saved["request"]["policy"]["deadline_at_ms"])
    if min(deadlines) != data["deadline_at_ms"]:
        raise JobError("owner_wait_evidence_corrupt")
    seen.clear()
    for ref in data["model_exits"]:
        if not isinstance(ref, dict) or not isinstance(ref.get("event_id"), str) or ref["event_id"] in seen:
            raise JobError("owner_wait_evidence_corrupt")
        seen.add(ref["event_id"])
        original = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (ref["event_id"],))
        if original is None or _hash(content_reference(original)) != _hash(ref):
            raise JobError("owner_wait_evidence_corrupt")


async def references(manager, tx, actor, owner, *, success):
    """Retain each original waiting phase in success and supervision proofs."""
    from .processing import require_event_handled

    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type IN "
        "('owner_waiting','owner_wait_resumed','owner_wait_expired')", (owner["job_id"],))
    refs, entries, endings = {}, {}, []
    for event in events:
        data = waits._object(event["payload_json"])
        proof = data.get("owner")
        if not isinstance(proof, dict) or type(proof.get("generation")) is not int:
            raise JobError("owner_wait_evidence_corrupt")
        if proof["generation"] != owner["generation"]:
            continue
        if event["event_type"] == "owner_waiting":
            await _validate_entry(manager, tx, actor, owner, event, data)
            entries[event["event_id"]] = data
            for ref in data["model_exits"]:
                refs[ref["event_id"]] = ref
        else:
            endings.append((event, data))
        refs[event["event_id"]] = content_reference(event)
    resumed, expired = set(), set()
    for event, data in endings:
        identity = data.get("waiting_id")
        entry = entries.get(identity)
        if entry is None or data.get("waiting_hash") != _hash(entry) or _hash(data.get("owner")) != _hash(entry["owner"]):
            raise JobError("owner_wait_evidence_corrupt")
        if event["event_type"] == "owner_wait_expired":
            expected = {"version": 1, "owner": entry["owner"], "waiting_id": identity,
                "waiting_hash": _hash(entry), "deadline_at_ms": entry["deadline_at_ms"]}
            _fact(event, owner, "owner_wait_expired", _id(manager, "owner_wait_end_", identity), expected)
            if event["created_at"] < entry["deadline_at_ms"]:
                raise JobError("owner_wait_evidence_corrupt")
            expired.add(identity)
            continue
        if (set(data) != {"version", "waiting_id", "waiting_hash", "owner", "inputs"}
                or type(data["version"]) is not int or data["version"] != 1
                or not isinstance(data["inputs"], list) or not data["inputs"]
                or event["created_at"] >= entry["deadline_at_ms"]):
            raise JobError("owner_wait_evidence_corrupt")
        _fact(event, owner, "owner_wait_resumed", _id(manager, "owner_wait_resume_", identity), data)
        seen = set()
        for ref in data["inputs"]:
            if not isinstance(ref, dict) or not isinstance(ref.get("event_id"), str) or ref["event_id"] in seen:
                raise JobError("owner_wait_evidence_corrupt")
            seen.add(ref["event_id"])
            original = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (ref["event_id"],))
            request = waits._object(original["payload_json"]).get("request", {}) if original else {}
            if (original is None or _hash(content_reference(original)) != _hash(ref)
                    or original["event_type"] != "operation" or original["action"] != "nudge"
                    or original["job_id"] != owner["job_id"] or request.get("session_id") != entry["session_id"]
                    or _hash(request.get("target")) != _hash(entry["owner"])):
                raise JobError("owner_wait_evidence_corrupt")
            if success:
                await require_event_handled(manager, tx, original, session_id=entry["session_id"], owner_job_id=owner["job_id"])
            refs[ref["event_id"]] = ref
        resumed.add(identity)
    if resumed & expired or success and (set(entries) != resumed or owner["status"] == "waiting"):
        raise JobError("owner_wait_unresolved")
    return [refs[key] for key in sorted(refs)]


async def _pending(manager, tx, actor, owner, sid=None):
    """A consumed input may commit its wait only with the owner's result."""
    from . import wait_effects as effects
    from .wait_notifications import decision_id

    rows = await tx.fetchall(f"SELECT j.*,e.payload_json AS creation_json FROM {manager.jobs} j LEFT JOIN {manager.events} e ON e.job_id=j.job_id "
        "AND e.event_type='operation' AND e.action='create' WHERE j.control_owner_job_id=? AND j.control_owner_generation=? AND j.kind='timer'",
        (owner["job_id"], owner["generation"]))
    pending = []
    for row in rows:
        creation = waits._object(row["creation_json"])
        request = creation.get("request", {})
        if not isinstance(request, dict):
            raise JobError("wait_evidence_corrupt")
        if "wait_binding" not in creation and "managed_wait" not in request:
            continue  # An explicit legacy adapter has no invented wait policy.
        job, event, saved = await waits._read(manager, tx, actor, row["job_id"])
        if _hash(saved["binding"]["owner"]) != _hash(waits._proof(owner)):
            raise JobError("owner_wait_binding_changed")
        if job["outcome"] == "succeeded":
            # The canonical success validator checks resolution and consumption.
            from .wait_terminal import references
            await references(manager, tx, job)
            continue
        if job["outcome"] is not None or job["cancel_requested_at"] is not None:
            raise JobError("required_wait_cancelled")
        if sid is not None and saved["binding"]["session_id"] != sid:
            raise JobError("owner_wait_session_mismatch")
        if saved["receipt_ref"] is None:
            raise JobError("wait_receipt_missing")
        main = JobActor(actor.tenant_id, actor.principal_id, saved["binding"]["main_session_id"])
        due = waits.due_event_id(manager, job["job_id"], event["event_id"])
        decision = await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (decision_id(manager, due),))
        if decision is not None:
            _, _, _, accepted, data = await effects._decision(manager, tx, main, due)
            effect = await effects._effect(manager, tx, main, job, saved, accepted, data)
            if effect is not None and data["request"]["decision"] == "continue":
                op = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (waits._object(effect["payload_json"])["operation_id"],))
                if op is not None and op["delivery_status"] == "handled":
                    from .processing import require_event_handled
                    await require_event_handled(manager, tx, op, session_id=saved["binding"]["session_id"], owner_job_id=owner["job_id"])
                    continue
        pending.append((job, event, saved))
    return pending


async def require_time(manager, tx, owner):
    if owner["kind"] not in ("subagent", "workflow_node"):
        return
    actor = JobActor(owner["tenant_id"], owner["principal_id"], owner["created_by_session_id"])
    timers = await tx.fetchall(f"SELECT j.job_id,e.payload_json FROM {manager.jobs} j LEFT JOIN {manager.events} e ON e.job_id=j.job_id "
        "AND e.event_type='operation' AND e.action='create' WHERE j.control_owner_job_id=? AND j.control_owner_generation=? AND j.kind='timer'",
        (owner["job_id"], owner["generation"]))
    for timer in timers:
        creation = waits._object(timer["payload_json"])
        request = creation.get("request", {})
        if not isinstance(request, dict):
            raise JobError("wait_evidence_corrupt")
        if "wait_binding" not in creation and "managed_wait" not in request:
            continue
        _, _, saved = await waits._read(manager, tx, actor, timer["job_id"])
        if manager.now() >= saved["request"]["policy"]["deadline_at_ms"]:
            raise JobError("owner_wait_deadline_exceeded")


async def require_dispatch(manager, tx, row):
    """Check original wait policies again at actual paid request admission."""
    seen = set()
    for _ in range(64):
        if row is None or row["job_id"] in seen:
            raise JobError("invalid_control_owner")
        seen.add(row["job_id"])
        await require_time(manager, tx, row)
        if row["control_owner_job_id"] is None:
            return
        row = await manager._row(tx, row["control_owner_job_id"])
    raise JobError("job_depth_exceeded")


async def check_deadline(binding):
    manager = binding.runtime.manager
    async with manager._write() as tx:
        row = await manager._fence(tx, binding.actor, binding.lease)
        await require_time(manager, tx, row)


async def _quiescent(manager, tx, owner):
    from .execution_membership import read, scope_id, seal_id
    from .result_revisions import result_id

    queue, seen, refs = [(owner, None)], set(), {}
    for _ in range(64):
        if not queue:
            return [refs[k] for k in sorted(refs)]
        following = []
        for row, parent in queue:
            key = (row["job_id"], row["generation"])
            if key in seen or len(seen) >= 10000:
                raise JobError("owner_wait_membership_corrupt")
            seen.add(key)
            member = await read(manager, tx, row, row["generation"])
            scope = member["scope"]
            if (scope["tenant_id"], scope["principal_id"]) != (owner["tenant_id"], owner["principal_id"]):
                raise JobError("owner_wait_membership_corrupt")
            ids = [scope_id(manager, *key)]
            if parent is not None:
                if _hash(scope["owner"]) != _hash(parent) or not member["sealed"]:
                    raise JobError("owner_wait_resources_unsettled")
                ids.append(seal_id(manager, *key))
                if scope["kind"] == "send":
                    if member["model"] is None:
                        raise JobError("owner_wait_resources_unsettled")
                    ids.extend(await require_stopped(manager, tx, historical_row(row, scope), member["model"]["payload"]))
                    ids.append(member["model"]["event_id"])
                else:
                    result = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result'", (result_id(*key),))
                    if result is None or waits._object(result["payload_json"]).get("resource_state") != "stopped":
                        raise JobError("owner_wait_resources_unsettled")
                    ids.append(result["event_id"])
            claimed = set()
            for edge in member["members"]:
                lease = edge["payload"]["member"]
                claimed.add((lease["job_id"], lease["generation"]))
                child = await manager._row(tx, lease["job_id"])
                if child is None:
                    raise JobError("owner_wait_membership_corrupt")
                original = await read(manager, tx, child, lease["generation"])
                if _hash(original["scope"]["lease"]) != _hash(lease):
                    raise JobError("owner_wait_membership_corrupt")
                if child["generation"] == lease["generation"] and (child["resource_state"] != "stopped" or child["intake"] != "closed"):
                    raise JobError("owner_wait_resources_unsettled")
                following.append((historical_row(child, original["scope"]), scope["lease"]))
                ids.append(edge["event_id"])
            owned = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE control_owner_job_id=? AND control_owner_generation=?", key)
            for child in owned:
                if child["kind"] == "timer" and child["generation"] == 0:
                    from .wait_terminal import admission
                    await admission(manager, tx, JobActor(owner["tenant_id"], owner["principal_id"], owner["created_by_session_id"]), child["job_id"])
                elif (child["job_id"], child["generation"]) not in claimed:
                    raise JobError("owner_wait_resources_unsettled")
            for identity in ids:
                event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (identity,))
                if event is None:
                    raise JobError("owner_wait_evidence_missing")
                refs[identity] = content_reference(event)
        queue = following
    raise JobError("owner_wait_depth_exceeded")


async def _inputs(manager, tx, actor, owner, sid):
    from .wait_effects import validate_input

    rows = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='nudge' AND delivery_status<>'handled'", (owner["job_id"],))
    ready = []
    for event in rows:
        request = waits._object(event["payload_json"]).get("request", {})
        if request.get("session_id") != sid or _hash(request.get("target")) != _hash(waits._proof(owner)):
            continue
        if event["call_ref_json"] is not None:
            if event["receipt_ref_json"] is None:
                continue  # The original caller may still be saving its receipt.
            caller = JobActor(actor.tenant_id, actor.principal_id, event["caller_session_id"])
            call, receipt = waits._ref(waits._object(event["call_ref_json"])), waits._ref(waits._object(event["receipt_ref_json"]))
            await manager._validate_call(tx, caller, call, event["tool_call_id"])
            await manager._validate_ref(tx, caller, receipt, role="tool")
            if (receipt.tool_call_id != event["tool_call_id"] or receipt.send_index != call.send_index
                    or receipt.round_index != call.round_index or receipt.seq <= call.seq
                    or event["request_hash"] != _hash([request, call.to_dict()])):
                raise JobError("owner_wait_input_corrupt")
        await validate_input(manager, tx, JobActor(actor.tenant_id, actor.principal_id, sid), event)
        inbox = await tx.fetchone(f"SELECT * FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND item_id=?", (sid, "nudge:" + event["event_id"]))
        if (inbox is None or inbox["kind"] != "relay" or inbox["mode"] != "steer" or inbox["content"] != request.get("message")
                or inbox["status"] not in ("pending", "claimed", "delivered")):
            raise JobError("owner_wait_input_corrupt")
        ready.append((event, inbox["status"] == "delivered"))
    return ready


async def _enter(manager, actor, lease, raw):
    from .execution_membership import _lease
    _lease(asdict(lease))
    async with manager._write() as tx:
        owner = await manager._fence(tx, actor, lease)
        sid = raw.get("session_id")
        pending = await _pending(manager, tx, actor, owner, sid)
        if not pending:
            return None
        if not isinstance(sid, str) or not sid:
            raise JobError("owner_wait_session_required")
        refs = await _quiescent(manager, tx, owner)
        data = {"version": 1, "owner": asdict(lease), "session_id": sid, "raw": raw,
            "waits": [{"job_id": j["job_id"], "creation_id": e["event_id"], "creation_hash": e["request_hash"]} for j, e, _ in pending],
            "deadline_at_ms": min(saved["request"]["policy"]["deadline_at_ms"] for _, _, saved in pending), "model_exits": refs}
        identity = _id(manager, "owner_wait_", data)
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (identity,))
        if old is None:
            if owner["status"] == "waiting":
                raise JobError("owner_wait_phase_changed")
            await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (identity, owner["job_id"], "owner_waiting", owner["version"]+1, _hash(data), _json(data), "not_applicable", manager.now()))
            await tx.execute(f"UPDATE {manager.jobs} SET status='waiting',version=version+1,updated_at=? WHERE job_id=?", (manager.now(), owner["job_id"]))
        else:
            await _validate_entry(manager, tx, actor, owner, old, data)
            if owner["status"] != "waiting":
                raise JobError("owner_wait_already_resumed")
        return identity, data


async def _step(manager, actor, lease, identity, saved):
    async with manager._write() as tx:
        owner = await manager._authorize(tx, actor, lease.job_id)
        manager._check_proof(owner, lease)
        event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (identity,))
        _fact(event, owner, "owner_waiting", identity, saved)
        if manager.now() >= saved["deadline_at_ms"]:
            await _validate_entry(manager, tx, actor, owner, event, saved)
            if _hash(await _quiescent(manager, tx, owner)) != _hash(saved["model_exits"]):
                raise JobError("owner_wait_resources_changed")
            payload = {"version": 1, "owner": asdict(lease), "waiting_id": identity, "waiting_hash": _hash(saved), "deadline_at_ms": saved["deadline_at_ms"]}
            key = _id(manager, "owner_wait_end_", identity)
            old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
            if old is None:
                await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (key, owner["job_id"], "owner_wait_expired", owner["version"], _hash(payload), _json(payload), "not_applicable", manager.now()))
            else:
                _fact(old, owner, "owner_wait_expired", key, payload)
            return "expired"
        owner = await manager._fence(tx, actor, lease)
        ready = await _inputs(manager, tx, actor, owner, saved["session_id"])
        if not ready:
            return None
        # Full immutable and physical proofs are checked at a transition, not
        # repeatedly while polling an unchanged waiting controller.
        await _validate_entry(manager, tx, actor, owner, event, saved)
        if _hash(await _quiescent(manager, tx, owner)) != _hash(saved["model_exits"]):
            raise JobError("owner_wait_resources_changed")
        payload = {"version": 1, "waiting_id": identity, "waiting_hash": _hash(saved), "owner": asdict(lease),
            "inputs": [content_reference(e) for e, _ in ready]}
        key = _id(manager, "owner_wait_resume_", identity)
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
        if old is None:
            if owner["status"] != "waiting":
                raise JobError("owner_wait_phase_changed")
            await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (key, owner["job_id"], "owner_wait_resumed", owner["version"]+1, _hash(payload), _json(payload), "not_applicable", manager.now()))
            await tx.execute(f"UPDATE {manager.jobs} SET status='running',version=version+1,updated_at=? WHERE job_id=?", (manager.now(), owner["job_id"]))
        else:
            _fact(old, owner, "owner_wait_resumed", key, payload)
            if owner["status"] != "running":
                raise JobError("owner_wait_phase_changed")
        return "history" if any(delivered for _, delivered in ready) else "input"


async def wait_ready(loop, raw, stop_event, *, deadline):
    """Whole-controller wait; no model call or replacement owner is made here."""
    from power_loop.workflow.admission import idle

    from .cancellation import CancellationToken
    from .task_runtime import _OWNER

    runtime = getattr(loop, "task_runtime", None)
    binding = _OWNER.get()
    if runtime is None or binding is None or binding.runtime is not runtime or raw.get("status") != "completed":
        return None
    if binding.stop_renewal.is_set() or not any(local["binding"] is binding and local["task"] is asyncio.current_task()
            for local in runtime._local_executions.values()):
        raise JobError("owner_wait_executor_required")
    entered = await _enter(runtime.manager, binding.actor, binding.lease, raw)
    if entered is None:
        return None
    identity, saved = entered
    token = CancellationToken.from_any(stop_event)
    try:
        async with idle():
            while True:
                if token.is_cancelled():
                    raise _WaitCancelled()
                try:
                    state = await _step(runtime.manager, binding.actor, binding.lease, identity, saved)
                except JobError as exc:
                    if exc.code == "cancel_requested":
                        raise _WaitCancelled() from exc
                    raise
                if state == "expired" or asyncio.get_running_loop().time() >= deadline:
                    raise WaitDeadline("original continuation wait deadline reached")
                if state:
                    break
                await asyncio.sleep(min(.1, max(0, deadline - asyncio.get_running_loop().time())))
        # idle() may have queued for the original leaf slot. The earlier
        # accepted input does not extend its deadline while that slot is busy.
        await check_deadline(binding)
        return state
    except _WaitCancelled:
        return "cancelled"
