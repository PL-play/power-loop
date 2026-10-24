"""Immutable defer cycles on one original timer; no execution or model dispatch.

History validation walks forward once. Already validated due evidence is passed
explicitly to the notification/decision readers so prior cycles cannot recurse.
"""
from __future__ import annotations

from . import timer_waits as waits
from .jobs import JobError, _hash, _json

MAX_REVIEWS = 256


def review_id(manager, effect_id):
    return "wait_review_" + _hash([manager.store.store_id, effect_id])[:48]


def _payload(manager, saved, previous, decision, effect, cycle):
    data = waits._object(decision["payload_json"])
    next_id = "wait_due_" + _hash([manager.store.store_id, saved["job_id"], saved["creation_operation_id"],
        previous["event_id"], effect["event_id"], cycle])[:48]
    return {"version": 1, "cycle": cycle, "creation_id": saved["creation_operation_id"],
        "previous_due_id": previous["event_id"], "previous_due_hash": previous["request_hash"],
        "decision_id": decision["event_id"], "decision_hash": decision["request_hash"],
        "effect_id": effect["event_id"], "effect_hash": effect["request_hash"],
        "review_at_ms": data["request"]["next_review_at_ms"], "next_due_id": next_id}


def _check_review(manager, job, saved, previous, decision, effect, cycle, event):
    expected = _payload(manager, saved, previous, decision, effect, cycle)
    review_at = expected["review_at_ms"]
    if (event is None or event["event_id"] != review_id(manager, effect["event_id"])
            or event["job_id"] != job["job_id"] or event["event_type"] != "wait_review"
            or event["caller_session_id"] != saved["binding"]["main_session_id"]
            or event["delivery_status"] != "not_applicable" or event["request_hash"] != _hash(expected)
            or _hash(waits._object(event["payload_json"])) != _hash(expected)
            or type(review_at) is not int or not decision["created_at"] < review_at < saved["request"]["policy"]["deadline_at_ms"]
            or type(event["created_at"]) is not int or event["created_at"] < decision["created_at"]
            or type(event["job_version"]) is not int or not 0 < event["job_version"] <= job["version"]):
        raise JobError("wait_review_evidence_corrupt")
    return expected


def _check_due(job, event, identity, payload, due_at):
    if (event is None or event["event_id"] != identity or event["event_type"] != "wait_due"
            or event["job_id"] != job["job_id"] or event["request_hash"] != _hash(payload)
            or _hash(waits._object(event["payload_json"])) != _hash(payload)
            or event["delivery_status"] not in ("pending_delivery", "delivered", "handled")
            or type(event["created_at"]) is not int or event["created_at"] < due_at
            or type(event["job_version"]) is not int or not 0 < event["job_version"] <= job["version"]):
        raise JobError("wait_due_evidence_corrupt")


async def history(manager, tx, actor, job, creation, saved):
    """Validate all admitted cycles without current execution authority."""
    from . import wait_effects as effects
    from . import wait_notifications as notices
    from .jobs import JobActor

    main = JobActor(actor.tenant_id, actor.principal_id, saved["binding"]["main_session_id"])
    rows = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='wait_review' LIMIT ?",
        (job["job_id"], MAX_REVIEWS + 1))
    if len(rows) > MAX_REVIEWS:
        raise JobError("wait_review_limit_exceeded")
    reviews = {}
    for row in rows:
        data = waits._object(row["payload_json"])
        number = data.get("cycle")
        if type(number) is not int or not 1 <= number <= MAX_REVIEWS or number in reviews:
            raise JobError("wait_review_chain_corrupt")
        reviews[number] = row
    if set(reviews) != set(range(1, len(rows) + 1)):
        raise JobError("wait_review_chain_corrupt")
    identity = waits.due_event_id(manager, job["job_id"], creation["event_id"])
    due_at, payload = saved["request"]["request"]["due_at_ms"], {"version": 1, "wait": saved}
    event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (identity,))
    expected_dues, events, refs, pending_defer = {}, {}, set(), False
    for cycle in range(len(rows) + 1):
        if event is not None:
            _check_due(job, event, identity, payload, due_at)
            events[identity] = event
            expected_dues[identity] = event
            refs.add(identity)
        if cycle == len(rows):
            # A handled defer must have transferred responsibility atomically.
            if event is not None:
                decision = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (notices.decision_id(manager, identity),))
                if decision:
                    data = waits._object(decision["payload_json"])
                    if not isinstance(data.get("request"), dict):
                        raise JobError("wait_decision_evidence_corrupt")
                    if data["request"].get("decision") == "defer":
                        if decision["delivery_status"] == "handled":
                            raise JobError("wait_review_responsibility_missing")
                        pending_defer = True
            break
        if event is None:
            raise JobError("wait_review_previous_due_missing")
        bundle = await notices.notification(manager, tx, main, identity, _due=(job, event, saved))
        _, _, _, decision, data = await effects._decision(manager, tx, main, identity, _notification=bundle)
        effect = await effects._effect(manager, tx, main, job, saved, decision, data, _review=False)
        if (effect is None or data["request"]["decision"] != "defer"
                or effect["delivery_status"] != "handled" or decision["delivery_status"] != "handled"):
            raise JobError("wait_review_predecessor_unsettled")
        row = reviews[cycle + 1]
        record = _check_review(manager, job, saved, event, decision, effect, cycle + 1, row)
        refs.update((row["event_id"], decision["event_id"], effect["event_id"], data["operation_id"],
            notices.notification_id(manager, identity), data["observation"]["observation_id"]))
        identity, due_at = record["next_due_id"], record["review_at_ms"]
        payload = {"version": 2, "wait": saved, "review": {"event_id": row["event_id"],
            "request_hash": row["request_hash"], "cycle": cycle + 1}}
        event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (identity,))
    all_due = await tx.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='wait_due'", (job["job_id"],))
    if {r["event_id"] for r in all_due} != set(expected_dues):
        raise JobError("wait_review_chain_corrupt")
    return {"cycle": len(rows), "event_id": identity, "event": event, "payload": payload,
        "due_at_ms": due_at, "events": events, "references": sorted(refs), "pending_defer": pending_defer}


async def require_link(manager, tx, job, saved, decision, effect):
    """Check the next durable responsibility without recursively reading history."""
    data = waits._object(decision["payload_json"])
    previous = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (data["event_id"],))
    row = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (review_id(manager, effect["event_id"]),))
    if previous is None or row is None:
        raise JobError("wait_review_responsibility_missing")
    raw = waits._object(row["payload_json"])
    cycle = raw.get("cycle")
    if type(cycle) is not int or not 1 <= cycle <= MAX_REVIEWS:
        raise JobError("wait_review_chain_corrupt")
    return _check_review(manager, job, saved, previous, decision, effect, cycle, row)


async def schedule(manager, tx, actor, job, saved, decision, effect):
    """Commit next responsibility and hand off the defer in this transaction."""
    _, creation, _ = await waits._read(manager, tx, actor, job["job_id"])
    state = await history(manager, tx, actor, job, creation, saved)
    data = waits._object(decision["payload_json"])
    if state["event"] is None or state["event_id"] != data["event_id"]:
        raise JobError("wait_due_state_changed")
    if state["cycle"] >= MAX_REVIEWS:
        raise JobError("wait_review_limit_exceeded")
    payload = _payload(manager, saved, state["event"], decision, effect, state["cycle"] + 1)
    await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,caller_session_id,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (review_id(manager, effect["event_id"]), job["job_id"], "wait_review", job["version"] + 1, actor.session_id,
            _hash(payload), _json(payload), "not_applicable", manager.now()))
    await tx.execute(f"UPDATE {manager.jobs} SET version=version+1,updated_at=? WHERE job_id=?", (manager.now(), job["job_id"]))
    await tx.execute(f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id IN (?,?)", (decision["event_id"], effect["event_id"]))


async def drive(loop, actor, job_id):
    """One repeatable SDK driver step; the host authorizes any actual main wake.

    It either retains the next due, queues one main notification, or applies an
    already receipted main decision. It never starts a model or replacement job.
    """
    from . import wait_effects as effects
    from . import wait_notifications as notices

    runtime = getattr(loop, "task_runtime", None)
    if runtime is None or runtime.actor(actor.session_id) != actor:
        raise JobError("task_runtime_required")
    manager = runtime.manager
    saved = await waits.read_wait(manager, actor, job_id)
    if actor.session_id != saved["binding"]["main_session_id"]:
        raise JobError("access_denied")
    due = await waits.prepare_due(manager, actor, job_id)
    if due["status"] == "armed":
        return due
    async with manager._write() as tx:
        decision = await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (notices.decision_id(manager, due["event_id"]),))
    if decision:
        return await effects.apply(manager, actor, due["event_id"])
    return await notices.enqueue_notification(loop, actor, due["event_id"])
