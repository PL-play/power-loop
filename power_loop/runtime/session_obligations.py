"""Pending-only discovery. Original task events remain the recovery authority."""
from __future__ import annotations

import json

from .jobs import JobActor, JobError, _hash, verify_stored
from .result_revisions import result_id

EVENTS = frozenset({
    "tool_call_abort", "source_abort_delivery", "pending_call_disposition",
    "controller_review_handoff", "pending_owner_transfer", "managed_effect_delivery",
    "controller_stopped", "wait_notification", "result", "supervision_reopened",
    "processing_decision", "input_disposition", "node_result_consumed", "wait_decision",
    "managed_effect_tool_receipt",
})
COLUMNS = ("session_id", "kind", "source_id", "job_id", "opened_version")
_INPUT_MODES = frozenset({"steer", "continue", "restore_input"})


def _queued_input(event, data):
    # Current wait continuations carry the original decision/effect handoff,
    # rather than the user-nudge mode. Generic caller operations are not input.
    return data.get("mode") in _INPUT_MODES or (
        event.get("call_ref_json") is None and "handoff" in data.get("request", {}))


def note(recorder, event):
    kind = event.get("event_type")
    input_write = (kind == "operation" and event.get("action") == "nudge"
                   and _queued_input(event, json.loads(event["payload_json"])))
    if kind in EVENTS or input_write:
        recorder.obligation_jobs.add(event["job_id"])
        if kind == "processing_decision":
            recorder.obligation_processed.add(event["job_id"])
    elif kind == "model_closed":
        data = json.loads(event["payload_json"])
        evidence = data.get("evidence", {})
        response = evidence.get("last_assistant_ref") if evidence.get("exit") == "completed" and not evidence.get("new_input_after_pass") else None
        recorder.obligation_decisions.update((key, response) for key in evidence.get("processing_decisions", ()))
    elif kind == "supervision_closed":
        data = json.loads(event["payload_json"])
        if any(ref["event_type"] in EVENTS for ref in data["proof"]["references"]):
            recorder.obligation_jobs.add(event["job_id"])


async def _handled(manager, tx, row, result, notice, *, aborted=False, clean=None):
    # A handled flag without the original handler's clean model exit is not an
    # acknowledgement. Keep its barrier until that close commits.
    if row["result_status"] not in ("handled", "suppressed"):
        return False
    if clean is not None:
        return clean
    if aborted:
        from .abort_outcomes import handled_references
    else:
        from .managed_delivery import handled_references
    try:
        await handled_references(manager, tx, row, result, notice)
    except JobError:
        return False
    return True


async def _desired(manager, tx, row, *, verify=False, clean=None):
    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND "
        "(event_type IN ('tool_call_abort','source_abort_delivery','pending_call_disposition',"
        "'controller_review_handoff','pending_owner_transfer','managed_effect_delivery',"
        "'controller_stopped','wait_notification','result') OR (event_type='operation' AND action='nudge'))",
        (row["job_id"],))
    if not events:
        return set()
    if row["supervision_status"] == "closed":
        if verify:
            from .finalization import closed
            await closed(manager, tx, row)
        # A generation-zero managed action can be physically closed without
        # its negative outcome having been reviewed by the original main.
        if row["kind"] != "background_tool" or row["generation"] != 0:
            return set()
    found = {event["event_id"]: event for event in events}
    current = found.get(result_id(row["job_id"], row["generation"]))
    if verify and row["kind"] in ("subagent", "workflow", "workflow_node", "background_tool") and row["result_status"] not in ("none", "suppressed"):
        if current is None or _hash(json.loads(current["payload_json"])["result"]) != _hash(json.loads(row["result_json"])):
            raise JobError("session_obligation_original_result_missing_or_changed")
    entries = set()

    def add(session, kind, event):
        if session:
            entries.add((session, kind, event["event_id"], row["job_id"], int(event["job_version"])))

    from .abort_outcomes import FIELD, MODE
    dispositions = [event for event in events if event["event_type"] == "pending_call_disposition"]
    routes = {row["session_id"]}
    routes.update(event["caller_session_id"] for event in events
                  if event["event_type"] in ("controller_review_handoff", "pending_owner_transfer"))
    aborts = [event for event in events if event["event_type"] in ("tool_call_abort", "source_abort_delivery")]
    negative = (row["kind"] == "send" and
        (FIELD in json.loads(row["result_json"] or "{}") or bool(aborts) and row["result_status"] != "none"
         or any(json.loads(event["payload_json"]).get("disposition") == MODE for event in dispositions)))
    handled = False
    if negative:
        if verify:
            from .abort_outcomes import notice, result_event
            current = await result_event(manager, tx, row)
            await notice(manager, tx, row, current)
        anchor = current or (aborts or dispositions)[0]
        if current is not None and row["result_status"] == "handled":
            from .abort_outcomes import notice
            delivery = await notice(manager, tx, row, current) if clean is None else None
            handled = await _handled(manager, tx, row, current, delivery, aborted=True, clean=clean)
        if not handled:
            for session in routes:
                add(session, "abort", anchor)
    if not handled:
        for event in dispositions:
            for session in routes:
                add(session, "disposition", event)
    if current is not None and row["kind"] in ("subagent", "workflow"):
        if row["result_status"] != "handled" and current["delivery_status"] != "handled":
            add(row["created_by_session_id"], "result", current)
    for event in events:
        kind = event["event_type"]
        recipient = event["caller_session_id"]
        if kind == "managed_effect_delivery" and current is not None and recipient:
            from .managed_delivery import read
            delivery = await read(manager, tx, row, current) if verify or clean is None and row["result_status"] in ("handled", "suppressed") else None
            if not await _handled(manager, tx, row, current, delivery, clean=clean):
                add(recipient, "managed", event)
        elif kind == "controller_stopped" and current is not None and row["result_status"] != "handled":
            if verify:
                from .controller_recovery import notice
                await notice(manager, tx, row)
            add(recipient, "controller", event)
        elif kind == "wait_notification" and event["delivery_status"] != "handled":
            if verify:
                from .wait_notifications import notification
                actor = JobActor(row["tenant_id"], row["principal_id"], recipient)
                await notification(manager, tx, actor, json.loads(event["payload_json"])["due_event_id"])
            add(recipient, "wait", event)
        elif (kind == "operation" and event["delivery_status"] == "pending_delivery"
              and row["kind"] in ("subagent", "workflow_node")):
            data = json.loads(event["payload_json"])
            if verify:
                from .wait_effects import validate_input
                actor = JobActor(row["tenant_id"], row["principal_id"], data["request"]["session_id"])
                await validate_input(manager, tx, actor, event)
            add(data["request"]["session_id"], "input", event)
    return entries


async def _replace(manager, tx, row, *, verify=False, clean=None):
    entries = await _desired(manager, tx, row, verify=verify, clean=clean)
    table = manager.prefix + "session_obligations"
    await tx.execute(f"DELETE FROM {table} WHERE job_id=?", (row["job_id"],))
    for entry in sorted(entries):
        await tx.execute(manager.db.dialect.insert_ignore(table, COLUMNS), entry)
    return entries


async def refresh(manager, tx, recorder):
    jobs = set(recorder.obligation_jobs)
    routes = sorted(recorder.obligation_routes)
    clean = {job_id: False for job_id in recorder.obligation_processed}
    for offset in range(0, len(routes), 100):
        part = routes[offset:offset+100]
        rows = await tx.fetchall(f"SELECT job_id FROM {manager.events} WHERE event_id IN (" +
            ",".join("?" for _ in part) + ")", tuple(part))
        jobs.update(row["job_id"] for row in rows)
    for job_id in sorted(jobs):
        row = await tx.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (job_id,))
        if row is not None:
            await _replace(manager, tx, row, clean=clean.get(job_id))


async def refresh_closed_decisions(manager, decisions, *, verify=False):
    """After the original close commits, discover and refresh one tree at a time.

    An interrupted refresh leaves a conservative pending entry. The immutable
    decision and actual close are sufficient for cold retry; no execution is
    admitted and no additional original record is necessary.
    """
    from .session_controls import ControlReadView

    reader = ControlReadView(manager.db, manager)
    roots = {}
    keys = sorted(decisions)
    for offset in range(0, len(keys), 100):
        part = keys[offset:offset+100]
        rows = await reader.fetchall(f"SELECT e.event_id,e.job_id,e.caller_session_id,e.payload_json,j.root_job_id "
            f"FROM {manager.events} e JOIN {manager.jobs} j ON j.job_id=e.job_id "
            "WHERE e.event_type='processing_decision' AND e.event_id IN (" +
            ",".join("?" for _ in part) + ")", tuple(part))
        for event in rows:
            payload = json.loads(event["payload_json"])
            if verify:
                await _desired(manager, reader, await manager._row(reader, event["job_id"]), verify=True)
            roots.setdefault(event["root_job_id"], []).append((event, payload,
                decisions[event["event_id"]] == payload["response_ref"]))
    for root_id in sorted(roots):
        await _refresh_closed_tree(manager, root_id, roots[root_id])


async def _refresh_closed_tree(manager, root_id, entries):
    async with manager._write() as tx:
        jobs = {}
        for event, payload, clean in entries:
            row = await manager._row(tx, event["job_id"])
            source = payload["event"]
            # A target can legitimately advance after the handler close and
            # before maintenance. Its new generation owns its own projection.
            if source.get("event_type", "result") != "operation" and source["event_id"] != result_id(row["job_id"], row["generation"]):
                continue
            jobs[row["job_id"]] = row, clean
            tx.sessions.add(event["caller_session_id"])
        for job_id, (row, clean) in jobs.items():
            tx.touch_job(job_id)
            await _replace(manager, tx, row, clean=clean)


@verify_stored
async def reconcile_closed(manager, actor, job_id):
    """Cold, mechanical retry of projections from an actual stored model exit."""
    from .session_controls import ControlReadView

    reader = ControlReadView(manager.db, manager)
    row = await manager._authorize(reader, actor, job_id)
    event = await reader.fetchone(f"SELECT payload_json FROM {manager.events} "
        "WHERE event_id=? AND job_id=? AND event_type='model_closed'", ("model_closed_"+job_id, job_id))
    if event is None:
        return False
    data = json.loads(event["payload_json"])
    if data["generation"] != row["generation"]:
        raise JobError("stale_execution")
    evidence = data["evidence"]
    response = evidence.get("last_assistant_ref") if evidence.get("exit") == "completed" and not evidence.get("new_input_after_pass") else None
    await refresh_closed_decisions(manager,
        dict.fromkeys(evidence.get("processing_decisions", ()), response), verify=True)
    return True


async def sources(manager, tx, actor, *kinds):
    return await tx.fetchall(f"SELECT DISTINCT j.* FROM {manager.prefix}session_obligations o "
        f"JOIN {manager.jobs} j ON j.job_id=o.job_id WHERE o.session_id=? AND o.kind IN (" +
        ",".join("?" for _ in kinds) + ") AND j.tenant_id=? AND j.principal_id=? ORDER BY j.job_id LIMIT 1001",
        (actor.session_id, *kinds, actor.tenant_id, actor.principal_id))


async def event_rows(manager, tx, actor, kind, *, job_id=None, limit=1001):
    guard = " AND o.job_id=?" if job_id is not None else ""
    bound = "" if limit is None else f" LIMIT {limit}"
    return await tx.fetchall(f"SELECT e.* FROM {manager.prefix}session_obligations o "
        f"JOIN {manager.events} e ON e.event_id=o.source_id JOIN {manager.jobs} j ON j.job_id=o.job_id "
        f"WHERE o.session_id=? AND o.kind=? AND j.tenant_id=? AND j.principal_id=?{guard} ORDER BY o.source_id{bound}",
        (actor.session_id, kind, actor.tenant_id, actor.principal_id, *((job_id,) if job_id is not None else ())))


async def rebuild_tx(manager, tx):
    """Explicit maintenance/recovery only, with all writers fenced by its caller."""
    old = await tx.fetchall(f"SELECT DISTINCT session_id FROM {manager.prefix}session_obligations")
    sessions = {row["session_id"] for row in old}
    await tx.execute(f"DELETE FROM {manager.prefix}session_obligations")
    cursor = ""
    while True:
        rows = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE job_id>? ORDER BY job_id LIMIT 100", (cursor,))
        if not rows:
            return sessions
        for row in rows:
            entries = await _replace(manager, tx, row, verify=True)
            sessions.update(entry[0] for entry in entries)
        cursor = rows[-1]["job_id"]


@verify_stored
async def rebuild(manager):
    async with manager._write() as tx:
        tx.sessions.update(await rebuild_tx(manager, tx))


@verify_stored
async def verify(manager, actor: JobActor):
    async with manager._write() as tx:
        rows = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? ORDER BY job_id",
                                (actor.tenant_id, actor.principal_id))
        expected = set()
        for row in rows:
            expected.update(entry for entry in await _desired(manager, tx, row, verify=True) if entry[0] == actor.session_id)
        actual = await tx.fetchall(f"SELECT {','.join(COLUMNS)} FROM {manager.prefix}session_obligations WHERE session_id=?",
                                  (actor.session_id,))
        if expected != {tuple(row[key] for key in COLUMNS) for row in actual}:
            raise JobError("session_obligation_projection_mismatch")
