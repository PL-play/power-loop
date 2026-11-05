"""Pending-only discovery. Original task events remain the recovery authority."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .jobs import JobActor, JobError, _hash, verify_stored
from .result_revisions import result_id

EVENTS = frozenset({
    "tool_call_abort", "source_abort_delivery", "pending_call_disposition",
    "controller_review_handoff", "pending_owner_transfer", "managed_effect_delivery",
    "controller_stopped", "wait_notification", "result", "supervision_reopened",
    "input_disposition", "node_result_consumed", "wait_decision",
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
        if kind == "result":
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


async def _processing_handled(manager, tx, row, event, session_id, owner_job_id, clean):
    if clean is not None:
        return clean
    from .processing import require_event_handled
    try:
        await require_event_handled(manager, tx, event, session_id=session_id, owner_job_id=owner_job_id)
    except JobError:
        return False
    return True


async def _desired(manager, tx, row, *, verify=False, clean=None, processed_inputs=frozenset(), projected_events=frozenset()):
    operation_filter = "action IN ('steer','nudge')" if verify else "action='nudge'"
    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND "
        "(event_type IN ('tool_call_abort','source_abort_delivery','pending_call_disposition',"
        "'controller_review_handoff','pending_owner_transfer','managed_effect_delivery',"
        f"'controller_stopped','wait_notification','result','task_control_request') OR (event_type='operation' AND {operation_filter}))",
        (row["job_id"],))
    if verify:
        from .store.operation_migration import verify_operation_records
        await verify_operation_records(tx, manager.prefix,
            records=[event for event in events if event['event_type'] == 'operation'])
    events = [{**event, "delivery_status": "handled"} if event["event_id"] in projected_events else event
        for event in events]
    if not events:
        return set()
    from .control_requests import request_obligations
    entries = await request_obligations(manager, tx, row, events, verify=verify)
    if row["supervision_status"] == "closed":
        if verify:
            from .finalization import closed
            await closed(manager, tx, row)
        # A generation-zero managed action can be physically closed without
        # its negative outcome having been reviewed by the original main.
        if row["kind"] != "background_tool" or row["generation"] != 0:
            return entries
    found = {event["event_id"]: event for event in events}
    current = found.get(result_id(row["job_id"], row["generation"]))
    if verify and row["kind"] in ("subagent", "workflow", "workflow_node", "background_tool") and row["result_status"] not in ("none", "suppressed"):
        if current is None or _hash(json.loads(current["payload_json"])["result"]) != _hash(json.loads(row["result_json"])):
            raise JobError("session_obligation_original_result_missing_or_changed")

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
        if current is not None:
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
        if not await _processing_handled(manager, tx, row, current,
                row["created_by_session_id"], row["control_owner_job_id"], clean):
            add(row["created_by_session_id"], "result", current)
    for event in events:
        kind = event["event_type"]
        recipient = event["caller_session_id"]
        if kind == "managed_effect_delivery" and current is not None and recipient:
            from .managed_delivery import read
            delivery = await read(manager, tx, row, current) if verify or clean is None else None
            if not await _handled(manager, tx, row, current, delivery, clean=clean):
                add(recipient, "managed", event)
        elif kind == "controller_stopped" and current is not None:
            if verify:
                from .controller_recovery import notice
                await notice(manager, tx, row)
            if not await _processing_handled(manager, tx, row, current, recipient, None, clean):
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
            if event["event_id"] in processed_inputs or await _processing_handled(manager, tx, row, event,
                    data["request"]["session_id"], row["job_id"], None):
                continue
            if verify:
                from .wait_effects import validate_input
                actor = JobActor(row["tenant_id"], row["principal_id"], data["request"]["session_id"])
                await validate_input(manager, tx, actor, event)
            add(data["request"]["session_id"], "input", event)
    return entries


async def _replace(manager, tx, row, *, verify=False, clean=None, processed_inputs=frozenset()):
    entries = await _desired(manager, tx, row, verify=verify, clean=clean, processed_inputs=processed_inputs)
    return await _write_entries(manager, tx, row, entries)


async def _write_entries(manager, tx, row, entries):
    table = manager.prefix + "session_obligations"
    await tx.execute(f"DELETE FROM {table} WHERE job_id=?", (row["job_id"],))
    for entry in sorted(entries):
        await tx.execute(manager.db.dialect.insert_ignore(table, COLUMNS), entry)
    return entries


async def refresh(manager, tx, recorder):
    jobs = set(recorder.obligation_jobs)
    routes = sorted(recorder.obligation_routes)
    clean = {job_id: False for job_id in recorder.obligation_processed}
    clean.update(dict.fromkeys(recorder.obligation_handled, True))
    for offset in range(0, len(routes), 100):
        part = routes[offset:offset+100]
        rows = await tx.fetchall(f"SELECT job_id FROM {manager.events} WHERE event_id IN (" +
            ",".join("?" for _ in part) + ")", tuple(part))
        jobs.update(row["job_id"] for row in rows)
    for job_id in sorted(jobs):
        row = await tx.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (job_id,))
        if row is not None:
            await _replace(manager, tx, row, clean=clean.get(job_id))
    table = manager.prefix+"session_obligations"
    for entry in sorted(recorder.obligation_close):
        await tx.execute(f"DELETE FROM {table} WHERE session_id=? AND kind=? AND source_id=? AND job_id=?", entry)
    for entry in sorted(recorder.obligation_open):
        await tx.execute(manager.db.dialect.insert_ignore(table, COLUMNS), entry)


class _ProjectionConflict(Exception):
    pass


@dataclass(frozen=True)
class _Projection:
    row: dict
    expected_version: int
    changed: bool
    result_changed: bool
    events: tuple
    entries: frozenset
    sessions: frozenset


async def refresh_closed_decisions(manager, decisions, *, verify=False):
    """Publish target projections only after the actual handler close commits."""
    from .execution_membership import _trusted
    from .session_controls import ControlReadView

    rows, missing = [], []
    for key in sorted(decisions):
        known = _trusted(manager, "_processing_originals", key)
        if known is None:
            missing.append(key)
        else:
            rows.append(known)
    reader = ControlReadView(manager.db, manager)
    for offset in range(0, len(missing), 100):
        part = missing[offset:offset+100]
        rows.extend(await reader.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='processing_decision' "
            "AND event_id IN ("+",".join("?" for _ in part)+")", tuple(part)))
    pending = [event for event in rows if decisions[event["event_id"]] == json.loads(event["payload_json"])["response_ref"]]
    while pending:
        reader = ControlReadView(manager.db, manager)
        roots = {}
        for event in pending:
            payload = json.loads(event["payload_json"])
            target = await manager._row(reader, payload["event"]["job_id"])
            if target["generation"] != payload["target"]["generation"]:
                continue
            handler = event.get("_handler_snapshot") or await manager._row(reader, event["job_id"])
            actor = JobActor(target["tenant_id"], target["principal_id"], event["caller_session_id"])
            from .message_ref import integer_id
            from .processing import _decision_target, require_event_handled
            target, source = await _decision_target(manager, reader, actor, handler, payload["event"],
                integer_id(payload["response_ref"]["seq"], "seq"))
            if verify:
                await _desired(manager, reader, target, verify=True)
                if payload["target"] != {key: target[key] for key in ("job_id", "root_job_id", "generation")}:
                    raise JobError("processing_target_changed")
                await require_event_handled(manager, reader, source, session_id=actor.session_id,
                    owner_job_id=handler["control_owner_job_id"], allow_suppressed=target["generation"] == 0)
            roots.setdefault(target["root_job_id"], []).append((event, payload, target, source))
        pending = []
        for root_id in sorted(roots):
            projections = await _prepare_closed_tree(manager, reader, roots[root_id])
            try:
                await _refresh_closed_tree(manager, root_id, projections)
            except _ProjectionConflict:
                # A real concurrent stop/resume changed a target after discovery.
                # Roll back this tree and rediscover without retaining any lock.
                pending.extend(entry[0] for entry in roots[root_id])


async def _prepare_closed_tree(manager, reader, entries):
    jobs = {}
    for event, payload, row, source in entries:
        jobs.setdefault(row["job_id"], (row, []))[1].append((event, payload, source))
    projections = []
    for row, sources in jobs.values():
        inputs = {source["event_id"] for _, payload, source in sources
            if payload["event"].get("event_type", "result") == "operation"}
        observed_result = any(source["event_id"] == result_id(row["job_id"], row["generation"])
            for _, _, source in sources)
        negative = row["kind"] == "background_tool" and row["generation"] == 0
        result_changed = observed_result and not negative and row["result_status"] != "handled"
        events = [source for _, _, source in sources if source["delivery_status"] != "handled" and not negative]
        changed = bool(events) or result_changed
        now = manager.now()
        expected_version = row["version"]
        if changed:
            row = {**row, "version": row["version"]+1, "updated_at": now,
                **({"result_status": "handled"} if result_changed else {})}
        projected = frozenset(source["event_id"] for source in events)
        entries = await _desired(manager, reader, row, clean=True if observed_result else None,
            processed_inputs=inputs, projected_events=projected)
        projections.append(_Projection(row, expected_version, changed, result_changed,
            tuple(sorted(projected)), frozenset(entries),
            frozenset(event["caller_session_id"] for event, _, _ in sources)))
    return projections


async def _refresh_closed_tree(manager, root_id, projections):
    async with manager._write() as tx:
        for projection in projections:
            row = projection.row
            version = "version+1" if projection.changed else "version"
            fields = ",result_status='handled'" if projection.result_changed else ""
            fields += ",updated_at=?" if projection.changed else ""
            params = ((row["updated_at"],) if projection.changed else ())+(row["job_id"], row["generation"], projection.expected_version)
            count = await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET version={version}{fields} "
                "WHERE job_id=? AND generation=? AND version=?", params)
            if count != 1:
                raise _ProjectionConflict()
            for event_id in projection.events:
                await tx.update_event(row["job_id"], f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=?",
                    (event_id,), obligations=False)
            tx.sessions.update(projection.sessions)
            await _write_entries(manager, tx, row, projection.entries)


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
    """Offline schema migration only, with all writers fenced by its caller."""
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


@dataclass(frozen=True)
class _RebuildJob:
    row: dict
    sessions: frozenset


async def _prepare_rebuild(manager, *, root_id=None):
    """Validate all originals before publishing any repair; hold no writer."""
    from .session_controls import ControlReadView

    async with manager.db.read_snapshot() as snapshot:
        reader = ControlReadView(snapshot, manager)
        jobs, expected, cursor = {}, {}, ""
        scope = " AND root_job_id=?" if root_id is not None else ""
        while True:
            rows = await reader.fetchall(f"SELECT * FROM {manager.jobs} WHERE job_id>?{scope} ORDER BY job_id LIMIT 100",
                (cursor, *((root_id,) if root_id is not None else ())))
            if not rows:
                break
            for row in rows:
                jobs[row["job_id"]] = row
                reader.remember_row(manager.jobs, "job_id", row["job_id"], row)
                expected[row["job_id"]] = frozenset(await _desired(manager, reader, row, verify=True))
            cursor = rows[-1]["job_id"]
        table = manager.prefix+"session_obligations"
        if root_id is None:
            indexed = await reader.fetchall(f"SELECT {','.join(COLUMNS)} FROM {table}")
        else:
            indexed = await reader.fetchall(f"SELECT {','.join('o.'+key for key in COLUMNS)} FROM {table} o "
                f"JOIN {manager.jobs} j ON j.job_id=o.job_id WHERE j.root_job_id=?", (root_id,))
        actual = {}
        for entry in indexed:
            actual.setdefault(entry["job_id"], set()).add(tuple(entry[key] for key in COLUMNS))
        changed = [row for job_id, row in jobs.items() if expected[job_id] != actual.get(job_id, set())]
        plans = {}
        for row in changed:
            entries = expected[row["job_id"]]
            recipients = frozenset(entry[0] for entry in entries | actual.get(row["job_id"], set()))
            plans.setdefault(row["root_job_id"], []).append(_RebuildJob(row, recipients))
        orphans = {job: frozenset(entry[0] for entry in entries) for job, entries in actual.items() if job not in jobs}
        return plans, orphans


async def _rebuild_tree(manager, root_id, plan):
    async with manager._write() as tx:
        for item in plan:
            row = item.row
            changed = await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET version=version "
                "WHERE job_id=? AND root_job_id=? AND generation=? AND version=?",
                (row["job_id"], root_id, row["generation"], row["version"]))
            if changed != 1:
                raise _ProjectionConflict()
            # Routing may change without advancing the job version. Rebuild
            # from current originals while this existing tree lock is held.
            current = await manager._row(tx, row['job_id'])
            entries = await _desired(manager, tx, current, verify=True)
            await _write_entries(manager, tx, current, entries)
            tx.sessions.update(item.sessions | {entry[0] for entry in entries})
    return frozenset(tx.sessions)


async def _remove_orphan_entries(manager, job_id, sessions):
    # No task original exists for these derived rows. Do not create a fake root
    # or remove the index of a task admitted after the snapshot.
    async with manager._write() as tx:
        await tx.discard_orphan_obligations(job_id, sessions)


@verify_stored
async def rebuild(manager):
    plans, orphans = await _prepare_rebuild(manager)
    for root_id in sorted(plans):
        plan = plans[root_id]
        while plan:
            try:
                await _rebuild_tree(manager, root_id, plan)
                break
            except _ProjectionConflict:
                # A concurrent writer changed a generation, job or route after
                # discovery. Revalidate this tree outside the rolled-back writer.
                current, _ = await _prepare_rebuild(manager, root_id=root_id)
                plan = current.get(root_id)
    for job_id in sorted(orphans):
        await _remove_orphan_entries(manager, job_id, orphans[job_id])


@verify_stored
async def verify(manager, actor: JobActor):
    from .session_controls import ControlReadView

    async with manager.db.read_snapshot() as snapshot:
        tx = ControlReadView(snapshot, manager)
        rows = await tx.fetchall(f"SELECT * FROM {manager.jobs} WHERE tenant_id=? AND principal_id=? ORDER BY job_id",
                                (actor.tenant_id, actor.principal_id))
        expected = set()
        for row in rows:
            expected.update(entry for entry in await _desired(manager, tx, row, verify=True) if entry[0] == actor.session_id)
        actual = await tx.fetchall(f"SELECT {','.join(COLUMNS)} FROM {manager.prefix}session_obligations WHERE session_id=?",
                                  (actor.session_id,))
        if expected != {tuple(row[key] for key in COLUMNS) for row in actual}:
            raise JobError("session_obligation_projection_mismatch")
