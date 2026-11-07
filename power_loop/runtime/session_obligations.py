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


def _handled(event):
    return event["delivery_status"] in ("handled", "suppressed")


async def _desired(manager, tx, row, *, verify=False):
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
            handled = _handled(current)
        if not handled:
            for session in routes:
                add(session, "abort", anchor)
    if not handled:
        for event in dispositions:
            for session in routes:
                add(session, "disposition", event)
    if (row["kind"] in ("subagent", "workflow")
            or row["kind"] in ("background_tool", "background_command") and row["session_id"] != row["created_by_session_id"]):
        for result in (event for event in events if event["event_type"] == "result"):
            if result["delivery_status"] in ("handled", "suppressed"):
                continue
            if not _handled(result):
                add(row["created_by_session_id"], "result", result)
    for event in events:
        kind = event["event_type"]
        recipient = event["caller_session_id"]
        if kind == "managed_effect_delivery" and current is not None and recipient:
            if not _handled(current):
                add(recipient, "managed", event)
        elif kind == "controller_stopped" and current is not None:
            if verify:
                from .controller_recovery import notice
                await notice(manager, tx, row)
            if not _handled(current):
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
            if _handled(event):
                continue
            if verify:
                from .wait_effects import validate_input
                actor = JobActor(row["tenant_id"], row["principal_id"], data["request"]["session_id"])
                await validate_input(manager, tx, actor, event)
            add(data["request"]["session_id"], "input", event)
    return entries


async def _replace(manager, tx, row, *, verify=False):
    entries = await _desired(manager, tx, row, verify=verify)
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
    for offset in range(0, len(routes), 100):
        part = routes[offset:offset+100]
        rows = await tx.fetchall(f"SELECT job_id FROM {manager.events} WHERE event_id IN (" +
            ",".join("?" for _ in part) + ")", tuple(part))
        jobs.update(row["job_id"] for row in rows)
    for job_id in sorted(jobs):
        row = await tx.fetchone(f"SELECT * FROM {manager.jobs} WHERE job_id=?", (job_id,))
        if row is not None:
            if row["supervision_status"] == "closed" and not (row["kind"] == "background_tool" and row["generation"] == 0):
                # Closure does not decide an outstanding control request. Its
                # own writer maintains that projection; retire other work here.
                await tx.execute(f"DELETE FROM {manager.prefix}session_obligations WHERE job_id=? AND kind<>'control_request'",
                    (job_id,))
            else:
                await _replace(manager, tx, row)
    table = manager.prefix+"session_obligations"
    for entry in sorted(recorder.obligation_close):
        await tx.execute(f"DELETE FROM {table} WHERE session_id=? AND kind=? AND source_id=? AND job_id=?", entry)
    for entry in sorted(recorder.obligation_open):
        await tx.execute(manager.db.dialect.insert_ignore(table, COLUMNS), entry)


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




class _RebuildConflict(Exception):
    pass


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
                raise _RebuildConflict()
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
            except _RebuildConflict:
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
