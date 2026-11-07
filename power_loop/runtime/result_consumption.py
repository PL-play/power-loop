"""Actual input consumption within one execution; no durable handler proof.

Original result/input events are the status authority. Fault recovery does not
reconstruct model rounds or replay unknown actions to repair this marker.
"""
from __future__ import annotations

import json

from .jobs import JobError
from .result_revisions import result_id


async def observe(binding, messages, *, round_index, send_index):
    manager, actor = binding.runtime.manager, binding.actor
    visible = {m["seq"]: m for m in messages}
    ids = list(dict.fromkeys(key for m in messages for key in m["item_ids"]
        if key not in binding.observed_inputs))
    if not ids:
        return
    from .session_controls import ControlReadView
    tx = ControlReadView(manager.db, manager)
    source = binding.resource_evidence["_control_source"]
    waits = []
    for offset in range(0, len(ids), 100):
        part = ids[offset:offset+100]
        rows = await tx.fetchall(f"SELECT * FROM {manager.store.t.follow_up_queue} "
            "WHERE session_id=? AND status='delivered' AND item_id IN (" + ",".join("?" for _ in part) + ")",
            (actor.session_id, *part))
        for inbox in rows:
            if inbox["delivered_seq"] not in visible:
                continue
            meta = json.loads(inbox["meta_json"] or "{}")
            reference = meta.get("result_ref")
            key = reference["result_id"] if reference else inbox["item_id"].removeprefix("nudge:")
            event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (key,))
            binding.observed_inputs.add(inbox["item_id"])
            if event is None:
                continue  # Ordinary append/steer input has no task result to consume.
            if event["event_type"] == "wait_notification":
                waits.append(event)
                binding.observed_inputs.discard(inbox["item_id"])
                continue
            if event["event_type"] in {"managed_effect_delivery", "source_abort_delivery", "controller_result_delivery"}:
                target = await manager._authorize(tx, actor, event["job_id"])
                if event["caller_session_id"] != actor.session_id:
                    raise JobError("access_denied")
                notice = json.loads(event["payload_json"])
                original_id = notice["result_id"] if event["event_type"] == "managed_effect_delivery" else notice["result"]["event_id"]
                event = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (original_id,))
            if event["event_type"] == "operation" and event["action"] == "nudge":
                from .node_inputs import input_target
                from .wait_effects import validate_input
                target = await input_target(manager, tx, actor, source, event)
                await validate_input(manager, tx, actor, event)
                request = json.loads(event["payload_json"])["request"]
                if request["session_id"] != actor.session_id or source["control_owner_job_id"] != target["job_id"]:
                    raise JobError("input_binding_mismatch")
                generation = request["target"]["generation"]
            elif event["event_type"] == "result":
                target = await manager._authorize(tx, actor, event["job_id"])
                generation = json.loads(event["payload_json"]).get("generation", target["generation"])
                if reference and (reference["job_id"] != target["job_id"] or reference["generation"] != generation
                        or target["created_by_session_id"] != actor.session_id):
                    raise JobError("result_binding_mismatch")
            else:
                continue
            if event["delivery_status"] != "handled":
                if len(binding.consumed_inputs) >= 1000 and event["event_id"] not in binding.consumed_inputs:
                    raise JobError("consumed_input_limit")
                binding.consumed_inputs[event["event_id"]] = (target["job_id"], target["root_job_id"], generation,
                    event["event_type"])
    if waits:
        # These observations authorize the real wait decision tool. They are
        # not consumption acknowledgements and have no handler-exit dependency.
        from .wait_notifications import prepare_request_observations, record_request_observations
        _, observations = await prepare_request_observations(manager, tx, actor, binding.lease,
            source=source, round_index=round_index, send_index=send_index, messages=messages, notices=waits)
        if observations:
            async with manager._write() as writer:
                current = await manager._fence(writer, actor, binding.lease)
                await record_request_observations(manager, writer, actor, binding.lease,
                    source=current, observations=observations)


async def finish(binding):
    """Mark only references actually used by this successful execution.

    Each task root commits separately. A failed/unknown commit can leave a
    partial marker; no later proof-repair pass claims otherwise.
    """
    manager = binding.runtime.manager
    roots = {}
    for event_id, (job_id, root_id, generation, kind) in binding.consumed_inputs.items():
        roots.setdefault(root_id, []).append((event_id, job_id, generation, kind))
    for entries in roots.values():
        async with manager._write() as tx:
            for event_id, job_id, generation, kind in entries:
                changed = await tx.update_event(job_id,
                    f"UPDATE {manager.events} SET delivery_status='handled' WHERE event_id=? AND job_id=? "
                    "AND delivery_status<>'handled'", (event_id, job_id), obligations=False)
                if changed:
                    if kind == "result":
                        await tx.write_job(job_id, f"UPDATE {manager.jobs} SET result_status='handled',version=version+1 "
                            "WHERE job_id=? AND generation=?", (job_id, generation))
                    await tx.execute(f"DELETE FROM {manager.prefix}session_obligations WHERE job_id=? AND source_id=?",
                        (job_id, event_id))
                    if kind == "result":
                        await tx.execute(f"DELETE FROM {manager.prefix}session_obligations WHERE job_id=? "
                            "AND kind IN ('managed','abort','controller','disposition') "
                            f"AND EXISTS (SELECT 1 FROM {manager.jobs} WHERE job_id=? AND generation=?)",
                            (job_id, job_id, generation))
    return tuple(dict.fromkeys(value[0] for value in binding.consumed_inputs.values()))


async def require_handled(manager, tx, child):
    from .jobs import JobError

    if child['kind'] in ('subagent', 'workflow_node', 'workflow'):
        from .controller_recovery import handled_references, read
        if await read(manager, tx, child) is not None:
            await handled_references(manager, tx, child)
            return

    result = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND event_type='result'", (result_id(child["job_id"], child["generation"]),))
    if result is None:
        raise JobError("required_result_unhandled")
    await require_event_handled(manager, tx, result, session_id=child["created_by_session_id"], owner_job_id=child["control_owner_job_id"])


async def require_inputs_handled(manager, tx, job):
    from .jobs import JobError, _hash
    events = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='operation' AND action='nudge'", (job["job_id"],))
    for event in events:
        try:
            sid = job["session_id"]
            if job["kind"] == "workflow_node":
                from .node_inputs import descriptor
                request = json.loads(event["payload_json"])["request"]
                sid = request["session_id"]
                generation = request["target"]["generation"]
                if isinstance(generation, bool) or not isinstance(generation, int) or not 1 <= generation <= job["generation"]:
                    raise JobError("input_binding_mismatch")
                if _hash(request.get("node")) != _hash(await descriptor(manager, tx, {**job, "generation": generation}, sid,
                        historical=generation < job["generation"] or job["resource_state"] == "stopped")):
                    raise JobError("input_binding_mismatch")
            await require_event_handled(manager, tx, event, session_id=sid, owner_job_id=job["job_id"])
        except JobError as exc:
            raise JobError("required_input_unhandled") from exc


async def require_event_handled(manager, tx, event, *, session_id, owner_job_id, allow_suppressed=False):
    if event.get("action") == "nudge":
        request = json.loads(event["payload_json"])["request"]
        if request["session_id"] != session_id or request["target"]["job_id"] != owner_job_id:
            raise JobError("required_input_unhandled")
        from .input_dispositions import read_tx
        if event["delivery_status"] != "handled" and await read_tx(manager, tx, event) is not None:
            return
    if event["delivery_status"] != "handled" and not (allow_suppressed and event["delivery_status"] == "suppressed"):
        raise JobError("required_result_unhandled")
