"""Persisted user interactions; answers resolve only their original tool call."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from power_loop.agent.runtime_store import InputRecord, now_ms
from power_loop.agent.runtime_types import (
    Command,
    InputRef,
    RuntimeDescriptor,
    RuntimeRequestError,
    TargetRef,
    canonical,
    plain,
)


async def save_wait(start, tx, binding, evidence):
    from .control_runtime import flush_binding, forget_completed
    port, ctx, manager = start.worker.inputs, start.context, binding.runtime.manager
    await flush_binding(binding, close=True, transaction=tx)
    state = await port.db.dialect.lock_state(tx, port.state, ctx.session_id)
    pending = json.loads(state["pending_json"])
    prior = start.record.meta.get("interaction") or {}
    usage = dict(prior.get("usage") or {})
    for key, value in evidence.get("usage", {}).items():
        usage[key] = usage.get(key, 0) + value
    pending["runtime_wait"] = {"job_ref": plain(ctx.job_ref), "model_lease": asdict(binding.lease),
        "send_index": start.send_index, "rounds": prior.get("rounds", 0) + evidence.get("rounds", 0),
        "usage": usage, "evidence": evidence, "input_ref": plain(ctx.input_ref)}
    await tx.execute(f"UPDATE {port.state} SET pending_json=?,runtime_version=runtime_version+1 WHERE session_id=?",
                     (canonical(pending), ctx.session_id))
    for job_id in {binding.lease.job_id, ctx.job_ref.job_id}:
        await tx.write_job(job_id, f"UPDATE {manager.jobs} SET status='waiting',resource_state='stopped',"
            "version=version+1 WHERE job_id=? AND outcome IS NULL", (job_id,))
    manager.after_outcome(lambda: forget_completed(binding), lambda: None)


async def answer(client, ref, *, interaction_id, wait_round, value, message_id, actor):
    from .jobs import JobManager
    port = client.inputs
    job = await port.resolve_job(port.db, ref, actor)
    if ref.generation is None or ref.generation <= 0:
        raise RuntimeRequestError("stale_generation")
    target = TargetRef(ref.store_id, job["session_id"], ref.job_id)
    item_id = "answer_" + hashlib.sha256(interaction_id.encode()).hexdigest()[:48]
    answer_ref = InputRef(ref.store_id, target.session_id, item_id)
    identity = {"id": interaction_id, "wait_round": wait_round, "value": plain(value), "job_ref": plain(ref)}
    canonical(identity)
    manager = JobManager(port.store)
    async with manager.joined_write() as tx:
        await port.authorize(tx, target, actor)
        state = await port.db.dialect.lock_state(tx, port.state, target.session_id)
        old = await port.input(tx, answer_ref)
        if old:
            if old.meta["interaction"]["answer"] != identity:
                raise RuntimeRequestError("interaction_conflict")
            return port.receipt(old)
        job = await manager._row(tx, ref.job_id)
        if job["generation"] != ref.generation:
            raise RuntimeRequestError("stale_generation")
        if job["cancel_requested_at"] is not None:
            raise RuntimeRequestError("target_cancelled")
        await manager._supervision_barriers(tx, job)
        pending = json.loads(state["pending_json"] or "{}")
        waiting = pending.get("runtime_wait")
        interactions = pending.get("pending_interactions", [])
        matches = [item for item in interactions if item["interaction_id"] == interaction_id]
        if (job["status"] != "waiting" or waiting is None or waiting["job_ref"] != plain(ref)
                or pending.get("round_index") != wait_round or len(matches) != 1):
            raise RuntimeRequestError("interaction_expired")
        interaction = matches[0]
        expires = interaction.get("metadata", {}).get("expires_at_ms")
        if expires is not None and expires <= now_ms():
            raise RuntimeRequestError("interaction_expired")
        original = await port.input(tx, InputRef(**waiting["input_ref"]))
        # The accepted answer and its tool response share the state lock and
        # transaction; another node cannot answer the same question differently.
        receipt = await port.accept_input(tx, Command(item_id, target, "answer", {"body": ""}, "control",
            expected_job_generation=ref.generation), actor,
            descriptor=RuntimeDescriptor(**original.meta["descriptor"]))
        seq, timestamp = state["next_seq"], now_ms()
        await tx.execute(f"INSERT INTO {port.store.t.messages} "
            "(session_id,seq,role,tool_call_id,name,content,state,round_index,send_index,meta_json,created_at) "
            "VALUES (?,?,'tool',?,?,?,'active',?,?,?,?)", (target.session_id, seq,
            interaction["tool_call_id"], interaction["tool_name"], value if isinstance(value, str) else canonical(value),
            wait_round, waiting["send_index"], canonical({"interaction_id": interaction_id}), timestamp))
        remaining = [call for call in pending.get("tool_calls", []) if call["id"] != interaction["tool_call_id"]]
        next_pending = ({**pending, "tool_calls": remaining, "tool_call_ids": [c["id"] for c in remaining],
            "pending_interactions": [item for item in interactions if item != interaction]} if remaining else None)
        await tx.execute(f"UPDATE {port.state} SET next_seq=next_seq+1,pending_json=?,runtime_version=runtime_version+1 "
            "WHERE session_id=?", (canonical(next_pending) if next_pending else None, target.session_id))
        accepted = await port.input(tx, answer_ref)
        meta = {**accepted.meta, "options": original.meta["options"], "interaction": {
            **waiting, "answer": identity, "answer_seq": seq, "request_id": message_id}}
        await tx.execute(f"UPDATE {port.q} SET meta_json=? WHERE session_id=? AND item_id=?",
                         (canonical(meta), target.session_id, item_id))
        receipt = port.receipt(InputRecord(answer_ref, accepted.row, meta), new=True)
    await client._notify(target, item_id, answer_ref, ["work", "activation"])
    return receipt


async def claim_answer(start, runtime, tx, row):
    from .jobs import JobLease
    from .task_runtime import Execution
    manager = runtime.manager
    if row["generation"] != start.record.row["expected_job_generation"]:
        raise RuntimeRequestError("stale_generation")
    if row["status"] != "waiting":
        raise RuntimeRequestError("interaction_expired")
    await manager._supervision_barriers(tx, row)
    waiting = start.record.meta["interaction"]
    previous = JobLease(**waiting["model_lease"])
    # Close a child's earlier physical model phase after its interaction has a
    # real tool response. A root send remains the original logical result owner.
    await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET status='running',resource_state='running',"
        "version=version+1 WHERE job_id=? AND generation=? AND version=?",
        (row["job_id"], row["generation"], row["version"]))
    if previous.job_id != row["job_id"]:
        await tx.write_job(previous.job_id, f"UPDATE {manager.jobs} SET resource_state='running',version=version+1 WHERE job_id=?",
                           (previous.job_id,))
        await manager.close_model_execution(runtime.actor(row["session_id"]), previous, evidence=waiting["evidence"])
    lease = JobLease(row["job_id"], row["executor_id"], row["boot_id"], row["generation"])
    return Execution(runtime, runtime.actor(row["session_id"]), lease)
