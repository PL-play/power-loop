"""Repair projections or retire an expired owner without replaying its actions."""
from __future__ import annotations

import json

from .control_runtime import project_events
from .execution_journal import ExecutionJournal
from .execution_state import ExecutionIdentity
from .jobs import JobError


async def inspect(manager, actor, job_id, *, generation, repair=False, retire_expired=False):
    from .managed_completion import _host_only
    _host_only()
    if type(generation) is not int or generation < 1 or type(repair) is not bool or type(retire_expired) is not bool:
        raise JobError("invalid_execution_recovery")
    await manager._authorize(manager.db, actor, job_id)
    head = await manager.db.fetchone(f"SELECT identity_json FROM {manager.prefix}execution_checkpoints WHERE job_id=? AND generation=?",
                                     (job_id, generation))
    if head is None:
        raise JobError("execution_checkpoint_missing")
    try:
        identity = ExecutionIdentity(**json.loads(head["identity_json"]))
    except (ValueError, TypeError) as exc:
        raise JobError("execution_checkpoint_corrupt") from exc
    if (identity.job_id, identity.generation, identity.tenant_id, identity.principal_id) != (
            job_id, generation, actor.tenant_id, actor.principal_id):
        raise JobError("access_denied")
    async def projector(tx, identity, records):
        await project_events(manager, tx, identity, records)
    journal = ExecutionJournal(manager.store, projector=projector)
    recovered = await journal.recover(identity, repair=repair)
    retired = False
    if retire_expired and not recovered["summary"]["closed"]:
        async with manager._write() as tx:
            row = await manager._authorize(tx, actor, job_id)
            if row["generation"] < generation:
                raise JobError("execution_checkpoint_corrupt")
            if row["generation"] == generation and row["lease_until_ms"] is not None and row["lease_until_ms"] > manager.now():
                raise JobError("execution_owner_still_live")
            head = await tx.fetchone(f"SELECT * FROM {manager.prefix}execution_checkpoints WHERE job_id=? AND generation=?",
                                     (job_id, generation))
            if head["batch_no"] != recovered["batch_no"] or head["chain_hash"] != recovered["chain_hash"]:
                raise JobError("execution_checkpoint_conflict")
            summary = {**recovered["summary"], "closed": True, "needs_reconciliation": True, "unknown_suffix": True}
            records = [{"kind": "lifecycle", "key": "retired_owner", "value": {
                "closed": True, "unknown_suffix": True, "execution_authority": False,
                "replay_allowed": False, "capacity_released": False}}]
            await journal.append(identity, recovered["batch_no"] + 1, recovered["chain_hash"], records, summary, transaction=tx)
            if row["generation"] == generation:
                # Owner retirement is not physical exit, tool success, or a
                # financial refund. Preserve those unresolved responsibilities.
                await tx.write_job(job_id, f"UPDATE {manager.jobs} SET intake='closed',resource_state='unknown',version=version+1,updated_at=? "
                                 "WHERE job_id=? AND generation=?", (manager.now(), job_id, generation))
            manager._control_retirements.get().add((job_id, generation))
        retired = True
        recovered = await journal.recover(identity)
    return dict(job_id=job_id, generation=generation, batch_no=recovered["batch_no"],
        summary=recovered["summary"], needs_reconciliation=recovered["needs_reconciliation"],
        repaired=repair, owner_retired=retired, execution_authority=False, replay_allowed=False, capacity_released=False)
