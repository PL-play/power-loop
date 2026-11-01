"""Explicit, temporary local changes prepared from one original-record view.

The plan is not a ledger or an executor. Publication applies its own tree's
conditional projections and originals together; it never interprets SQL.
"""
from __future__ import annotations

from .jobs import JobError


class PlanChanged(JobError):
    def __init__(self):
        super().__init__("finalization_execution_changed")


class Preparation:
    def __init__(self, manager, reader):
        self.manager, self.reader = manager, reader
        self.jobs = {}
        self.events = {}
        self.statuses = {}
        self.queues = {}
        self.signals = set()
        self.inputs = {}
        self.terminals = {}
        self.dispositions = {}

    def change_job(self, row, **fields):
        key = row["job_id"]
        previous = self.jobs.get(key)
        base = previous[0] if previous is not None else dict(row)
        projected = {**row, **fields, "version": row["version"]+1, "updated_at": self.manager.now()}
        self.jobs[key] = base, projected
        self.reader.remember_row(self.manager.jobs, "job_id", key, projected)
        return projected

    def append(self, event):
        self.events[event["event_id"]] = event
        return event

    def handled(self, job_id, event_id):
        self.statuses[event_id] = job_id, "handled"

    def withdraw(self, queue):
        self.queues[queue["id"]] = dict(queue)
        return {**queue, "status": "void", "claim_token": None}

    async def apply(self, tx):
        for base, row in self.jobs.values():
            fields = [key for key in row if row[key] != base[key]]
            changed = await tx.write_job(row["job_id"], f"UPDATE {self.manager.jobs} SET "
                + ','.join(key+'=?' for key in fields)
                + " WHERE job_id=? AND root_job_id=? AND generation=? AND version=?",
                (*[row[key] for key in fields], row["job_id"], base["root_job_id"], base["generation"], base["version"]),
                obligations=True)
            if changed != 1:
                raise PlanChanged()
        for queue in self.queues.values():
            token = queue["claim_token"]
            condition = "claim_token IS NULL" if token is None else "claim_token=?"
            params = (queue["id"], queue["status"])+(() if token is None else (token,))
            changed = await tx.execute(f"UPDATE {self.manager.store.t.follow_up_queue} SET status='void',claim_token=NULL "
                "WHERE id=? AND status=? AND "+condition, params)
            if changed != 1:
                raise PlanChanged()
        for sid in sorted(self.signals):
            await self.manager.store.signal_changes(tx, sid, "inbox")
        for event in self.events.values():
            await tx.append_event(tuple(event), tuple(event.values()))
        for event_id, (job_id, status) in self.statuses.items():
            await tx.update_event(job_id, f"UPDATE {self.manager.events} SET delivery_status=? WHERE event_id=?", (status, event_id))


async def execute(manager, reader, operation):
    """The same preparation algorithm also serves an existing local writer."""
    current = manager._finalization_preparation.get()
    if current is not None:
        return await operation(current)
    prepared = Preparation(manager, reader)
    token = manager._finalization_preparation.set(prepared)
    try:
        result = await operation(prepared)
        await prepared.apply(reader)
        return result
    finally:
        manager._finalization_preparation.reset(token)
