"""Committed session changes shared by the round's inbox and runtime readers."""
from __future__ import annotations

from contextvars import ContextVar

COLUMNS = ("inbox_version", "background_version", "runtime_version")
ACTIVE = ContextVar("power_loop_session_changes", default=None)


async def bump(store, tx, session_id, kind):
    # Acquire the session row before mutating its queue/runtime rows. The append
    # path takes these locks in the same order when it marks input delivered.
    column = kind + "_version"
    from power_loop.contracts.errors import SessionLeaseLost

    from . import send_reads
    sql, args, guarded = store._state_update(session_id, [f"{column}={column}+1"], ())
    changed = await tx.execute(sql, args)
    send_reads.lease_writer(store, session_id)
    if guarded and changed != 1:
        raise SessionLeaseLost(session_id)


async def migration_steps(tx, db, prefix):
    from .schema import _column_exists
    integer = "INTEGER" if db.dialect.name == "sqlite" else "BIGINT"
    missing = [column for column in COLUMNS
               if not await _column_exists(tx, db.dialect.name, prefix + "session_state", column)]
    steps = [f"ALTER TABLE {prefix}session_state ADD COLUMN {column} {integer} NOT NULL DEFAULT 0"
             for column in missing]
    # Maintenance imports existing obligations once. Live readers then use the
    # counters, with no old-format recognition or history scans.
    for column, table in zip(COLUMNS, ("follow_up_queue", "background_tasks", "session_runtime_state"), strict=True):
        steps.append(f"UPDATE {prefix}session_state SET {column}=1 WHERE {column}=0 AND EXISTS "
                     f"(SELECT 1 FROM {prefix}{table} c WHERE c.session_id={prefix}session_state.session_id)")
    return steps


class ChangeReader:
    def __init__(self, store, session_id):
        self.store, self.session_id = store, session_id
        self.versions = (0, 0, 0)
        self.model_versions = (0, 0, 0)
        self.inbox_seen, self.inbox_retry_at = 0, None
        self.background_seen, self.background = 0, []
        self.runtime = {}
        self.snapshot_seq = None

    async def refresh(self, *, force=False):
        snapshot = self.store._holding(self.session_id).change_snapshot
        if not force and snapshot is not None and snapshot[0] != self.snapshot_seq:
            self.snapshot_seq, self.versions = snapshot
            return
        row = await self.store._db.fetchone(
            f"SELECT {','.join(COLUMNS)} FROM {self.store.t.session_state} WHERE session_id=?", (self.session_id,))
        self.versions = tuple(row[column] for column in COLUMNS)
        if snapshot is not None:
            self.snapshot_seq = snapshot[0]

    def inbox_due(self, now):
        return (self.versions[0] != self.inbox_seen
                or self.inbox_retry_at is not None and now >= self.inbox_retry_at)


def reader(store, session_id):
    active = ACTIVE.get()
    return active if active is not None and active.store is store and active.session_id == session_id else None
