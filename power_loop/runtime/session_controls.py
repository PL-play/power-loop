"""Explicit control write intents and transaction-local primary-key reads.

Original events are authoritative. Writers declare event, job, executor and
recipient changes; commit refreshes pending-only discovery and session versions.
No SQL text is inspected to infer a write's meaning or invalidate cached rows.
"""
from __future__ import annotations

OBLIGATION_EVENTS = frozenset({
    "tool_call_abort", "source_abort_delivery", "pending_call_disposition",
    "controller_review_handoff", "pending_owner_transfer", "managed_effect_delivery",
    "controller_stopped", "wait_notification", "result", "supervision_reopened",
})
GLOBAL = "*"


class ControlReadView:
    """A bounded discovery's primary-key view, with no mutation port or lock.

    Discovery is evidence for a later conditional publication, never authority
    to write. Its caller must fence again when publishing its own observations.
    """

    def __init__(self, reader, manager):
        self._tx = reader
        self.dialect = manager.db.dialect
        self._memo_sql = {f"SELECT * FROM {table} WHERE {key}=?": table for table, key in (
            (manager.jobs, "job_id"), (manager.prefix + "task_executors", "executor_key"))}
        self._memo: dict[tuple[str, tuple], dict | None] = {}

    def forget_reads(self):
        self._memo.clear()

    def remember_row(self, table, key, value, row):
        self._memo[f"SELECT * FROM {table} WHERE {key}=?", (value,)] = dict(row)

    async def fetchone(self, sql, params=()):
        if sql in self._memo_sql:
            key = (sql, tuple(params))
            if key not in self._memo:
                row = await self._tx.fetchone(sql, params)
                self._memo[key] = None if row is None else dict(row)
            row = self._memo[key]
            return None if row is None else dict(row)
        return await self._tx.fetchone(sql, params)

    async def fetchall(self, sql, params=()):
        return await self._tx.fetchall(sql, params)


class ControlTransaction(ControlReadView):
    """One transaction's declared writes, read view and derived-state work."""

    def __init__(self, tx, manager):
        super().__init__(tx, manager)
        self._events = manager.events
        self._jobs = manager.jobs
        self._executors = manager.prefix + "task_executors"
        self._obligations = manager.prefix + "session_obligations"
        self._dialect = manager.db.dialect
        self.jobs: set[str] = set()
        self.sessions: set[str] = set()
        self.touched_jobs: set[str] = set()
        self.writes = 0
        self.obligation_jobs: set[str] = set()
        self.obligation_routes: set[str] = set()
        self.obligation_decisions: dict[str, dict | None] = {}
        self.obligation_processed: set[str] = set()
        self.obligation_open: set[tuple] = set()
        self.obligation_close: set[tuple] = set()

    def __getattr__(self, name):
        return getattr(self._tx, name)

    def touch_job(self, job_id, *, obligations=False, notify=False):
        self.touched_jobs.add(job_id)
        self._memo.pop((f"SELECT * FROM {self._jobs} WHERE job_id=?", (job_id,)), None)
        if obligations:
            self.obligation_jobs.add(job_id)
        if notify:
            self.jobs.add(job_id)

    async def write_job(self, job_id, sql, params=(), *, obligations=False):
        self.touch_job(job_id, obligations=obligations)
        return await self._execute(sql, params)

    async def write_executor(self, executor_key, sql, params=()):
        self._memo.pop((f"SELECT * FROM {self._executors} WHERE executor_key=?", (executor_key,)), None)
        return await self._execute(sql, params)

    async def append_event(self, columns, values, *, ignore=False):
        """Append a declared original; its meaning is supplied as fields."""
        self.note_event(dict(zip(columns, values, strict=True)))
        sql = (self._dialect.insert_ignore(self._events, columns) if ignore else
               f"INSERT INTO {self._events} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})")
        return await self._execute(sql, values)

    async def append_event_if_current(self, event, row):
        """Append mechanical evidence without changing a superseding owner."""
        self.note_event(event)
        columns = tuple(event)
        sql = f"INSERT INTO {self._events} ({','.join(columns)}) SELECT " + ','.join('?' for _ in columns)
        sql += f" FROM {self._jobs} WHERE job_id=? AND root_job_id=? AND generation=? AND version=?"
        return await self._execute(sql, (*event.values(), row["job_id"], row["root_job_id"], row["generation"], row["version"]))

    def note_event(self, event):
        self.touched_jobs.add(event["job_id"])
        from .session_obligations import note
        note(self, event)
        if event["event_type"] in OBLIGATION_EVENTS:
            self.jobs.add(event["job_id"])
            recipient = event.get("caller_session_id")
            if recipient is not None:
                self.sessions.add(recipient)

    async def project_events(self, columns, rows, sql, params):
        """Declare the fields of a journal projection's batched event write."""
        for row in rows:
            self.note_event(dict(zip(columns, row, strict=True)))
        return await self._execute(sql, params)

    async def update_event(self, job_id, sql, params=(), *, obligations=True, notify=False):
        """Declare an original event's state change without re-reading it."""
        self.touched_jobs.add(job_id)
        if obligations:
            self.obligation_jobs.add(job_id)
        if notify:
            self.jobs.add(job_id)
        return await self._execute(sql, params)

    async def set_event_recipient(self, event_id: str, session_id: str):
        changed = await self._execute(
            f"UPDATE {self._events} SET caller_session_id=? WHERE event_id=?", (session_id, event_id))
        if changed:
            self.sessions.add(session_id)
            self.obligation_routes.add(event_id)
        return changed

    async def discard_orphan_obligations(self, job_id, sessions):
        changed = await self._execute(f"DELETE FROM {self._obligations} WHERE job_id=? AND "
            f"NOT EXISTS (SELECT 1 FROM {self._jobs} WHERE job_id=?)", (job_id, job_id))
        if changed:
            self.sessions.update(sessions)
        return changed

    def open_obligation(self, *, session_id, kind, source_id, job_id, opened_version):
        self.touched_jobs.add(job_id)
        self.sessions.add(session_id)
        self.obligation_open.add((session_id, kind, source_id, job_id, opened_version))

    def close_obligation(self, *, session_id, kind, source_id, job_id):
        self.touched_jobs.add(job_id)
        self.sessions.add(session_id)
        self.obligation_close.add((session_id, kind, source_id, job_id))

    async def execute(self, sql, params=()):
        """Execute a declared mutation; control writers use the typed methods."""
        return await self._execute(sql, params)

    async def _execute(self, sql, params=()):
        self.writes += 1
        return await self._tx.execute(sql, params)

    async def control_statement(self, sql):
        """Savepoint operations are transaction control, not business writes."""
        return await self._tx.execute(sql)

    async def execute_flag(self, sql, params=()):
        self.writes += 1
        self.forget_reads()
        return await self._tx.execute_flag(sql, params)

async def bump(manager, tx, recorder):
    sessions = set(recorder.sessions)
    jobs = sorted(recorder.jobs)
    for offset in range(0, len(jobs), 100):
        part = jobs[offset:offset + 100]
        rows = await tx.fetchall(f"SELECT session_id,created_by_session_id FROM {manager.jobs} WHERE job_id IN (" +
                                 ",".join("?" for _ in part) + ")", tuple(part))
        for row in rows:
            sessions.update(value for value in (row["session_id"], row["created_by_session_id"]) if value)
    table = manager.prefix + "session_controls"
    for session_id in sorted(sessions):
        if not await tx.execute(f"UPDATE {table} SET version=version+1 WHERE session_id=?", (session_id,)):
            await tx.execute(manager.db.dialect.insert_ignore(table, ("session_id", "version")), (session_id, 1))


async def version(manager, reader, session_id):
    rows = await reader.fetchall(f"SELECT session_id,version FROM {manager.prefix}session_controls "
                                 "WHERE session_id IN (?,?)", (session_id, GLOBAL))
    found = {row["session_id"]: int(row["version"]) for row in rows}
    return found.get(session_id, 0), found.get(GLOBAL, 0)


def is_clean(manager, session_id, seen):
    return manager._trust_own_writes.get() and manager.__dict__.get("_clean_sessions", {}).get(session_id) == seen


def mark_clean(manager, session_id, seen):
    cache = manager.__dict__.setdefault("_clean_sessions", {})
    if len(cache) >= 4096 and session_id not in cache:
        cache.pop(next(iter(cache)))
    cache[session_id] = seen


def forget(manager, session_id):
    manager.__dict__.get("_clean_sessions", {}).pop(session_id, None)
