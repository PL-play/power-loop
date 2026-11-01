"""Per-session control version: O(1) discovery of prior-send control obligations.

Every committed control transaction that inserts an event able to add work for
a session's discovery scans (abort reviews, managed deliveries, child results,
controller stops, wait notifications, reopened supervision) bumps that
session's version in the same transaction. The recording happens at the
transaction layer, so individual writers cannot forget it.

A live send compares the version with the last version this process fully
scanned and found clean; only a changed version runs the detailed scans.
Writes that only resolve obligations do not need a bump. Events remain the
authority; recovery and audit always scan.
"""
from __future__ import annotations

import re

OBLIGATION_EVENTS = frozenset({
    "tool_call_abort", "source_abort_delivery", "pending_call_disposition",
    "controller_review_handoff", "pending_owner_transfer", "managed_effect_delivery",
    "controller_stopped", "wait_notification", "result", "supervision_reopened",
})
GLOBAL = "*"
_INSERT = re.compile(r"^\s*INSERT\s+(?:IGNORE\s+)?INTO\s+(\S+)\s*\(([^)]*)\)\s*VALUES\s*", re.I)


class RecordingTransaction:
    """Delegating control transaction.

    Notes sessions touched by obligation events, and memoizes primary-key row
    reads inside this one transaction: a write to the same table drops that
    table's memo, so a read never returns something this transaction changed.
    """

    def __init__(self, tx, events_table, memo_tables=()):
        self._tx = tx
        self._events = events_table
        self._memo_sql = {f"SELECT * FROM {table} WHERE {key}=?": table for table, key in memo_tables}
        self._memo: dict[tuple[str, tuple], dict | None] = {}
        self.jobs: set[str] = set()
        self.sessions: set[str] = set()
        self.writes = 0
        self.obligation_jobs: set[str] = set()
        self.obligation_routes: set[str] = set()
        self.obligation_decisions: dict[str, dict | None] = {}
        self.obligation_processed: set[str] = set()

    def __getattr__(self, name):
        return getattr(self._tx, name)

    def forget_reads(self):
        self._memo = {}

    def remember_row(self, table, key, value, row):
        """Seed the memo with a row this transaction just wrote in full."""
        self._memo[f"SELECT * FROM {table} WHERE {key}=?", (value,)] = dict(row)

    async def set_event_recipient(self, event_id: str, session_id: str):
        """Route a notice and invalidate its recipient in this same transaction."""
        changed = await self.execute(
            f"UPDATE {self._events} SET caller_session_id=? WHERE event_id=?", (session_id, event_id))
        if changed:
            self.sessions.add(session_id)
            self.obligation_routes.add(event_id)
        return changed

    def _written(self, sql):
        head = sql.lstrip()[:7].upper()
        if self._memo and head.startswith(("UPDATE", "INSERT", "DELETE", "REPLACE")):
            touched = {table for table in self._memo_sql.values() if re.search(rf"\b{re.escape(table)}\b", sql)}
            if touched:
                self._memo = {key: row for key, row in self._memo.items() if self._memo_sql[key[0]] not in touched}

    async def execute(self, sql, params=()):
        if not sql.lstrip()[:9].upper().startswith(("SELECT", "SAVEPOINT", "RELEASE", "ROLLBACK")):
            self.writes += 1
        if self._events in sql:
            self._observe(sql, tuple(params))
        self._written(sql)
        return await self._tx.execute(sql, params)

    async def execute_flag(self, sql, params=()):
        self.writes += 1
        self.forget_reads()
        return await self._tx.execute_flag(sql, params)

    async def fetchone(self, sql, params=()):
        if sql in self._memo_sql:
            key = (sql, tuple(params))
            if key not in self._memo:
                row = await self._tx.fetchone(sql, params)
                self._memo[key] = None if row is None else dict(row)
            row = self._memo[key]
            return None if row is None else dict(row)
        self._written(sql)
        return await self._tx.fetchone(sql, params)

    async def fetchall(self, sql, params=()):
        self._written(sql)
        return await self._tx.fetchall(sql, params)

    def _observe(self, sql, params):
        matched = _INSERT.match(sql)
        if matched is None or matched[1] != self._events:
            return
        columns = [c.strip() for c in matched[2].split(",")]
        rows = _value_rows(sql[matched.end():], params)
        if rows is None:
            self.sessions.add(GLOBAL)
            return
        for values in rows:
            if len(values) != len(columns):
                self.sessions.add(GLOBAL)
                continue
            row = dict(zip(columns, values, strict=True))
            from .session_obligations import note
            note(self, row)
            if row.get("event_type") not in OBLIGATION_EVENTS:
                continue
            if isinstance(row.get("job_id"), str):
                self.jobs.add(row["job_id"])
            if isinstance(row.get("caller_session_id"), str):
                self.sessions.add(row["caller_session_id"])


def _value_rows(text, params):
    """Map placeholders/literals of each VALUES tuple; None when not understood."""
    rows, index, position = [], 0, 0
    while True:
        match = re.compile(r"\s*\(([^()]*)\)\s*,?").match(text, position)
        if match is None:
            break
        values = []
        for token in (t.strip() for t in match[1].split(",")):
            if token == "?":
                if index >= len(params):
                    return None
                values.append(params[index])
                index += 1
            elif len(token) >= 2 and token[0] == token[-1] == "'":
                values.append(token[1:-1])
            else:
                values.append(None)
        rows.append(values)
        position = match.end()
    return rows if rows and index == len(params) else None


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
