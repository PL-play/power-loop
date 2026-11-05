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
        self._manager = manager
        self.root_job_id = None
        self._root_members: set[str] = set()
        self._events = manager.events
        self._jobs = manager.jobs
        self._executors = manager.prefix + "task_executors"
        self._executor_groups = manager.prefix + "task_executor_admissions"
        self._principal_admissions = manager.prefix + "task_principal_admissions"
        self._context_events = manager.prefix + "task_context_events"
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
        self.obligation_handled: set[str] = set()
        self.obligation_open: set[tuple] = set()
        self.obligation_close: set[tuple] = set()
        self._executor_admissions: dict[str, bool] = {}
        self._executor_group_locks: dict[str, tuple[bool, bool]] = {}
        self._budget_policy_admissions: dict[str, dict | None] = {}
        self._principal_admission_hashes: dict[str, str | None] = {}

    def __getattr__(self, name):
        return getattr(self._tx, name)

    def touch_job(self, job_id, *, obligations=False, notify=False):
        self.touched_jobs.add(job_id)
        self._memo.pop((f"SELECT * FROM {self._jobs} WHERE job_id=?", (job_id,)), None)
        if obligations:
            self.obligation_jobs.add(job_id)
        if notify:
            self.jobs.add(job_id)

    def handled(self, job_id):
        """Declare an original result's already-proved handling for this write."""
        self.touch_job(job_id, obligations=True)
        self.obligation_handled.add(job_id)

    async def _lock_root(self, job_id):
        """The lookup proves the target's root without fetching its full row."""
        import time

        suffix = '' if self.dialect.name == 'sqlite' else ' FOR UPDATE'
        root = await self._tx.fetchone(f'SELECT root.* FROM {self._jobs} root '
            f'WHERE root.job_id=(SELECT root_job_id FROM {self._jobs} WHERE job_id=?)'+suffix, (job_id,))
        if root is None:
            return False
        self.root_job_id = root['job_id']
        self._root_members.update((root['job_id'], job_id))
        self.remember_row(self._jobs, 'job_id', root['job_id'], root)
        if (timing := self._manager._control_lock_timing.get()) is not None:
            timing['acquired'] = time.perf_counter()
        return True

    async def job_row(self, job_id):
        """Acquire this writer's one root before reading mutable job state."""
        sql = f'SELECT * FROM {self._jobs} WHERE job_id=?'
        if self.root_job_id is None and not await self._lock_root(job_id):
            return None
        memo = sql, (job_id,)
        if memo not in self._memo:
            # A writer never acquires another root. The scope predicate also
            # fences a prepared deterministic ID which was recreated elsewhere.
            row = await self._tx.fetchone(sql+' AND root_job_id=?', (job_id, self.root_job_id))
            self._memo[memo] = row
        row = self._memo[memo]
        if row is not None:
            self._root_members.add(job_id)
        return None if row is None else dict(row)

    async def insert_job(self, job_id, root_job_id, sql, params=()):
        """A fresh root's INSERT owns its row;children use the parent's writer."""
        import time

        from .jobs import JobError

        if self.root_job_id is not None and self.root_job_id != root_job_id:
            raise JobError('task_tree_scope_mismatch')
        if job_id != root_job_id:
            await self.job_row(root_job_id)
        else:
            # A no-op upsert takes the actual root row lock on both initial
            # creation and a simultaneous retry,without a shared-lock upgrade.
            sql += (' ON DUPLICATE KEY UPDATE job_id=VALUES(job_id)' if self.dialect.name == 'mysql' else
                ' ON CONFLICT(job_id) DO UPDATE SET job_id=excluded.job_id')
        self.touch_job(job_id)
        changed = await self._execute(sql, params)
        self._root_members.add(job_id)
        if job_id == root_job_id:
            self.root_job_id = root_job_id
            if (timing := self._manager._control_lock_timing.get()) is not None:
                timing['acquired'] = time.perf_counter()
        return changed

    async def write_job(self, job_id, sql, params=(), *, obligations=False):
        await self.require_job(job_id)
        self.touch_job(job_id, obligations=obligations)
        return await self._execute(sql, params)

    async def require_job(self, job_id):
        from .jobs import JobError

        # Root membership is immutable. A job-state update invalidates its
        # row cache,not the membership this transaction already locked/proved.
        if job_id in self._root_members:
            return
        if self.root_job_id is None:
            if await self._lock_root(job_id):
                return
        elif await self.job_row(job_id) is not None:
            return
        raise JobError('task_tree_scope_mismatch')

    async def renew_job(self, job_id, sql, params):
        """A lease update has the same root scope as other job writes."""
        await self.require_job(job_id)
        from .session_ownership import require_current_ownership
        require_current_ownership(self._manager.store)
        self.touch_job(job_id)
        self.writes += 1
        return await self._tx.execute_flag(sql, params)

    async def write_executor(self, executor_key, sql, params=()):
        self._memo.pop((f"SELECT * FROM {self._executors} WHERE executor_key=?", (executor_key,)), None)
        return await self._execute(sql, params)

    async def write_executor_group(self, group_key, sql, params=()):
        self._executor_group_locks.pop(group_key, None)
        return await self._execute(sql, params)

    async def write_principal_admission(self, policy_key, sql, params=()):
        return await self._execute(sql, params)

    async def executor_admission(self, executor_key, *, exclusive=False, refresh=False):
        """Hold this boot's admission until the tree publication commits.

        Drain and activation close the boot with an exclusive metadata write
        before discovering roots. Concurrent trees share this read lock.
        """
        sql = f"SELECT * FROM {self._executors} WHERE executor_key=?"
        if refresh or executor_key not in self._executor_admissions or exclusive and not self._executor_admissions[executor_key]:
            suffix = '' if self.dialect.name == 'sqlite' else ' FOR UPDATE' if exclusive else ' FOR SHARE'
            row = await self._tx.fetchone(sql+suffix, (executor_key,))
            self._memo[sql, (executor_key,)] = None if row is None else dict(row)
            self._executor_admissions[executor_key] = exclusive
        return await self.fetchone(sql, (executor_key,))

    async def unbudgeted_executor_admission(self, executor_key, policy_key):
        """Share both existing metadata gates in the boot admission query."""
        suffix = '' if self.dialect.name == 'sqlite' else ' FOR SHARE'
        row = await self._tx.fetchone(
            f'SELECT boot.*,principal.policy_hash AS admission_hash FROM {self._executors} boot '
            f'JOIN {self._principal_admissions} principal ON principal.policy_key=? '
            'WHERE boot.executor_key=?'+suffix, (policy_key, executor_key))
        if row is not None:
            self._principal_admission_hashes[policy_key] = row.pop('admission_hash')
        self._memo[f'SELECT * FROM {self._executors} WHERE executor_key=?', (executor_key,)] = row
        self._executor_admissions[executor_key] = False
        return row

    async def append_event(self, columns, values, *, ignore=False):
        """Append a declared original; its meaning is supplied as fields."""
        event = dict(zip(columns, values, strict=True))
        await self.require_job(event['job_id'])
        self.note_event(event)
        sql = (self._dialect.insert_ignore(self._events, columns) if ignore else
               f"INSERT INTO {self._events} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})")
        return await self._execute(sql, values)

    async def append_context_event(self, columns, values):
        """Append a host task-context original, before any root need exist."""
        event = dict(zip(columns, values, strict=True))
        self.sessions.add(event['session_id'])
        return await self._execute(f"INSERT INTO {self._context_events} ({','.join(columns)}) VALUES ("
            + ','.join('?' for _ in columns) + ')', values)

    async def append_event_if_current(self, event, row):
        """Append mechanical evidence without changing a superseding owner."""
        await self.require_job(event['job_id'])
        self.note_event(event)
        columns = tuple(event)
        sql = f"INSERT INTO {self._events} ({','.join(columns)}) SELECT " + ','.join('?' for _ in columns)
        sql += f" FROM {self._jobs} WHERE job_id=? AND root_job_id=? AND generation=? AND version=?"
        return await self._execute(sql, (*event.values(), row["job_id"], row["root_job_id"], row["generation"], row["version"]))

    async def append_event_if_tree_current(self, event, tree):
        """Publish a prepared original only if no job in its tree changed.

        Job versions only advance. Count plus their sum also catches a new or
        resumed descendant which was absent from the prepared closure plans.
        The tree writer remains held until all conditional projections commit.
        Original-only publications can leave job versions unchanged; their
        original key must also remain absent when this statement executes.
        """
        await self.require_job(event['job_id'])
        self.note_event(event)
        columns = tuple(event)
        sql = f"INSERT INTO {self._events} ({','.join(columns)}) SELECT " + ','.join('?' for _ in columns)
        sql += (f" WHERE EXISTS (SELECT 1 FROM {self._jobs} WHERE job_id=? AND generation=?)"
            f" AND (SELECT COUNT(*) FROM {self._jobs} WHERE root_job_id=?)=?"
            f" AND (SELECT COALESCE(SUM(version),0) FROM {self._jobs} WHERE root_job_id=?)=?"
            f" AND NOT EXISTS (SELECT 1 FROM {self._events} WHERE event_id=?)")
        return await self._execute(sql, (*event.values(), tree.root_id, tree.generation,
            tree.root_id, tree.count, tree.root_id, tree.version_sum,event['event_id']))

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
            event = dict(zip(columns, row, strict=True))
            await self.require_job(event['job_id'])
            self.note_event(event)
        return await self._execute(sql, params)

    async def update_event(self, job_id, sql, params=(), *, obligations=True, notify=False):
        """Declare an original event's state change without re-reading it."""
        await self.require_job(job_id)
        self.touched_jobs.add(job_id)
        if obligations:
            self.obligation_jobs.add(job_id)
        if notify:
            self.jobs.add(job_id)
        return await self._execute(sql, params)

    async def set_event_recipient(self, event_id: str, session_id: str):
        row = await self.fetchone(f'SELECT job_id FROM {self._events} WHERE event_id=?', (event_id,))
        await self.require_job(row['job_id'])
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
        from .session_ownership import require_current_ownership
        require_current_ownership(self._manager.store)
        self.writes += 1
        return await self._tx.execute(sql, params)

    async def control_statement(self, sql):
        """Savepoint operations are transaction control, not business writes."""
        return await self._tx.execute(sql)

    async def execute_flag(self, sql, params=()):
        from .session_ownership import require_current_ownership
        require_current_ownership(self._manager.store)
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
