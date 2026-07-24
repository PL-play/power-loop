"""Regression tests for the BUG_REVIEW_2.0 confirmed-bug fixes (G9–G18, C2).

Backend-agnostic cases run on SQLite; the PG/MySQL-specific bugs (G7/G9 race, G18) are
additionally covered by the parity suites against real servers when available.
"""

from __future__ import annotations

import asyncio
import operator
import sqlite3

import pytest

from power_loop.runtime.store.backends.mysql import _args
from power_loop.runtime.store.backends.sqlite import SqliteDatabase
from power_loop.runtime.store.schema import (
    SchemaPolicy,
    StoreSchemaError,
    ensure_schema,
    provisioning_ddl,
    validate_table_prefix,
)
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.store.types import SubagentLifecycle

pytestmark = pytest.mark.unit


# ── G13: set_pending({}) normalizes empty → SQL NULL (oracle parity) ─────────


async def test_set_pending_empty_dict_normalizes_to_none():
    s = await SessionStore.open(":memory:")
    sid = await s.create_session(session_id="s")
    await s.set_pending(sid, {"round_index": 3})
    assert (await s.get_state(sid)).pending == {"round_index": 3}
    # An empty (falsy) dict must round-trip to None, not {} — matches the legacy oracle.
    await s.set_pending(sid, {})
    assert (await s.get_state(sid)).pending is None
    await s.set_pending(sid, None)
    assert (await s.get_state(sid)).pending is None
    await s.close()


# ── G9: timer/note id allocation still works (and tolerates no session row) ──


async def test_timer_and_note_ids_sequential_under_lock_state():
    s = await SessionStore.open(":memory:")
    sid = await s.create_session(session_id="s")
    t1 = await s.create_timer(sid, due_at=1, note="a")
    t2 = await s.create_timer(sid, due_at=2, note="b")
    assert (t1.timer_id, t2.timer_id) == (1, 2)
    n1 = await s.add_note(sid, "n1")
    n2 = await s.add_note(sid, "n2")
    assert (n1.note_id, n2.note_id) == (1, 2)
    await s.close()


async def test_note_without_session_state_row_is_tolerated():
    """Legacy allowed notes for a session id with no session_state row; the new per-session
    lock (lock_state) must not turn that into a hard ValueError."""
    s = await SessionStore.open(":memory:")
    n = await s.add_note("sess_never_created", "orphan")  # no create_session
    assert n.note_id == 1
    await s.close()


# ── G10: provisioning_ddl version stamp is idempotent ────────────────────────


async def test_provisioning_ddl_stamp_is_idempotent_sqlite():
    db = SqliteDatabase.open(":memory:")
    ddl = provisioning_ddl(db, "pl_")
    assert "ON CONFLICT" in ddl[-1].upper()  # not a plain INSERT
    for _ in range(2):  # re-applying the whole script must not raise
        for stmt in ddl:
            await db.execute(stmt)
    row = await db.fetchone("SELECT COUNT(*) AS c FROM pl_schema_migrations")
    assert row["c"] == 1
    await db.close()


def test_provisioning_ddl_stamp_idempotent_mysql_shape():
    class _MyDialect:
        name = "mysql"

        def ddl(self, prefix):
            return []

    class _Db:
        dialect = _MyDialect()

    ddl = provisioning_ddl(_Db(), "pl_")
    assert ddl[-1].startswith("INSERT IGNORE INTO pl_schema_migrations")


# ── G12: auto_vacuum=INCREMENTAL actually takes (PRAGMA ordering) ────────────


async def test_auto_vacuum_is_incremental(tmp_path):
    db = SqliteDatabase.open(str(tmp_path / "av.db"))
    row = await db.fetchone("PRAGMA auto_vacuum")
    assert next(iter(row.values())) == 2  # 2 == INCREMENTAL (0 == NONE = the bug)
    await db.close()


# ── G16: VERIFY detects a dropped data table, not just the version row ───────


async def test_verify_detects_dropped_data_table(tmp_path):
    path = str(tmp_path / "s.db")
    s = await SessionStore.open(path, schema=SchemaPolicy.AUTO_CREATE)
    await s.close()
    raw = SqliteDatabase.open(path)
    await raw.execute("DROP TABLE pl_timers")
    await raw.close()
    with pytest.raises(StoreSchemaError) as ei:
        await SessionStore.open(path, schema=SchemaPolicy.VERIFY)
    msg = str(ei.value)
    assert "pl_timers" in msg and "incomplete" in msg


# ── G15: VERIFY surfaces a real read error instead of 'not initialized' ──────


async def test_verify_propagates_real_read_error(monkeypatch):
    db = SqliteDatabase.open(":memory:")
    await ensure_schema(db, "pl_", policy=SchemaPolicy.AUTO_CREATE)  # provision fully

    orig = db.fetchone

    async def boom(sql, params=()):
        if sql.strip().upper().startswith("SELECT VERSION"):
            raise RuntimeError("connection reset by peer")
        return await orig(sql, params)

    monkeypatch.setattr(db, "fetchone", boom)
    # The version table exists, so a failure READING it is a real fault and must propagate
    # as itself — not be swallowed and reported as 'schema not initialized'.
    with pytest.raises(RuntimeError, match="connection reset"):
        await ensure_schema(db, "pl_", policy=SchemaPolicy.VERIFY)
    await db.close()


# ── G11: a failed ROLLBACK must not mask the caller's original exception ─────


async def test_failed_rollback_does_not_mask_original_error(monkeypatch):
    db = SqliteDatabase.open(":memory:")
    orig_exec = db._exec

    async def exec_(sql, params=()):
        if sql == "ROLLBACK":
            raise sqlite3.OperationalError("rollback boom")
        return await orig_exec(sql, params)

    monkeypatch.setattr(db, "_exec", exec_)
    with pytest.raises(ValueError, match="business error"):
        async with db.transaction():
            raise ValueError("business error")
    await db.close()


# ── C2: a failed COMMIT must not wedge the shared connection ─────────────────


async def test_failed_commit_does_not_wedge_connection(monkeypatch):
    db = SqliteDatabase.open(":memory:")
    await db.execute("CREATE TABLE t (x INTEGER)")  # autocommit, no manual COMMIT

    commits = {"n": 0}
    orig_exec = db._exec

    async def exec_(sql, params=()):
        if sql == "COMMIT":
            commits["n"] += 1
            if commits["n"] == 1:
                raise sqlite3.OperationalError("commit boom")
        return await orig_exec(sql, params)

    monkeypatch.setattr(db, "_exec", exec_)
    with pytest.raises(sqlite3.OperationalError, match="commit boom"):
        async with db.transaction() as tx:
            await tx.execute("INSERT INTO t VALUES (1)")
    # The connection must NOT be stuck in an open transaction: a fresh one commits.
    async with db.transaction() as tx:
        await tx.execute("INSERT INTO t VALUES (2)")
    rows = await db.fetchall("SELECT x FROM t ORDER BY x")
    assert [r["x"] for r in rows] == [2]  # the rolled-back 1 is gone; 2 committed
    await db.close()


# ── G18: MySQL _args always returns a tuple so the %-collapse pass runs ───────


def test_mysql_args_always_tuple():
    assert _args(()) == ()
    assert _args([]) == ()
    assert _args([1, 2]) == (1, 2)
    # With a real tuple (even empty), the driver's `query % args` pass runs and collapses
    # the `%%` that dialect.translate emits back to a single literal `%`. With the old
    # `None` return that pass was skipped and `%%` leaked verbatim.
    assert operator.mod("LIKE 'pl\\_%%'", _args(())) == "LIKE 'pl\\_%'"


# ── C1: table_prefix is validated (interpolated into SQL identifiers) ────────


def test_validate_table_prefix_accepts_safe_and_empty():
    for ok in ("", "pl_", "Pl_2", "_x", "tenant42_"):
        assert validate_table_prefix(ok) == ok


def test_validate_table_prefix_rejects_unsafe():
    for bad in ('x"; DROP TABLE t; --', "a b", "1abc", "a-b", "pl_;", "pl_%"):
        with pytest.raises(ValueError, match="table_prefix"):
            validate_table_prefix(bad)


async def test_open_with_malicious_prefix_raises():
    with pytest.raises(ValueError, match="table_prefix"):
        await SessionStore.open(":memory:", table_prefix='x"; DROP TABLE pl_sessions; --')


# ── C4: cascade close returns every deleted id (loop clears their state) ─────


async def test_close_session_tree_returns_linked_subtree_ids():
    s = await SessionStore.open(":memory:")
    p = await s.create_session(session_id="p")
    c1 = await s.create_session(
        session_id="c1", parent_session_id="p", lifecycle=SubagentLifecycle.LINKED
    )
    gc = await s.create_session(
        session_id="gc", parent_session_id="c1", lifecycle=SubagentLifecycle.LINKED
    )
    det = await s.create_session(
        session_id="det", parent_session_id="p", lifecycle=SubagentLifecycle.DETACHED
    )
    deleted = await s.close_session_tree("p", cascade=True)
    # parent + LINKED descendants are deleted (and returned); DETACHED is re-parented, not.
    assert set(deleted) == {p, c1, gc}
    assert det not in deleted
    assert await s.get_session("det") is not None  # survived, re-parented
    assert await s.get_session("p") is None
    await s.close()


async def test_close_session_count_unchanged_by_id_collection():
    s = await SessionStore.open(":memory:")
    await s.create_session(session_id="p")
    await s.create_session(
        session_id="c1", parent_session_id="p", lifecycle=SubagentLifecycle.LINKED
    )
    n = await s.close_session("p", cascade=True)  # still returns an int count
    assert n == 2
    await s.close()


# ── C5: maintenance ops raise a clear error after close (no opaque driver err) ──


async def test_maintenance_ops_raise_after_close(tmp_path):
    db = SqliteDatabase.open(str(tmp_path / "m.db"))
    await db.close()
    with pytest.raises(RuntimeError, match="closed"):
        await db.checkpoint()
    with pytest.raises(RuntimeError, match="closed"):
        await db.vacuum()
    with pytest.raises(RuntimeError, match="closed"):
        await db.backup(str(tmp_path / "dest.db"))


# ── C3: two file handles serialize via BEGIN IMMEDIATE (no 'database is locked') ──


async def test_concurrent_file_handles_serialize_not_deadlock(tmp_path):
    path = str(tmp_path / "c3.db")
    seed = await SessionStore.open(path, schema=SchemaPolicy.AUTO_CREATE)
    sid = await seed.create_session(session_id="s")
    await seed.close()

    # Two independent handles to the SAME file → independent in-process write locks, so they
    # actually contend at the SQLite file level. A DEFERRED BEGIN would deadlock on the
    # SELECT→write lock upgrade ('database is locked'); BEGIN IMMEDIATE serializes them.
    a = await SessionStore.open(path, schema=SchemaPolicy.VERIFY)
    b = await SessionStore.open(path, schema=SchemaPolicy.VERIFY)
    try:
        results = await asyncio.gather(
            *(a.append_message(sid, role="user", content=f"a{i}") for i in range(10)),
            *(b.append_message(sid, role="user", content=f"b{i}") for i in range(10)),
            return_exceptions=True,
        )
        errs = [r for r in results if isinstance(r, BaseException)]
        assert not errs, f"writes failed (expected serialized success): {errs[:3]}"
        seqs = [r for r in results if not isinstance(r, BaseException)]
        assert len(set(seqs)) == 20  # every append got a unique seq
    finally:
        await a.close()
        await b.close()


# ── cascade=False: audit-trail close (workflow driver keeps its leaves) ──────


async def test_close_session_no_cascade_keeps_children_and_reparents():
    s = await SessionStore.open(":memory:")
    await s.create_session(session_id="driver")
    await s.create_session(
        session_id="leaf1", parent_session_id="driver", lifecycle=SubagentLifecycle.LINKED
    )
    await s.create_session(
        session_id="leaf2", parent_session_id="driver", lifecycle=SubagentLifecycle.LINKED
    )
    await s.append_message("leaf1", role="assistant", content="finding A")

    deleted = await s.close_session_tree("driver", cascade=False)
    assert deleted == ["driver"]
    assert await s.get_session("driver") is None
    # LINKED leaves survive as the audit trail, re-parented to NULL (no dangling refs).
    for leaf in ("leaf1", "leaf2"):
        row = await s.get_session(leaf)
        assert row is not None and row.parent_session_id is None
    msgs = await s.load_all_messages("leaf1")
    assert [m.content for m in msgs] == ["finding A"]
    await s.close()
