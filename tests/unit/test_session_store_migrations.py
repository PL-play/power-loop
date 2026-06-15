"""OPS-1: PRAGMA user_version gate + ordered additive migration ladder.

The on-disk SessionStore is the product's value prop, so an existing ``.db`` must
survive a power_loop upgrade. Before this, the only migration logic was a hardcoded
timers-only ``_micro_migrate`` with no version gate: any future schema change would
silently never land on an existing file (CREATE TABLE IF NOT EXISTS no-ops), and a
DB written by newer code could be opened with a mismatched schema. These tests pin
the versioned ladder: fresh DBs are stamped, legacy DBs are upgraded idempotently,
re-open is a no-op, and a newer-than-code DB is refused.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from power_loop.runtime.session_store import CURRENT_SCHEMA_VERSION, SessionStore

pytestmark = pytest.mark.unit

# every table SCHEMA_SQL is responsible for
_EXPECTED_TABLES = {
    "sessions", "messages", "compactions", "usage_rounds", "session_state",
    "session_runtime_state", "shared_state", "background_tasks", "session_stats",
    "timers", "notes",
}


def _user_version(db_path: str) -> int:
    raw = sqlite3.connect(db_path)
    try:
        return int(raw.execute("PRAGMA user_version").fetchone()[0])
    finally:
        raw.close()


def _columns(db_path: str, table: str) -> set[str]:
    raw = sqlite3.connect(db_path)
    try:
        return {r[1] for r in raw.execute(f"PRAGMA table_info({table})")}
    finally:
        raw.close()


def _tables(db_path: str) -> set[str]:
    raw = sqlite3.connect(db_path)
    try:
        return {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        raw.close()


def test_fresh_db_is_stamped_and_fully_created(tmp_path: Path) -> None:
    """Opening a brand-new file stamps user_version == CURRENT and creates every table."""
    db = str(tmp_path / "fresh.db")
    store = SessionStore.open(db)
    store.close()
    assert _user_version(db) == CURRENT_SCHEMA_VERSION
    assert _EXPECTED_TABLES <= _tables(db)
    # a fresh timers table already has the migration-1 columns from SCHEMA_SQL
    assert {"interval_s", "fire_count", "last_fired_at"} <= _columns(db, "timers")


def test_legacy_db_is_upgraded_idempotently(tmp_path: Path) -> None:
    """A pre-columns, user_version=0 DB gains the timers columns and is stamped.

    Faithfully simulates a legacy file: create the full current schema, then surgically
    downgrade (drop the migration-1 timers columns + reset user_version=0) so the only
    difference from a real ≤0.14.1 file is exactly what migration 1 repairs."""
    db = str(tmp_path / "legacy.db")
    SessionStore.open(db).close()  # full current schema
    raw = sqlite3.connect(db)
    for col in ("interval_s", "fire_count", "last_fired_at"):
        raw.execute(f"ALTER TABLE timers DROP COLUMN {col}")
    raw.execute("PRAGMA user_version = 0")  # legacy marker
    raw.commit()
    raw.close()
    assert not {"interval_s", "fire_count", "last_fired_at"} & _columns(db, "timers")
    assert _user_version(db) == 0

    SessionStore.open(db).close()  # should run migration 1

    assert _user_version(db) == CURRENT_SCHEMA_VERSION
    assert {"interval_s", "fire_count", "last_fired_at"} <= _columns(db, "timers")


def test_reopen_is_a_noop(tmp_path: Path) -> None:
    """Opening an already-current DB a second time changes nothing and never raises
    a duplicate-column error (idempotent migration steps + version short-circuit)."""
    db = str(tmp_path / "reopen.db")
    SessionStore.open(db).close()
    v1 = _user_version(db)
    cols1 = _columns(db, "timers")
    # second open must short-circuit (current >= CURRENT) without re-ALTERing
    SessionStore.open(db).close()
    assert _user_version(db) == v1 == CURRENT_SCHEMA_VERSION
    assert _columns(db, "timers") == cols1


def test_newer_than_code_db_is_refused(tmp_path: Path) -> None:
    """A DB stamped by a NEWER build is refused with a clear error, not silently
    operated on with a mismatched schema."""
    db = str(tmp_path / "future.db")
    SessionStore.open(db).close()  # create + stamp at CURRENT
    raw = sqlite3.connect(db)
    raw.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}")
    raw.commit()
    raw.close()

    with pytest.raises(RuntimeError, match="newer than this power_loop build"):
        SessionStore.open(db)


def test_migrations_ladder_is_consistent() -> None:
    """The ladder's max target equals CURRENT_SCHEMA_VERSION and steps are ordered &
    unique — a guard against forgetting to bump the version when appending a step."""
    from power_loop.runtime.session_store import MIGRATIONS

    targets = [t for t, _ in MIGRATIONS]
    assert targets == sorted(targets), "migration steps must be in ascending order"
    assert len(targets) == len(set(targets)), "duplicate migration target versions"
    assert max(targets) == CURRENT_SCHEMA_VERSION, (
        "CURRENT_SCHEMA_VERSION must equal the highest migration target"
    )
