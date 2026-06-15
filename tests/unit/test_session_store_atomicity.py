"""Regression: SessionStore writers are transactional (rollback on mid-block error)."""
from __future__ import annotations

import pytest

from power_loop import SessionStore

pytestmark = pytest.mark.unit


def test_write_block_rolls_back_on_failure():
    s = SessionStore.open(":memory:")
    try:
        sid = s.create_session()
        # Simulate a multi-write method that raises partway through its
        # `with self._conn:` block: the partial write MUST roll back.
        with pytest.raises(RuntimeError):
            with s._conn:
                s._conn.execute(
                    "INSERT INTO session_runtime_state(session_id,key,value_json,updated_at)"
                    " VALUES (?,?,?,?)",
                    (sid, "k", '"v"', 1),
                )
                raise RuntimeError("boom mid-method")
        # autocommit (isolation_level=None) would have persisted the row;
        # deferred isolation rolls it back.
        assert s.get_runtime_state(sid, "k", default=None) is None
    finally:
        s.close()


def test_write_block_commits_on_success():
    s = SessionStore.open(":memory:")
    try:
        sid = s.create_session()
        with s._conn:
            s._conn.execute(
                "INSERT INTO session_runtime_state(session_id,key,value_json,updated_at)"
                " VALUES (?,?,?,?)",
                (sid, "k", '"v"', 1),
            )
        assert s.get_runtime_state(sid, "k") == "v"
    finally:
        s.close()


def test_leading_select_then_dml_rolls_back_atomically():
    """The read-modify-write shape used by append_message/record_compaction etc.:
    a leading SELECT (which under deferred isolation runs in autocommit) followed by
    DML that then fails must still roll back the DML — the `with self._conn:` block
    covers every write after the first DML. (H1.8.)"""
    s = SessionStore.open(":memory:")
    try:
        sid = s.create_session()
        s.append_message(sid, role="user", content="first")
        before = s._conn.execute(
            "SELECT next_seq FROM session_state WHERE session_id=?", (sid,)
        ).fetchone()["next_seq"]

        with pytest.raises(RuntimeError):
            with s._lock, s._conn:
                # leading SELECT — the C7 case: begins NO transaction by itself
                s._conn.execute(
                    "SELECT next_seq FROM session_state WHERE session_id=?", (sid,)
                ).fetchone()
                # a DML opens the transaction …
                s._conn.execute(
                    "UPDATE session_state SET next_seq=next_seq+5 WHERE session_id=?", (sid,)
                )
                raise RuntimeError("boom after leading select + dml")

        after = s._conn.execute(
            "SELECT next_seq FROM session_state WHERE session_id=?", (sid,)
        ).fetchone()["next_seq"]
        assert after == before  # the UPDATE rolled back atomically
    finally:
        s.close()
