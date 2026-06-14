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
