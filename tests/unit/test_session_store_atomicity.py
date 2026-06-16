"""Regression: the async SessionStore's transactions are atomic — a block that
raises partway through rolls back every write in it (explicit BEGIN/ROLLBACK in
the SQLite backend, serialized by the store's writer lock)."""
from __future__ import annotations

import pytest

from power_loop import SessionStore

pytestmark = pytest.mark.unit


async def test_write_block_rolls_back_on_failure():
    s = await SessionStore.open(":memory:")
    try:
        sid = await s.create_session()
        # A multi-write transaction that raises partway through MUST roll back the
        # partial write.
        with pytest.raises(RuntimeError):
            async with s._db.transaction() as tx:
                await tx.execute(
                    f"INSERT INTO {s.t.session_runtime_state}"
                    "(session_id,key,value_json,updated_at) VALUES (?,?,?,?)",
                    (sid, "k", '"v"', 1),
                )
                raise RuntimeError("boom mid-transaction")
        assert await s.get_runtime_state(sid, "k", default=None) is None
    finally:
        await s.close()


async def test_write_block_commits_on_success():
    s = await SessionStore.open(":memory:")
    try:
        sid = await s.create_session()
        async with s._db.transaction() as tx:
            await tx.execute(
                f"INSERT INTO {s.t.session_runtime_state}"
                "(session_id,key,value_json,updated_at) VALUES (?,?,?,?)",
                (sid, "k", '"v"', 1),
            )
        assert await s.get_runtime_state(sid, "k") == "v"
    finally:
        await s.close()


async def test_leading_select_then_dml_rolls_back_atomically():
    """The read-modify-write shape used by append_message/record_compaction etc.:
    a leading SELECT followed by DML that then fails must still roll back the DML —
    the transaction block covers every write in it."""
    s = await SessionStore.open(":memory:")
    try:
        sid = await s.create_session()
        await s.append_message(sid, role="user", content="first")
        before_row = await s._db.fetchone(
            f"SELECT next_seq FROM {s.t.session_state} WHERE session_id=?", (sid,)
        )
        before = before_row["next_seq"]

        with pytest.raises(RuntimeError):
            async with s._db.transaction() as tx:
                # leading SELECT inside the transaction
                await tx.fetchone(
                    f"SELECT next_seq FROM {s.t.session_state} WHERE session_id=?", (sid,)
                )
                # a DML write …
                await tx.execute(
                    f"UPDATE {s.t.session_state} SET next_seq=next_seq+5 WHERE session_id=?",
                    (sid,),
                )
                raise RuntimeError("boom after leading select + dml")

        after_row = await s._db.fetchone(
            f"SELECT next_seq FROM {s.t.session_state} WHERE session_id=?", (sid,)
        )
        assert after_row["next_seq"] == before  # the UPDATE rolled back atomically
    finally:
        await s.close()
