"""SCALE-2: optional read-only WAL connection pool.

Reads go through a pool of extra connections (when ``read_pool_size>0`` on a file DB) so
they run concurrently with the single writer instead of serializing behind its lock. The
writer stays the sole source of truth; pooled readers see every committed write and can
never themselves write (``query_only=ON``). Default off; ``:memory:`` falls back.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from power_loop.runtime.session_store import SessionStore

pytestmark = pytest.mark.unit


def _seed(store: SessionStore, sid: str, n: int = 5) -> list[int]:
    return [store.append_message(sid, role="user", content=f"m{i}", round_index=i) for i in range(n)]


def test_pooled_reads_return_the_same_data_as_the_writer(tmp_path: Path) -> None:
    store = SessionStore.open(str(tmp_path / "p.db"), read_pool_size=2)
    try:
        sid = store.create_session(session_id="s")
        _seed(store, sid, 5)
        active = store.load_active_messages(sid)
        assert [m.content for m in active] == [f"m{i}" for i in range(5)]
        # a write committed under the writer is immediately visible to a pooled reader
        store.append_message(sid, role="assistant", content="after", round_index=99)
        assert store.load_active_messages(sid)[-1].content == "after"
        assert len(store.load_all_messages(sid)) == 6
    finally:
        store.close()


def test_pooled_reader_cannot_write(tmp_path: Path) -> None:
    """query_only=ON: a checked-out pool connection physically refuses writes."""
    store = SessionStore.open(str(tmp_path / "ro.db"), read_pool_size=1)
    try:
        store.create_session(session_id="s")
        with store._read_cursor() as c:
            with pytest.raises(Exception):  # sqlite3.OperationalError: attempt to write a readonly db
                c.execute("INSERT INTO notes (session_id, note_id, content, created_at, updated_at) "
                          "VALUES ('s', 1, 'x', 0, 0)")
    finally:
        store.close()


def test_in_memory_ignores_read_pool(caplog) -> None:
    """:memory: can't share connections, so the pool is declined (reads use the writer)."""
    store = SessionStore.open(":memory:", read_pool_size=4)
    try:
        assert store._read_pool is None  # declined
        sid = store.create_session(session_id="s")
        _seed(store, sid, 3)
        assert len(store.load_active_messages(sid)) == 3  # still works via the writer
    finally:
        store.close()


def test_default_has_no_read_pool(tmp_path: Path) -> None:
    store = SessionStore.open(str(tmp_path / "d.db"))
    try:
        assert store._read_pool is None  # off by default → historical behavior
    finally:
        store.close()


def test_pooled_reads_run_concurrently_with_a_held_write_lock(tmp_path: Path) -> None:
    """The headline win: a long write holding the writer RLock does NOT block pooled
    reads. With no pool, the same read would serialize behind the writer."""
    store = SessionStore.open(str(tmp_path / "c.db"), read_pool_size=2)
    try:
        sid = store.create_session(session_id="s")
        _seed(store, sid, 5)

        release = threading.Event()
        holding = threading.Event()

        def hold_writer() -> None:
            # simulate a long write transaction holding the writer RLock
            with store._lock:
                holding.set()
                release.wait(timeout=5)

        t = threading.Thread(target=hold_writer)
        t.start()
        assert holding.wait(timeout=2)  # writer lock is now held

        # a pooled read completes promptly despite the writer lock being held
        t0 = time.perf_counter()
        rows = store.load_active_messages(sid)
        elapsed = time.perf_counter() - t0
        assert len(rows) == 5
        assert elapsed < 1.0, f"pooled read blocked on the write lock ({elapsed:.2f}s)"

        release.set()
        t.join(timeout=5)
    finally:
        store.close()
