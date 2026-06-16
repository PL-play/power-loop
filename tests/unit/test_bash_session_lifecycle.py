"""Regression tests for the persistent-bash-session manager (C8/C9):

- C8: get-or-create is lock-guarded, so concurrent calls for one workspace key make
  exactly ONE session (no orphaned bash processes/threads/fds).
- C9: the session cache and FILE_READ_STATE are bounded (LRU); eviction closes the
  evicted session; close_all_bash_sessions() reclaims everything.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools import default_tools as dt

pytestmark = pytest.mark.unit


class _FakeSession:
    """Stands in for BashSession: counts constructions, records close(), and sleeps
    in __init__ to widen the get-or-create race window."""

    created = 0
    lock = threading.Lock()

    def __init__(self, workspace_dir: Path, backend) -> None:
        with _FakeSession.lock:
            _FakeSession.created += 1
        time.sleep(0.01)  # widen the race window the lock must close
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(dt, "BashSession", _FakeSession)
    monkeypatch.setattr(dt, "_BASH_SESSIONS", type(dt._BASH_SESSIONS)())
    _FakeSession.created = 0
    yield


async def test_concurrent_get_or_create_makes_one_session(tmp_path: Path) -> None:
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
        # asyncio.to_thread copies the runtime-env contextvar into each worker thread,
        # exactly like the real sync-tool dispatch path.
        results = await asyncio.gather(
            *[asyncio.to_thread(dt._bash_for_current_workspace) for _ in range(16)]
        )
    assert _FakeSession.created == 1  # was >1 (orphans) without the lock
    assert len({id(r) for r in results}) == 1  # all callers share the one session


def test_cache_is_lru_bounded_and_closes_evicted(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dt, "_MAX_BASH_SESSIONS", 2)
    sessions = []
    for i in range(4):
        ws = tmp_path / f"ws{i}"
        ws.mkdir()
        with runtime_env_context(RuntimeEnv(workspace_dir=ws)):
            sessions.append(dt._bash_for_current_workspace())

    assert len(dt._BASH_SESSIONS) == 2  # capped
    # the 2 least-recently-used were evicted AND closed
    assert sessions[0].closed and sessions[1].closed
    assert not sessions[2].closed and not sessions[3].closed


def test_close_all_bash_sessions(tmp_path: Path) -> None:
    made = []
    for i in range(3):
        ws = tmp_path / f"w{i}"
        ws.mkdir()
        with runtime_env_context(RuntimeEnv(workspace_dir=ws)):
            made.append(dt._bash_for_current_workspace())
    dt.close_all_bash_sessions()
    assert len(dt._BASH_SESSIONS) == 0
    assert all(s.closed for s in made)


def test_file_read_state_is_bounded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dt, "_MAX_FILE_READ_STATE", 3)
    monkeypatch.setattr(dt, "FILE_READ_STATE", type(dt.FILE_READ_STATE)())
    for i in range(5):
        f = tmp_path / f"f{i}.txt"
        f.write_text("x")
        dt._remember_read(f)
    assert len(dt.FILE_READ_STATE) == 3  # oldest evicted, not unbounded growth
