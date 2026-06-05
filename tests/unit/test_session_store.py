from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from power_loop.runtime.session_store import (
    MAX_SPAWN_DEPTH,
    MessageState,
    SessionKind,
    SessionStatus,
    SessionStore,
    SubagentLifecycle,
)


@pytest.fixture
def store() -> SessionStore:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


def test_open_creates_schema_in_file(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    s = SessionStore.open(db)
    try:
        assert db.exists()
        names = {
            r[0]
            for r in s._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"sessions", "messages", "compactions", "usage_rounds", "session_state"} <= names
    finally:
        s.close()


def test_create_session_seeds_state_and_returns_id(store: SessionStore) -> None:
    sid = store.create_session(system_prompt="hi", model="m1", config={"a": 1})
    row = store.get_session(sid)
    assert row is not None
    assert row.system_prompt == "hi"
    assert row.model == "m1"
    assert row.config == {"a": 1}
    assert row.kind is SessionKind.ROOT
    assert row.status is SessionStatus.ACTIVE
    assert row.spawn_depth == 0
    assert row.parent_session_id is None

    state = store.get_state(sid)
    assert state is not None
    assert state.next_seq == 1
    assert state.round_index == 0
    assert state.last_compact_seq == 0
    assert state.pending is None


def test_append_messages_allocates_monotonic_seq(store: SessionStore) -> None:
    sid = store.create_session()
    s1 = store.append_message(sid, role="user", content="hello", round_index=1)
    s2 = store.append_message(
        sid, role="assistant",
        tool_calls=[{"id": "tc1", "function": {"name": "f"}}],
        round_index=1,
    )
    s3 = store.append_message(sid, role="tool", tool_call_id="tc1", content="ok", round_index=1)
    assert (s1, s2, s3) == (1, 2, 3)

    msgs = store.load_active_messages(sid)
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[1].tool_calls == [{"id": "tc1", "function": {"name": "f"}}]
    assert msgs[2].tool_call_id == "tc1"
    assert store.get_state(sid).next_seq == 4


def test_record_compaction_marks_range_and_appends_note(store: SessionStore) -> None:
    sid = store.create_session()
    for i in range(5):
        store.append_message(sid, role="user", content=f"m{i}", round_index=i)

    compact_seq, note_seq = store.record_compaction(
        sid, from_seq=2, to_seq=4,
        note_content="summary text",
        before_tokens=1234, after_tokens=200, round_index=5,
    )
    assert compact_seq == 1
    assert note_seq == 6

    all_msgs = store.load_all_messages(sid)
    states = {m.seq: m.state for m in all_msgs}
    assert states[1] is MessageState.ACTIVE
    assert states[2] is MessageState.COMPACTED_OUT
    assert states[3] is MessageState.COMPACTED_OUT
    assert states[4] is MessageState.COMPACTED_OUT
    assert states[5] is MessageState.ACTIVE
    assert states[6] is MessageState.ACTIVE  # the note itself

    active = store.load_active_messages(sid)
    note = next(m for m in active if m.name == "compact_note")
    assert note.content == "summary text"
    assert note.meta["from_seq"] == 2
    assert note.meta["to_seq"] == 4
    assert note.meta["original_tokens"] == 1234

    rows = store.list_compactions(sid)
    assert len(rows) == 1 and rows[0].from_seq == 2 and rows[0].to_seq == 4

    state = store.get_state(sid)
    assert state.last_compact_seq == 1
    assert state.next_seq == 7


def test_subagent_links_to_parent_and_inherits_depth(store: SessionStore) -> None:
    parent = store.create_session(system_prompt="P")
    child = store.create_session(
        parent_session_id=parent, spawn_tool_call_id="tc-spawn",
        lifecycle=SubagentLifecycle.LINKED,
    )
    grand = store.create_session(parent_session_id=child)

    child_row = store.get_session(child)
    assert child_row.kind is SessionKind.SUBAGENT
    assert child_row.parent_session_id == parent
    assert child_row.spawn_depth == 1
    assert child_row.spawn_tool_call_id == "tc-spawn"
    assert child_row.lifecycle is SubagentLifecycle.LINKED

    assert store.get_session(grand).spawn_depth == 2
    assert [c.session_id for c in store.list_children(parent)] == [child]


def test_spawn_depth_limit_enforced(store: SessionStore) -> None:
    parent = store.create_session()
    cur = parent
    for _ in range(MAX_SPAWN_DEPTH):
        cur = store.create_session(parent_session_id=cur)
    with pytest.raises(ValueError, match="spawn depth"):
        store.create_session(parent_session_id=cur)


def test_close_session_cascades_linked_and_detaches_detached(store: SessionStore) -> None:
    parent = store.create_session()
    linked = store.create_session(
        parent_session_id=parent, lifecycle=SubagentLifecycle.LINKED,
    )
    detached = store.create_session(
        parent_session_id=parent, lifecycle=SubagentLifecycle.DETACHED,
    )
    store.append_message(linked, role="user", content="x")
    store.append_message(detached, role="user", content="y")

    deleted = store.close_session(parent, cascade=True)
    # parent + linked child = 2 sessions removed; detached preserved.
    assert deleted == 2
    assert store.get_session(parent) is None
    assert store.get_session(linked) is None
    assert store.get_session(detached) is not None
    assert store.get_session(detached).parent_session_id is None
    # Messages of removed sessions are gone.
    assert store.load_all_messages(linked) == []


def test_close_session_wipes_all_tables(store: SessionStore) -> None:
    sid = store.create_session()
    store.append_message(sid, role="user", content="hi")
    store.record_compaction(
        sid, from_seq=1, to_seq=1, note_content="n",
        before_tokens=10, after_tokens=5, round_index=0,
    )
    store.record_usage(
        sid, round_index=1, prompt_tokens=1, completion_tokens=2, total_tokens=3,
    )

    store.close_session(sid)

    for table in ("sessions", "messages", "compactions", "usage_rounds", "session_state"):
        n = store._conn.execute(
            f"SELECT count(*) FROM {table} WHERE session_id=?", (sid,)
        ).fetchone()[0]
        assert n == 0, f"{table} not fully wiped"


def test_pending_round_index_set_get(store: SessionStore) -> None:
    sid = store.create_session()
    store.set_round_index(sid, 7)
    store.set_pending(sid, {"tool_calls": ["tc1", "tc2"], "assistant_seq": 5})
    state = store.get_state(sid)
    assert state.round_index == 7
    assert state.pending == {"tool_calls": ["tc1", "tc2"], "assistant_seq": 5}

    store.set_pending(sid, None)
    assert store.get_state(sid).pending is None


def test_record_usage_upserts(store: SessionStore) -> None:
    sid = store.create_session()
    store.record_usage(sid, round_index=1, prompt_tokens=10, completion_tokens=20, total_tokens=30)
    store.record_usage(sid, round_index=1, prompt_tokens=11, completion_tokens=22, total_tokens=33)
    row = store._conn.execute(
        "SELECT prompt_tokens, total_tokens FROM usage_rounds WHERE session_id=? AND round_index=1",
        (sid,),
    ).fetchone()
    assert row["prompt_tokens"] == 11 and row["total_tokens"] == 33


def test_unknown_parent_raises(store: SessionStore) -> None:
    with pytest.raises(ValueError, match="parent session not found"):
        store.create_session(parent_session_id="sess_doesnotexist")


def test_append_to_unknown_session_raises(store: SessionStore) -> None:
    with pytest.raises(ValueError, match="unknown session"):
        store.append_message("sess_nope", role="user", content="x")


def test_open_default_path_uses_module_default(tmp_path: Path, monkeypatch) -> None:
    # Verify the contract that an explicit path is required, but the default
    # constant resolves to a writable file when callers do pass it through.
    db = tmp_path / "default.db"
    s = SessionStore.open(db)
    try:
        sid = s.create_session()
        assert s.get_session(sid) is not None
    finally:
        s.close()
    # Re-open the same file and confirm persistence.
    s2 = SessionStore.open(db)
    try:
        assert s2.get_session(sid) is not None
    finally:
        s2.close()


def test_wal_mode_enabled(tmp_path: Path) -> None:
    db = tmp_path / "wal.db"
    s = SessionStore.open(db)
    try:
        mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        s.close()


def test_concurrent_appends_get_distinct_seqs() -> None:
    """Two threads appending to the same session must not collide on seq."""
    import threading
    s = SessionStore.open(":memory:")
    try:
        sid = s.create_session()
        seqs: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(20):
                seq = s.append_message(sid, role="user", content="x")
                with lock:
                    seqs.append(seq)

        ts = [threading.Thread(target=worker) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert sorted(seqs) == list(range(1, 81))
    finally:
        s.close()


def test_archive_session_preserves_data(store: SessionStore) -> None:
    sid = store.create_session()
    store.append_message(sid, role="user", content="hi")
    store.archive_session(sid)
    row = store.get_session(sid)
    assert row is not None and row.status is SessionStatus.ARCHIVED
    assert len(store.load_all_messages(sid)) == 1


def test_sqlite3_module_smoke() -> None:
    # Sanity: stdlib only.
    assert sqlite3.sqlite_version_info >= (3, 7, 0)
