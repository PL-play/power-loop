"""OPS-4: full-session export/import round-trip + whole-store backup.

``export_session`` must capture enough durable state that ``import_session`` into a new
id reproduces the session faithfully — active-message ordering (incl. a compact_note's
logical ``ord``), the compacted originals, compactions, timers, notes — so "archive then
prune" and cross-store moves are lossless.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from power_loop.runtime.session_store import (
    CURRENT_SCHEMA_VERSION,
    MessageState,
    SessionStore,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def store() -> Generator[SessionStore, None, None]:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


def _build_rich_session(store: SessionStore) -> str:
    sid = store.create_session(system_prompt="SP", model="m", metadata={"k": "v"})
    seqs = [store.append_message(sid, role="user", content=f"m{i}", round_index=i) for i in range(6)]
    # fold the first 4 → a compact_note carrying meta['ord'] + compacted originals
    fold = seqs[:4]
    store.record_compaction(
        sid, from_seq=min(fold), to_seq=max(fold), note_content="summary",
        before_tokens=100, after_tokens=10, round_index=6, fold_seqs=fold, order_key=fold[0],
    )
    store.add_note(sid, "remember this", pinned=True)
    store.create_timer(sid, due_at=999_999, note="recurring", interval_s=60)
    store.record_usage(sid, round_index=0, prompt_tokens=5, completion_tokens=3, total_tokens=8)
    store.set_runtime_state(sid, "scratch", {"a": 1})
    return sid


def test_export_import_round_trip(store: SessionStore) -> None:
    sid = _build_rich_session(store)
    src_active = [(r.role, r.name, r.content) for r in store.load_active_messages(sid)]
    src_compactions = [(c.from_seq, c.to_seq, c.note_seq) for c in store.list_compactions(sid)]
    src_compacted = sorted(
        r.seq for r in store.load_all_messages(sid) if r.state is MessageState.COMPACTED_OUT
    )

    blob = store.export_session(sid)
    assert blob["schema_version"] == CURRENT_SCHEMA_VERSION
    # must be JSON-serializable (this is the archival contract)
    blob = json.loads(json.dumps(blob))

    new_id = store.import_session(blob, new_session_id="restored")

    assert new_id == "restored"
    # active window — including compact_note ordering — matches exactly
    assert [(r.role, r.name, r.content) for r in store.load_active_messages(new_id)] == src_active
    # the first active row is the compact_note at its logical position
    assert store.load_active_messages(new_id)[0].name == "compact_note"
    # compacted originals + compactions + ancillary state all came across
    assert sorted(
        r.seq for r in store.load_all_messages(new_id) if r.state is MessageState.COMPACTED_OUT
    ) == src_compacted
    assert [(c.from_seq, c.to_seq, c.note_seq) for c in store.list_compactions(new_id)] == src_compactions
    assert [n.content for n in store.list_notes(new_id)] == ["remember this"]
    assert [t.note for t in store.list_timers(new_id)] == ["recurring"]
    assert store.get_runtime_state(new_id, "scratch") == {"a": 1}
    sess = store.get_session(new_id)
    assert sess is not None and sess.system_prompt == "SP" and sess.metadata == {"k": "v"}


def test_import_rejects_unknown_target_collision(store: SessionStore) -> None:
    sid = _build_rich_session(store)
    blob = store.export_session(sid)
    with pytest.raises(ValueError, match="already exists"):
        store.import_session(blob, new_session_id=sid)  # collides with the source


def test_import_refuses_newer_schema(store: SessionStore) -> None:
    sid = _build_rich_session(store)
    blob = store.export_session(sid)
    blob["schema_version"] = CURRENT_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="newer than this power_loop build"):
        store.import_session(blob)


def test_export_unknown_session_raises(store: SessionStore) -> None:
    with pytest.raises(ValueError, match="unknown session"):
        store.export_session("nope")


def test_backup_produces_openable_copy(tmp_path: Path) -> None:
    src = SessionStore.open(str(tmp_path / "src.db"))
    sid = _build_rich_session(src)
    dest = str(tmp_path / "backup.db")
    src.backup(dest)
    src.close()
    # the backup is a complete, openable store with the same session
    restored = SessionStore.open(dest)
    try:
        assert restored.get_session(sid) is not None
        assert restored.load_active_messages(sid)[0].name == "compact_note"
    finally:
        restored.close()
    # and it is a valid SQLite file stamped at the current schema version
    raw = sqlite3.connect(dest)
    try:
        assert int(raw.execute("PRAGMA user_version").fetchone()[0]) == CURRENT_SCHEMA_VERSION
    finally:
        raw.close()
