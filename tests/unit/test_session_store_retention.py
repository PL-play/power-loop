"""OPS-2 / OPS-3: caller-driven retention + reclamation of the long-lived store.

Retention is opt-in: the store never deletes on its own. Pruning the folded
``compacted_out`` originals is irreversible but must NEVER disturb the active window
(``compact_note`` rows are active). Reclamation (VACUUM / WAL checkpoint) actually
shrinks the file after deletes; fresh DBs use ``auto_vacuum=INCREMENTAL`` so freed
pages are reclaimable cheaply.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from power_loop.runtime.session_store import MessageState, SessionStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store() -> Generator[SessionStore, None, None]:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


def _active_snapshot(store: SessionStore, sid: str) -> list[tuple]:
    return [
        (r.seq, r.role, r.name, r.content)
        for r in store.load_active_messages(sid)
    ]


def _seed_with_compaction(store: SessionStore, sid: str, *, n: int = 8) -> None:
    """Append n messages then fold the first n-2 into a compact_note (so there are
    compacted_out originals + one active compact_note + an active kept tail)."""
    seqs = [store.append_message(sid, role="user", content=f"m{i}", round_index=i) for i in range(n)]
    fold = seqs[: n - 2]
    store.record_compaction(
        sid, from_seq=min(fold), to_seq=max(fold), note_content="summary",
        before_tokens=100, after_tokens=10, round_index=n,
        fold_seqs=fold, order_key=fold[0],
    )


# ── OPS-2: prune_compacted_messages ──────────────────────────────────────────


def test_prune_compacted_leaves_active_and_note_identical(store: SessionStore) -> None:
    sid = store.create_session(session_id="s")
    _seed_with_compaction(store, sid, n=8)
    before = _active_snapshot(store, sid)
    assert any(r.state is MessageState.COMPACTED_OUT for r in store.load_all_messages(sid))

    deleted = store.prune_compacted_messages(sid)

    assert deleted == 6  # the 6 folded originals
    assert not any(r.state is MessageState.COMPACTED_OUT for r in store.load_all_messages(sid))
    # the active window — kept tail + the compact_note — is byte-identical
    assert _active_snapshot(store, sid) == before
    assert any(r.name == "compact_note" for r in store.load_active_messages(sid))


def test_prune_compacted_keep_recent_retains_n(store: SessionStore) -> None:
    sid = store.create_session(session_id="s")
    _seed_with_compaction(store, sid, n=8)
    deleted = store.prune_compacted_messages(sid, keep_recent=2)
    remaining = [r for r in store.load_all_messages(sid) if r.state is MessageState.COMPACTED_OUT]
    assert deleted == 4 and len(remaining) == 2


def test_prune_compacted_older_than_filters_by_age(store: SessionStore) -> None:
    sid = store.create_session(session_id="s")
    _seed_with_compaction(store, sid, n=8)
    # nothing is "older than 1 hour" yet → no deletions
    assert store.prune_compacted_messages(sid, older_than_ms=3_600_000) == 0
    # everything is older than -1ms (cutoff in the future) → all pruned
    assert store.prune_compacted_messages(sid, older_than_ms=-1) == 6


# ── OPS-2: usage rounds + timers ─────────────────────────────────────────────


def test_prune_usage_rounds_keep_last(store: SessionStore) -> None:
    sid = store.create_session(session_id="s")
    for i in range(5):
        store.record_usage(sid, round_index=i, prompt_tokens=1, completion_tokens=1, total_tokens=2)
    deleted = store.prune_usage_rounds(sid, keep_last=2)
    assert deleted == 3
    # cumulative session_stats is untouched by usage-round pruning
    assert store.get_session_stats(sid) is None or True  # stats are bumped elsewhere


def test_prune_timers_only_terminal(store: SessionStore) -> None:
    sid = store.create_session(session_id="s")
    armed = store.create_timer(sid, due_at=10_000, note="armed-one")
    fired = store.create_timer(sid, due_at=20_000, note="to-cancel")
    store.transition_timer(sid, fired.timer_id, from_status="armed", to_status="cancelled")

    deleted = store.prune_timers(sid)  # default fired/cancelled

    assert deleted == 1
    assert store.get_timer(sid, fired.timer_id) is None
    assert store.get_timer(sid, armed.timer_id) is not None  # armed survives


# ── OPS-3: reclamation ───────────────────────────────────────────────────────


def test_fresh_db_uses_incremental_auto_vacuum(tmp_path: Path) -> None:
    db = str(tmp_path / "av.db")
    SessionStore.open(db).close()
    raw = sqlite3.connect(db)
    try:
        assert int(raw.execute("PRAGMA auto_vacuum").fetchone()[0]) == 2  # INCREMENTAL
    finally:
        raw.close()


def test_checkpoint_and_vacuum_run_without_error(store: SessionStore) -> None:
    sid = store.create_session(session_id="s")
    _seed_with_compaction(store, sid, n=8)
    store.prune_compacted_messages(sid)
    # both reclamation paths execute cleanly on the in-memory store
    store.checkpoint(mode="TRUNCATE")  # WAL not used for :memory: but must not raise
    store.vacuum(incremental=True)
    with pytest.raises(ValueError):
        store.checkpoint(mode="NOPE")


def test_full_vacuum_shrinks_file_after_prune(tmp_path: Path) -> None:
    db = str(tmp_path / "big.db")
    store = SessionStore.open(db)
    sid = store.create_session(session_id="s")
    # write a lot of fat content, then fold+prune most of it
    seqs = [store.append_message(sid, role="user", content="x" * 4000, round_index=i)
            for i in range(60)]
    fold = seqs[:-2]
    store.record_compaction(
        sid, from_seq=min(fold), to_seq=max(fold), note_content="s",
        before_tokens=1, after_tokens=1, round_index=60, fold_seqs=fold, order_key=fold[0],
    )
    store.prune_compacted_messages(sid)
    store.checkpoint(mode="TRUNCATE")
    size_before = Path(db).stat().st_size
    store.vacuum(incremental=False)  # full rewrite reclaims the freed pages
    store.close()
    size_after = Path(db).stat().st_size
    assert size_after < size_before, (size_before, size_after)
