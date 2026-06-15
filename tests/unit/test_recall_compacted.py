"""Unit tests for the recall_compacted tool (H7 Phase 1).

Surfaces THIS session's compacted-out (folded) messages on demand: keyword/seq
filters, most-recent cap, session isolation, active rows excluded.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.session_store import SessionStore
from power_loop.tools.default_tools import RECALL_COMPACTED_CONTENT_CHARS, run_recall_compacted

pytestmark = pytest.mark.unit


@contextmanager
def _active(store: SessionStore, sid: str):
    """Make (store, sid) the current tool runtime context (as a live loop would)."""
    loop = SimpleNamespace(store=store, config=None)
    t_loop = set_current_loop(loop)  # type: ignore[arg-type]
    t_sid = set_session_id(sid)
    try:
        yield
    finally:
        reset_session_id(t_sid)
        reset_current_loop(t_loop)


def _seed(store: SessionStore, sid: str, *, contents: list[tuple[str, str]]) -> list[int]:
    seqs = []
    for i, (role, content) in enumerate(contents):
        seqs.append(store.append_message(sid, role=role, content=content, round_index=i))
    return seqs


@pytest.fixture
def store():
    s = SessionStore.open(":memory:")
    yield s
    s.close()


def test_returns_folded_rows_and_excludes_active(store: SessionStore) -> None:
    sid = store.create_session(system_prompt="S")
    seqs = _seed(store, sid, contents=[
        ("user", "first question about apples"),
        ("assistant", "apples are red"),
        ("user", "second question about oranges"),
        ("assistant", "oranges are orange"),
        ("user", "latest live question"),  # stays ACTIVE
    ])
    # fold the first four (seqs[0..3]); seqs[4] stays active
    store.record_compaction(sid, from_seq=seqs[0], to_seq=seqs[3],
                            note_content="summary", before_tokens=100, after_tokens=10, round_index=0)

    with _active(store, sid):
        out = run_recall_compacted()

    assert "apples are red" in out and "oranges are orange" in out
    assert "latest live question" not in out  # ACTIVE row never surfaces
    assert "4 of 4 compacted" in out  # header reports the count


def test_query_filter_is_case_insensitive_substring(store: SessionStore) -> None:
    sid = store.create_session(system_prompt="S")
    seqs = _seed(store, sid, contents=[
        ("user", "tell me about APPLES"),
        ("assistant", "a fruit"),
        ("user", "tell me about cars"),
        ("assistant", "a vehicle"),
    ])
    store.record_compaction(sid, from_seq=seqs[0], to_seq=seqs[3],
                            note_content="n", before_tokens=1, after_tokens=1, round_index=0)

    with _active(store, sid):
        out = run_recall_compacted(query="apple")

    assert "APPLES" in out
    assert "cars" not in out and "vehicle" not in out


def test_seq_range_filter(store: SessionStore) -> None:
    sid = store.create_session(system_prompt="S")
    seqs = _seed(store, sid, contents=[
        ("user", "alpha"), ("assistant", "one"),
        ("user", "beta"), ("assistant", "two"),
    ])
    store.record_compaction(sid, from_seq=seqs[0], to_seq=seqs[3],
                            note_content="n", before_tokens=1, after_tokens=1, round_index=0)

    with _active(store, sid):
        out = run_recall_compacted(from_seq=seqs[2])  # only beta/two

    assert "beta" in out and "two" in out
    assert "alpha" not in out and "one" not in out


def test_limit_caps_to_most_recent(store: SessionStore) -> None:
    sid = store.create_session(system_prompt="S")
    contents = [("user" if i % 2 == 0 else "assistant", f"folded-{i}") for i in range(10)]
    seqs = _seed(store, sid, contents=contents)
    store.record_compaction(sid, from_seq=seqs[0], to_seq=seqs[-1],
                            note_content="n", before_tokens=1, after_tokens=1, round_index=0)

    with _active(store, sid):
        out = run_recall_compacted(limit=3)

    assert "3 of 10 compacted" in out
    assert "most recent" in out
    assert "folded-9" in out and "folded-7" in out  # newest kept
    assert "folded-0" not in out                     # oldest dropped


def test_empty_when_nothing_compacted(store: SessionStore) -> None:
    sid = store.create_session(system_prompt="S")
    _seed(store, sid, contents=[("user", "only active"), ("assistant", "reply")])
    with _active(store, sid):
        out = run_recall_compacted()
    assert "No compacted" in out


def test_session_isolation(store: SessionStore) -> None:
    sid_a = store.create_session(system_prompt="A")
    sid_b = store.create_session(system_prompt="B")
    a = _seed(store, sid_a, contents=[("user", "secret-A"), ("assistant", "ra")])
    b = _seed(store, sid_b, contents=[("user", "secret-B"), ("assistant", "rb")])
    store.record_compaction(sid_a, from_seq=a[0], to_seq=a[1],
                            note_content="n", before_tokens=1, after_tokens=1, round_index=0)
    store.record_compaction(sid_b, from_seq=b[0], to_seq=b[1],
                            note_content="n", before_tokens=1, after_tokens=1, round_index=0)

    with _active(store, sid_a):
        out = run_recall_compacted()

    assert "secret-A" in out
    assert "secret-B" not in out  # never leaks another session's folded rows


def test_no_active_session_raises(store: SessionStore) -> None:
    # outside any loop/session context the tool errors clearly rather than silently
    with pytest.raises(RuntimeError, match="No active"):
        run_recall_compacted()


def test_long_content_is_truncated(store: SessionStore) -> None:
    sid = store.create_session(system_prompt="S")
    big = "x" * (RECALL_COMPACTED_CONTENT_CHARS + 500)
    seqs = _seed(store, sid, contents=[("user", big), ("assistant", "ok")])
    store.record_compaction(sid, from_seq=seqs[0], to_seq=seqs[1],
                            note_content="n", before_tokens=1, after_tokens=1, round_index=0)

    with _active(store, sid):
        out = run_recall_compacted()

    assert "…[truncated]" in out
    assert out.count("x") <= RECALL_COMPACTED_CONTENT_CHARS + 10  # not the full blob
