"""Phase 1 vertical slice: the async SessionStore on the SQLite backend.

Proves the new abstraction end-to-end for the hardest paths — transactional per-session
seq allocation, compaction (explicit fold-set + logical ord ordering), atomic rollback,
the ``pl_`` table prefix, and the schema-init contract — and asserts behavior PARITY with
the legacy synchronous store for those operations.
"""

from __future__ import annotations

import pytest

from power_loop.runtime.session_store import SessionStore as LegacyStore
from power_loop.runtime.store.schema import CURRENT_SCHEMA_VERSION, StoreSchemaError
from power_loop.runtime.store.store import SessionStore

pytestmark = pytest.mark.unit


async def test_create_append_load_and_seq_monotonic() -> None:
    async with await SessionStore.open(":memory:") as store:
        sid = await store.create_session(system_prompt="hi", metadata={"k": "v"})
        sess = await store.get_session(sid)
        assert sess is not None and sess.system_prompt == "hi" and sess.metadata == {"k": "v"}

        seqs = []
        for i in range(4):
            seqs.append(await store.append_message(
                sid, role="user" if i % 2 == 0 else "assistant", content=f"m{i}", round_index=i))
        assert seqs == [1, 2, 3, 4]  # per-session, gap-free, monotonic
        assert (await store.get_state(sid)).next_seq == 5

        active = await store.load_active_messages(sid)
        assert [m.content for m in active] == ["m0", "m1", "m2", "m3"]


async def test_compaction_marks_exact_set_and_orders_note_logically() -> None:
    async with await SessionStore.open(":memory:") as store:
        sid = await store.create_session()
        for i in range(4):
            await store.append_message(sid, role="user", content=f"m{i}")
        # fold the first two; the note logically sits at ord=1 (front)
        compact_seq, note_seq = await store.record_compaction(
            sid, from_seq=1, to_seq=2, note_content="summary", before_tokens=100,
            after_tokens=10, round_index=1, fold_seqs=[1, 2], order_key=1)
        assert (compact_seq, note_seq) == (1, 5)

        active = await store.load_active_messages(sid)
        # note (ord=1) sorts before the kept tail despite its high identity seq=5
        assert [(m.name or m.role, m.seq) for m in active] == [
            ("compact_note", 5), ("user", 3), ("user", 4)]
        all_msgs = await store.load_all_messages(sid)
        folded = {m.seq for m in all_msgs if m.state.value == "compacted_out"}
        assert folded == {1, 2}
        comps = await store.list_compactions(sid)
        assert len(comps) == 1 and comps[0].note_seq == 5


async def test_transaction_rolls_back_on_error() -> None:
    async with await SessionStore.open(":memory:") as store:
        with pytest.raises(ValueError, match="unknown session"):
            await store.append_message("ghost", role="user", content="x")
        # the failed append left no row and no seq bump (state row never existed)
        assert await store.get_state("ghost") is None


async def test_pl_prefix_isolates_tables() -> None:
    async with await SessionStore.open(":memory:") as store:
        await store.create_session(session_id="s")
        names = {r["name"] for r in await store._db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "pl_sessions" in names and "pl_messages" in names
        assert "sessions" not in names  # unprefixed names are NOT created


async def test_create_schema_false_on_fresh_db_raises(tmp_path) -> None:
    db_path = str(tmp_path / "fresh.db")
    with pytest.raises(StoreSchemaError, match="not initialized"):
        await SessionStore.open(db_path, create_schema=False)
    # provisioning then reopening read-only works
    s = await SessionStore.open(db_path, create_schema=True)
    await s.close()
    s2 = await SessionStore.open(db_path, create_schema=False)
    assert s2.schema_version == CURRENT_SCHEMA_VERSION
    await s2.close()


async def test_parity_with_legacy_store_for_compaction() -> None:
    """Same ops on the legacy sync store and the new async store yield identical
    active-message logical ordering + folded set."""
    def drive_legacy() -> list[tuple[str, int]]:
        st = LegacyStore.open(":memory:")
        try:
            sid = st.create_session()
            for i in range(4):
                st.append_message(sid, role="user", content=f"m{i}")
            st.record_compaction(sid, from_seq=1, to_seq=2, note_content="s",
                                 before_tokens=9, after_tokens=1, round_index=1,
                                 fold_seqs=[1, 2], order_key=1)
            return [(m.name or m.role, m.seq) for m in st.load_active_messages(sid)]
        finally:
            st.close()

    legacy = drive_legacy()
    async with await SessionStore.open(":memory:") as store:
        sid = await store.create_session()
        for i in range(4):
            await store.append_message(sid, role="user", content=f"m{i}")
        await store.record_compaction(sid, from_seq=1, to_seq=2, note_content="s",
                                      before_tokens=9, after_tokens=1, round_index=1,
                                      fold_seqs=[1, 2], order_key=1)
        new = [(m.name or m.role, m.seq) for m in await store.load_active_messages(sid)]

    assert new == legacy
