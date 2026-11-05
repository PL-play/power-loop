"""Store surface for the send-context projection layer (pl_project_messages, schema v2).

pl_messages stays the immutable audit log; this derived table holds one row per
projected send (kind=user/project) plus append-only compact rows. The core CRUD is
exercised via :func:`exercise_project_messages_crud` so the SAME assertions run against
SQLite (here) and against real Postgres/MySQL (test_store_parity_pg/_mysql). SQLite-only
here: cascade-delete and refusal of old formats during ordinary opening.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from power_loop.runtime.store.store import SessionStore


async def exercise_hook_events_crud(store: SessionStore) -> None:
    """Backend-agnostic CRUD conformance for the hook_events audit table: append a message with a
    hook injection → it lands as a linked event; per-session monotonic event_id; filter by
    message_seq; cleanup via close_session_tree. create_session mints a fresh id (shared-DB safe)."""
    sid = await store.create_session()
    hi = {"hook_point": "LLM_BEFORE", "hook": "builtin.memory_recall", "position": "tail",
          "kind": "inject", "payload": {"v": 1, "items": [{"name": "memory_0", "chars": 5}]}}
    seq1 = await store.append_message(
        sid, role="assistant", content="a", round_index=0, send_index=1, hook_injected=hi
    )
    await store.append_message(sid, role="user", content="next")  # no event
    seq2 = await store.append_message(
        sid, role="assistant", content="b", round_index=1, send_index=1, hook_injected=hi
    )
    evs = await store.list_hook_events(sid)
    assert [e.event_id for e in evs] == [1, 2]  # per-session monotonic
    assert [e.message_seq for e in evs] == [seq1, seq2]
    assert evs[0].hook == "builtin.memory_recall" and evs[0].position == "tail"
    assert evs[0].payload["items"][0]["name"] == "memory_0"
    assert len(await store.list_hook_events(sid, message_seq=seq1)) == 1
    await store.close_session_tree(sid)
    assert await store.list_hook_events(sid) == []


async def exercise_project_messages_crud(store: SessionStore) -> None:
    """Backend-agnostic CRUD conformance: upsert/load round-trip (incl. a follow_up human
    list), idempotent re-finalize (preserve created_at), and the compact read cursor.
    Uses distinct session_ids so it is safe to run against a shared, freshly-truncated DB."""
    # round-trip: (user list) + (project) for one send, loaded in kind order
    sid = "pm_roundtrip"
    await store.upsert_project_message(
        sid, send_index=1, kind="user", content={"at": "12:00", "human": ["hi", "still me"]},
        source_seq_lo=1, source_seq_hi=1, projector_version=3, token_estimate=5,
    )
    await store.upsert_project_message(
        sid, send_index=1, kind="project",
        content={"tools": [{"name": "bash", "s": "exit 0"}], "said": "ok"},
        rendered_text="工具: bash(exit 0). 我: ok", source_seq_lo=2, source_seq_hi=9,
        projector_version=3, token_estimate=12,
    )
    rows = await store.load_project_messages(sid)
    # conversation order: user before project within a send
    assert [(r.send_index, r.kind) for r in rows] == [(1, "user"), (1, "project")]
    assert next(r for r in rows if r.kind == "user").content == {
        "at": "12:00", "human": ["hi", "still me"]
    }
    proj = next(r for r in rows if r.kind == "project")
    assert proj.rendered_text == "工具: bash(exit 0). 我: ok" and proj.content["said"] == "ok"
    assert proj.source_seq_hi == 9 and proj.projector_version == 3

    # idempotent re-finalize: same (sid, send_index, kind) replaces, created_at preserved
    sid2 = "pm_idem"
    await store.upsert_project_message(sid2, send_index=2, kind="project", content={"v": 1})
    first = (await store.load_project_messages(sid2))[0]
    await store.upsert_project_message(sid2, send_index=2, kind="project", content={"v": 2})
    rows2 = await store.load_project_messages(sid2)
    assert len(rows2) == 1 and rows2[0].content == {"v": 2}
    assert rows2[0].created_at == first.created_at

    # compact cursor: latest_project_compact + after_send_index read window
    sid3 = "pm_compact"
    for n in (1, 2, 3, 4):
        await store.upsert_project_message(sid3, send_index=n, kind="project", content={"n": n})
    await store.upsert_project_message(
        sid3, send_index=2, kind="compact", content={"sum": "folded 1-2"},
        compact_from_send=1, compact_to_send=2,
    )
    latest = await store.latest_project_compact(sid3)
    assert latest is not None and latest.compact_from_send == 1 and latest.compact_to_send == 2
    after = await store.load_project_messages(sid3, after_send_index=latest.compact_to_send)
    assert sorted(r.send_index for r in after if r.kind == "project") == [3, 4]


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_project_messages_crud_sqlite(store: SessionStore) -> None:
    await exercise_project_messages_crud(store)


@pytest.mark.asyncio
async def test_close_session_cascades_project_messages(store: SessionStore) -> None:
    sid = await store.create_session()
    await store.upsert_project_message(sid, send_index=1, kind="project", content={"x": 1})
    assert await store.load_project_messages(sid)
    await store.close_session(sid)
    assert await store.load_project_messages(sid) == []


async def exercise_export_import_send_index(store: SessionStore) -> None:
    """Backend-agnostic (regression for #7): a v2 session round-trips with its send_index column
    intact (incl. a NULL legacy row), so projection still partitions by send after a restore.
    Distinct session ids → safe against a shared, freshly-truncated DB (run on PG/MySQL too)."""
    sid = "ei_si_src"
    await store.create_session(session_id=sid, system_prompt="S")
    await store.append_message(sid, role="user", content="u1", send_index=1)
    await store.append_message(sid, role="assistant", content="a1", send_index=1)
    await store.append_message(sid, role="user", content="u2", send_index=2)
    await store.append_message(sid, role="user", content="legacy")  # NULL send_index
    blob = await store.export_session(sid)
    new_id = await store.import_session(blob, new_session_id="ei_si_restored")
    rows = await store.load_active_messages(new_id)
    assert [(r.content, r.send_index) for r in rows] == [
        ("u1", 1), ("a1", 1), ("u2", 2), ("legacy", None),
    ]


@pytest.mark.asyncio
async def test_export_import_preserves_send_index(store: SessionStore) -> None:
    await exercise_export_import_send_index(store)




@pytest.mark.asyncio
async def test_v1_store_is_refused_without_rewriting_originals(tmp_path) -> None:
    import sqlite3

    from power_loop.runtime.store.schema import StoreSchemaError
    path = str(tmp_path / "m.db")
    s = await SessionStore.open(path)
    sid = await s.create_session()
    await s.append_message(sid, role="user", content="original")
    async with s._db.transaction() as tx:
        await tx.execute("DROP VIEW pl_usage_rounds")
        await tx.execute("DROP TABLE pl_project_messages")
        await tx.execute("DROP TABLE pl_hook_events")
        await tx.execute("ALTER TABLE pl_messages DROP COLUMN send_index")
        await tx.execute("UPDATE pl_schema_migrations SET version=1 WHERE id=1")
    await s.close()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as original:
        before = list(original.iterdump())
    with pytest.raises(StoreSchemaError, match="fresh store"):
        await SessionStore.open(path)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as original:
        assert list(original.iterdump()) == before
        assert original.execute("SELECT version FROM pl_schema_migrations").fetchone() == (1,)


@pytest.mark.asyncio
async def test_v2_store_is_refused_without_rewriting_originals(tmp_path) -> None:
    import sqlite3

    from power_loop.runtime.store.schema import StoreSchemaError
    path = str(tmp_path / "h.db")
    s = await SessionStore.open(path)
    sid = await s.create_session()
    await s.append_message(sid, role="user", content="original", send_index=7)
    async with s._db.transaction() as tx:
        await tx.execute("DROP TABLE pl_hook_events")
        await tx.execute("UPDATE pl_schema_migrations SET version=2 WHERE id=1")
    await s.close()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as original:
        before = list(original.iterdump())
    with pytest.raises(StoreSchemaError, match="fresh store"):
        await SessionStore.open(path)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as original:
        assert list(original.iterdump()) == before
        assert original.execute("SELECT version FROM pl_schema_migrations").fetchone() == (2,)

@pytest.mark.asyncio
async def test_load_active_messages_can_filter_to_one_send(store: SessionStore) -> None:
    """end-of-send 投影只要这一个 send 的行——以前是读全表再在 Python 里过滤。

    投影模式下 pl_messages 永不压缩，那张表只会一直长（真实会话已到 2332 行 / 2.7MB），
    而每个 send 都要为一小段做一次 O(会话) 的读。send_index=None 的行（compact note）
    不属于任何 send，必须排除——与它替代的那句 Python 过滤语义一致。
    """
    sid = await store.create_session()
    await store.append_message(sid, role="user", content="s1-u", round_index=0, send_index=1)
    await store.append_message(sid, role="assistant", content="s1-a", round_index=1, send_index=1)
    await store.append_message(sid, role="user", content="s2-u", round_index=0, send_index=2)
    await store.append_message(sid, role="assistant", content="no-send", round_index=0)

    all_rows = await store.load_active_messages(sid)
    assert len(all_rows) == 4

    s1 = await store.load_active_messages(sid, send_index=1)
    assert [r.content for r in s1] == ["s1-u", "s1-a"]
    s2 = await store.load_active_messages(sid, send_index=2)
    assert [r.content for r in s2] == ["s2-u"]
    assert await store.load_active_messages(sid, send_index=99) == []
    # 与它替代的 Python 过滤逐字等价
    assert [r.seq for r in s1] == [r.seq for r in all_rows if r.send_index == 1]
    await store.close_session_tree(sid)

