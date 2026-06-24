"""Store surface for the send-context projection layer (pl_project_messages, schema v2).

pl_messages stays the immutable audit log; this derived table holds one row per
projected send (kind=user/project) plus append-only compact rows. The core CRUD is
exercised via :func:`exercise_project_messages_crud` so the SAME assertions run against
SQLite (here) and against real Postgres/MySQL (test_store_parity_pg/_mysql). SQLite-only
here: cascade-delete and the v1→v2 migration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from power_loop.runtime.store.schema import CURRENT_SCHEMA_VERSION
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
async def test_migration_ddl_for_display_includes_alter(store: SessionStore) -> None:
    # The migration-FAILURE error must surface the ACTUAL steps incl. the ADD COLUMN that
    # provisioning_ddl omits (regression for #4).
    from power_loop.runtime.store.schema import migration_ddl_for_display
    joined = " ".join(migration_ddl_for_display(store._db, "pl_", from_version=1))
    assert "ADD COLUMN send_index" in joined and "project_messages" in joined
    assert "hook_events" in joined  # v2→v3 step is in the full ladder from v1


@pytest.mark.asyncio
async def test_v1_to_current_migration_adds_all_tables_and_columns(tmp_path) -> None:
    path = str(tmp_path / "m.db")
    s = await SessionStore.open(path)  # fresh → current
    assert s.schema_version == CURRENT_SCHEMA_VERSION
    # Simulate a true pre-existing v1 store: drop every post-v1 artifact (the v2 project_messages
    # table + send_index column, the v3 hook_events table), then roll the version stamp back to 1.
    async with s._db.transaction() as tx:
        await tx.execute("DROP TABLE pl_project_messages")
        await tx.execute("DROP TABLE pl_hook_events")
        await tx.execute("ALTER TABLE pl_messages DROP COLUMN send_index")
        await tx.execute("UPDATE pl_schema_migrations SET version=1 WHERE id=1")
    await s.close()

    s2 = await SessionStore.open(path)  # AUTO_CREATE runs the full v1→current migration ladder
    assert s2.schema_version == CURRENT_SCHEMA_VERSION
    # (a) the project_messages table is back
    await s2.upsert_project_message("sess_m", send_index=1, kind="project", content={"ok": True})
    assert (await s2.load_project_messages("sess_m"))[0].content == {"ok": True}
    # (b) the send_index column was re-added by the ALTER (write + read it back)
    sid = await s2.create_session()
    await s2.append_message(sid, role="user", content="hi", send_index=7)
    assert (await s2.load_active_messages(sid))[0].send_index == 7
    # (c) the hook_events table is back and writable
    seq = await s2.append_message(
        sid, role="assistant", content="a", round_index=0, send_index=7,
        hook_injected={"hook_point": "LLM_BEFORE", "hook": "x", "position": "tail",
                       "kind": "inject", "payload": {"v": 1, "items": []}},
    )
    evs = await s2.list_hook_events(sid)
    assert len(evs) == 1 and evs[0].message_seq == seq
    await s2.close()

    # idempotent: reopening (already current) is a no-op and everything survives
    s3 = await SessionStore.open(path)
    assert s3.schema_version == CURRENT_SCHEMA_VERSION
    assert (await s3.load_active_messages(sid))[0].send_index == 7
    assert len(await s3.list_hook_events(sid)) == 1
    await s3.close()


@pytest.mark.asyncio
async def test_v2_to_v3_migration_adds_hook_events(tmp_path) -> None:
    path = str(tmp_path / "h.db")
    s = await SessionStore.open(path)  # fresh → current (v3)
    # Simulate a pre-existing v2 store: drop the v3 hook_events table, roll the stamp back to 2.
    async with s._db.transaction() as tx:
        await tx.execute("DROP TABLE pl_hook_events")
        await tx.execute("UPDATE pl_schema_migrations SET version=2 WHERE id=1")
    await s.close()

    s2 = await SessionStore.open(path)  # AUTO_CREATE runs just the v2→v3 step
    assert s2.schema_version == CURRENT_SCHEMA_VERSION
    sid = await s2.create_session()
    seq = await s2.append_message(
        sid, role="assistant", content="a", round_index=0,
        hook_injected={"hook_point": "LLM_BEFORE", "hook": "builtin.memory_recall",
                       "position": "tail", "kind": "inject",
                       "payload": {"v": 1, "items": [{"name": "memory_0", "chars": 3}]}},
    )
    evs = await s2.list_hook_events(sid)
    assert len(evs) == 1
    assert evs[0].message_seq == seq and evs[0].hook == "builtin.memory_recall"
    assert evs[0].payload["items"][0]["name"] == "memory_0"
    await s2.close()
