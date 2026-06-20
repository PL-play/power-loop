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
    assert [(r.send_index, r.kind) for r in rows] == [(1, "project"), (1, "user")]
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


@pytest.mark.asyncio
async def test_v1_to_v2_migration_adds_project_messages(tmp_path) -> None:
    path = str(tmp_path / "m.db")
    s = await SessionStore.open(path)  # fresh → v2 (table present)
    assert s.schema_version == CURRENT_SCHEMA_VERSION == 2
    # Simulate a pre-existing v1 store: drop the v2 table and roll the stamp back to 1.
    async with s._db.transaction() as tx:
        await tx.execute("DROP TABLE pl_project_messages")
        await tx.execute("UPDATE pl_schema_migrations SET version=1 WHERE id=1")
    await s.close()
    s2 = await SessionStore.open(path)  # AUTO_CREATE runs the v1→v2 migration ladder
    assert s2.schema_version == 2
    await s2.upsert_project_message("sess_m", send_index=1, kind="project", content={"ok": True})
    assert (await s2.load_project_messages("sess_m"))[0].content == {"ok": True}
    await s2.close()
