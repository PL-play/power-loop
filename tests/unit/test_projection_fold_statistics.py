"""Fold statistics preserve the real render threshold, cursors and revision fencing.

The same cases run on SQLite and an explicitly configured disposable PostgreSQL
database. Counters measure finalization alone, independently of context loading.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import urlparse

import pytest

from power_loop import AgentLoopConfig, LLMResponse, StatefulAgentLoop
from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.fold import FoldResult
from power_loop.runtime.representation import ProjectedRepresentation, ProjectionFoldPolicy
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.projection_fold import OWNER_PREFIX, STATE_KEY, read_state
from power_loop.runtime.store.schema import _STORE_TABLES
from tests.unit.test_stateful_loop import _Scripted

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture(params=["sqlite", "postgres"])
async def fold_store(request, tmp_path):
    import os

    if request.param == "sqlite":
        dsn = str(tmp_path / "fold.sqlite")
    else:
        dsn = os.environ.get("POWER_LOOP_FOLD_TEST_PG_DSN")
        if not dsn:
            pytest.skip("Set POWER_LOOP_FOLD_TEST_PG_DSN to an isolated *_test database")
        assert urlparse(dsn).path.endswith("_test"), "refusing a non-test PostgreSQL database"
    store = await open_store(dsn, table_prefix="pf_")
    store.test_dsn = dsn
    try:
        if request.param == "postgres":
            actual = await store._db.fetchone("SELECT current_database() AS name")
            assert actual["name"].endswith("_test")
            await store._db.execute("TRUNCATE " + ",".join(getattr(store.t, t) for t in _STORE_TABLES))
        yield store
    finally:
        await store.close()


class ProjectionReads:
    def __init__(self, store, monkeypatch):
        self.reads = []
        table = store.t.project_messages

        def record(sql, result, locked):
            if table in sql and "SELECT *" in sql and "kind='compact'" not in sql:
                self.reads.append((len(result), locked, sql))

        original_fetch = store._db.fetchall

        async def fetch(sql, params=()):
            result = await original_fetch(sql, params)
            record(sql, result, False)
            return result

        original_transaction = store._db.transaction

        @asynccontextmanager
        async def transaction():
            async with original_transaction() as tx:
                class Transaction:
                    execute = tx.execute
                    fetchone = tx.fetchone

                    async def fetchall(self, sql, params=()):
                        result = await tx.fetchall(sql, params)
                        record(sql, result, True)
                        return result

                yield Transaction()

        monkeypatch.setattr(store._db, "fetchall", fetch)
        monkeypatch.setattr(store._db, "transaction", transaction)


def rows(n, *, text="answer", key=None):
    tools = [{"k": key, "s": "read result"}] if key else []
    return [("user", {"input": [f"question {n}"]}, None),
            ("project", {"tools": tools, "final_text": text}, None)]


async def append(store, sid, n, *, projector=None, threshold=10**9, keep=2, content=None):
    projector = projector or ProjectedRepresentation()
    return await store.prepare_send_projection_fold(
        sid, send_index=n, rows=content if content is not None else rows(n),
        source_seq_lo=None, source_seq_hi=None, projector_version=projector.version,
        projector=projector, keep_last_sends=keep, threshold=threshold,
    )


async def assert_exact(store, sid, projector):
    state = await read_state(store, store._db, sid)
    prior = await store.latest_project_compact(sid)
    live = await store.load_project_messages(sid, after_send_index=prior.compact_to_send if prior else None)
    sends = {r.send_index for r in live if r.kind in ("user", "project")}
    projector.stamp_render_context(live, max(sends, default=0) + 1)
    expected = estimate_tokens(projector.render(([prior] if prior else []) + live))
    assert state["tokens"] == expected
    assert state["sends"] == len(sends)
    assert state["rows"] == sum(r.kind in ("user", "project") for r in live)
    assert state["maximum"] == max(sends, default=0)
    return state


@pytest.mark.parametrize("history", [20, 400, 1200])
async def test_normal_finalization_does_not_read_historical_bodies(fold_store, monkeypatch, history):
    sid = await fold_store.create_session()
    for n in range(1, history + 1):
        assert await append(fold_store, sid, n) is None
    probe = ProjectionReads(fold_store, monkeypatch)
    assert await append(fold_store, sid, history + 1) is None
    assert probe.reads == []
    await assert_exact(fold_store, sid, ProjectedRepresentation())


@dataclass
class RecencyProjector(ProjectedRepresentation):
    hot_window: int = 2

    def fold_statistics_policy(self):
        return ProjectionFoldPolicy.for_renderer(self, config={"tag": self.render_config.user_tag},
                                                  hot_window=self.hot_window, deduplicate=True)

    def render_project_row(self, row):
        ctx = row.render_ctx
        visible = [t["s"] for t in row.content["tools"]
                   if ctx["latest_key"].get(t["k"], row.send_index) <= row.send_index]
        final = row.content["final_text"]
        if row.recency >= ctx["hot_window"]:
            final = final[:3]
        return {"role": "assistant", "content": " ".join([*visible, final])}


async def test_recency_and_cross_send_dedup_update_only_affected_rows(fold_store, monkeypatch):
    sid = await fold_store.create_session()
    projector = RecencyProjector()
    for n in [1, 2, 3, 7, 8, 9, 10]:
        with monkeypatch.context() as patch:
            probe = ProjectionReads(fold_store, patch)
            await append(fold_store, sid, n, projector=projector,
                         content=rows(n, text="long answer " * 200, key="file:/same"))
            assert sum(count for count, _, _ in probe.reads) <= 5
        await assert_exact(fold_store, sid, projector)


@pytest.mark.parametrize("offset", [0, 1])
async def test_exact_threshold_preserves_original_trigger(fold_store, offset):
    sid = await fold_store.create_session()
    projector = ProjectedRepresentation()
    for n in range(1, 4):
        await append(fold_store, sid, n, projector=projector)
    total = (await assert_exact(fold_store, sid, projector))["tokens"]
    # Re-finalize a send to verify that replacement does not double-count it.
    candidate = await append(fold_store, sid, 3, projector=projector, threshold=total + offset)
    assert (candidate is not None) == (offset == 0)
    await assert_exact(fold_store, sid, projector)


async def test_missing_statistics_and_render_version_rebuild_once(fold_store, monkeypatch):
    sid = await fold_store.create_session()
    projector = ProjectedRepresentation()
    for n in range(1, 6):
        for kind, content, text in rows(n):
            await fold_store.upsert_project_message(sid, send_index=n, kind=kind, content=content,
                                                    rendered_text=text, projector_version=1)
    for n in [6, 8, 10]:
        if n == 8:
            projector.render_config.user_tag = "longer-tag-{n}: "
        if n == 10:
            projector.version = 2
        with monkeypatch.context() as patch:
            probe = ProjectionReads(fold_store, patch)
            await append(fold_store, sid, n, projector=projector)
            assert probe.reads and all(not locked for _, locked, _ in probe.reads)
        await assert_exact(fold_store, sid, projector)
        with monkeypatch.context() as patch:
            probe = ProjectionReads(fold_store, patch)
            await append(fold_store, sid, n + 1, projector=projector)
            assert probe.reads == []


async def test_rewrite_restores_the_previous_dedup_owner(fold_store):
    sid = await fold_store.create_session()
    projector = RecencyProjector()
    for n, key in [(1, "a"), (2, "a"), (3, "b"), (2, None), (4, "a")]:
        await append(fold_store, sid, n, projector=projector,
                     content=rows(n, text="answer" * 200, key=key))
        await assert_exact(fold_store, sid, projector)
    owners = await fold_store._db.fetchall(
        f"SELECT state_key FROM {fold_store.t.session_runtime_state} WHERE session_id=?", (sid,),
    )
    assert sum(r["state_key"].startswith(OWNER_PREFIX) for r in owners) == 2


async def test_owner_generation_cleanup_spans_multiple_batches(fold_store):
    sid = await fold_store.create_session()
    projector = RecencyProjector()
    for n in range(1, 6):
        content = rows(n)
        content[1][1]["tools"] = [{"k": f"{n}/{i}", "s": "result"} for i in range(80)]
        await append(fold_store, sid, n, projector=projector, content=content)
    await append(fold_store, sid, 2, projector=projector, content=rows(2))
    await assert_exact(fold_store, sid, projector)
    owners = await fold_store._db.fetchall(
        f"SELECT state_key FROM {fold_store.t.session_runtime_state} WHERE session_id=?", (sid,),
    )
    assert sum(r["state_key"].startswith(OWNER_PREFIX) for r in owners) == 320


async def test_undeclared_global_renderer_uses_an_unlocked_exact_check(fold_store, monkeypatch):
    class GlobalRenderer(ProjectedRepresentation):
        def render(self, rows):
            return [{"role": "user", "content": "whole view: " + str([r.content for r in rows])}]

    projector = GlobalRenderer()
    assert projector.fold_statistics_policy() is None
    sid = await fold_store.create_session()
    for n in range(1, 4):
        assert await append(fold_store, sid, n, projector=projector) is None
    with monkeypatch.context() as patch:
        probe = ProjectionReads(fold_store, patch)
        assert await append(fold_store, sid, 4, projector=projector) is None
        assert probe.reads and all(not locked for _, locked, _ in probe.reads)
    candidate = await append(fold_store, sid, 5, projector=projector, threshold=0)
    assert len(candidate.rows) == 10


async def test_projection_and_statistics_rollback_together(fold_store, monkeypatch):
    sid = await fold_store.create_session()
    projector = RecencyProjector()
    await append(fold_store, sid, 1, projector=projector, content=rows(1, key="a"))
    before = await assert_exact(fold_store, sid, projector)
    original = fold_store._upsert_project_message_tx

    async def fail_after_user(*args, **kw):
        if kw["kind"] == "project":
            raise RuntimeError("crash before project row")
        return await original(*args, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(fold_store, "_upsert_project_message_tx", fail_after_user)
        with pytest.raises(RuntimeError, match="crash"):
            await append(fold_store, sid, 2, projector=projector, content=rows(2, key="a"))
    assert await read_state(fold_store, fold_store._db, sid) == before
    assert len(await fold_store.load_project_messages(sid)) == 2
    await append(fold_store, sid, 2, projector=projector, content=rows(2, key="a"))
    await assert_exact(fold_store, sid, projector)


async def test_changed_projection_rejects_a_fold_without_cursor_change(fold_store):
    sid = await fold_store.create_session()
    for n in range(1, 3):
        await append(fold_store, sid, n)
    candidate = await append(fold_store, sid, 3, threshold=0)
    other = await open_store(fold_store.test_dsn, table_prefix="pf_")
    try:
        await other.upsert_project_message(sid, send_index=1, kind="user", content={"input": ["corrected"]})
        assert await fold_store.commit_projection_fold(
            sid, content={"summary": "stale"}, rendered_text=None, from_send=1, to_send=1,
            projector_version=1, expected_prior_to_send=None,
            expected_projection_revision=candidate.revision,
        ) is False
        assert await fold_store.latest_project_compact(sid) is None
    finally:
        await other.close()


async def test_candidate_read_does_not_hold_the_session_lock(fold_store, monkeypatch):
    sid = await fold_store.create_session()
    for n in range(1, 140):
        await append(fold_store, sid, n)
    original = fold_store._db.fetchall
    other = await open_store(fold_store.test_dsn, table_prefix="pf_")
    changed = False

    async def concurrent_write(sql, params=()):
        nonlocal changed
        result = await original(sql, params)
        if "ORDER BY send_index,kind LIMIT" in sql and not changed:
            changed = True
            await asyncio.wait_for(other.upsert_project_message(
                sid, send_index=1, kind="user", content={"input": ["concurrent correction"]},
            ), timeout=2)
        return result

    try:
        monkeypatch.setattr(fold_store._db, "fetchall", concurrent_write)
        assert await append(fold_store, sid, 140, threshold=0) is None
        assert changed and await fold_store.latest_project_compact(sid) is None
        await append(fold_store, sid, 141)
        await assert_exact(fold_store, sid, ProjectedRepresentation())
    finally:
        await other.close()


@dataclass
class CaptureFold:
    keep_last_sends: int = 2
    trigger_ratio: float = 1
    fold_id: str = "capture"
    seen: list = field(default_factory=list)

    async def fold(self, rows, *, context):
        self.seen = rows
        return FoldResult({"summary": "all selected history"}, max(r.send_index for r in rows))


async def test_fold_reads_all_candidates_despite_history_cap_and_pages(fold_store, monkeypatch):
    sid = await fold_store.create_session(metadata={"projection_migrated": 0})
    for n in range(1, 322):
        await append(fold_store, sid, n)
        await fold_store.append_message(sid, role="user", content=f"original {n}", send_index=n)
    await fold_store.set_runtime_state(sid, "send_index", 321)
    fold = CaptureFold()
    loop = StatefulAgentLoop(store=fold_store, llm=_Scripted([LLMResponse(raw_text="done")]),
        config=AgentLoopConfig(system_prompt="S", representation=ProjectedRepresentation(),
                               fold_strategy=fold, context_budget_tokens=1, max_context_rows=6))
    probe = ProjectionReads(fold_store, monkeypatch)
    try:
        assert (await loop.send("next", sid)).status == "completed"
        assert {r.send_index for r in fold.seen} == set(range(1, 321))
        assert (await fold_store.latest_project_compact(sid)).compact_to_send == 320
        assert len(await fold_store.load_all_messages(sid)) == 323  # originals never pruned
        candidate_reads = [(count, locked) for count, locked, sql in probe.reads
                           if "ORDER BY send_index,kind LIMIT" in sql]
        assert len(candidate_reads) == 3 and sum(n for n, _ in candidate_reads) == 644
        assert all(not locked for _, locked in candidate_reads)
        await append(fold_store, sid, 323)
        await assert_exact(fold_store, sid, ProjectedRepresentation())
    finally:
        await loop.aclose()


async def test_statistics_survive_reopening_the_store(fold_store, monkeypatch):
    sid = await fold_store.create_session()
    for n in range(1, 6):
        await append(fold_store, sid, n)
    other = await open_store(fold_store.test_dsn, table_prefix="pf_")
    try:
        probe = ProjectionReads(other, monkeypatch)
        await append(other, sid, 6)
        assert probe.reads == []
        await assert_exact(other, sid, ProjectedRepresentation())
    finally:
        await other.close()


async def test_derived_statistics_are_not_exported_as_recovery_state(fold_store):
    sid = await fold_store.create_session()
    await fold_store.set_runtime_state(sid, "todo", {"text": "keep this"})
    await append(fold_store, sid, 1, projector=RecencyProjector(), content=rows(1, key="a"))
    exported = await fold_store.export_session(sid)
    assert [r["state_key"] for r in exported["tables"]["session_runtime_state"]] == ["todo"]
    restored = await fold_store.import_session(exported)
    assert await fold_store.get_runtime_state(restored, "todo") == {"text": "keep this"}
    assert await fold_store.get_runtime_state(restored, STATE_KEY) is None


async def test_corrupt_statistics_are_rebuilt_without_losing_projections(fold_store):
    sid = await fold_store.create_session()
    await append(fold_store, sid, 1)
    await fold_store._db.execute(
        f"UPDATE {fold_store.t.session_runtime_state} SET value_json=? WHERE session_id=? AND state_key=?",
        ("invalid JSON", sid, STATE_KEY),
    )
    await append(fold_store, sid, 2)
    await assert_exact(fold_store, sid, ProjectedRepresentation())
    assert len(await fold_store.load_project_messages(sid)) == 4


async def test_long_single_send_is_not_truncated_or_split(fold_store):
    sid = await fold_store.create_session()
    text = "tool transcript " * 20_000
    assert await append(fold_store, sid, 1, keep=1, threshold=1, content=rows(1, text=text)) is None
    candidate = await append(fold_store, sid, 2, keep=1, threshold=1)
    assert candidate.rows[1].content["final_text"] == text
    assert {r.send_index for r in candidate.rows} == {1, 2}


async def test_fold_cannot_advance_over_unprocessed_sends(fold_store):
    class BadCursorFold(CaptureFold):
        async def fold(self, rows, *, context):
            return FoldResult({"summary": "invalid cursor"}, max(r.send_index for r in rows) + 1)

    sid = await fold_store.create_session(metadata={"projection_migrated": 0})
    for n in range(1, 4):
        await append(fold_store, sid, n)
    await fold_store.set_runtime_state(sid, "send_index", 3)
    loop = StatefulAgentLoop(store=fold_store, llm=_Scripted([LLMResponse(raw_text="done")]),
        config=AgentLoopConfig(system_prompt="S", representation=ProjectedRepresentation(),
                               fold_strategy=BadCursorFold(), context_budget_tokens=1))
    try:
        assert (await loop.send("next", sid)).status == "completed"
        assert await fold_store.latest_project_compact(sid) is None
        assert len(await fold_store.load_project_messages(sid)) == 8
    finally:
        await loop.aclose()


@pytest.mark.parametrize("options", [{}, {"max_context_rows": None}, {"max_context_rows": 0},
                                     {"max_context_rows": 20}])
async def test_history_limit_contract_on_each_backend(fold_store, options):
    from tests.unit.test_max_context_rows import test_history_limit_is_explicit_beyond_previous_defaults

    await test_history_limit_is_explicit_beyond_previous_defaults(fold_store, options)
