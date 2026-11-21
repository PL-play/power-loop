"""Fold statistics preserve the real render threshold, cursors and revision fencing.

The same cases run on SQLite and an explicitly configured disposable PostgreSQL
database. Counters measure finalization alone, independently of context loading.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest

from power_loop import AgentLoopConfig, LLMResponse
from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.fold import FoldResult
from power_loop.runtime.representation import ProjectedRepresentation, ProjectionFoldPolicy
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.projection_fold import STATE_KEY, read_state
from tests.unit.session_native_fixture import ACTOR as STORE_ACTOR
from tests.unit.session_native_fixture import DESCRIPTOR as STORE_DESCRIPTOR
from tests.unit.test_stateful_loop import _Scripted

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]





@pytest.fixture
async def fold_store(registry_store, tmp_path):
    import os
    store, _ = registry_store
    backend = store._db.dialect.name
    store.test_dsn = str(tmp_path / 'registry.db') if backend == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if backend == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    return store


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
                    def __getattr__(self, name):
                        return getattr(tx, name)

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


async def append(store, sid, n, *, projector=None, threshold=10**9, keep=2, content=None, window=None):
    projector = projector or ProjectedRepresentation()
    return await store.prepare_send_projection_fold(
        sid, send_index=n, rows=content if content is not None else rows(n),
        source_seq_lo=None, source_seq_hi=None, projector_version=projector.version,
        projector=projector, keep_last_sends=keep, threshold=threshold, window=window,
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
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
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
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
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
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    projector = ProjectedRepresentation()
    for n in range(1, 4):
        await append(fold_store, sid, n, projector=projector)
    total = (await assert_exact(fold_store, sid, projector))["tokens"]
    # Re-finalize a send to verify that replacement does not double-count it.
    candidate = await append(fold_store, sid, 3, projector=projector, threshold=total + offset)
    assert (candidate is not None) == (offset == 0)
    await assert_exact(fold_store, sid, projector)


async def test_missing_statistics_and_render_version_rebuild_once(fold_store, monkeypatch):
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
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
    from power_loop.runtime.store.projection_fold import fold_dedup_hash

    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    projector = RecencyProjector()
    for n, key in [(1, "a"), (2, "a"), (3, "b"), (2, None), (4, "a")]:
        await append(fold_store, sid, n, projector=projector,
                     content=rows(n, text="answer" * 200, key=key))
        await assert_exact(fold_store, sid, projector)
    # Design 217 M0 Decision 2: 0 rows in session_runtime_state; inline dedup in sessions.fold_stats_json.
    kv_rows = await fold_store._db.fetchall(
        f"SELECT state_key FROM {fold_store.t.session_runtime_state} WHERE session_id=?", (sid,),
    )
    assert kv_rows == []
    state = await read_state(fold_store, fold_store._db, sid)
    assert state["v"] == 3
    assert state["dedup"] == {fold_dedup_hash("a"): 4, fold_dedup_hash("b"): 3}


async def test_owner_generation_cleanup_spans_multiple_batches(fold_store):
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    projector = RecencyProjector()
    for n in range(1, 6):
        content = rows(n)
        content[1][1]["tools"] = [{"k": f"{n}/{i}", "s": "result"} for i in range(80)]
        await append(fold_store, sid, n, projector=projector, content=content)
    await append(fold_store, sid, 2, projector=projector, content=rows(2))
    await assert_exact(fold_store, sid, projector)
    kv_rows = await fold_store._db.fetchall(
        f"SELECT state_key FROM {fold_store.t.session_runtime_state} WHERE session_id=?", (sid,),
    )
    assert kv_rows == []
    state = await read_state(fold_store, fold_store._db, sid)
    assert len(state["dedup"]) == 320
    assert all(len(k) == 12 for k in state["dedup"])


async def test_undeclared_global_renderer_uses_an_unlocked_exact_check(fold_store, monkeypatch):
    class GlobalRenderer(ProjectedRepresentation):
        def render(self, rows):
            return [{"role": "user", "content": "whole view: " + str([r.content for r in rows])}]

    projector = GlobalRenderer()
    assert projector.fold_statistics_policy() is None
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for n in range(1, 4):
        assert await append(fold_store, sid, n, projector=projector) is None
    with monkeypatch.context() as patch:
        probe = ProjectionReads(fold_store, patch)
        assert await append(fold_store, sid, 4, projector=projector) is None
        assert probe.reads and all(not locked for _, locked, _ in probe.reads)
    candidate = await append(fold_store, sid, 5, projector=projector, threshold=0)
    assert len(candidate.rows) == 10


async def test_projection_and_statistics_rollback_together(fold_store, monkeypatch):
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    projector = RecencyProjector()
    await append(fold_store, sid, 1, projector=projector, content=rows(1, key="a"))
    before = await assert_exact(fold_store, sid, projector)
    window = await fold_store.load_projection_window(sid, current_send_index=2)
    original = fold_store._upsert_project_messages_tx

    async def fail_after_user(tx, rows):
        if any(row[2] == "project" for row in rows):
            raise RuntimeError("crash before project row")
        return await original(tx, rows)

    with monkeypatch.context() as patch:
        patch.setattr(fold_store, "_upsert_project_messages_tx", fail_after_user)
        with pytest.raises(RuntimeError, match="crash"):
            await append(fold_store, sid, 2, projector=projector, content=rows(2, key="a"), window=window)
    assert await read_state(fold_store, fold_store._db, sid) == before
    assert len(await fold_store.load_project_messages(sid)) == 2
    await append(fold_store, sid, 2, projector=projector, content=rows(2, key="a"))
    await assert_exact(fold_store, sid, projector)


async def test_changed_projection_rejects_a_fold_without_cursor_change(fold_store):
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for n in range(1, 3):
        await append(fold_store, sid, n)
    candidate = await append(fold_store, sid, 3, threshold=0)
    other = await open_store(fold_store.test_dsn, table_prefix=fold_store.table_prefix)
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
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for n in range(1, 140):
        await append(fold_store, sid, n)
    original = fold_store._db.fetchall
    other = await open_store(fold_store.test_dsn, table_prefix=fold_store.table_prefix)
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


async def test_fold_reads_all_candidates_despite_history_cap_and_pages(fold_store, monkeypatch, runtime_agent):
    fold = CaptureFold()
    loop = await runtime_agent(store=fold_store, model=_Scripted([LLMResponse(raw_text="done")]),
        config=AgentLoopConfig(system_prompt="S", representation=ProjectedRepresentation(),
                               fold_strategy=fold, context_budget_tokens=1, max_context_rows=6))
    sid = loop.target.session_id
    for n in range(1, 322):
        await append(fold_store, sid, n)
        await fold_store.append_message(sid, role="user", content=f"original {n}", send_index=n)
    await fold_store._db.execute(
        f"UPDATE {fold_store.t.sessions} SET send_index=? WHERE session_id=?", (321, sid))

    probe = ProjectionReads(fold_store, monkeypatch)
    try:
        assert (await loop.send("next")).status == "completed"
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
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for n in range(1, 6):
        await append(fold_store, sid, n)
    other = await open_store(fold_store.test_dsn, table_prefix=fold_store.table_prefix)
    try:
        probe = ProjectionReads(other, monkeypatch)
        await append(other, sid, 6)
        assert probe.reads == []
        await assert_exact(other, sid, ProjectedRepresentation())
    finally:
        await other.close()


async def test_derived_statistics_are_not_exported_as_recovery_state(fold_store):
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    await fold_store.set_runtime_state(sid, "todo", {"text": "keep this"})
    await append(fold_store, sid, 1, projector=RecencyProjector(), content=rows(1, key="a"))
    exported = await fold_store.export_session(sid)
    assert [r["state_key"] for r in exported["tables"]["session_runtime_state"]] == ["todo"]
    # Schema 17 keeps export for audit but deliberately does not emulate the old
    # id-copying import (design 215); the derived statistics are absent either way.
    from power_loop.agent.runtime_types import RuntimeRequestError
    with pytest.raises(RuntimeRequestError, match="capability_unavailable"):
        await fold_store.import_session(exported)
    assert STATE_KEY not in str(exported["tables"]["session_runtime_state"])


async def test_corrupt_statistics_are_rebuilt_without_losing_projections(fold_store):
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    await append(fold_store, sid, 1)
    await fold_store._db.execute(
        f"UPDATE {fold_store.t.sessions} SET fold_stats_json=? WHERE session_id=?",
        ("invalid JSON", sid),
    )
    await append(fold_store, sid, 2)
    await assert_exact(fold_store, sid, ProjectedRepresentation())
    assert len(await fold_store.load_project_messages(sid)) == 4


async def test_long_single_send_is_not_truncated_or_split(fold_store):
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    text = "tool transcript " * 20_000
    assert await append(fold_store, sid, 1, keep=1, threshold=1, content=rows(1, text=text)) is None
    candidate = await append(fold_store, sid, 2, keep=1, threshold=1)
    assert candidate.rows[1].content["final_text"] == text
    assert {r.send_index for r in candidate.rows} == {1, 2}


async def test_fold_cannot_advance_over_unprocessed_sends(fold_store, runtime_agent):
    class BadCursorFold(CaptureFold):
        async def fold(self, rows, *, context):
            return FoldResult({"summary": "invalid cursor"}, max(r.send_index for r in rows) + 1)

    loop = await runtime_agent(store=fold_store, model=_Scripted([LLMResponse(raw_text="done")]),
        config=AgentLoopConfig(system_prompt="S", representation=ProjectedRepresentation(),
                               fold_strategy=BadCursorFold(), context_budget_tokens=1))
    sid = loop.target.session_id
    for n in range(1, 4):
        await append(fold_store, sid, n)
    await fold_store._db.execute(
        f"UPDATE {fold_store.t.sessions} SET send_index=? WHERE session_id=?", (3, sid))

    try:
        assert (await loop.send("next")).status == "completed"
        assert await fold_store.latest_project_compact(sid) is None
        assert len(await fold_store.load_project_messages(sid)) == 8
    finally:
        await loop.aclose()


async def test_waiting_fold_rejects_a_rewrite_committed_during_lock_wait(fold_store):
    """A real state-lock wait must see the revision the previous holder committed.

    PostgreSQL keeps the joined statistics row from before the wait; MySQL re-reads it under
    FOR UPDATE. The SQLite backend serializes writers, so this interleaving is not constructible.
    """
    if fold_store._db.dialect.name == "sqlite":
        pytest.skip("BEGIN IMMEDIATE serializes the writer/reader; no same-row lock interleaving")
    from tests.unit.test_session_inbox_batch_215 import _await_database_lock_wait

    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for n in (1, 2):
        await append(fold_store, sid, n)
    candidate = await append(fold_store, sid, 3, threshold=0)
    assert candidate is not None
    writer = await open_store(fold_store.test_dsn, table_prefix=fold_store.table_prefix)
    reader = await open_store(fold_store.test_dsn, table_prefix=fold_store.table_prefix)
    observer = await open_store(fold_store.test_dsn, table_prefix=fold_store.table_prefix)
    release, written = asyncio.Event(), asyncio.Event()
    original_tx = writer._db.transaction
    tasks = []
    detail = {}

    @asynccontextmanager
    async def held_tx():
        async with original_tx() as tx:
            yield tx
            detail["rewrite_revision"] = (await read_state(writer, tx, sid))["revision"]
            written.set()
            await release.wait()

    writer._db.transaction = held_tx
    try:
        rewrite = asyncio.create_task(writer.upsert_project_message(
            sid, send_index=1, kind="user", content={"input": ["concurrent corrected fact"]}))
        tasks.append(rewrite)
        await asyncio.wait_for(written.wait(), 10)
        fold = asyncio.create_task(reader.commit_projection_fold(
            sid, content={"summary": "OLD SNAPSHOT"}, rendered_text=None,
            from_send=1, to_send=1, projector_version=1, expected_prior_to_send=None,
            expected_projection_revision=candidate.revision))
        tasks.append(fold)
        # MySQL refreshes information_schema.innodb_trx on its own cadence; a short settle
        # makes the backend's LOCK WAIT evidence observable without treating a timeout as success.
        await asyncio.sleep(0.2)
        await _await_database_lock_wait(observer)
        release.set()
        await asyncio.wait_for(rewrite, 10)
        accepted = await asyncio.wait_for(fold, 10)
        assert accepted is False
        compact = await observer.latest_project_compact(sid)
        assert compact is None, compact.content if compact else None
        final = await read_state(observer, observer._db, sid)
        assert final["revision"] == detail["rewrite_revision"]
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer._db.transaction = original_tx
        await writer.close()
        await reader.close()
        await observer.close()


@pytest.mark.parametrize("carry_facts", [False, True])
async def test_waiting_persist_reads_statistics_after_the_lock(fold_store, carry_facts):
    from tests.unit.test_session_inbox_batch_215 import _await_database_lock_wait

    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    import json
    projector = RecencyProjector(hot_window=4)
    for n in range(1, 7):
        await append(fold_store, sid, n, projector=projector,
                     content=rows(n, text="original " * 80, key="shared"))
    window = await fold_store.load_projection_window(sid, current_send_index=8) if carry_facts else None
    writer = await open_store(fold_store.test_dsn, table_prefix=fold_store.table_prefix)
    observer = await open_store(fold_store.test_dsn, table_prefix=fold_store.table_prefix)
    release, written = asyncio.Event(), asyncio.Event()
    original_tx = writer._db.transaction
    tasks = []

    await writer.upsert_project_message(sid, send_index=4, kind="project",
        content=rows(4, text="CORRECTED " * 700, key="shared")[1][1], projector_version=1)

    @asynccontextmanager
    async def held_tx():
        published = False
        async with original_tx() as tx:
            class Tx:
                def __getattr__(self, name):
                    return getattr(tx, name)

                async def execute(self, sql, params=()):
                    nonlocal published
                    affected = await tx.execute(sql, params)
                    if f"UPDATE {writer.t.sessions} SET fold_stats_json=" in sql and params:
                        value = json.loads(params[0])
                        published = value.get("v") == 3 and value.get("maximum") == 7
                    return affected

            yield Tx()
            if published:
                written.set()
                await release.wait()

    writer._db.transaction = held_tx
    try:
        async with cas_trace(fold_store, sid, f"real-lock-wait-{carry_facts}") as probe:
            rewrite = asyncio.create_task(append(writer, sid, 7, projector=projector,
                content=rows(7, text="writer " * 80, key="shared")))
            tasks.append(rewrite)
            await asyncio.wait_for(written.wait(), 10)
            persisted = asyncio.create_task(append(fold_store, sid, 8, projector=projector,
                content=rows(8, text="waiter " * 80, key="shared"), window=window))
            tasks.append(persisted)
            await asyncio.sleep(0.2)
            if fold_store._db.dialect.name == "sqlite":
                assert writer._db._conn.in_transaction and probe.transactions == 1
                assert not probe.records and not persisted.done()
            else:
                await _await_database_lock_wait(observer)
            release.set()
            await asyncio.wait_for(rewrite, 10)
            await asyncio.wait_for(persisted, 10)
        assert not any(fold_store.t.session_runtime_state in r["sql"] for r in probe.records)
        await assert_exact(fold_store, sid, projector)
        final_rows = await fold_store.load_project_messages(sid)
        assert next(r for r in final_rows if r.send_index == 4 and r.kind == "project").content["final_text"] == "CORRECTED " * 700
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer._db.transaction = original_tx
        await writer.close()
        await observer.close()


@pytest.mark.parametrize("options", [{}, {"max_context_rows": None}, {"max_context_rows": 0},
                                     {"max_context_rows": 20}])
async def test_history_limit_contract_on_each_backend(fold_store, options, runtime_agent):
    from tests.unit.test_max_context_rows import test_history_limit_is_explicit_beyond_previous_defaults

    await test_history_limit_is_explicit_beyond_previous_defaults(fold_store, options, runtime_agent)


@asynccontextmanager
async def cas_trace(store, sid, label):
    """Keep actual SQL statements and transaction exits for inline fold_stats_json verification."""
    import json
    import os
    import time
    from pathlib import Path

    from .test_send_sql_profile import Profiled

    class Trace(Profiled):
        effects, exits, rollback_images = None, None, None

        @asynccontextmanager
        async def transaction(self):
            start = time.perf_counter()
            try:
                async with super().transaction() as tx:
                    self.enter_seconds.append(time.perf_counter() - start)
                    class Tx:
                        def __getattr__(self, name):
                            return getattr(tx, name)

                        async def execute(self, sql, params=()):
                            n = await tx.execute(sql, params)
                            self_trace.effects.append((sql, n))
                            return n

                        async def fetchone(self, sql, params=()):
                            started = time.perf_counter()
                            row = await tx.fetchone(sql, params)
                            self_trace.records[-1]["elapsed_s"] = time.perf_counter() - started
                            return row

                    self_trace = self
                    yield Tx()
                self.exits.append("commit")
            except BaseException as exc:
                self.exits.append(type(exc).__name__)
                raise

    original = store._db
    probe = Trace(original)
    probe.effects, probe.exits, probe.rollback_images = [], [], []
    probe.enter_seconds = []
    store._db = probe
    try:
        yield probe
    finally:
        store._db = original
        if path := os.environ.get("C02_TRACE_OUT"):
            with Path(path).open("a") as stream:
                stream.write(json.dumps({"backend": original.dialect.name, "case": label,
                    "records": probe.records, "effects": probe.effects,
                    "exits": probe.exits, "rollback_images": probe.rollback_images,
                    "transaction_enter_s": probe.enter_seconds}) + "\n")


def is_projection_cas(sql):
    return sql.startswith("UPDATE ") and "fold_stats_json=" in sql


def cas_effects(probe):
    return [n for sql, n in probe.effects if is_projection_cas(sql)]


@pytest.mark.parametrize("history", [0, 20])
async def test_inline_fold_stats_zero_kv_sql(fold_store, history):
    """Design 217 M1b: persist writes inline sessions.fold_stats_json with 0 session_runtime_state SQL."""
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for n in range(1, history + 1):
        await append(fold_store, sid, n)
    window = await fold_store.load_projection_window(sid, current_send_index=history + 1)
    async with cas_trace(fold_store, sid, f"inline-{history}") as probe:
        await append(fold_store, sid, history + 1, window=window)
    assert probe.transactions == 1
    assert not any(fold_store.t.session_runtime_state in r["sql"] for r in probe.records)
    await assert_exact(fold_store, sid, ProjectedRepresentation())


@pytest.mark.parametrize("bad", ["missing", "null", "json", "store", "policy", "version", "cutoff", "rewrite"])
async def test_cold_or_rewritten_inline_stats_keeps_history(fold_store, bad):
    from dataclasses import replace

    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    projector = ProjectedRepresentation()
    await append(fold_store, sid, 1, projector=projector)
    if bad in ("missing", "null", "json"):
        val = None if bad in ("missing", "null") else "{broken"
        await fold_store._db.execute(
            f"UPDATE {fold_store.t.sessions} SET fold_stats_json=? WHERE session_id=?",
            (val, sid),
        )
    window = await fold_store.load_projection_window(sid, current_send_index=2)
    if bad in ("store", "policy", "cutoff"):
        import json
        state = json.loads(window.stats.value_json)
        state[{"store": "store_id", "policy": "policy", "cutoff": "compact_to_send"}[bad]] = "wrong"
        window = replace(window, stats=replace(window.stats, value_json=json.dumps(state)))
    if bad == "version":
        projector.version += 1
    n = 1 if bad == "rewrite" else 2
    async with cas_trace(fold_store, sid, f"cold-{bad}") as probe:
        await append(fold_store, sid, n, projector=projector, window=window)
    assert not any(fold_store.t.session_runtime_state in r["sql"] for r in probe.records)
    await assert_exact(fold_store, sid, projector)
    assert {r.send_index for r in await fold_store.load_project_messages(sid)} == {1, n}


@pytest.mark.parametrize("compact", [None, 3])
@pytest.mark.parametrize("current,limit", [(13, None), (13, 5), (13, 2), (16, 5)])
async def test_inline_window_and_missing_ranges_match_full_statistics(fold_store, current, limit, compact):
    from dataclasses import replace

    from power_loop.runtime.store.projection_fold import fold_dedup_hash

    projector = RecencyProjector(hot_window=4)

    def content(index):
        value = rows(index, text=f"answer {index} " * 80, key=f"unique-{index}")
        value[1][1]["tools"] = [{"k": key, "s": f"result {key}"} for key in (
            ["old", "recent"] if index >= 13 else ["old" if index == 6 else "recent" if index == 11 else f"unique-{index}"])]
        return value

    results = []
    for reuse in (True, False):
        sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
        for n in range(1, 13):
            await append(fold_store, sid, n, projector=projector, content=content(n))
        if compact:
            state = await read_state(fold_store, fold_store._db, sid)
            assert await fold_store.commit_projection_fold(sid, content={"summary": "prefix"},
                rendered_text="prefix", from_send=1, to_send=compact, projector_version=1,
                expected_prior_to_send=None, expected_projection_revision=state["revision"])
            await append(fold_store, sid, 12, projector=projector, content=content(12))
        window = await fold_store.load_projection_window(sid, current_send_index=current,
            send_limit=limit) if reuse else None
        if window is not None:
            projector.stamp_render_context(window.rows, current)
            projector.render(window.rows)
            for row in window.rows:
                row.recency, row.render_ctx = -999, {"latest_key": {"old": 999}, "hot_window": -999}
            window = replace(window, rows=list(window.rows))
        async with cas_trace(fold_store, sid, f"reuse-{current}-{limit}-{compact}-{reuse}") as probe:
            candidate = await append(fold_store, sid, current, projector=projector,
                content=content(current), window=window, threshold=0)
        assert candidate is not None
        assert not any(fold_store.t.session_runtime_state in r["sql"] for r in probe.records)
        state = await assert_exact(fold_store, sid, projector)
        expected = {fold_dedup_hash(t["k"]): r.send_index for r in candidate.rows if r.kind == "project" for t in r.content["tools"]}
        assert state["dedup"] == expected
        results.append(({k: state[k] for k in ("tokens", "rows", "sends", "maximum", "compact_to_send")}, state["dedup"],
            [(r.send_index, r.kind, r.content) for r in candidate.rows],
            candidate.prior.content if candidate.prior else None))
    assert results[0] == results[1]


async def test_inline_covered_empty_send_exact(fold_store):
    projector = RecencyProjector(hot_window=4)
    sid = await fold_store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for n in (1, 2, 4, 5, 6):
        await append(fold_store, sid, n, projector=projector)
    window = await fold_store.load_projection_window(sid, current_send_index=7)
    async with cas_trace(fold_store, sid, "reuse-covered-empty") as probe:
        await append(fold_store, sid, 7, projector=projector, window=window)
    assert not any(fold_store.t.session_runtime_state in r["sql"] for r in probe.records)
    state = await assert_exact(fold_store, sid, projector)
    assert state["sends"] == 6 and state["rows"] == 12 and state["maximum"] == 7
