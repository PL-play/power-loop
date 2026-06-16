"""Conformance: the new async SessionStore must behave IDENTICALLY to the legacy sync
store across every method group. Each scenario runs the same operations on both (the
legacy store wrapped in a thin async adapter) and asserts structural equality of every
return value AND a final-state snapshot — the legacy store is the oracle, so any
translation drift is caught before callers are switched over.

Wall-clock ``*_at`` fields are zeroed for comparison (they differ run-to-run); everything
structural — seqs, ids, statuses, counts, ordering, folded sets, cascade counts,
export shape — is compared exactly.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from power_loop.runtime.session_store import SessionStore as LegacyStore
from power_loop.runtime.store.store import SessionStore as AsyncStore
from power_loop.runtime.store.types import SubagentLifecycle as Lc

pytestmark = pytest.mark.unit

_TS = {"created_at", "updated_at", "last_send_at", "first_send_at", "last_fired_at",
       "due_at", "last_seen_at"}


def _norm(v: Any) -> Any:
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: (0 if f.name in _TS else _norm(getattr(v, f.name)))
                for f in dataclasses.fields(v)}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: (0 if k in _TS else _norm(x)) for k, x in v.items()}
    return v


class _LegacyAsync:
    """Expose the sync legacy store with the same async surface (each call → to_thread)."""

    def __init__(self, store: LegacyStore) -> None:
        self._s = store

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._s, name)
        if callable(attr):
            async def _w(*a: Any, **k: Any) -> Any:
                return await asyncio.to_thread(attr, *a, **k)
            return _w
        return attr


async def _run_parity(
    scenario: Callable[[Any], Awaitable[list[Any]]],
    snapshot: Callable[[Any, str], Awaitable[Any]] | None = None,
    *,
    sid: str = "s",
) -> None:
    legacy = LegacyStore.open(":memory:")
    new = await AsyncStore.open(":memory:")
    try:
        legacy_out = await scenario(_LegacyAsync(legacy))
        new_out = await scenario(new)
        assert _norm(new_out) == _norm(legacy_out), "return-value parity"
        if snapshot is not None:
            assert _norm(await snapshot(new, sid)) == _norm(await asyncio.to_thread(
                lambda: _sync_snapshot(legacy, sid))), "final-state parity"
    finally:
        legacy.close()
        await new.close()


def _sync_snapshot(store: LegacyStore, sid: str) -> Any:
    return (
        store.load_active_messages(sid), store.load_all_messages(sid),
        store.get_state(sid), store.list_compactions(sid),
        store.list_timers(sid, statuses=("armed", "firing", "fired", "cancelled")),
        store.list_notes(sid), store.get_session_stats(sid),
        store.list_background_tasks(sid),
    )


async def _async_snapshot(store: AsyncStore, sid: str) -> Any:
    return (
        await store.load_active_messages(sid), await store.load_all_messages(sid),
        await store.get_state(sid), await store.list_compactions(sid),
        await store.list_timers(sid, statuses=("armed", "firing", "fired", "cancelled")),
        await store.list_notes(sid), await store.get_session_stats(sid),
        await store.list_background_tasks(sid),
    )


# ── scenarios ───────────────────────────────────────────────────────────────


async def test_parity_messages_and_compaction() -> None:
    async def scn(st: Any) -> list[Any]:
        await st.create_session(session_id="s")
        out = [await st.append_message("s", role="user", content=f"m{i}", round_index=i)
               for i in range(5)]
        out.append(await st.record_compaction(
            "s", from_seq=1, to_seq=2, note_content="sum", before_tokens=9, after_tokens=1,
            round_index=2, fold_seqs=[1, 2], order_key=1))
        return out
    await _run_parity(scn, _wrap_snapshot())


async def test_parity_timers() -> None:
    async def scn(st: Any) -> list[Any]:
        await st.create_session(session_id="s")
        out: list[Any] = []
        out.append((await st.create_timer("s", due_at=1000, note="a")).timer_id)
        out.append((await st.create_timer("s", due_at=2000, note="b", interval_s=60)).timer_id)
        out.append(await st.transition_timer("s", 1, from_status="armed", to_status="firing"))
        out.append(await st.transition_timer("s", 1, from_status="armed", to_status="firing"))  # lost
        out.append(await st.finish_firing_timer("s", 1))  # one-shot → fired
        out.append(await st.transition_timer("s", 2, from_status="armed", to_status="firing"))
        out.append(await st.finish_firing_timer("s", 2))  # recurring → re-armed
        out.append([t.timer_id for t in await st.list_timers("s")])
        out.append([t.timer_id for t in await st.due_timers(now=5000)])
        out.append(await st.prune_timers("s"))
        return out
    await _run_parity(scn, _wrap_snapshot())


async def test_parity_notes() -> None:
    async def scn(st: Any) -> list[Any]:
        await st.create_session(session_id="s")
        out: list[Any] = []
        out.append((await st.add_note("s", "first")).note_id)
        out.append((await st.add_note("s", "second", pinned=True)).note_id)
        out.append(await st.update_note("s", 1, content="edited"))
        out.append(await st.update_note("s", 99, pinned=True))  # missing → False
        out.append(await st.count_notes("s"))
        out.append([(n.note_id, n.content, n.pinned) for n in await st.list_notes("s")])
        out.append(await st.delete_note("s", 2))
        return out
    await _run_parity(scn, _wrap_snapshot())


async def test_parity_usage_and_stats() -> None:
    async def scn(st: Any) -> list[Any]:
        await st.create_session(session_id="s")
        await st.record_usage("s", round_index=0, prompt_tokens=10, completion_tokens=5,
                              total_tokens=15, model="m")
        await st.record_usage("s", round_index=0, prompt_tokens=11, completion_tokens=6,
                              total_tokens=17)  # overwrite same round
        await st.bump_session_stats("s", {"calls": 1, "prompt_tokens": 10,
                                          "completion_tokens": 5, "total_tokens": 15},
                                    rounds=2, tool_calls=3)
        await st.bump_session_stats("s", {"calls": 2, "prompt_tokens": 20,
                                          "completion_tokens": 10, "total_tokens": 30},
                                    rounds=1, tool_calls=1)  # accumulate
        s = await st.get_session_stats("s")
        return [s.sends, s.rounds, s.llm_calls, s.tool_calls, s.prompt_tokens,
                s.completion_tokens, s.total_tokens]
    await _run_parity(scn, _wrap_snapshot())


async def test_parity_runtime_and_shared_state() -> None:
    async def scn(st: Any) -> list[Any]:
        await st.create_session(session_id="s")
        out: list[Any] = []
        await st.set_runtime_state("s", "k", {"a": 1})
        out.append(await st.get_runtime_state("s", "k"))
        await st.set_runtime_state("s", "k", {"a": 2})  # overwrite
        out.append(await st.get_runtime_state("s", "k"))
        await st.delete_runtime_state("s", "k")
        out.append(await st.get_runtime_state("s", "k", default="gone"))
        await st.set_shared_state("owner1", "x", [1, 2, 3])
        out.append(await st.get_shared_state("owner1", "x"))
        return out
    await _run_parity(scn)


async def test_parity_background_tasks() -> None:
    async def scn(st: Any) -> list[Any]:
        await st.create_session(session_id="s")
        out: list[Any] = []
        await st.upsert_background_task("s", task_id="t1", command="cmd", status="running")
        out.append([t.task_id for t in await st.list_unseen_background_updates("s")])
        await st.mark_background_seen("s", ["t1"])
        out.append([t.task_id for t in await st.list_unseen_background_updates("s")])  # seen → []
        await st.upsert_background_task("s", task_id="t1", command="cmd", status="done",
                                        return_code=0, output_tail="ok")
        # monotonic bump: the update must reappear as unseen even if wall-clock didn't move
        out.append([t.task_id for t in await st.list_unseen_background_updates("s")])
        bt = await st.get_background_task("s", "t1")
        out.append((bt.status, bt.return_code, bt.output_tail))
        return out
    await _run_parity(scn, _wrap_snapshot())


async def test_parity_session_cascade() -> None:
    async def scn(st: Any) -> list[Any]:
        await st.create_session(session_id="p")
        await st.create_session(session_id="linked", parent_session_id="p", lifecycle=Lc.LINKED)
        await st.create_session(session_id="eph", parent_session_id="p", lifecycle=Lc.EPHEMERAL)
        await st.create_session(session_id="det", parent_session_id="p", lifecycle=Lc.DETACHED)
        removed = await st.close_session("p", cascade=True)
        survivors = [bool(await st.get_session(x)) for x in ("p", "linked", "eph", "det")]
        det = await st.get_session("det")
        return [removed, survivors, det.parent_session_id if det else None]
    await _run_parity(scn)


async def test_parity_export_import_roundtrip() -> None:
    async def scn(st: Any) -> list[Any]:
        await st.create_session(session_id="src")
        for i in range(3):
            await st.append_message("src", role="user", content=f"m{i}")
        await st.add_note("src", "note")
        await st.create_timer("src", due_at=1000, note="t")
        blob = await st.export_session("src")
        new_id = await st.import_session(blob, new_session_id="restored")
        msgs = [m.content for m in await st.load_active_messages("restored")]
        sess = await st.get_session("restored")
        return [new_id, sorted(blob["tables"].keys()), msgs,
                sess.parent_session_id, sess.kind.value]
    await _run_parity(scn)


# ── snapshot wrapper (the new store's async snapshot; legacy uses the sync one) ──
def _wrap_snapshot() -> Callable[[Any, str], Awaitable[Any]]:
    return _async_snapshot
