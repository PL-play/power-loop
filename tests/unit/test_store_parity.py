"""Backend conformance against the current async SQLite implementation.

The same scenarios run on PostgreSQL/MySQL and a fresh current SQLite store,
comparing every return value and final-state snapshot. This retains dialect,
transaction and data-size coverage without a synchronous runtime or adapter.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from power_loop.runtime.store.store import SessionStore as AsyncStore
from tests.unit.session_native_fixture import ACTOR as STORE_ACTOR
from tests.unit.session_native_fixture import DESCRIPTOR as STORE_DESCRIPTOR

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


async def _async_snapshot(store: AsyncStore, sid: str) -> Any:
    return (
        await store.load_active_messages(sid), await store.load_all_messages(sid),
        await store.get_state(sid), await store.list_compactions(sid),
        await store.list_timers(sid, statuses=("armed", "firing", "fired", "cancelled")),
        await store.list_notes(sid), await store.get_session_stats(sid),
    )


async def run_parity(
    scenario: Callable[[Any], Awaitable[list[Any]]],
    snapshot_sid: str | None = None,
    *,
    new_store: AsyncStore,
) -> None:
    """Compare a server backend to the current SQLite contract."""
    reference = await AsyncStore.open(":memory:")
    try:
        reference_out = await scenario(reference)
        new_out = await scenario(new_store)
        assert _norm(new_out) == _norm(reference_out), "return-value parity"
        if snapshot_sid is not None:
            assert _norm(await _async_snapshot(new_store, snapshot_sid)) == _norm(
                await _async_snapshot(reference, snapshot_sid)), "final-state parity"
    finally:
        await reference.close()


# ── scenarios (backend-agnostic; reused by the PG/MySQL conformance suites) ──


async def scn_messages(st: Any) -> list[Any]:
    await st.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id="s")
    out = [await st.append_message("s", role="user", content=f"m{i}", round_index=i)
           for i in range(5)]
    out.append(await st.record_compaction(
        "s", from_seq=1, to_seq=2, note_content="sum", before_tokens=9, after_tokens=1,
        round_index=2, fold_seqs=[1, 2], order_key=1))
    return out


async def scn_timers(st: Any) -> list[Any]:
    await st.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id="s")
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


async def scn_notes(st: Any) -> list[Any]:
    await st.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id="s")
    out: list[Any] = []
    out.append((await st.add_note("s", "first")).note_id)
    out.append((await st.add_note("s", "second", pinned=True)).note_id)
    out.append(await st.update_note("s", 1, content="edited"))
    out.append(await st.update_note("s", 99, pinned=True))  # missing → False
    out.append(await st.count_notes("s"))
    out.append([(n.note_id, n.content, n.pinned) for n in await st.list_notes("s")])
    out.append(await st.delete_note("s", 2))
    return out


async def scn_usage_stats(st: Any) -> list[Any]:
    await st.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id="s")
    await st.record_usage("s", round_index=0, prompt_tokens=10, completion_tokens=5,
                          total_tokens=15, model="m")
    await st.record_usage("s", round_index=0, prompt_tokens=11, completion_tokens=6,
                          total_tokens=17)  # overwrite same round (same send 0)
    # v6: round_index resets per send — the SAME round in a LATER send must NOT clobber
    # send 0's row (the old (session_id, round_index) key kept only the last send).
    await st.record_usage("s", round_index=0, send_index=2, prompt_tokens=20,
                          completion_tokens=8, total_tokens=28)
    await st.bump_session_stats("s", {"calls": 1, "prompt_tokens": 10,
                                      "completion_tokens": 5, "total_tokens": 15},
                                rounds=2, tool_calls=3)
    await st.bump_session_stats("s", {"calls": 2, "prompt_tokens": 20,
                                      "completion_tokens": 10, "total_tokens": 30},
                                rounds=1, tool_calls=1)  # accumulate
    s = await st.get_session_stats("s")
    return [s.sends, s.rounds, s.llm_calls, s.tool_calls, s.prompt_tokens,
            s.completion_tokens, s.total_tokens]


async def scn_runtime_shared(st: Any) -> list[Any]:
    await st.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id="s")
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


async def scn_bgtasks(st: Any) -> list[Any]:
    """Schema 18 background tasks execute as durable child sessions (kind='background')
    and deliver completion via River into parent.follow_up_queue (kind='background_result')."""
    import json

    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_types import TargetRef, TrustedActor
    from power_loop.runtime.background import load_record

    await st.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id="s")
    port = RuntimeInputStore(st)
    seq = await st.append_message("s", role="user", content="start bg")
    call_ref = await st.message_ref("s", seq)
    parent_actor = TrustedActor(STORE_ACTOR.tenant_id, STORE_ACTOR.principal_id, "s", call_ref, "bg1")
    frozen = {
        "tool": "echo",
        "arguments": {"text": "cmd"},
        "definition_hash": "parity-bg-hash",
        "call_ref": call_ref.to_dict(),
        "call_id": "bg1",
    }
    bg_desc = dataclasses.replace(
        STORE_DESCRIPTOR,
        kind="background",
        config={**dict(STORE_DESCRIPTOR.config), "background": frozen},
    )
    child_target, start_ref, receipt = await port.create_child(
        bg_desc,
        parent=TargetRef(st.store_id, "s"),
        actor=parent_actor,
        call_ref=call_ref,
        tool_call_id="bg1",
        body=frozen,
        result_kind="background_result",
        require_assistant_call=False,
    )
    out: list[Any] = [receipt.newly_accepted]
    children = await st.list_children("s")
    out.append([c.session_kind for c in children])
    before_rec = await load_record(st, "s", child_target.session_id)
    out.append((before_rec["tool"], before_rec["args"], before_rec["status"], before_rec["result"]))

    batch = await port.claim(child_target, node_id="parity-node", boot_id="parity-boot")
    assert batch is not None
    await port.final(batch, status="completed", final_text="ok")

    after_rec = await load_record(st, "s", child_target.session_id)
    out.append((after_rec["tool"], after_rec["args"], after_rec["status"], after_rec["result"]))
    child_res = await port.get_result(start_ref, actor=STORE_ACTOR)
    out.append((child_res.status, child_res.final_text))

    parent_q = await st._db.fetchall(
        f"SELECT kind,status,content_json FROM {st.t.follow_up_queue} "
        "WHERE session_id=? AND kind='background_result'",
        ("s",),
    )
    parent_row = await st._db.fetchone(
        f"SELECT inbox_version,background_version,execution_state FROM {st.t.sessions} WHERE session_id=?",
        ("s",),
    )
    body = json.loads(parent_q[0]["content_json"])
    out.append((parent_q[0]["kind"], parent_q[0]["status"], body["status"], body["final_text"]))
    out.append((int(parent_row["inbox_version"]), int(parent_row["background_version"]), parent_row["execution_state"]))
    return out


async def scn_cascade(st: Any) -> list[Any]:
    await st.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id="p")
    await st.create_session(
        STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id="linked",
        parent_session_id="p", cancel_propagation="linked",
    )
    await st.create_session(
        STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id="det",
        parent_session_id="p", cancel_propagation="independent",
    )
    removed = await st.close_session("p", cascade=True)
    survivors = [bool(await st.get_session(x)) for x in ("p", "linked", "det")]
    det = await st.get_session("det")
    return [removed, survivors, det.parent_session_id if det else None]


async def scn_export_import(st: Any) -> list[Any]:
    """Schema 18: export_session succeeds with identical table/message payloads;
    import_session raises capability_unavailable identically across backends."""
    from power_loop.agent.runtime_types import RuntimeRequestError

    await st.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id="src")
    for i in range(3):
        await st.append_message("src", role="user", content=f"m{i}")
    await st.add_note("src", "note")
    await st.create_timer("src", due_at=1000, note="t")
    blob = await st.export_session("src")
    msgs = [r["content"] for r in blob["tables"]["messages"]]
    sess = await st.get_session("src")
    import_code = None
    try:
        await st.import_session(blob, new_session_id="restored")
    except RuntimeRequestError as exc:
        import_code = exc.code
    return [import_code, sorted(blob["tables"].keys()), msgs, sess.parent_session_id, sess.session_kind]


async def scn_prune(st: Any) -> list[Any]:
    """Retention with a keep-N bound. This exercises the ORDER BY/LIMIT-in-subquery DELETE
    that MySQL rejects (err 1235/1093) unless the subquery is materialized in a derived
    table — a backend-divergent SQL path that the other scenarios never touch, so it must
    run on EVERY backend (this is what makes the PG/MySQL conformance suites catch it)."""
    await st.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id="s")
    out: list[Any] = []
    # usage rounds: keep the 2 most recent of 5 → 3 deleted
    for i in range(5):
        await st.record_usage("s", round_index=i, prompt_tokens=i, completion_tokens=i,
                              total_tokens=2 * i, model="m")
    out.append(await st.prune_usage_rounds("s", keep_last=2))
    # compacted-out messages: fold 4 of 6, keep the 2 most recent compacted rows → 2 deleted
    for i in range(6):
        await st.append_message("s", role="user", content=f"m{i}", round_index=i)
    await st.record_compaction("s", from_seq=1, to_seq=4, note_content="sum", before_tokens=9,
                               after_tokens=1, round_index=4, fold_seqs=[1, 2, 3, 4], order_key=1)
    out.append(await st.prune_compacted_messages("s", keep_recent=2))
    out.append(sorted(m.seq for m in await st.load_all_messages("s")))
    return out


#: (id, scenario, snapshot_sid|None) — reused by every backend's conformance suite.
SCENARIOS: list[tuple[str, Callable[[Any], Awaitable[list[Any]]], str | None]] = [
    ("messages", scn_messages, "s"),
    ("timers", scn_timers, "s"),
    ("notes", scn_notes, "s"),
    ("usage_stats", scn_usage_stats, "s"),
    ("runtime_shared", scn_runtime_shared, None),
    ("bgtasks", scn_bgtasks, "s"),
    ("cascade", scn_cascade, None),
    ("export_import", scn_export_import, None),
    ("prune", scn_prune, "s"),
]


# ── H2 (BUG_REVIEW_3.4): large (>64 KiB) free-text/JSON payloads must round-trip on every
# backend. MySQL's old TEXT columns capped at 64 KiB (DataError 1406); SQLite/PG TEXT is
# unbounded. Shared so the PG/MySQL conformance suites import and run it against real servers.
async def exercise_large_payload(store: Any) -> None:
    big = "A" * 100_000          # 100 KiB — comfortably past MySQL's 64 KiB TEXT cap
    big_prompt = "P" * 100_000
    big_args = '{"text":"' + ("x" * 100_000) + '"}'
    await store.create_session(dataclasses.replace(STORE_DESCRIPTOR, config={"system_prompt": big_prompt}), actor=STORE_ACTOR, session_id="big")
    await store.append_message(
        "big", role="assistant", content=big,
        tool_calls=[{"id": "c1", "function": {"name": "echo", "arguments": big_args}}],
        round_index=0,
    )
    await store.append_message("big", role="tool", content=big, tool_call_id="c1", round_index=0)
    await store.add_note("big", "N" * 100_000)

    rows = await store.load_active_messages("big")
    asst = next(r for r in rows if r.role == "assistant")
    tool = next(r for r in rows if r.role == "tool")
    assert asst.content == big and len(asst.content) == 100_000          # not truncated
    assert asst.tool_calls[0]["function"]["arguments"] == big_args
    assert tool.content == big
    sess = await store.get_session("big")
    assert sess.descriptor.config["system_prompt"] == big_prompt
    notes = await store.list_notes("big")
    assert notes and len(notes[0].content) == 100_000


async def test_sqlite_large_payload() -> None:
    store = await AsyncStore.open(":memory:")
    try:
        await exercise_large_payload(store)
    finally:
        await store.close()

