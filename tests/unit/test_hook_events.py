"""Tests for the hook-events audit log (pl_hook_events): the per-round capture of EPHEMERAL
LLM_BEFORE hook injections (e.g. recalled memory), recorded for observability ONLY.

The load-bearing property is the LEAK GUARD: what is captured for audit must never re-enter the
loop's history or the LLM request (which would accrete + bust prefix-caching — the exact thing the
ephemeral-tail design avoids).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.core.pipeline import AgentPipeline

_summarize = AgentPipeline._summarize_hook_injection


# ── pure helper: the inject diff/summary ─────────────────────────────────────


def test_summarize_off_or_no_snapshot_returns_none() -> None:
    final = [{"role": "user", "content": "hi"}]
    assert _summarize(final, None, "off") is None
    assert _summarize(final, {id(final[0])}, "off") is None  # mode off wins even with a snapshot


def test_summarize_nothing_injected_returns_none() -> None:
    final = [{"role": "user", "content": "hi"}]
    pre = {id(m) for m in final}
    assert _summarize(final, pre, "full") is None


def test_summarize_tail_injection_metadata_omits_content() -> None:
    orig = [{"role": "system", "content": "sp"}, {"role": "user", "content": "hi"}]
    pre = {id(m) for m in orig}
    injected = {"role": "system", "name": "memory_0", "content": "secret fact"}
    audit = _summarize([*orig, injected], pre, "metadata")
    assert audit["hook_point"] == "LLM_BEFORE" and audit["kind"] == "inject"
    assert audit["position"] == "tail"
    assert audit["hook"] == "builtin.memory_recall"
    item = audit["payload"]["items"][0]
    assert item == {"role": "system", "name": "memory_0", "source": "builtin.memory_recall", "chars": 11}
    assert "content" not in item  # metadata mode must NOT store the text
    assert audit["payload"]["item_count"] == 1 and audit["payload"]["total_chars"] == 11


def test_summarize_full_includes_content() -> None:
    orig = [{"role": "user", "content": "hi"}]
    pre = {id(m) for m in orig}
    injected = {"role": "system", "name": "memory_0", "content": "abc"}
    audit = _summarize([*orig, injected], pre, "full")
    assert audit["payload"]["items"][0]["content"] == "abc"


def test_summarize_front_position_detected() -> None:
    orig = [{"role": "system", "content": "sp"}, {"role": "user", "content": "hi"}]
    pre = {id(m) for m in orig}
    injected = {"role": "system", "name": "memory_0", "content": "x"}
    # injected lands BEFORE the user message (front)
    final = [orig[0], injected, orig[1]]
    audit = _summarize(final, pre, "full")
    assert audit["position"] == "front"


def test_summarize_multi_item_and_non_memory_source() -> None:
    orig = [{"role": "user", "content": "hi"}]
    pre = {id(m) for m in orig}
    a = {"role": "system", "name": "memory_0", "content": "11"}
    b = {"role": "system", "name": "custom_block", "content": "222"}
    audit = _summarize([*orig, a, b], pre, "metadata")
    assert audit["payload"]["item_count"] == 2 and audit["payload"]["total_chars"] == 5
    srcs = {it["source"] for it in audit["payload"]["items"]}
    assert srcs == {"builtin.memory_recall", "llm_before"}
    assert audit["hook"] == "llm_before"  # mixed sources → coarse label


def test_summarize_rebind_failsafe_does_not_dump_conversation() -> None:
    # A hook that REPLACES ctx.messages with fresh copies makes every id novel. The diff must NOT
    # mislabel the whole prompt as injected (data bloat / privacy) — it records a small marker.
    orig = [{"role": "system", "content": "sp"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "prev"}]
    pre = {id(m) for m in orig}
    rebound = [dict(m) for m in orig] + [{"role": "system", "name": "memory_0", "content": "MEM"}]
    audit = _summarize(rebound, pre, "full")
    assert audit["kind"] == "inject_unresolved" and audit["position"] == "unknown"
    assert audit["payload"]["items"] == [] and audit["payload"]["rebound"] is True
    # crucially: no conversation text was captured
    assert "prev" not in str(audit) and "MEM" not in str(audit)


# ── store round-trip + cleanup ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_records_and_lists_hook_events() -> None:
    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session()
        seq = await store.append_message(
            sid, role="assistant", content="a", round_index=2, send_index=5,
            hook_injected={"hook_point": "LLM_BEFORE", "hook": "builtin.memory_recall",
                           "position": "tail", "kind": "inject",
                           "payload": {"v": 1, "items": [{"name": "memory_0", "chars": 3}]}},
        )
        # a plain append (no hook_injected) writes NO event
        await store.append_message(sid, role="user", content="next")
        evs = await store.list_hook_events(sid)
        assert len(evs) == 1
        ev = evs[0]
        assert ev.message_seq == seq and ev.round_index == 2 and ev.send_index == 5
        assert ev.hook == "builtin.memory_recall" and ev.position == "tail"
        assert ev.payload["items"][0]["name"] == "memory_0"
        # filter by message_seq
        assert len(await store.list_hook_events(sid, message_seq=seq)) == 1
        assert await store.list_hook_events(sid, message_seq=99999) == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_event_ids_are_monotonic_per_session() -> None:
    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session()
        hi = {"hook_point": "LLM_BEFORE", "hook": "h", "position": "tail", "kind": "inject",
              "payload": {"v": 1, "items": []}}
        await store.append_message(sid, role="assistant", content="a", hook_injected=hi)
        await store.append_message(sid, role="assistant", content="b", hook_injected=hi)
        evs = await store.list_hook_events(sid)
        assert [e.event_id for e in evs] == [1, 2]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_hook_events_excluded_from_export_by_design() -> None:
    # hook_events is an audit-only sidecar, intentionally NOT part of session export/import (the
    # audit lives in the live store). Pin that decision so it can't silently change.
    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session()
        await store.append_message(
            sid, role="assistant", content="a", round_index=0, send_index=1,
            hook_injected={"hook_point": "LLM_BEFORE", "hook": "builtin.memory_recall",
                           "position": "tail", "kind": "inject",
                           "payload": {"v": 1, "items": [{"name": "memory_0", "chars": 3}]}},
        )
        blob = await store.export_session(sid)
        assert "hook_events" not in blob["tables"]
        new_id = await store.import_session(blob, new_session_id="he_restored")
        assert await store.list_hook_events(new_id) == []  # not carried over
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_close_session_tree_deletes_hook_events() -> None:
    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session()
        await store.append_message(
            sid, role="assistant", content="a",
            hook_injected={"hook_point": "LLM_BEFORE", "hook": "h", "position": "tail",
                           "kind": "inject", "payload": {"v": 1, "items": []}},
        )
        assert len(await store.list_hook_events(sid)) == 1
        await store.close_session_tree(sid)
        assert await store.list_hook_events(sid) == []
    finally:
        await store.close()


# ── loop-level capture + config modes + the LEAK GUARD ───────────────────────


@dataclass
class _RecLLM:
    seen: list[list[dict]] = field(default_factory=list)

    async def complete(self, request, **kwargs):
        from power_loop._vendor.llm_client.interface import LLMResponse

        self.seen.append([dict(m) for m in request.messages])
        return LLMResponse(raw_text="ok")

    async def stream(self, request):  # pragma: no cover
        raise NotImplementedError

    async def close(self):  # pragma: no cover
        pass


@dataclass
class _FakeProvider:
    to_recall: list[dict] = field(default_factory=list)

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        return [dict(m) for m in self.to_recall]

    async def remember(self, *, snapshot, session_id):  # pragma: no cover - unused
        return None


async def _run_with(record_mode: str, *, recall_text="the-secret-memory"):
    store = await SessionStore.open(":memory:")
    llm = _RecLLM()
    loop = StatefulAgentLoop(
        llm=llm, store=store,
        config=AgentLoopConfig(
            max_rounds=1, compactor=None,
            memory=_FakeProvider(to_recall=[{"content": recall_text}]),
            record_hook_events=record_mode,
        ),
    )
    sid = await loop.new_session()
    await loop.send("hi", session_id=sid)
    return store, llm, sid


@pytest.mark.asyncio
async def test_loop_off_by_default_records_nothing() -> None:
    store, llm, sid = await _run_with("off")
    try:
        # memory WAS injected into the LLM request...
        assert any(str(m.get("name") or "").startswith("memory_") for m in llm.seen[0])
        # ...but with auditing off, no event was recorded.
        assert await store.list_hook_events(sid) == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_loop_full_mode_records_injected_content() -> None:
    store, llm, sid = await _run_with("full")
    try:
        evs = await store.list_hook_events(sid)
        assert len(evs) == 1
        ev = evs[0]
        assert ev.hook_point == "LLM_BEFORE" and ev.hook == "builtin.memory_recall"
        assert ev.payload["items"][0]["content"] == "the-secret-memory"
        # linked to a real assistant message row
        msgs = await store.load_all_messages(sid)
        assert any(m.seq == ev.message_seq and m.role == "assistant" for m in msgs)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_loop_metadata_mode_omits_content() -> None:
    store, llm, sid = await _run_with("metadata")
    try:
        ev = (await store.list_hook_events(sid))[0]
        assert ev.payload["items"][0]["chars"] == len("the-secret-memory")
        assert "content" not in ev.payload["items"][0]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_loop_records_one_event_per_round_in_multi_round_send() -> None:
    """A multi-round send (tool call on round 0 → text on round 1) records ONE audit row per round,
    each linked to its own assistant message, with monotonic event_ids — and a 2nd send keeps
    counting forward (no mislink/double-count)."""
    from tests.unit.test_stateful_loop import _echo_registry, _Scripted, _tool_resp

    store = await SessionStore.open(":memory:")
    try:
        llm = _Scripted(responses=[
            _tool_resp("c1", "echo", '{"text": "hi"}'),  # round 0 → tool call
            LLMResponse(raw_text="done"),                # round 1 → finish
        ])
        loop = StatefulAgentLoop(
            llm=llm, store=store, tool_registry=_echo_registry(),
            config=AgentLoopConfig(
                max_rounds=4, compactor=None,
                memory=_FakeProvider(to_recall=[{"content": "M"}]),
                record_hook_events="full",
            ),
        )
        sid = await loop.new_session()
        await loop.send("go", session_id=sid)
        evs = await store.list_hook_events(sid)
        assert len(evs) == 2  # one per round
        assert [e.event_id for e in evs] == [1, 2]  # monotonic
        assert len({e.message_seq for e in evs}) == 2  # distinct assistant rows
        asst_seqs = {m.seq for m in await store.load_all_messages(sid) if m.role == "assistant"}
        assert all(e.message_seq in asst_seqs for e in evs)

        # a SECOND send continues the id sequence (no reattach/double-count of prior events)
        await loop.send("again", session_id=sid)
        evs2 = await store.list_hook_events(sid)
        assert [e.event_id for e in evs2][:2] == [1, 2]
        assert len(evs2) > 2 and evs2[-1].event_id == len(evs2)
    finally:
        await store.close()


def test_config_rejects_invalid_record_hook_events() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError, match="record_hook_events"):
        AgentLoopConfig(record_hook_events="on")  # typo / not in the enum
    # case-insensitive normalization
    assert AgentLoopConfig(record_hook_events="FULL").record_hook_events == "full"
    assert AgentLoopConfig(record_hook_events="off").record_hook_events == "off"


@pytest.mark.asyncio
async def test_leak_guard_audit_never_enters_history_or_request() -> None:
    """The load-bearing test: captured audit must NOT be on the persisted message row, must NOT
    appear when the row is converted into an LLM-request message, and must NOT accrete into the
    next request's history."""
    from power_loop.runtime.representation import _row_to_loop_dict

    store = await SessionStore.open(":memory:")
    llm = _RecLLM()
    loop = StatefulAgentLoop(
        llm=llm, store=store,
        config=AgentLoopConfig(
            max_rounds=1, compactor=None,
            memory=_FakeProvider(to_recall=[{"content": "SECRET"}]),
            record_hook_events="full",
        ),
    )
    try:
        sid = await loop.new_session()
        await loop.send("first", session_id=sid)
        # (a) the audit row exists
        assert len(await store.list_hook_events(sid)) == 1
        # (b) the assistant MessageRow carries NO hook payload (no field, nothing in meta)
        rows = await store.load_all_messages(sid)
        asst = next(r for r in rows if r.role == "assistant")
        assert not hasattr(asst, "hook_injected")
        assert "hook_injected" not in (asst.meta or {})
        assert "SECRET" not in (asst.content or "")
        # (c) converting the row to an LLM message yields ONLY the protocol keys
        d = _row_to_loop_dict(asst)
        assert set(d.keys()) <= {"role", "content", "tool_calls", "tool_call_id", "name"}
        # (d) a SECOND send: the history portion (everything before the freshly re-injected memory)
        # must not contain the recorded audit/SECRET except as the live ephemeral re-injection.
        llm.seen.clear()
        await loop.send("second", session_id=sid)
        sent = llm.seen[0]
        secret_msgs = [m for m in sent if "SECRET" in str(m.get("content") or "")]
        # SECRET appears ONLY as the live memory_* re-injection, never as accreted history.
        assert all(str(m.get("name") or "").startswith("memory_") for m in secret_msgs)
    finally:
        await store.close()
