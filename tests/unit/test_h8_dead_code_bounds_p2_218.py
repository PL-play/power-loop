"""Design 218 H8 regression tests: retention bounds, tool_context pruning/batching,
memory/FD caps, hot-path zero-hash resolve, release_claim cancel, redaction, and boundary fixes.
"""
from __future__ import annotations

import asyncio
import io
import os
import subprocess
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import (
    AgentEvent,
    AgentEventBus,
    AgentEventType,
    AgentHooks,
    AgentLoopConfig,
    ExecutionEnvironment,
    InMemoryWakeTransport,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    SessionStore,
    ToolDefinition,
    ToolRegistry,
)
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.system_prompt import build_skill_section
from power_loop.contracts.errors import SessionLeaseLost
from power_loop.contrib._redact import DEFAULT_VALUE_PATTERNS, resolve_redact, sanitize
from power_loop.contrib.jsonl_sink import attach_jsonl_sink
from power_loop.contrib.otel_sink import OtelSpanBridge
from power_loop.core.pipeline import AgentPipeline
from power_loop.core.state import ContextManager
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.pricing import Free
from power_loop.runtime.process_control import OwnedProcess
from power_loop.runtime.store.schema import CURRENT_SCHEMA_VERSION
from power_loop.runtime.store.session_native_ddl import INDEXES
from power_loop.runtime.tool_archive import _pairs, prune_tool_context_state
from power_loop.runtime.tool_contracts import (
    FunctionRuntimeTool,
    ToolRuntimeContract,
    resolve,
)

from .session_native_fixture import ACTOR, DESCRIPTOR, target

pytestmark = pytest.mark.unit


# ── 1. Active open session retention, superseded projections, orphan shared:* ──


@pytest.mark.asyncio
async def test_h8_active_open_session_retention_and_safety_three_dbs(registry_store):
    """Active open sessions clean expired terminal inbox, superseded projections & orphan shared:*
    without ever deleting pending/waiting_continuation, workflow_driver start input, or raw messages.
    """
    store, _ = registry_store
    port = RuntimeInputStore(store)

    now = int(time.time() * 1000)
    old_ts = now - 20 * 86_400_000  # 20 days ago

    # Active open session with recent updated_at AND a child session (previously doubly blocked).
    parent = await target(port, "h8-open-parent")
    _child = await target(port, "h8-open-child", parent=parent)

    # 1a) Expired fast-path delivered completed inbox row on active open session -> MUST be purged.
    r_old = await port.accept(parent, "old turn", message_id="m-old-done", actor=ACTOR)
    batch = await port.claim(parent, node_id="n1", boot_id="b1")
    await port.final(batch, final_text="ok-old")
    await store._db.execute(
        f"UPDATE {store.t.follow_up_queue} SET delivered_at=? WHERE session_id=? AND item_id=?",
        (old_ts, parent.session_id, r_old.message_id),
    )

    # 1b) Expired void/cancelled terminal inbox row (terminal_status IS NOT NULL, completed_at < cutoff) -> MUST be purged.
    r_void = await port.accept(parent, "void turn", message_id="m-old-void", actor=ACTOR)
    await port.cancel(r_void.input_ref, actor=ACTOR)
    await store._db.execute(
        f"UPDATE {store.t.follow_up_queue} SET completed_at=? WHERE session_id=? AND item_id=?",
        (old_ts, parent.session_id, r_void.message_id),
    )

    # 2) Waiting-continuation row with old timestamps -> MUST be preserved.
    r_wait = await port.accept(parent, "wait turn", message_id="m-wait", actor=ACTOR)
    batch2 = await port.claim(parent, node_id="n1", boot_id="b1")
    await port.final(batch2, final_text="ok-wait")
    await store._db.execute(
        f"UPDATE {store.t.follow_up_queue} SET status='waiting_continuation', delivered_at=?,"
        "terminal_status=NULL,result_send_index=NULL,result_seq=NULL,completed_at=NULL "
        "WHERE session_id=? AND item_id=?",
        (old_ts, parent.session_id, r_wait.message_id),
    )

    # 3) Pending inbox row -> MUST be preserved.
    r_pend = await port.accept(parent, "pending turn", message_id="m-pend", actor=ACTOR)

    # 4) workflow_driver's start_input_message_id -> MUST be preserved even when settled & old.
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    drv = await port.create_session(drv_desc, actor=ACTOR, parent=parent, options={"session_id": "h8-wf-driver"})
    r_drv = await port.accept(drv, "start wf", message_id="m-drv-start", actor=ACTOR)
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET start_input_message_id=? WHERE session_id=?",
        (r_drv.message_id, drv.session_id),
    )
    drv_batch = await port.claim(drv, node_id="n1", boot_id="b1")
    await port.final(drv_batch, final_text="wf-done")
    await store._db.execute(
        f"UPDATE {store.t.follow_up_queue} SET delivered_at=? WHERE session_id=? AND item_id=?",
        (old_ts, drv.session_id, r_drv.message_id),
    )

    # 5) Superseded projections (send_index <= compact_to_send=2) vs live projection (send_index=3).
    await store.upsert_project_message(parent.session_id, send_index=1, kind="project", content={"t": 1})
    await store.upsert_project_message(parent.session_id, send_index=2, kind="project", content={"t": 2})
    await store.upsert_project_message(
        parent.session_id, send_index=2, kind="compact", content={"summary": "s12"},
        compact_from_send=1, compact_to_send=2,
    )
    await store.upsert_project_message(parent.session_id, send_index=3, kind="project", content={"t": 3})
    await store._db.execute(
        f"UPDATE {store.t.project_messages} SET created_at=? WHERE session_id=?",
        (old_ts, parent.session_id),
    )

    # 6) Orphaned shared:* vs live-session-owned shared:*.
    await store.set_shared_state("h8-orphan-scope", "k1", {"stale": True})
    await store.set_shared_state(parent.session_id, "k2", {"alive": True})
    await store._db.execute(
        f"UPDATE {store.t.session_runtime_state} SET updated_at=? WHERE state_key LIKE 'shared:%'",
        (old_ts,),
    )

    # Keep parent session fresh (updated_at = now) AND verify raw messages count before purge.
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET updated_at=? WHERE session_id IN (?,?)",
        (now, parent.session_id, drv.session_id),
    )
    raw_msgs_before = await store.load_all_messages(parent.session_id)
    assert len(raw_msgs_before) >= 2

    # Run paged retention with small limit=2 to verify supplementary cursor pagination terminates cleanly.
    totals: dict[str, int] = {}
    cursor = None
    for _ in range(10):
        page = await store.purge_expired_page(
            now,
            limit=2,
            retention_days={"sessions": 30, "messages": 30, "inbox": 7, "runtime": 7, "projections": 7},
            after_session_id=cursor,
        )
        for k, v in page.deleted.items():
            totals[k] = totals.get(k, 0) + v
        cursor = page.next_after
        if cursor is None:
            break
    assert cursor is None
    assert totals.get("inbox", 0) == 2
    assert totals.get("projections", 0) == 2
    assert totals.get("runtime", 0) == 1
    assert totals.get("messages", 0) == 0

    # Verify invariants after purge:
    q_items = {
        r["item_id"]
        for r in await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?",
            (parent.session_id,),
        )
    }
    assert r_old.message_id not in q_items
    assert r_void.message_id not in q_items
    assert {r_wait.message_id, r_pend.message_id} <= q_items

    drv_items = {
        r["item_id"]
        for r in await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?",
            (drv.session_id,),
        )
    }
    assert r_drv.message_id in drv_items

    # Raw messages of active open session are completely untouched.
    assert len(await store.load_all_messages(parent.session_id)) == len(raw_msgs_before)

    # Superseded projections deleted; compact row and send_index=3 preserved.
    proj_rows = await store._db.fetchall(
        f"SELECT send_index, kind FROM {store.t.project_messages} WHERE session_id=? ORDER BY send_index, kind",
        (parent.session_id,),
    )
    assert [(int(r["send_index"]), r["kind"]) for r in proj_rows] == [(2, "compact"), (3, "project")]

    # Orphaned shared:* deleted; live-session shared:* preserved.
    assert await store.get_shared_state("h8-orphan-scope", "k1") is None
    assert await store.get_shared_state(parent.session_id, "k2") == {"alive": True}


def test_h8_schema_version_unchanged_and_unused_index_removed():
    assert CURRENT_SCHEMA_VERSION == 18
    assert "idx_messages_retention" not in INDEXES


# ── 2. release_claim concurrent cancel must NOT delete user message ───────────


@pytest.mark.asyncio
async def test_h8_release_claim_concurrent_cancel_preserves_user_message_three_dbs(registry_store):
    """When a concurrent cancel arrives before release_claim settles via _final_tx,
    the delivered user message row MUST remain intact in messages.
    """
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port, "h8-release-cancel")
    receipt = await port.accept(root, "keep my user row", message_id="m-rc-1", actor=ACTOR)
    batch = await port.claim(root, node_id="n1", boot_id="b1")
    assert batch is not None and batch.message_seq >= 1

    # Concurrent cancel bumps cancel_seq while session is running.
    await port.cancel(root, actor=ACTOR, reason="user_stop")

    # release_claim takes the _final_tx branch and must NOT delete the user message.
    await port.release_claim(batch)

    user_row = await store.get_message(root.session_id, batch.message_seq)
    assert user_row is not None
    assert user_row.role == "user"
    assert user_row.send_index == batch.send_index

    result = await port.get_result(receipt.input_ref, actor=ACTOR)
    assert result.status == "cancelled"


# ── 3. tool_context.v1 pruning & _pairs() batch fetch (zero N+1) ─────────────


@pytest.mark.asyncio
async def test_h8_tool_context_pruning_and_pairs_batch_fetch():
    state = {
        "send_policies": {str(i): {"v": i} for i in range(1, 51)},
        "operations": {f"op_{i}": {"text": str(i)} for i in range(300)},
        "archive": {
            "patches": {
                "1:2": {"send_index": 1},
                "50:2": {"send_index": 50},
            },
            "exposures": {"1:2": 5, "50:2": 1},
            "view_groups": [{"send_index": 1, "seqs": [2, 3]}, {"send_index": 50, "seqs": [2, 3]}],
            "events": [{"i": i} for i in range(100)],
            "groups": {f"g{i}": {"send_index": i} for i in range(300)},
        },
    }
    prune_tool_context_state(state, current_send_index=50)
    assert len(state["send_policies"]) == 32 and "50" in state["send_policies"] and "1" not in state["send_policies"]
    assert len(state["operations"]) == 256
    assert list(state["archive"]["patches"].keys()) == ["50:2"]
    assert list(state["archive"]["exposures"].keys()) == ["50:2"]
    assert [g["send_index"] for g in state["archive"]["view_groups"]] == [50]
    assert len(state["archive"]["events"]) == 64
    assert len(state["archive"]["groups"]) == 256

    # Verify _pairs uses single batch get_messages_by_seqs instead of N+1 get_message calls.
    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session(DESCRIPTOR, actor=ACTOR)
        history = []
        seqs = []
        for idx in range(4):
            call_id = f"c-{idx}"
            tc = [{"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a"}'}}]
            a_seq = await store.append_message(sid, role="assistant", content=None, tool_calls=tc, send_index=1)
            t_seq = await store.append_message(sid, role="tool", tool_call_id=call_id, name="read_file", content=f"out-{idx}", send_index=1)
            history.append({"role": "assistant", "tool_calls": tc})
            history.append({"role": "tool", "tool_call_id": call_id, "content": f"out-{idx}"})
            seqs.extend([a_seq, t_seq])

        single_calls = 0
        batch_calls = 0
        orig_get = store.get_message
        orig_batch = store.get_messages_by_seqs

        async def _count_get(*a, **kw):
            nonlocal single_calls
            single_calls += 1
            return await orig_get(*a, **kw)

        async def _count_batch(*a, **kw):
            nonlocal batch_calls
            batch_calls += 1
            return await orig_batch(*a, **kw)

        store.get_message = _count_get  # type: ignore[method-assign]
        store.get_messages_by_seqs = _count_batch  # type: ignore[method-assign]

        engine = SimpleNamespace(sid=sid, store=store)
        pipeline = SimpleNamespace(history=history, send_index=1, sink=SimpleNamespace(_history_seqs=seqs))
        pairs = [p async for p in _pairs(engine, pipeline)]
        assert len(pairs) == 4
        assert batch_calls == 1
        assert single_calls == 0
    finally:
        await store.close()


# ── 4. Unbounded containers, unsubscribe & FD cleanup ────────────────────────


@pytest.mark.asyncio
async def test_h8_bounded_containers_unsubscribe_and_fd_cleanup(tmp_path):
    # InMemoryWakeTransport bounded & cleared on aclose()
    wt = InMemoryWakeTransport(max_history=5)
    for i in range(12):
        await wt.publish_wake(f"s-{i}")
        await wt.publish_cancel(f"s-{i}")
    assert len(wt.published_wakes) == 5 and wt.published_wakes[-1][0] == "s-11"
    assert len(wt.published_cancels) == 5 and wt.published_cancels[-1][0] == "s-11"
    await wt.aclose()
    assert len(wt.published_wakes) == 0 and len(wt.published_cancels) == 0

    # OtelSpanBridge caps open spans, cleans child spans on SESSION_ENDED, unsubscribes on close()
    ended_spans: list[str] = []

    class _FakeSpan:
        def __init__(self, name: str):
            self.name = name

        def set_attribute(self, *_a, **_kw):
            pass

        def end(self):
            ended_spans.append(self.name)

    class _FakeTracer:
        def start_span(self, name, **_kw):
            return _FakeSpan(name)

    bus = AgentEventBus()
    bridge = OtelSpanBridge(_FakeTracer(), bus=bus, max_open_spans=3)
    bus.subscribe(None, bridge.handle)
    for i in range(5):
        bus.publish(AgentEvent(type=AgentEventType.SESSION_STARTED, session_id=f"s-{i}"))
    assert len(bridge._session) == 3
    bus.publish(AgentEvent(type=AgentEventType.ROUND_STARTED, session_id="s-4", payload={"round_index": 0}))
    bus.publish(AgentEvent(type=AgentEventType.TOOL_CALL_STARTED, session_id="s-4", payload={"tool_call_id": "t1"}))
    bus.publish(AgentEvent(type=AgentEventType.SESSION_ENDED, session_id="s-4"))
    assert "s-4" not in bridge._session
    assert not any(k[0] == "s-4" for k in bridge._round)
    assert not any(k[0] == "s-4" for k in bridge._tool)
    bridge.close()
    ended_before = len(ended_spans)
    bus.publish(AgentEvent(type=AgentEventType.SESSION_STARTED, session_id="after-close"))
    assert len(bridge._session) == 0 and len(ended_spans) == ended_before

    # JsonlSink unsubscribes on close()
    jpath = tmp_path / "events.jsonl"
    jsink = attach_jsonl_sink(bus, jpath)
    bus.publish(AgentEvent(type=AgentEventType.SESSION_STARTED, session_id="j1"))
    jsink.close()
    assert len(bus._global_handlers) == 0

    # OwnedProcess closes _result_fd and _control on drain_stopped_output timeout
    r_ctrl, w_ctrl = os.pipe()
    r_res, w_res = os.pipe()
    os.close(r_ctrl)
    os.close(w_res)
    proc = OwnedProcess.__new__(OwnedProcess)
    proc._control = w_ctrl
    proc._result_fd = r_res
    streams = [io.StringIO() for _ in range(3)]
    proc.process = SimpleNamespace(
        poll=lambda: 0,
        communicate=lambda *, timeout: (_ for _ in ()).throw(subprocess.TimeoutExpired("cmd", timeout)),
        stdin=streams[0], stdout=streams[1], stderr=streams[2],
    )
    with pytest.raises(subprocess.TimeoutExpired):
        proc.drain_stopped_output(timeout=0.1)
    assert proc._control is None and proc._result_fd is None
    assert all(s.closed for s in streams)
    for fd in (w_ctrl, r_res):
        with pytest.raises(OSError):
            os.fstat(fd)


# ── 5. Hot-path zero redundant to_openai_tool / SHA-256 hashing ──────────────


@pytest.mark.asyncio
async def test_h8_hot_path_tool_resolve_and_invoke_avoid_rehash():
    to_openai_calls = 0
    version_calls = 0

    class _CountingFree(Free):
        def version(self) -> str:
            nonlocal version_calls
            version_calls += 1
            return super().version()

    class _CountingDef(ToolDefinition):
        def to_openai_tool(self):
            nonlocal to_openai_calls
            to_openai_calls += 1
            return super().to_openai_tool()

    reg = ToolRegistry()
    tdef = _CountingDef(name="echo", description="echo", pricing=_CountingFree("local"))
    impl = FunctionRuntimeTool(lambda text="": text, ToolRuntimeContract("test.echo"))
    reg.register(tdef, impl)
    assert version_calls == 1 and to_openai_calls == 0

    rt = reg.get("echo")
    for _ in range(10):
        assert resolve(rt) is impl
        assert await reg.invoke_async("echo", {"text": "hi"}) == "hi"
    assert version_calls == 1
    assert to_openai_calls == 0


# ── 6. Redact-before-truncate, tuple/set, logger name, skill errors, monotonic claim ──


def test_h8_redact_scrubs_before_truncating_and_supports_tuple_set():
    secret = "sk-1234567890abcdefghijklmnop"
    # Place the secret so it straddles limit=20 (prefix "sk-12345" would leak if truncated first).
    raw = "prefix msg: " + secret + " tail"
    sanitized = sanitize(raw, 20, resolve_redact(None), value_patterns=DEFAULT_VALUE_PATTERNS)
    assert "sk-12345" not in sanitized
    assert "***" in sanitized

    payload = {
        "tuple_val": ("ok", f"Bearer {'A' * 20}"),
        "set_val": {f"ghp_{'B' * 24}"},
    }
    cleaned = sanitize(payload, 200, resolve_redact(None), value_patterns=DEFAULT_VALUE_PATTERNS)
    assert isinstance(cleaned["tuple_val"], tuple)
    assert cleaned["tuple_val"] == ("ok", "***")
    assert isinstance(cleaned["set_val"], set)
    assert cleaned["set_val"] == {"***"}


def test_h8_context_window_probe_logger_and_skill_section_error_propagation(monkeypatch):
    from power_loop.runtime import context_window_probe, skills

    assert context_window_probe.log.name == "power_loop.runtime.context_window_probe"
    assert build_skill_section("/nonexistent/h8/skills/dir") == ""

    def _boom(_dir):
        raise ValueError("malformed skill config")

    monkeypatch.setattr(skills, "SkillLoader", _boom)
    with pytest.raises(ValueError, match="malformed skill config"):
        build_skill_section("/some/dir")


@pytest.mark.asyncio
async def test_h8_claim_monotonic_timestamp_and_pipeline_image_rounds_cleanup():
    store = await SessionStore.open(":memory:")
    try:
        port = RuntimeInputStore(store)
        root = await target(port, "h8-mono")
        await port.accept(root, "hello", message_id="m1", actor=ACTOR)
        t_before = time.monotonic()
        batch = await port.claim(root, node_id="n1", boot_id="b1")
        t_after = time.monotonic()
        assert batch is not None and batch.claimed_at_mono is not None
        assert t_before <= batch.claimed_at_mono <= t_after
    finally:
        await store.close()

    # Verify _retire_stale_images pops retired entries from _image_rounds.
    class _NoopLLM(LLMService):
        async def complete(self, request, **kwargs):
            return LLMResponse(raw_text="ok")

        async def close(self):
            pass

    pipe = AgentPipeline(
        llm=_NoopLLM(),
        config=AgentLoopConfig(image_retention_rounds=1),
        hooks=AgentHooks(),
        bus=AgentEventBus(),
        ctx=ContextManager(),
    )
    msg = {"role": "user", "content": [{"type": "attachment", "attachment": {"name": "a.png", "kind": "image"}}]}
    pipe.history = [msg]
    pipe._image_rounds = {id(msg): 0}
    assert pipe._retire_stale_images(round_index=1, keep_rounds=1) == 1
    assert id(msg) not in pipe._image_rounds


@pytest.mark.asyncio
async def test_h8_pipeline_does_not_swallow_cancelled_error_and_rechecks_cancel_on_final_drain():
    class _SingleReplyLLM(LLMService):
        async def complete(self, request, **kwargs):
            return LLMResponse(raw_text="final answer")

        async def close(self):
            pass

    # 1) CancelledError inside LLM_RESPONSE_RECEIVED observer must propagate.
    bus = AgentEventBus()
    bus.subscribe(
        AgentEventType.LLM_RESPONSE_RECEIVED,
        lambda _e: (_ for _ in ()).throw(asyncio.CancelledError("observer cancelled")),
    )
    pipe = AgentPipeline(
        llm=_SingleReplyLLM(),
        config=AgentLoopConfig(),
        hooks=AgentHooks(),
        bus=bus,
        ctx=ContextManager(),
    )
    with pytest.raises(asyncio.CancelledError):
        await pipe.run([{"role": "user", "content": "hi"}])

    # 2) Cancel triggered during _drain_follow_ups_final yields cancelled status (not completed).
    token = CancellationToken()

    async def _cancel_on_final_drain():
        token.cancel("cancelled_at_final_drain")
        return []

    pipe2 = AgentPipeline(
        llm=_SingleReplyLLM(),
        config=AgentLoopConfig(),
        hooks=AgentHooks(),
        bus=AgentEventBus(),
        ctx=ContextManager(),
        stop_event=token,
        drain_follow_ups_final=_cancel_on_final_drain,
    )
    res = await pipe2.run([{"role": "user", "content": "hi"}])
    assert res.status == "cancelled"


@pytest.mark.asyncio
async def test_h8_shutdown_timeout_cancelled_in_final_not_marked_internal_error():
    """When shutdown times out while a send is inside final(), CancelledError/StoreError
    must NOT be converted into failed/internal_error.
    """
    store = await SessionStore.open(":memory:")
    entered_final = asyncio.Event()
    block_final = asyncio.Event()

    class _FastLLM(LLMService):
        async def complete(self, request, **kwargs):
            return LLMResponse(raw_text="done")

        async def close(self):
            pass

    async def _build(_d, _c):
        return ExecutionEnvironment(_FastLLM(), ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9))

    factory = RegistryRuntimeFactory()
    factory.register(DESCRIPTOR.definition_ref, _build)
    runtime = Runtime(store=store, factory=factory, node_id="n-shut", boot_id="b-shut")
    orig_final = runtime.worker.inputs.final

    async def _slow_final(*args, **kwargs):
        entered_final.set()
        await block_final.wait()
        return await orig_final(*args, **kwargs)

    runtime.worker.inputs.final = _slow_final  # type: ignore[method-assign]
    await runtime.worker.start()
    try:
        sid_target = await runtime.open_session(DESCRIPTOR, actor=ACTOR)
        receipt = await runtime.client.send(sid_target, "hello", message_id="m-shut", actor=ACTOR)
        await asyncio.wait_for(entered_final.wait(), timeout=3.0)
        shut = await runtime.worker.shutdown(timeout_s=0.05)
        assert shut.drained is False

        row = await store._db.fetchone(
            f"SELECT terminal_status, error_code FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
            (sid_target.session_id, receipt.message_id),
        )
        assert row is not None
        assert row["error_code"] != "internal_error"
    finally:
        block_final.set()
        await runtime.aclose(timeout_s=1.0)


@pytest.mark.asyncio
async def test_h8_stateful_loop_projection_write_does_not_swallow_session_lease_lost():
    """_run_session_send must re-raise SessionLeaseLost from projection write."""
    from power_loop.agent.stateful_loop import _ExecutionLoop

    class _FastLLM(LLMService):
        async def complete(self, request, **kwargs):
            return LLMResponse(raw_text="done")

        async def close(self):
            pass

    store = await SessionStore.open(":memory:")
    try:
        port = RuntimeInputStore(store)
        root = await target(port, "h8-lease-proj")
        await port.accept(root, "hello", message_id="m1", actor=ACTOR)
        batch = await port.claim(root, node_id="n1", boot_id="b1")
        assert batch is not None
        loop = _ExecutionLoop(llm=_FastLLM(), store=store, config=AgentLoopConfig(context_budget_tokens=10**9))

        async def _raise_lease_lost(*_a, **_kw):
            raise SessionLeaseLost(root.session_id)

        loop._prepare_or_write_session_projection = _raise_lease_lost  # type: ignore[method-assign]
        with pytest.raises(SessionLeaseLost):
            await loop._run_session_send(
                root.session_id,
                batch=batch,
                ownership=SimpleNamespace(
                    key=(store.store_id, root.session_id), epoch=batch.owner_epoch,
                    lost=False, released=False, require_owned=lambda: None,
                ),
                config=loop.config,
                tool_registry=ToolRegistry(),
                hooks=loop.hooks,
                event_bus=loop.event_bus,
                stop_event=CancellationToken(),
            )
    finally:
        await store.close()
