"""Design 218 H5 focused regression tests (P1-7 ~ P1-13).

Covers:
- P1-7: sanitize_request_messages remaps tool rows for duplicate / missing tool_call_id
- P1-8: trim_history terminates on orphan tool boundaries without breaking pair atomicity
- P1-9: multi-tool batch steer/urgent interruption wired via ctx.urgent_event, orphan _SessionSync.steer removed
- P1-10: _cancel_pre_tasks awaits cancellation + try/finally prevents leaked concurrent tasks;
         serial TOOL_ERROR SHORT_CIRCUIT honors cancel_token via _race and re-raises NativeReturnPending
- P1-11: truncated / think-only exhaustion stops without cascading into empty retries; normal text resets
         _truncated_streak; hit_round_limit uses actual round_idx, tools=None, and _race cancellation
- P1-12: HTTP 400 folding preserves send_index=0 and non-tool assistant final text; aligned historical_end_idx
- P1-13: build_agent_system_prompt preserves custom builder sections/overrides without mutating source builder
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentHooks,
    AgentLoopConfig,
    SessionStore,
    SystemPromptBuilder,
    SystemPromptContext,
    ToolDefinition,
    build_agent_system_prompt,
    trim_history,
)
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
    LLMTokenUsage,
)
from power_loop.agent.stateful_loop import _ExecutionLoop, _SessionSync
from power_loop.contracts.hook_contexts import CompleteDecideCtx, ToolBeforeCtx
from power_loop.contracts.hooks import HookDirective, HookPoint
from power_loop.core.pipeline import AgentPipeline
from power_loop.core.state import ContextManager
from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.history_sanitize import (
    fold_tool_rounds_by_send_index,
    sanitize_request_messages,
)
from power_loop.runtime.native_returns import NativeReturnPending
from power_loop.runtime.pricing import Free
from power_loop.runtime.representation import ProjectedRepresentation, resolve_send_projections
from power_loop.runtime.store.types import ProjectMessageRow
from power_loop.tools.registry import ToolRegistry
from tests.unit.test_request_sanitize import assert_valid_request

USAGE = LLMTokenUsage(prompt_tokens=40, completion_tokens=10, total_tokens=50)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _tc(cid: str | None, name: str = "probe", args: str = "{}") -> dict[str, Any]:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


class _Choice:
    def __init__(self, reason: str) -> None:
        self.finish_reason = reason


class _Completion:
    def __init__(self, reason: str) -> None:
        self.choices = [_Choice(reason)]


@dataclass
class _ScriptedLLM(LLMService):
    steps: list[Any] = field(default_factory=list)
    requests: list[LLMRequest] = field(default_factory=list)

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text=None,
        on_chunk_think=None,
        on_stream_end=None,
    ) -> LLMResponse:
        self.requests.append(request)
        if not self.steps:
            return LLMResponse(raw_text="done", token_usage=USAGE)
        step = self.steps.pop(0)
        if callable(step):
            return await step(request)
        if isinstance(step, Exception):
            raise step
        return step

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _empty()

    async def close(self) -> None:
        return None


def _build_pipeline(
    llm: LLMService,
    *,
    registry: ToolRegistry | None = None,
    hooks: AgentHooks | None = None,
    bus: AgentEventBus | None = None,
    config: AgentLoopConfig | None = None,
    cancel_token: CancellationToken | None = None,
    session_id: str = "h5-sess",
) -> AgentPipeline:
    return AgentPipeline(
        llm=llm,
        config=config or AgentLoopConfig(system_prompt="S", max_rounds=6, context_budget_tokens=10**9),
        tool_registry=registry,
        hooks=hooks or AgentHooks(),
        bus=bus or AgentEventBus(),
        ctx=ContextManager(),
        session_id=session_id,
        stop_event=cancel_token,
    )


# ── P1-7: sanitize_request_messages duplicate & missing tool_call_id remapping ──


def test_h5_p1_7_duplicate_tool_call_ids_preserve_all_real_results_in_order() -> None:
    messages = [
        {"role": "user", "content": "run three probes"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tc("dup", "p1"), _tc("dup", "p2"), _tc("dup", "p3")],
        },
        {"role": "tool", "tool_call_id": "dup", "name": "p1", "content": "r1"},
        {"role": "tool", "tool_call_id": "dup", "name": "p2", "content": "r2"},
        {"role": "tool", "tool_call_id": "dup", "name": "p3", "content": "r3"},
    ]
    out = sanitize_request_messages(messages)
    assert_valid_request(out)
    call_ids = [tc["id"] for tc in out[1]["tool_calls"]]
    assert len(set(call_ids)) == 3
    tool_rows = [m for m in out if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_rows] == call_ids
    assert [m["content"] for m in tool_rows] == ["r1", "r2", "r3"]
    assert sanitize_request_messages(out) == out


def test_h5_p1_7_missing_and_cross_turn_duplicate_and_suffix_collision_ids() -> None:
    messages = [
        {"role": "user", "content": "turn 1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tc(None, "m1"), _tc("", "m2"), _tc("c1", "a"), _tc("c1_dedup_1", "b")],
        },
        {"role": "tool", "tool_call_id": None, "name": "m1", "content": "miss_1"},
        {"role": "tool", "tool_call_id": "", "name": "m2", "content": "miss_2"},
        {"role": "tool", "tool_call_id": "c1", "name": "a", "content": "res_a"},
        {"role": "tool", "tool_call_id": "c1_dedup_1", "name": "b", "content": "res_b"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "", "tool_calls": [_tc("c1", "a2")]},
        {"role": "tool", "tool_call_id": "c1", "name": "a2", "content": "res_a2"},
    ]
    out = sanitize_request_messages(messages)
    assert_valid_request(out)
    tool_contents = [m["content"] for m in out if m["role"] == "tool"]
    assert tool_contents == ["miss_1", "miss_2", "res_a", "res_b", "res_a2"]
    all_call_ids = [tc["id"] for m in out if m.get("tool_calls") for tc in m["tool_calls"]]
    assert len(set(all_call_ids)) == 5


# ── P1-8: trim_history orphan tool boundary termination ───────────────────────


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [
                {"role": "user", "content": "q1"},
                {"role": "tool", "tool_call_id": "orphan_1", "content": "ghost"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
            id="user-orphan-tool-user-assistant",
        ),
        pytest.param(
            [
                {"role": "tool", "tool_call_id": "t1", "content": "x"},
                {"role": "tool", "tool_call_id": "t2", "content": "y"},
            ],
            id="all-tool",
        ),
        pytest.param(
            [
                {"role": "tool", "tool_call_id": "t1", "content": "x"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            id="leading-orphan-tool",
        ),
        pytest.param([], id="empty"),
    ],
)
def test_h5_p1_8_trim_history_terminates_on_degenerate_histories(messages: list[dict[str, Any]]) -> None:
    # Must terminate immediately (never spin infinitely when walking back hits user/start).
    result = trim_history(messages, max_tokens=1, keep_last_n=1)
    assert isinstance(result, list)
    # Also verify with a realistic budget that fits the last exchange.
    if messages:
        res2 = trim_history(messages, max_tokens=20, keep_last_n=1)
        assert isinstance(res2, list)


def test_h5_p1_8_trim_history_still_preserves_valid_tool_pair_at_boundary() -> None:
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old " * 40},
        {"role": "assistant", "content": "call", "tool_calls": [_tc("tc1", "echo")]},
        {"role": "tool", "tool_call_id": "tc1", "name": "echo", "content": "res"},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "done"},
    ]
    # Budget fits system + tool pair + latest exchange, not the big old user message.
    budget = estimate_tokens(msgs[:1]) + estimate_tokens(msgs[2:]) + 2
    result = trim_history(msgs, max_tokens=budget, keep_last_n=1)
    has_tc = any(m.get("tool_calls") for m in result)
    has_tool = any(m.get("role") == "tool" for m in result)
    assert has_tc and has_tool
    assert msgs[-2] in result and msgs[-1] in result


# ── P1-9: multi-tool steer/urgent wiring & orphan _SessionSync.steer removal ──


def test_h5_p1_9_orphan_session_sync_steer_removed() -> None:
    sync = _SessionSync()
    assert not hasattr(sync, "steer")
    assert not hasattr(_ExecutionLoop, "steer_event")


@pytest.mark.asyncio
async def test_h5_p1_9_urgent_interrupts_second_tool_and_resets_for_next_round(runtime_agent) -> None:
    bus = AgentEventBus()
    steer_events: list[dict[str, Any]] = []
    bus.subscribe(AgentEventType.STEER_INTERRUPTED, lambda e: steer_events.append(dict(e.payload or {})))

    tool1_started = asyncio.Event()
    tool1_release = asyncio.Event()
    tool2_calls = 0
    followup_tool_calls = 0

    async def tool1(**kw: Any) -> str:
        tool1_started.set()
        await tool1_release.wait()
        return "tool1_ok"

    def tool2(**kw: Any) -> str:
        nonlocal tool2_calls
        tool2_calls += 1
        return "tool2_should_not_run"

    def after_steer_tool(**kw: Any) -> str:
        nonlocal followup_tool_calls
        followup_tool_calls += 1
        return "after_steer_ok"

    reg = ToolRegistry()
    for name, fn in (("tool1", tool1), ("tool2", tool2), ("after_steer", after_steer_tool)):
        reg.register(
            ToolDefinition(
                name=name,
                description=name,
                input_schema={"type": "object", "properties": {}},
                pricing=Free("test_fixture"),
            ),
            fn,
        )

    llm = _ScriptedLLM(
        steps=[
            LLMResponse(raw_text="", tool_calls=[_tc("c1", "tool1"), _tc("c2", "tool2")], token_usage=USAGE),
            # Round 1 after urgent message is drained: tool calls MUST execute (urgent hint cleared!)
            LLMResponse(raw_text="", tool_calls=[_tc("c3", "after_steer")], token_usage=USAGE),
            LLMResponse(raw_text="handled urgent", token_usage=USAGE),
        ]
    )
    store = await SessionStore.open(":memory:")
    try:
        agent = await runtime_agent(
            model=llm,
            store=store,
            registry=reg,
            event_bus=bus,
            config=AgentLoopConfig(system_prompt="S", max_rounds=5, context_budget_tokens=10**9),
        )
        run = asyncio.create_task(agent.send("start batch"))
        await asyncio.wait_for(tool1_started.wait(), 5)
        # Deliver priority="urgent" while tool1 is executing
        await agent.runtime.client.send(
            agent.target, "URGENT_STOP_BATCH", priority="urgent", message_id="urg-1", actor=agent.actor
        )
        tool1_release.set()
        res = await asyncio.wait_for(run, 5)
        assert res.status == "completed"
        assert res.final_text == "handled urgent"
        assert tool2_calls == 0
        assert followup_tool_calls == 1
        assert len(steer_events) == 1
        assert steer_events[0]["where"] == "between_tools"
        assert steer_events[0]["tool_name"] == "tool2"
        # Round 1 request saw the urgent message
        assert any("URGENT_STOP_BATCH" in str(m.get("content")) for m in llm.requests[1].messages)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_h5_p1_9_normal_priority_does_not_skip_second_tool(runtime_agent) -> None:
    tool1_started = asyncio.Event()
    tool1_release = asyncio.Event()
    tool2_calls = 0

    async def tool1(**kw: Any) -> str:
        tool1_started.set()
        await tool1_release.wait()
        return "tool1_ok"

    def tool2(**kw: Any) -> str:
        nonlocal tool2_calls
        tool2_calls += 1
        return "tool2_ok"

    reg = ToolRegistry()
    for name, fn in (("tool1", tool1), ("tool2", tool2)):
        reg.register(
            ToolDefinition(
                name=name,
                description=name,
                input_schema={"type": "object", "properties": {}},
                pricing=Free("test_fixture"),
            ),
            fn,
        )

    llm = _ScriptedLLM(
        steps=[
            LLMResponse(raw_text="", tool_calls=[_tc("c1", "tool1"), _tc("c2", "tool2")], token_usage=USAGE),
            LLMResponse(raw_text="done", token_usage=USAGE),
        ]
    )
    store = await SessionStore.open(":memory:")
    try:
        agent = await runtime_agent(
            model=llm,
            store=store,
            registry=reg,
            config=AgentLoopConfig(system_prompt="S", max_rounds=4, context_budget_tokens=10**9),
        )
        run = asyncio.create_task(agent.send("start batch"))
        await asyncio.wait_for(tool1_started.wait(), 5)
        await agent.runtime.client.send(
            agent.target, "NORMAL_MSG", priority="normal", message_id="norm-1", actor=agent.actor
        )
        tool1_release.set()
        res = await asyncio.wait_for(run, 5)
        assert res.status == "completed"
        assert tool2_calls == 1
    finally:
        await store.close()


# ── P1-10: concurrent task leak prevention & TOOL_ERROR retry cancel/NativeReturn ──


@pytest.mark.asyncio
async def test_h5_p1_10_tool_before_exception_cancels_and_awaits_started_concurrent_tasks() -> None:
    started = asyncio.Event()
    cancelled_and_cleaned = asyncio.Event()

    async def slow_async(**kw: Any) -> str:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            cancelled_and_cleaned.set()
            raise
        return "never"

    reg = ToolRegistry()
    for name in ("a1", "a2"):
        reg.register(
            ToolDefinition(
                name=name,
                description=name,
                input_schema={"type": "object", "properties": {}},
                async_capable=True,
                pricing=Free("test_fixture"),
            ),
            slow_async,
        )

    hooks = AgentHooks()

    async def boom_on_second(ctx: ToolBeforeCtx) -> None:
        if ctx.tool_name == "a2":
            await asyncio.wait_for(started.wait(), 2)
            raise RuntimeError("hook boom on a2")

    hooks.register(HookPoint.TOOL_BEFORE, boom_on_second, name="t.boom_a2")

    llm = _ScriptedLLM(
        steps=[LLMResponse(raw_text="", tool_calls=[_tc("c1", "a1"), _tc("c2", "a2")], token_usage=USAGE)]
    )
    pipeline = _build_pipeline(
        llm,
        registry=reg,
        hooks=hooks,
        config=AgentLoopConfig(
            system_prompt="S",
            max_rounds=3,
            tool_batch_concurrency=2,
            context_budget_tokens=10**9,
        ),
    )
    with pytest.raises(RuntimeError, match="hook boom on a2"):
        await pipeline.run([{"role": "user", "content": "go"}])
    # Already awaited inside try/finally before run() raised — zero leaked background tasks!
    assert cancelled_and_cleaned.is_set()


@pytest.mark.asyncio
async def test_h5_p1_10_tool_after_break_awaits_cancelled_concurrent_sibling() -> None:
    sibling_started = asyncio.Event()
    sibling_cleaned = asyncio.Event()

    async def fast_tool(**kw: Any) -> str:
        await asyncio.wait_for(sibling_started.wait(), 5)
        return "fast_ok"

    async def slow_sibling(**kw: Any) -> str:
        sibling_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            sibling_cleaned.set()
            raise
        return "never"

    reg = ToolRegistry()
    for name, fn in (("fast_a", fast_tool), ("slow_b", slow_sibling)):
        reg.register(
            ToolDefinition(
                name=name,
                description=name,
                input_schema={"type": "object", "properties": {}},
                async_capable=True,
                pricing=Free("test_fixture"),
            ),
            fn,
        )
    hooks = AgentHooks()
    hooks.register(
        HookPoint.TOOL_AFTER,
        lambda ctx: setattr(ctx, "directive", HookDirective.BREAK) if ctx.tool_name == "fast_a" else None,
        name="t.break_after_fast",
    )
    llm = _ScriptedLLM(
        steps=[
            LLMResponse(raw_text="", tool_calls=[_tc("c1", "fast_a"), _tc("c2", "slow_b")], token_usage=USAGE),
            LLMResponse(raw_text="finished after break", token_usage=USAGE),
        ]
    )
    pipeline = _build_pipeline(
        llm,
        registry=reg,
        hooks=hooks,
        config=AgentLoopConfig(
            system_prompt="S",
            max_rounds=3,
            tool_batch_concurrency=2,
            context_budget_tokens=10**9,
        ),
    )
    res = await asyncio.wait_for(pipeline.run([{"role": "user", "content": "go"}]), 5)
    assert res.status == "completed"
    assert sibling_cleaned.is_set()


@pytest.mark.asyncio
async def test_h5_p1_10_serial_tool_error_short_circuit_retry_honors_cancel_and_reraises_native_return() -> None:
    # Sub-case 1: serial TOOL_ERROR SHORT_CIRCUIT retry is cancellable via _race
    retry_started = asyncio.Event()
    retry_cancelled = asyncio.Event()
    attempts = 0

    async def flaky_then_hang(**kw: Any) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient_first")
        retry_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            retry_cancelled.set()
            raise
        return "never"

    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="flaky",
            description="flaky",
            input_schema={"type": "object", "properties": {}},
            interrupt="abort",
            pricing=Free("test_fixture"),
        ),
        flaky_then_hang,
    )
    hooks = AgentHooks()
    hooks.register(
        HookPoint.TOOL_ERROR,
        lambda ctx: setattr(ctx, "directive", HookDirective.SHORT_CIRCUIT),
        name="t.retry",
    )
    tok = CancellationToken()
    llm = _ScriptedLLM(steps=[LLMResponse(raw_text="", tool_calls=[_tc("c1", "flaky")], token_usage=USAGE)])
    pipeline = _build_pipeline(llm, registry=reg, hooks=hooks, cancel_token=tok)

    task = asyncio.create_task(pipeline.run([{"role": "user", "content": "run"}]))
    await asyncio.wait_for(retry_started.wait(), 5)
    tok.cancel("stop during retry")
    res = await asyncio.wait_for(task, 5)
    assert res.status == "cancelled"
    assert retry_cancelled.is_set()

    # Sub-case 2: serial TOOL_ERROR SHORT_CIRCUIT retry re-raises NativeReturnPending
    attempts2 = 0

    async def flaky_then_native(**kw: Any) -> str:
        nonlocal attempts2
        attempts2 += 1
        if attempts2 == 1:
            raise RuntimeError("transient_first")
        raise NativeReturnPending()

    reg2 = ToolRegistry()
    reg2.register(
        ToolDefinition(
            name="flaky2",
            description="flaky2",
            input_schema={"type": "object", "properties": {}},
            pricing=Free("test_fixture"),
        ),
        flaky_then_native,
    )
    llm2 = _ScriptedLLM(steps=[LLMResponse(raw_text="", tool_calls=[_tc("c2", "flaky2")], token_usage=USAGE)])
    pipeline2 = _build_pipeline(llm2, registry=reg2, hooks=hooks)
    with pytest.raises(NativeReturnPending):
        await pipeline2.run([{"role": "user", "content": "run"}])


# ── P1-11: truncation exhaustion isolation & hit_round_limit fixes ────────────


def _think_only_truncated(think: str = "deep thinking...") -> LLMResponse:
    r = LLMResponse(raw_text="", think=think, tool_calls=[], token_usage=USAGE)
    r.raw_completion = _Completion("length")
    return r


@pytest.mark.asyncio
async def test_h5_p1_11_think_only_truncation_exhaustion_does_not_cascade_into_empty_retries() -> None:
    llm = _ScriptedLLM(steps=[_think_only_truncated() for _ in range(10)])
    pipeline = _build_pipeline(
        llm,
        registry=ToolRegistry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=10, max_tokens=2000, context_budget_tokens=10**9),
    )
    res = await pipeline.run([{"role": "user", "content": "solve"}])
    # _TRUNCATED_MAX_RETRIES == 2 -> 1 initial + 2 nudged retries = 3 calls total (NOT 3 + 3 = 6!)
    assert res.status == "completed"
    assert len(llm.requests) == 3
    assert pipeline._empty_response_streak == 0


@pytest.mark.asyncio
async def test_h5_p1_11_normal_text_resets_truncated_streak() -> None:
    # Truncated once, then normal text (triggers COMPLETE_DECIDE continuation), then truncated twice more:
    # if normal text reset _truncated_streak, both post-continuation truncations get nudged (not closed early).
    hooks = AgentHooks()
    fired = False

    def continue_once(ctx: CompleteDecideCtx) -> None:
        nonlocal fired
        if not fired and ctx.final_text == "normal answer":
            fired = True
            ctx.directive = HookDirective.SHORT_CIRCUIT
            ctx.inject = "please continue"
            ctx.extra_rounds = 6

    hooks.register(HookPoint.COMPLETE_DECIDE, continue_once, name="t.cont")
    llm = _ScriptedLLM(
        steps=[
            _think_only_truncated("t1"),
            LLMResponse(raw_text="normal answer", tool_calls=[], token_usage=USAGE),
            _think_only_truncated("t2"),
            _think_only_truncated("t3"),
            LLMResponse(raw_text="final recovered", tool_calls=[], token_usage=USAGE),
        ]
    )
    pipeline = _build_pipeline(
        llm,
        registry=ToolRegistry(),
        hooks=hooks,
        config=AgentLoopConfig(system_prompt="S", max_rounds=8, max_tokens=2000, context_budget_tokens=10**9),
    )
    res = await pipeline.run([{"role": "user", "content": "go"}])
    assert res.status == "completed"
    assert res.final_text == "final recovered"
    assert len(llm.requests) == 5


@pytest.mark.asyncio
async def test_h5_p1_11_hit_round_limit_uses_actual_round_idx_disables_tools_and_honors_cancel() -> None:
    bus = AgentEventBus()
    events: list[Any] = []
    bus.subscribe(None, events.append)

    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo",
            description="echo",
            input_schema={"type": "object", "properties": {}},
            pricing=Free("test_fixture"),
        ),
        lambda **kw: "ok",
    )

    # Extend max_rounds from 1 -> 2 via COMPLETE_DECIDE so actual round_idx == 2 != config.max_rounds (1)
    hooks = AgentHooks()
    extended = False

    def extend_once(ctx: CompleteDecideCtx) -> None:
        nonlocal extended
        if ctx.reason == "hit_round_limit" and not extended:
            extended = True
            ctx.directive = HookDirective.SHORT_CIRCUIT
            ctx.inject = "one more round"
            ctx.extra_rounds = 1

    hooks.register(HookPoint.COMPLETE_DECIDE, extend_once, name="t.extend")

    llm = _ScriptedLLM(
        steps=[
            LLMResponse(raw_text="", tool_calls=[_tc("c1", "echo")], token_usage=USAGE),
            LLMResponse(raw_text="", tool_calls=[_tc("c2", "echo")], token_usage=USAGE),
            LLMResponse(raw_text="wrapped up cleanly", token_usage=USAGE),
        ]
    )
    pipeline = _build_pipeline(
        llm,
        registry=reg,
        hooks=hooks,
        bus=bus,
        config=AgentLoopConfig(system_prompt="S", max_rounds=1, context_budget_tokens=10**9),
    )
    res = await pipeline.run([{"role": "user", "content": "go"}])
    assert res.status == "hit_round_limit"
    assert res.rounds == 2
    assert "wrapped up cleanly" in res.final_text
    # Summary LLM call MUST have tools=None and tool_choice=None
    summary_req = llm.requests[-1]
    assert summary_req.tools is None
    assert summary_req.tool_choice is None
    # Status payload & LLM_CALL_STARTED for summary round use actual round_idx=2
    hit_events = [e for e in events if e.type == AgentEventType.STATUS_CHANGED and e.payload.get("kind") == "hit_round_limit"]
    assert hit_events and hit_events[-1].payload["max_rounds"] == 2
    llm_starts = [e for e in events if e.type == AgentEventType.LLM_CALL_STARTED]
    assert llm_starts[-1].round_index == 2

    # Sub-case 2: hit_round_limit summary call responds to mid-flight cancellation via _race
    summary_started = asyncio.Event()
    summary_cancelled = asyncio.Event()

    async def hang_summary(req: LLMRequest) -> LLMResponse:
        assert req.tools is None
        summary_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            summary_cancelled.set()
            raise
        return LLMResponse(raw_text="never", token_usage=USAGE)

    tok = CancellationToken()
    llm_cancel = _ScriptedLLM(
        steps=[
            LLMResponse(raw_text="", tool_calls=[_tc("c1", "echo")], token_usage=USAGE),
            hang_summary,
        ]
    )
    pipeline_cancel = _build_pipeline(
        llm_cancel,
        registry=reg,
        cancel_token=tok,
        config=AgentLoopConfig(system_prompt="S", max_rounds=1, context_budget_tokens=10**9),
    )
    run_task = asyncio.create_task(pipeline_cancel.run([{"role": "user", "content": "go"}]))
    await asyncio.wait_for(summary_started.wait(), 5)
    tok.cancel("cancel summary")
    res_cancel = await asyncio.wait_for(run_task, 5)
    assert res_cancel.status == "cancelled"
    assert summary_cancelled.is_set()


# ── P1-12: HTTP 400 folding send_index=0 & final assistant text preservation ──


def test_h5_p1_12_fold_tool_rounds_preserves_send_index_zero_and_final_text() -> None:
    messages = [
        {"role": "user", "content": "q0", "_source_send_index": 0},
        {"role": "assistant", "content": "", "_source_send_index": 0, "tool_calls": [_tc("t0", "probe")]},
        {"role": "tool", "tool_call_id": "t0", "name": "probe", "content": "r1", "_source_send_index": 0},
        {"role": "assistant", "content": "FINAL_ANSWER_0", "_source_send_index": 0},
        {"role": "user", "content": "q2"},
    ]
    cleaned, per_send, cur_txt = fold_tool_rounds_by_send_index(
        messages, current_send_index=2, historical_end_idx=4
    )
    assert cur_txt == ""
    assert set(per_send.keys()) == {0}
    assert "[folded_tool:probe] r1" in per_send[0]
    assert "FINAL_ANSWER_0" in per_send[0]
    assert all(m.get("role") != "tool" and not m.get("tool_calls") for m in cleaned)
    assert any("FINAL_ANSWER_0" in str(m.get("content")) for m in cleaned)


def test_h5_p1_12_resolve_send_projections_repaired_supersedes_seed_and_merges_project_final() -> None:
    sid = "s-p1-12"
    seed_row = ProjectMessageRow(
        sid, 0, "seed", {"messages": [{"role": "user", "content": "q0"}]}, "q0",
        None, None, None, None, 0, None, 0, id=1,
    )
    rep_seed = ProjectMessageRow(
        sid, 0, "repaired", {"folded_on_400": True, "text": "[folded_tool:p] r\nFINAL_0"},
        "[folded_tool:p] r\nFINAL_0", None, None, None, None, 0, None, 0, id=2,
    )
    _, resolved_seed = resolve_send_projections([seed_row, rep_seed])
    assert [r.kind for r in resolved_seed[0]] == ["repaired"]

    u_row = ProjectMessageRow(sid, 1, "user", {"input": ["q1"]}, "q1", None, None, None, None, 1, None, 0, id=10)
    p_row = ProjectMessageRow(
        sid, 1, "project", {"tools": [], "final_text": "PRESERVED_FINAL_1"},
        "PRESERVED_FINAL_1", None, None, None, None, 1, None, 0, id=11,
    )
    rep_row = ProjectMessageRow(
        sid, 1, "repaired", {"folded_on_400": True, "text": "[folded_tool:probe] r1"},
        "[folded_tool:probe] r1", None, None, None, None, 0, None, 0, id=12,
    )
    _, resolved_turn = resolve_send_projections([u_row, p_row, rep_row])
    rendered = ProjectedRepresentation().render(resolved_turn[1])
    assert any("PRESERVED_FINAL_1" in str(m.get("content")) for m in rendered)
    assert any("[folded_tool:probe] r1" in str(m.get("content")) for m in rendered)


@pytest.mark.asyncio
async def test_h5_p1_12_e2e_http_400_fold_with_aligned_boundary_and_reload(runtime_agent) -> None:
    class _Format400Error(Exception):
        status_code = 400

        def __init__(self) -> None:
            super().__init__("invalid tool_call_id messages format")

    calls_seen: list[list[dict[str, Any]]] = []
    attempt = 0

    async def scripted(req: LLMRequest) -> LLMResponse:
        nonlocal attempt
        attempt += 1
        calls_seen.append([dict(m) for m in req.messages])
        if attempt == 1:
            raise _Format400Error()
        return LLMResponse(raw_text=f"ok_{attempt}", token_usage=USAGE)

    llm = _ScriptedLLM(steps=[scripted, scripted, scripted])
    store = await SessionStore.open(":memory:")
    try:
        agent = await runtime_agent(
            model=llm,
            store=store,
            registry=ToolRegistry(),
            config=AgentLoopConfig(system_prompt="SYS", max_rounds=4, context_budget_tokens=10**9),
        )
        sid = agent.target.session_id
        # Seed send_index=0 with a dangling tool_call (so align_tool_calls inserts a placeholder, shifting
        # historical_end_idx!) followed by a real tool exchange + non-tool final assistant reply.
        await store.upsert_repaired_projection(
            sid,
            send_index=0,
            content={
                "messages": [
                    {"role": "user", "content": "seed_q0"},
                    {"role": "assistant", "content": "dangling", "tool_calls": [_tc("d0", "probe")]},
                    {"role": "user", "content": "seed_q0b"},
                    {"role": "assistant", "content": "", "tool_calls": [_tc("t0", "probe")]},
                    {"role": "tool", "tool_call_id": "t0", "name": "probe", "content": "seed_r0"},
                    {"role": "assistant", "content": "SEED_FINAL_REPLY_0"},
                ],
            },
            rendered_text="seed turn 0",
        )
        await store.upsert_repaired_projection(
            sid,
            send_index=1,
            content={
                "messages": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "", "tool_calls": [_tc("t1", "probe")]},
                    {"role": "tool", "tool_call_id": "t1", "name": "probe", "content": "r1"},
                    {"role": "assistant", "content": "TURN1_FINAL_REPLY"},
                ],
            },
            rendered_text="turn 1",
        )
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET send_index=1,next_seq=10 WHERE session_id=?", (sid,)
        )

        # Send 2 hits HTTP 400 on attempt 1, folds historical turns 0 and 1, succeeds on attempt 2.
        res2 = await agent.send("live turn 2")
        assert res2.status == "completed" and res2.final_text == "ok_2"
        projs = await store.load_project_messages(sid)
        rep_by_si = {p.send_index: p for p in projs if p.kind == "repaired"}
        assert 0 in rep_by_si and 1 in rep_by_si
        assert "SEED_FINAL_REPLY_0" in (rep_by_si[0].rendered_text or "")
        assert "TURN1_FINAL_REPLY" in (rep_by_si[1].rendered_text or "")

        # Send 3 reloads the repaired projections — both final assistant replies MUST still be visible!
        res3 = await agent.send("live turn 3")
        assert res3.status == "completed" and res3.final_text == "ok_3"
        turn3_text = "\n".join(str(m.get("content") or "") for m in calls_seen[-1])
        assert "SEED_FINAL_REPLY_0" in turn3_text
        assert "TURN1_FINAL_REPLY" in turn3_text
    finally:
        await store.close()


# ── P1-13: build_agent_system_prompt custom builder preservation & no mutation ──


def test_h5_p1_13_build_agent_system_prompt_preserves_custom_and_overridden_sections_without_mutation() -> None:
    custom_builder = (
        SystemPromptBuilder()
        .remove("security")
        .add("identity", lambda ctx: f"CUSTOM_IDENTITY:{ctx.model}")
        .add("my_custom_rules", lambda ctx: f"CUSTOM_RULE_IN_{ctx.workspace_dir}", after="identity")
    )
    before_names = custom_builder.section_names()
    ctx = SystemPromptContext(model="gpt-test", workspace_dir="/ws/h5")

    prompt = build_agent_system_prompt(ctx, extra="EXTRA_INSTRUCTION", builder=custom_builder)
    assert "CUSTOM_IDENTITY:gpt-test" in prompt
    assert "CUSTOM_RULE_IN_/ws/h5" in prompt
    assert "EXTRA_INSTRUCTION" in prompt
    assert "You are an interactive coding agent." not in prompt
    assert "# Security" not in prompt

    # Source builder MUST remain untouched (no side-effect mutation).
    assert custom_builder.section_names() == before_names
    assert "extra" not in custom_builder.section_names()
    base_prompt = custom_builder.build(ctx)
    assert "EXTRA_INSTRUCTION" not in base_prompt
    assert "CUSTOM_IDENTITY:gpt-test" in base_prompt
    assert "CUSTOM_RULE_IN_/ws/h5" in base_prompt
