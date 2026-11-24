"""Unit tests for unified typed AgentHooks control-flow mechanism."""
from __future__ import annotations

import asyncio

from power_loop.contracts.hook_contexts import (
    CompactBeforeCtx,
    LlmBeforeCtx,
    RoundDecideCtx,
    RoundStartCtx,
    ToolBeforeCtx,
)
from power_loop.contracts.hooks import HookContext, HookDirective, HookPoint
from power_loop.core.hooks import AgentHooks


def test_typed_hook_context_mutation_and_default_continue():
    hooks = AgentHooks()

    def handler(ctx: HookContext) -> None:
        ctx.values["touched"] = True

    hooks.register(HookPoint.ROUND_START, handler)
    ctx = HookContext(values={"round": 1})
    hooks.run_typed(HookPoint.ROUND_START, ctx)
    assert ctx.directive == HookDirective.CONTINUE
    assert ctx.values["touched"] is True
    assert ctx.values["round"] == 1


def test_typed_hook_skip_directive():
    hooks = AgentHooks()

    def skip_handler(ctx: ToolBeforeCtx) -> HookDirective:
        ctx.output = "[blocked by policy]"
        return HookDirective.SKIP

    hooks.register(HookPoint.TOOL_BEFORE, skip_handler)
    ctx = ToolBeforeCtx(tool_name="bash", tool_args={"cmd": "rm -rf /"})
    hooks.run_typed(HookPoint.TOOL_BEFORE, ctx)
    assert ctx.directive == HookDirective.SKIP
    assert ctx.output == "[blocked by policy]"


def test_typed_hook_break_directive():
    hooks = AgentHooks()

    def budget_guard(ctx: RoundStartCtx) -> HookDirective:
        ctx.reason = "token_budget_exceeded"
        return HookDirective.BREAK

    hooks.register(HookPoint.ROUND_START, budget_guard)
    ctx = RoundStartCtx(round_index=5)
    hooks.run_typed(HookPoint.ROUND_START, ctx)
    assert ctx.directive == HookDirective.BREAK
    assert ctx.reason == "token_budget_exceeded"


def test_typed_hook_short_circuit():
    hooks = AgentHooks()

    class FakeResponse:
        raw_text = "cached answer"

    def cache_hit(ctx: LlmBeforeCtx) -> HookDirective:
        ctx.response = FakeResponse()
        return HookDirective.SHORT_CIRCUIT

    hooks.register(HookPoint.LLM_BEFORE, cache_hit)
    ctx = LlmBeforeCtx(messages=[])
    hooks.run_typed(HookPoint.LLM_BEFORE, ctx)
    assert ctx.directive == HookDirective.SHORT_CIRCUIT
    assert ctx.response.raw_text == "cached answer"


def test_multiple_handlers_last_directive_wins_and_none_preserves():
    hooks = AgentHooks()

    def first(ctx: HookContext) -> HookDirective:
        ctx.values["first"] = True
        return HookDirective.SKIP

    def observer(ctx: HookContext) -> None:
        ctx.values["observed"] = True

    def override(ctx: HookContext) -> HookDirective:
        ctx.values["override"] = True
        return HookDirective.CONTINUE

    hooks.register(HookPoint.TOOL_BEFORE, first, order=0)
    hooks.register(HookPoint.TOOL_BEFORE, observer, order=1)
    ctx = HookContext()
    hooks.run_typed(HookPoint.TOOL_BEFORE, ctx)
    assert ctx.directive == HookDirective.SKIP
    assert ctx.values["observed"] is True

    hooks.register(HookPoint.TOOL_BEFORE, override, order=2)
    ctx2 = HookContext()
    hooks.run_typed(HookPoint.TOOL_BEFORE, ctx2)
    assert ctx2.directive == HookDirective.CONTINUE
    assert ctx2.values["override"] is True


def test_async_handler_directive():
    hooks = AgentHooks()

    async def async_break(ctx: HookContext) -> HookDirective:
        ctx.values["async"] = True
        return HookDirective.BREAK

    hooks.register(HookPoint.TOOL_AFTER, async_break)
    ctx = HookContext(values={"tool_name": "bash"})
    asyncio.run(hooks.run_typed_async(HookPoint.TOOL_AFTER, ctx))
    assert ctx.directive == HookDirective.BREAK
    assert ctx.values["async"] is True


def test_round_decide_and_compact_before_skip():
    hooks = AgentHooks()
    hooks.register(HookPoint.ROUND_DECIDE, lambda ctx: HookDirective.SKIP)
    hooks.register(HookPoint.COMPACT_BEFORE, lambda ctx: HookDirective.SKIP)

    r_ctx = RoundDecideCtx(tool_calls=[{"function": {"name": "bash", "arguments": "{}"}}])
    hooks.run_typed(HookPoint.ROUND_DECIDE, r_ctx)
    assert r_ctx.directive == HookDirective.SKIP

    c_ctx = CompactBeforeCtx(round_index=3)
    hooks.run_typed(HookPoint.COMPACT_BEFORE, c_ctx)
    assert c_ctx.directive == HookDirective.SKIP
