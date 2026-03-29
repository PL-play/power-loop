"""Unit tests for HookDirective / HookResult control-flow mechanism."""
from __future__ import annotations

from power_loop.contracts.hooks import HookContext, HookDirective, HookPoint, HookResult
from power_loop.core.hooks import AgentHooks


# ---------------------------------------------------------------------------
# 1. Basic: handlers that return HookContext still work (backward compat)
# ---------------------------------------------------------------------------

def test_backward_compat_hook_context_return():
    hooks = AgentHooks()

    def handler(ctx: HookContext) -> HookContext:
        ctx.values["touched"] = True
        return ctx

    hooks.register(HookPoint.ROUND_START, handler)
    hr = hooks.run(HookPoint.ROUND_START, HookContext(values={"round": 1}))
    assert isinstance(hr, HookResult)
    assert hr.directive == HookDirective.CONTINUE
    assert hr.context.values["touched"] is True
    assert hr.context.values["round"] == 1


# ---------------------------------------------------------------------------
# 2. Basic: handlers that return dict still work (backward compat)
# ---------------------------------------------------------------------------

def test_backward_compat_dict_return():
    hooks = AgentHooks()

    def handler(ctx: HookContext) -> dict:
        return {**ctx.values, "enriched": True}

    hooks.register(HookPoint.LLM_BEFORE, handler)
    hr = hooks.run(HookPoint.LLM_BEFORE, HookContext(values={"model": "gpt-4"}))
    assert hr.directive == HookDirective.CONTINUE
    assert hr.context.values["enriched"] is True
    assert hr.context.values["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# 3. Handler returns HookResult with SKIP directive
# ---------------------------------------------------------------------------

def test_hook_result_skip_directive():
    hooks = AgentHooks()

    def skip_handler(ctx: HookContext) -> HookResult:
        ctx.values["tool_output"] = "[blocked by policy]"
        return HookResult(context=ctx, directive=HookDirective.SKIP)

    hooks.register(HookPoint.TOOL_BEFORE, skip_handler)
    hr = hooks.run(HookPoint.TOOL_BEFORE, HookContext(values={"tool_name": "bash", "tool_args": {"cmd": "rm -rf /"}}))
    assert hr.directive == HookDirective.SKIP
    assert hr.context.values["tool_output"] == "[blocked by policy]"


# ---------------------------------------------------------------------------
# 4. Handler returns HookResult with BREAK directive
# ---------------------------------------------------------------------------

def test_hook_result_break_directive():
    hooks = AgentHooks()

    def budget_guard(ctx: HookContext) -> HookResult:
        ctx.values["reason"] = "token_budget_exceeded"
        return HookResult(context=ctx, directive=HookDirective.BREAK)

    hooks.register(HookPoint.ROUND_START, budget_guard)
    hr = hooks.run(HookPoint.ROUND_START, HookContext(values={"round_index": 5}))
    assert hr.directive == HookDirective.BREAK
    assert hr.context.values["reason"] == "token_budget_exceeded"


# ---------------------------------------------------------------------------
# 5. Handler returns HookResult with SHORT_CIRCUIT directive
# ---------------------------------------------------------------------------

def test_hook_result_short_circuit():
    hooks = AgentHooks()

    class FakeResponse:
        raw_text = "cached answer"
        def get_tool_calls(self):
            return []

    def cache_hit(ctx: HookContext) -> HookResult:
        ctx.values["response"] = FakeResponse()
        return HookResult(context=ctx, directive=HookDirective.SHORT_CIRCUIT)

    hooks.register(HookPoint.LLM_BEFORE, cache_hit)
    hr = hooks.run(HookPoint.LLM_BEFORE, HookContext(values={"messages": []}))
    assert hr.directive == HookDirective.SHORT_CIRCUIT
    assert hr.context.values["response"].raw_text == "cached answer"


# ---------------------------------------------------------------------------
# 6. Multiple handlers — last non-CONTINUE directive wins
# ---------------------------------------------------------------------------

def test_multiple_handlers_last_directive_wins():
    hooks = AgentHooks()

    def first(ctx: HookContext) -> HookResult:
        ctx.values["first"] = True
        return HookResult(context=ctx, directive=HookDirective.SKIP)

    def second(ctx: HookContext) -> HookResult:
        # Override: allow this tool after all
        ctx.values["second"] = True
        return HookResult(context=ctx, directive=HookDirective.CONTINUE)

    hooks.register(HookPoint.TOOL_BEFORE, first, order=0)
    hooks.register(HookPoint.TOOL_BEFORE, second, order=1)
    hr = hooks.run(HookPoint.TOOL_BEFORE, HookContext(values={}))
    # second handler explicitly set CONTINUE, overriding first's SKIP
    assert hr.directive == HookDirective.CONTINUE
    assert hr.context.values["first"] is True
    assert hr.context.values["second"] is True


# ---------------------------------------------------------------------------
# 7. Multiple handlers — CONTINUE does not override previous non-CONTINUE
#    when the handler returns HookContext (not HookResult)
# ---------------------------------------------------------------------------

def test_multiple_handlers_context_return_preserves_directive():
    hooks = AgentHooks()

    def skipper(ctx: HookContext) -> HookResult:
        return HookResult(context=ctx, directive=HookDirective.SKIP)

    def observer(ctx: HookContext) -> HookContext:
        # Returns HookContext (no directive) — should NOT reset the SKIP
        ctx.values["observed"] = True
        return ctx

    hooks.register(HookPoint.TOOL_BEFORE, skipper, order=0)
    hooks.register(HookPoint.TOOL_BEFORE, observer, order=1)
    hr = hooks.run(HookPoint.TOOL_BEFORE, HookContext(values={}))
    assert hr.directive == HookDirective.SKIP
    assert hr.context.values["observed"] is True


# ---------------------------------------------------------------------------
# 8. Async handlers
# ---------------------------------------------------------------------------

import asyncio

def test_async_handler_directive():
    hooks = AgentHooks()

    async def async_skip(ctx: HookContext) -> HookResult:
        ctx.values["async"] = True
        return HookResult(context=ctx, directive=HookDirective.BREAK)

    hooks.register(HookPoint.TOOL_AFTER, async_skip)

    async def _run():
        return await hooks.run_async(HookPoint.TOOL_AFTER, HookContext(values={"tool_name": "bash"}))

    hr = asyncio.run(_run())
    assert hr.directive == HookDirective.BREAK
    assert hr.context.values["async"] is True


# ---------------------------------------------------------------------------
# 9. No handlers registered — returns CONTINUE
# ---------------------------------------------------------------------------

def test_no_handlers_returns_continue():
    hooks = AgentHooks()
    hr = hooks.run(HookPoint.SESSION_START, HookContext(values={"x": 1}))
    assert hr.directive == HookDirective.CONTINUE
    assert hr.context.values["x"] == 1


# ---------------------------------------------------------------------------
# 10. Handler returns None — no change
# ---------------------------------------------------------------------------

def test_handler_returns_none():
    hooks = AgentHooks()

    def noop(ctx: HookContext) -> None:
        pass

    hooks.register(HookPoint.SESSION_END, noop)
    hr = hooks.run(HookPoint.SESSION_END, HookContext(values={"reason": "done"}))
    assert hr.directive == HookDirective.CONTINUE
    assert hr.context.values["reason"] == "done"


# ---------------------------------------------------------------------------
# 11. New hook points exist and are usable
# ---------------------------------------------------------------------------

def test_new_hook_points_exist():
    assert HookPoint.TOOL_ERROR.value == "tool.error"
    assert HookPoint.ROUND_DECIDE.value == "round.decide"
    assert HookPoint.COMPACT_BEFORE.value == "compact.before"
    assert HookPoint.COMPACT_AFTER.value == "compact.after"


def test_round_decide_skip():
    """ROUND_DECIDE with SKIP should be expressible."""
    hooks = AgentHooks()

    def approval_gate(ctx: HookContext) -> HookResult:
        tool_calls = ctx.values.get("tool_calls", [])
        has_dangerous = any("rm" in str(tc) for tc in tool_calls)
        if has_dangerous:
            ctx.values["tool_output"] = "[blocked: dangerous tool call]"
            return HookResult(context=ctx, directive=HookDirective.SKIP)
        return HookResult(context=ctx, directive=HookDirective.CONTINUE)

    hooks.register(HookPoint.ROUND_DECIDE, approval_gate)
    hr = hooks.run(HookPoint.ROUND_DECIDE, HookContext(values={
        "tool_calls": [{"function": {"name": "bash", "arguments": '{"cmd": "rm -rf /"}'}}],
    }))
    assert hr.directive == HookDirective.SKIP
    assert "[blocked" in hr.context.values["tool_output"]


def test_compact_before_skip():
    """COMPACT_BEFORE with SKIP prevents compaction."""
    hooks = AgentHooks()

    def no_compact(ctx: HookContext) -> HookResult:
        return HookResult(context=ctx, directive=HookDirective.SKIP)

    hooks.register(HookPoint.COMPACT_BEFORE, no_compact)
    hr = hooks.run(HookPoint.COMPACT_BEFORE, HookContext(values={"round_index": 3}))
    assert hr.directive == HookDirective.SKIP


def test_tool_error_skip_swallows_error():
    """TOOL_ERROR with SKIP replaces the error with a custom output."""
    hooks = AgentHooks()

    def error_handler(ctx: HookContext) -> HookResult:
        ctx.values["tool_output"] = f"[graceful fallback for {ctx.values['tool_name']}]"
        return HookResult(context=ctx, directive=HookDirective.SKIP)

    hooks.register(HookPoint.TOOL_ERROR, error_handler)
    hr = hooks.run(HookPoint.TOOL_ERROR, HookContext(values={
        "tool_name": "web_search",
        "error": TimeoutError("request timed out"),
        "error_message": "request timed out",
    }))
    assert hr.directive == HookDirective.SKIP
    assert "graceful fallback" in hr.context.values["tool_output"]


if __name__ == "__main__":
    test_backward_compat_hook_context_return()
    test_backward_compat_dict_return()
    test_hook_result_skip_directive()
    test_hook_result_break_directive()
    test_hook_result_short_circuit()
    test_multiple_handlers_last_directive_wins()
    test_multiple_handlers_context_return_preserves_directive()
    test_async_handler_directive()
    test_no_handlers_returns_continue()
    test_handler_returns_none()
    test_new_hook_points_exist()
    test_round_decide_skip()
    test_compact_before_skip()
    test_tool_error_skip_swallows_error()
    print("All hook directive tests passed!")
