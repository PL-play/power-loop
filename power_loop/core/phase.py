"""Declarative ``@phase`` decorator for agent pipeline methods.

The decorator automatically wraps a pipeline method with:
- **before** hook (``hooks.run_async``) — may modify context values or return a directive
- **after** hook (``hooks.run_async``) — may modify output or return a directive
- **error** hook (optional) — fires on exception, may suppress/retry
- **event publishing** — emits start/complete events on the bus

This keeps each pipeline method focused on pure business logic while
cross-cutting concerns (hooks, events, directives) are handled declaratively.

Usage::

    class MyPipeline(AgentPipeline):
        @phase(
            before=HookPoint.LLM_BEFORE,
            after=HookPoint.LLM_AFTER,
            start_event=AgentEventType.STREAM_STARTED,
            end_event=AgentEventType.STREAM_COMPLETED,
        )
        async def call_llm(self, ctx: PhaseContext) -> Any:
            return await self.llm.complete(...)
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.hooks import HookContext, HookDirective, HookPoint, HookResult


@dataclass
class PhaseContext:
    """Mutable bag of values flowing through a phase.

    Pipeline methods receive this; the ``@phase`` decorator populates it
    from the before-hook result and passes it to the method.
    """
    values: Dict[str, Any] = field(default_factory=dict)
    round_index: int = 0
    session_id: str | None = None


@dataclass
class PhaseResult:
    """Return value of a ``@phase``-decorated method.

    The decorator merges hook directives and the method's raw output into this.
    The caller only needs to check ``directive`` and read ``output`` / ``values``.
    """
    output: Any = None
    directive: HookDirective = HookDirective.CONTINUE
    values: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_break(self) -> bool:
        return self.directive == HookDirective.BREAK

    @property
    def should_skip(self) -> bool:
        return self.directive == HookDirective.SKIP

    @property
    def is_short_circuit(self) -> bool:
        return self.directive == HookDirective.SHORT_CIRCUIT


def phase(
    *,
    before: HookPoint | None = None,
    after: HookPoint | None = None,
    error: HookPoint | None = None,
    start_event: AgentEventType | None = None,
    end_event: AgentEventType | None = None,
) -> Callable:
    """Decorator that wraps a pipeline method with hooks, events, and directive handling.

    The decorated method signature must be::

        async def method(self, ctx: PhaseContext) -> Any

    The ``self`` must be an ``AgentPipeline`` instance (has ``.hooks``, ``.bus``,
    ``.session_id`` attributes).

    Execution flow:

    1. Run **before** hook → get ``HookResult``
       - If directive is ``SKIP``: return ``PhaseResult(directive=SKIP, values=...)`` immediately
       - If directive is ``SHORT_CIRCUIT``: return ``PhaseResult(output=values["output"], ...)``
       - Otherwise: merge modified values back into ``ctx``
    2. Publish **start_event** (if set)
    3. Call the actual method
    4. Publish **end_event** (if set)
    5. Run **after** hook → get ``HookResult``
       - If directive is ``BREAK``: set ``PhaseResult.directive = BREAK``
       - May replace output via ``values["output"]``
    6. Return ``PhaseResult``

    On exception:
    - If **error** hook is set, run it.
      - ``SKIP`` → swallow error, use ``values["output"]`` as fallback
      - ``SHORT_CIRCUIT`` → retry the method once
      - Otherwise → re-package as ``PhaseResult`` with error info
    """

    def decorator(method: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[PhaseResult]]:

        @functools.wraps(method)
        async def wrapper(self: Any, ctx: PhaseContext) -> PhaseResult:
            hooks = self.hooks
            bus = self.bus

            event_meta = {
                "session_id": ctx.session_id,
                "round_index": ctx.round_index,
            }

            # ── before hook ──
            if before is not None:
                hr = await hooks.run_async(before, context=HookContext(values=dict(ctx.values)))
                # Merge hook-modified values back
                ctx.values.update(hr.context.values)

                if hr.directive == HookDirective.SKIP:
                    return PhaseResult(
                        output=hr.context.values.get("output"),
                        directive=HookDirective.SKIP,
                        values=hr.context.values,
                    )
                if hr.directive == HookDirective.SHORT_CIRCUIT:
                    return PhaseResult(
                        output=hr.context.values.get("output"),
                        directive=HookDirective.SHORT_CIRCUIT,
                        values=hr.context.values,
                    )
                if hr.directive == HookDirective.BREAK:
                    return PhaseResult(
                        output=hr.context.values.get("output"),
                        directive=HookDirective.BREAK,
                        values=hr.context.values,
                    )

            # ── start event ──
            if start_event is not None:
                bus.publish(AgentEvent(
                    type=start_event,
                    payload=ctx.values.get("event_payload", {}),
                    **event_meta,
                ))

            # ── execute method ──
            raw_output: Any = None
            try:
                raw_output = await method(self, ctx)
            except Exception as exc:
                if error is not None:
                    err_hr = await hooks.run_async(
                        error,
                        context=HookContext(values={
                            **ctx.values,
                            "error": exc,
                            "error_message": str(exc),
                        }),
                    )
                    if err_hr.directive == HookDirective.SKIP:
                        raw_output = err_hr.context.values.get("output", f"Error: {exc}")
                    elif err_hr.directive == HookDirective.SHORT_CIRCUIT:
                        # Retry once
                        try:
                            raw_output = await method(self, ctx)
                        except Exception as retry_exc:
                            raw_output = f"Error (retry failed): {retry_exc}"
                            return PhaseResult(output=raw_output, values={**ctx.values, "failed": True})
                    else:
                        return PhaseResult(
                            output=f"Error: {exc}",
                            values={**ctx.values, "error": exc, "failed": True},
                        )
                else:
                    raise

            # ── end event ──
            if end_event is not None:
                bus.publish(AgentEvent(
                    type=end_event,
                    payload=ctx.values.get("event_payload", {}),
                    **event_meta,
                ))

            # ── after hook ──
            result_directive = HookDirective.CONTINUE
            result_values = dict(ctx.values)
            if after is not None:
                after_ctx_vals = {**ctx.values, "output": raw_output}
                ahr = await hooks.run_async(after, context=HookContext(values=after_ctx_vals))
                result_values = ahr.context.values
                result_directive = ahr.directive
                # after hook may replace output
                if "output" in ahr.context.values:
                    raw_output = ahr.context.values["output"]

            return PhaseResult(
                output=raw_output,
                directive=result_directive,
                values=result_values,
            )

        # Preserve the original hook metadata for introspection
        wrapper._phase_before = before  # type: ignore[attr-defined]
        wrapper._phase_after = after  # type: ignore[attr-defined]
        wrapper._phase_error = error  # type: ignore[attr-defined]
        return wrapper

    return decorator
