"""AgentPipeline — the core agent loop refactored into discrete, hookable phases.

Phase methods (``prepare_round``, ``call_llm``, ``execute_tool``) are pure
business logic with explicit parameters and return types.  All hook
orchestration, directive checks, and event publishing live in ``run()``.

The old ``agent_loop_async`` function is preserved in ``agent.py`` as a thin
wrapper that delegates to ``AgentPipeline.run()``.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from llm_client.interface import LLMRequest, LLMResponse, LLMService
from power_loop.agent.sink import MessageSink, NullSink
from power_loop.agent.system_prompt import (
    DEFAULT_AGENT_SYSTEM_PROMPT,
    SystemPromptContext,
    format_tool_catalog,
    section_skills,
)
from power_loop.agent.types import AgentLoopConfig, AgentLoopResult, LoopMessage
from power_loop.contracts.errors import (
    CancellationRequested,
    LLMRetryExhausted,
    LLMTimeout,
    ToolNotFound,
    ToolValidationError,
)
from power_loop.contracts.event_payloads import (
    AutoCompactStatusPayload,
    BaseEventPayload,
    HitRoundLimitStatusPayload,
    LlmDegradedPayload,
    LlmRetryAttemptedPayload,
    LoopCancelledPayload,
    MemoryFailedPayload,
    MemoryRecalledPayload,
    RoundCompletedPayload,
    RoundStartedPayload,
    RoundToolsPresentPayload,
    RoundUsageStatusPayload,
    SessionEndedPayload,
    SessionStartedPayload,
    StreamCompletedPayload,
    StreamDeltaPayload,
    StreamStartedPayload,
    ToolCallCompletedPayload,
    ToolCallFailedPayload,
    ToolCallStartedPayload,
    UsageUpdatedPayload,
)
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.hook_contexts import (
    CompactAfterCtx,
    CompactBeforeCtx,
    LlmAfterCtx,
    LlmBeforeCtx,
    MemoryRecalledCtx,
    MessageAppendCtx,
    RoundDecideCtx,
    RoundEndCtx,
    RoundStartCtx,
    SessionEndCtx,
    SessionStartCtx,
    ToolAfterCtx,
    ToolBeforeCtx,
    ToolErrorCtx,
    ToolsBatchAfterCtx,
    ToolsBatchBeforeCtx,
)
from power_loop.contracts.hooks import HookDirective, HookPoint
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.state import ContextManager
from power_loop.runtime.cancellation import CancellationLike, CancellationToken
from power_loop.runtime.memory import MemorySnapshot, tag_as_memory
from power_loop.runtime.retry import with_retry
from power_loop.runtime.skills import SkillLoader
from power_loop.tools.registry import ToolRegistry

RESULT_MAX_CHARS = 50000


# ── Utility functions (unchanged from old agent.py) ──

def _truncate_result(output: Any) -> str:
    s = str(output)
    if len(s) <= RESULT_MAX_CHARS:
        return s
    return s[: RESULT_MAX_CHARS - 50] + f"\n... (truncated, {len(s)} total chars)"


def _tool_call_name(tool_call: Mapping[str, Any]) -> str:
    fn = tool_call.get("function")
    if isinstance(fn, Mapping):
        return str(fn.get("name") or "unknown")
    return str(tool_call.get("name") or "unknown")


def _tool_call_args(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    fn = tool_call.get("function")
    if not isinstance(fn, Mapping):
        return {}
    args = fn.get("arguments")
    if isinstance(args, Mapping):
        return dict(args)
    if not isinstance(args, str):
        return {}
    text = args.strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
        return dict(loaded) if isinstance(loaded, Mapping) else {}
    except Exception:
        try:
            repaired = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            loaded = json.loads(repaired)
            return dict(loaded) if isinstance(loaded, Mapping) else {}
        except Exception:
            return {}


def _sanitize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        tc2: dict[str, Any] = dict(tc)
        fn = tc2.get("function")
        if isinstance(fn, Mapping):
            fn2 = dict(fn)
            args = fn2.get("arguments")
            if isinstance(args, Mapping):
                fn2["arguments"] = json.dumps(dict(args), ensure_ascii=False)
            elif isinstance(args, str):
                try:
                    json.loads(args)
                except Exception:
                    try:
                        repaired = args.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                        json.loads(repaired)
                        fn2["arguments"] = repaired
                    except Exception:
                        fn2["arguments"] = "{}"
            elif args is None:
                fn2["arguments"] = "{}"
            tc2["function"] = fn2
        out.append(tc2)
    return out


def _is_cancelled(token: CancellationToken | None) -> bool:
    return bool(token is not None and token.is_cancelled())


def _round_usage_payload(*, round_index: int, max_rounds: int, usage: dict[str, Any]) -> RoundUsageStatusPayload:
    def _g(*keys: str) -> int | None:
        for k in keys:
            if k in usage and usage[k] is not None:
                return int(usage[k])
        return None
    return RoundUsageStatusPayload(
        time_iso=datetime.now().isoformat(timespec="seconds"),
        round_index=round_index,
        round_number=round_index + 1,
        max_rounds=max_rounds,
        prompt_tokens=_g("prompt_tokens", "input"),
        completion_tokens=_g("completion_tokens", "output"),
        cache_read_tokens=_g("cache_read_tokens", "cache_read"),
        reasoning_tokens=_g("reasoning_tokens", "reasoning"),
    )


# ── AgentPipeline ──

class AgentPipeline:
    """Agent loop as a pipeline of hookable phases.

    Attributes set by the caller (or by ``from_context``):
        llm, config, tool_registry, hooks, bus, ctx, session_id, stop_event

    Mutable session state:
        history, rounds_since_todo, system_prompt, runtime_tools
    """

    def __init__(
        self,
        *,
        llm: LLMService,
        config: AgentLoopConfig,
        tool_registry: ToolRegistry | None = None,
        hooks: AgentHooks,
        bus: AgentEventBus,
        ctx: ContextManager,
        session_id: str | None = None,
        stop_event: CancellationLike = None,
        sink: MessageSink | None = None,
        store: Any | None = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.tool_registry = tool_registry
        self.hooks = hooks
        self.bus = bus
        self.ctx = ctx
        self.session_id = session_id
        # Normalise to CancellationToken once; pipeline only ever sees this shape.
        self.cancel_token: CancellationToken = CancellationToken.from_any(stop_event)
        # Legacy attribute kept for hook ctx fields (RoundStartCtx.stop_event etc.).
        self.stop_event = stop_event if isinstance(stop_event, threading.Event) else None
        self.sink: MessageSink = sink if sink is not None else NullSink()
        self.store = store

        base_prompt = (config.system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT).strip()
        self.runtime_tools = tool_registry.to_openai_tools() if tool_registry is not None else None

        # Auto-inject tool catalog into system prompt (M1.10).
        # The catalog lives on self.system_prompt (a plain string attribute),
        # NOT in self.history — the compactor never touches it.
        if config.inject_tool_descriptions and tool_registry is not None:
            catalog = format_tool_catalog(
                tool_registry, header=config.tool_catalog_header,
            )
            if catalog:
                base_prompt = f"{base_prompt}\n\n{catalog}"

        skill_section = self._build_skill_prompt_section()
        if skill_section:
            base_prompt = f"{base_prompt}\n\n{skill_section}"

        self.system_prompt = base_prompt
        self.history: list[LoopMessage] = []
        self.rounds_since_todo = 0
        self._completed_rounds = 0

    def _build_skill_prompt_section(self) -> str:
        if not self.config.skills_dir:
            return ""
        try:
            loader = SkillLoader(self.config.skills_dir)
        except Exception:
            return ""
        section = section_skills(
            SystemPromptContext(
                skills_dir=str(loader.skills_dir),
                skill_descriptions=loader.get_descriptions(),
            )
        )
        return section or ""

    # ── Helper: emit event ──

    def _emit(self, event_type: AgentEventType, data: BaseEventPayload,
              *, round_index: int | None = None, stream_id: str | None = None) -> None:
        self.bus.publish(AgentEvent(
            type=event_type,
            data=data,
            session_id=self.session_id,
            round_index=round_index,
            stream_id=stream_id,
        ))

    # ── Helper: append message (with MESSAGE_APPEND hook) ──

    async def _append_message(self, msg: LoopMessage, *, round_index: int | None = None) -> None:
        ctx = MessageAppendCtx(
            round_index=round_index or 0,
            message=dict(msg),
            session_id=self.session_id,
        )
        await self.hooks.run_typed_async(HookPoint.MESSAGE_APPEND, ctx)
        self.history.append(ctx.message)
        self.sink.on_message_appended(ctx.message, round_index=round_index)

    # ── Helper: finalize session ──

    async def _finalize(self, reason: str, *, final_text: str | None = None,
                        rounds: int | None = None) -> None:
        if rounds is not None:
            self._completed_rounds = rounds
        ctx = SessionEndCtx(
            scope="main", reason=reason,
            messages=self.history, final_text=final_text,
        )
        await self.hooks.run_typed_async(HookPoint.SESSION_END, ctx)
        self._emit(AgentEventType.SESSION_ENDED, SessionEndedPayload(reason=reason))
        await self._maybe_remember(reason=reason, final_text=final_text or "")

    # ── Memory: recall at start, remember at end (M1.9) ──

    async def _maybe_recall(self) -> None:
        """Call ``config.memory.recall`` and inject results into ``self.history``.

        Injection position: after any leading ``role=system`` messages
        (which includes a ``compact_note`` if one is present). Recalled
        messages are tagged ``role=system, name=memory_*`` so they share
        the system region's compactor protection.

        Soft-fails on any exception by emitting ``MEMORY_FAILED`` and
        continuing with no injection.
        """
        provider = self.config.memory
        if provider is None:
            return
        budget = int(self.config.memory_budget_tokens or 0)
        try:
            recalled = await provider.recall(
                messages=self.history, session_id=self.session_id, budget_tokens=budget,
            )
        except Exception as exc:
            self._emit(
                AgentEventType.MEMORY_FAILED,
                MemoryFailedPayload(
                    phase="recall", error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                ),
            )
            return

        recalled = list(recalled or [])
        returned = len(recalled)
        hook_ctx = MemoryRecalledCtx(
            recalled=recalled, session_id=self.session_id, budget_tokens=budget,
        )
        await self.hooks.run_typed_async(HookPoint.MEMORY_RECALLED, hook_ctx)
        if hook_ctx.directive == HookDirective.SKIP:
            self._emit(
                AgentEventType.MEMORY_RECALLED,
                MemoryRecalledPayload(returned=returned, injected=0, budget_tokens=budget),
            )
            return

        tagged = tag_as_memory(list(hook_ctx.recalled or []))
        if not tagged:
            self._emit(
                AgentEventType.MEMORY_RECALLED,
                MemoryRecalledPayload(returned=returned, injected=0, budget_tokens=budget),
            )
            return

        insert_at = 0
        n = len(self.history)
        while insert_at < n and self.history[insert_at].get("role") == "system":
            insert_at += 1
        self.history[insert_at:insert_at] = tagged

        self._emit(
            AgentEventType.MEMORY_RECALLED,
            MemoryRecalledPayload(returned=returned, injected=len(tagged), budget_tokens=budget),
        )

    async def _maybe_remember(self, *, reason: str, final_text: str) -> None:
        provider = self.config.memory
        if provider is None:
            return
        snapshot = MemorySnapshot(
            session_id=self.session_id or "",
            messages=list(self.history),
            final_text=final_text,
            rounds=self._completed_rounds,
            status=reason,
        )
        try:
            await provider.remember(snapshot=snapshot, session_id=self.session_id)
        except Exception as exc:
            self._emit(
                AgentEventType.MEMORY_FAILED,
                MemoryFailedPayload(
                    phase="remember", error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                ),
            )

    def _make_result(self, status: str, *, final_text: str = "", rounds: int = 0,
                     pending_tool_calls: list | None = None) -> AgentLoopResult:
        self._completed_rounds = rounds  # for MemorySnapshot
        return AgentLoopResult(
            status=status,  # type: ignore[arg-type]
            final_text=final_text,
            rounds=rounds,
            pending_tool_calls=pending_tool_calls or [],
            messages=self.history,
        )

    # ══════════════════════════════════════════════════════════════
    # Phase methods — pure business logic with explicit parameters.
    # Hook orchestration is handled entirely by run().
    # ══════════════════════════════════════════════════════════════

    async def prepare_round(self, round_index: int) -> None:
        """Prepare a new round: todo reminders, then run the pluggable
        compactor if one is configured on the loop config."""
        # Microcompact (legacy: dump large tool outputs to disk and replace
        # with a short pointer — orthogonal to LLM-based compaction).
        self.ctx.microcompact(self.history)

        compactor = self.config.compactor
        if compactor is None:
            return

        compact_before = CompactBeforeCtx(
            round_index=round_index,
            messages=self.history,
        )
        await self.hooks.run_typed_async(HookPoint.COMPACT_BEFORE, compact_before)
        if compact_before.directive == HookDirective.SKIP:
            return

        plan = await compactor.maybe_compact(
            self.history,
            llm=self.llm,
            max_tokens=int(self.config.max_tokens or 0),
            round_index=round_index,
        )
        if plan is None:
            return

        self._emit(
            AgentEventType.STATUS_CHANGED,
            AutoCompactStatusPayload(
                phase="started",
                round_index=round_index,
                trigger="compactor_plan_emitted",
                before_tokens=plan.before_tokens,
                after_tokens=plan.after_tokens,
            ),
            round_index=round_index,
        )

        # Apply plan IN-MEMORY first, then persist via the sink.
        note_msg = {"role": "system", "name": "compact_note", "content": plan.summary_text}
        before_len = len(self.history)
        self.history = (
            self.history[: plan.fold_start_idx]
            + [note_msg]
            + self.history[plan.fold_end_idx + 1 :]
        )
        # Persist (no-op for NullSink).
        self.sink.on_compaction(
            fold_start_idx=plan.fold_start_idx,
            fold_end_idx=plan.fold_end_idx,
            summary_text=plan.summary_text,
            before_tokens=plan.before_tokens,
            after_tokens=plan.after_tokens,
            round_index=round_index,
        )
        compact_after = CompactAfterCtx(
            round_index=round_index,
            messages=self.history,
            messages_before_count=before_len,
            messages_after_count=len(self.history),
        )
        await self.hooks.run_typed_async(HookPoint.COMPACT_AFTER, compact_after)

    def _runtime_messages_for_round(self, round_index: int) -> list[LoopMessage]:
        """Build transient runtime-state messages for the next LLM call.

        These messages are deliberately not appended through ``_append_message``:
        SQLite runtime state is the authority, and the message is just the
        current projection the model needs for this round.
        """
        if self.store is None or self.session_id is None:
            todo_snap = self.ctx.todo.snapshot_for_prompt()
            return [{"role": "user", "content": todo_snap}] if todo_snap else []

        messages: list[LoopMessage] = []
        for projector in self.config.runtime_projectors:
            projected = projector.project(
                store=self.store,
                session_id=self.session_id,
                round_index=round_index,
                context=self.ctx,
            )
            messages.extend(dict(msg) for msg in projected)
        return messages

    async def call_llm(
        self,
        round_index: int,
        *,
        messages: list[LoopMessage],
        system_prompt: str,
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Call the LLM and return its response.

        If ``config.retry_policy`` is set, transient failures retry under
        :func:`with_retry`; exhaustion raises :class:`LLMRetryExhausted` /
        :class:`LLMTimeout`, which :meth:`run` catches and degrades from.
        Cancellation during retry sleep raises :class:`CancellationRequested`.
        """
        def _on_delta(text: str) -> None:
            if text:
                self._emit(AgentEventType.STREAM_DELTA,
                           StreamDeltaPayload(text=text, is_think=False),
                           round_index=round_index, stream_id="main")

        def _on_think(text: str) -> None:
            if text:
                self._emit(AgentEventType.STREAM_THINK_DELTA,
                           StreamDeltaPayload(text=text, is_think=True),
                           round_index=round_index, stream_id="main")

        request = LLMRequest(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice="auto" if tools else None,  # DashScope rejects "auto" with no tools
            max_tokens=max_tokens,
            temperature=temperature,
        )

        async def _do_call() -> LLMResponse:
            # STREAM_STARTED is emitted **per attempt** so subscribers know
            # a fresh stream is beginning; STREAM_COMPLETED only on success
            # of the attempt that returns.
            self._emit(AgentEventType.STREAM_STARTED, StreamStartedPayload(),
                       round_index=round_index, stream_id="main")
            return await self.llm.complete(
                request,
                on_chunk_delta_text=_on_delta,
                on_chunk_think=_on_think,
            )

        policy = self.config.retry_policy
        if policy is None:
            response = await _do_call()
        else:
            def _on_retry(attempt: int, exc: BaseException, sleep_s: float) -> None:
                self._emit(
                    AgentEventType.LLM_RETRY_ATTEMPTED,
                    LlmRetryAttemptedPayload(
                        attempt=attempt,
                        max_attempts=policy.max_attempts,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:500],
                        next_sleep_seconds=sleep_s,
                    ),
                    round_index=round_index,
                    stream_id="main",
                )

            response = await with_retry(
                _do_call, policy=policy, token=self.cancel_token, on_retry=_on_retry,
            )

        self._emit(AgentEventType.STREAM_COMPLETED, StreamCompletedPayload(),
                   round_index=round_index, stream_id="main")

        return response

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> tuple[str, bool]:
        """Execute a single tool and return ``(output_string, failed)``.

        Catches :class:`ToolNotFound` / :class:`ToolValidationError` from
        the registry and returns them as error strings (failed=True), making
        them visible to the LLM so it can self-correct.
        """
        if self.tool_registry is None:
            return (f"Error: tool '{tool_name}' requested but no tool registry configured", True)
        try:
            validation_err = self.tool_registry.validate(tool_name, tool_args)
            if validation_err is not None:
                return (validation_err, True)

            result = await self.tool_registry.invoke_async(tool_name, tool_args)
        except (ToolNotFound, ToolValidationError) as exc:
            return (str(exc), True)
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        return (str(result), False)

    # ══════════════════════════════════════════════════════════════
    # Main orchestrator — loop, hooks, directive checks, events
    # ══════════════════════════════════════════════════════════════

    async def run(self, messages: list[LoopMessage]) -> AgentLoopResult:
        """Run the full agent loop. Returns when done, cancelled, or hit round limit."""
        self.history = [dict(m) for m in messages]

        # ── Session start ──
        session_ctx = SessionStartCtx(
            scope="main", messages=self.history, stop_event=self.stop_event,
        )
        await self.hooks.run_typed_async(HookPoint.SESSION_START, session_ctx)
        if isinstance(session_ctx.messages, list):
            self.history = session_ctx.messages
        self._emit(AgentEventType.SESSION_STARTED, SessionStartedPayload(scope="main"))

        # ── Memory recall (M1.9) ──
        await self._maybe_recall()

        # ── Round loop ──
        for round_idx in range(int(self.config.max_rounds)):
            # Track for MemorySnapshot: how many round attempts we made.
            self._completed_rounds = round_idx
            if _is_cancelled(self.cancel_token):
                await self._finalize("cancelled")
                return self._make_result("cancelled", final_text="[cancelled by user]", rounds=round_idx)

            # ── Hook: ROUND_START ──
            round_ctx = RoundStartCtx(
                round_index=round_idx, messages=self.history,
                stop_event=self.stop_event,
            )
            await self.hooks.run_typed_async(HookPoint.ROUND_START, round_ctx)
            if round_ctx.directive == HookDirective.BREAK:
                await self._finalize(round_ctx.reason or "hook_break")
                return self._make_result("completed", rounds=round_idx)
            if round_ctx.directive == HookDirective.SKIP:
                continue

            # Apply hook-modified messages
            if isinstance(round_ctx.messages, list):
                self.history = round_ctx.messages

            # ── Business logic: prepare round ──
            await self.prepare_round(round_idx)

            self.sink.on_round_started(round_idx)
            self._emit(AgentEventType.ROUND_STARTED, RoundStartedPayload(round_index=round_idx), round_index=round_idx)

            runtime_messages = self._runtime_messages_for_round(round_idx)
            llm_messages = [*self.history, *runtime_messages]

            # ── Hook: LLM_BEFORE ──
            llm_before = LlmBeforeCtx(
                round_index=round_idx,
                messages=llm_messages,
                system_prompt=self.system_prompt,
                tools=self.runtime_tools,
                max_tokens=int(self.config.max_tokens or 8000),
                temperature=float(self.config.temperature or 0),
            )
            await self.hooks.run_typed_async(HookPoint.LLM_BEFORE, llm_before)

            if llm_before.directive == HookDirective.SHORT_CIRCUIT:
                response = llm_before.output
                if not isinstance(response, LLMResponse):
                    raise ValueError("LLM_BEFORE SHORT_CIRCUIT but no valid LLMResponse")
            elif llm_before.directive == HookDirective.BREAK:
                await self._finalize("hook_break")
                return self._make_result("completed", rounds=round_idx)
            else:
                # ── Business logic: call LLM (with retry/timeout/cancel) ──
                try:
                    response = await self.call_llm(
                        round_idx,
                        messages=llm_before.messages,
                        system_prompt=llm_before.system_prompt,
                        tools=llm_before.tools,
                        max_tokens=llm_before.max_tokens,
                        temperature=llm_before.temperature,
                    )
                except CancellationRequested as exc:
                    self._emit(
                        AgentEventType.LOOP_CANCELLED,
                        LoopCancelledPayload(reason=exc.reason, round_index=round_idx),
                        round_index=round_idx,
                    )
                    await self._finalize("cancelled")
                    return self._make_result(
                        "cancelled", final_text=f"[cancelled: {exc.reason}]", rounds=round_idx,
                    )
                except (LLMRetryExhausted, LLMTimeout) as exc:
                    reason = "timeout" if isinstance(exc, LLMTimeout) else "retry_exhausted"
                    inner = getattr(exc, "last_error", exc)
                    self._emit(
                        AgentEventType.LLM_DEGRADED,
                        LlmDegradedPayload(
                            reason=reason,
                            attempts=getattr(exc, "attempts", 0),
                            error_type=type(inner).__name__,
                            error_message=str(inner)[:500],
                        ),
                        round_index=round_idx,
                    )
                    msg = f"[degraded: LLM {reason} — {type(inner).__name__}: {str(inner)[:200]}]"
                    await self._append_message({"role": "assistant", "content": msg}, round_index=round_idx)
                    await self._finalize("degraded", final_text=msg, rounds=round_idx + 1)
                    return self._make_result("degraded", final_text=msg, rounds=round_idx + 1)

                # ── Hook: LLM_AFTER ──
                llm_after = LlmAfterCtx(
                    round_index=round_idx,
                    output=response,
                    messages=self.history,
                )
                await self.hooks.run_typed_async(HookPoint.LLM_AFTER, llm_after)
                if llm_after.directive == HookDirective.BREAK:
                    text = (getattr(response, "raw_text", "") or "").strip()
                    await self._append_message({"role": "assistant", "content": text}, round_index=round_idx)
                    await self._finalize("hook_break", final_text=text, rounds=round_idx + 1)
                    return self._make_result("completed", final_text=text, rounds=round_idx + 1)
                # After hook may replace the response
                if isinstance(llm_after.output, LLMResponse):
                    response = llm_after.output

            # ── Post-LLM processing ──
            usage = self.ctx.update_usage(response)
            self._emit(AgentEventType.STATUS_CHANGED, _round_usage_payload(
                round_index=round_idx, max_rounds=int(self.config.max_rounds), usage=usage,
            ), round_index=round_idx)
            self._emit(AgentEventType.USAGE_UPDATED, UsageUpdatedPayload(usage=usage), round_index=round_idx)

            assistant_text = (getattr(response, "raw_text", "") or getattr(response, "content_text", "") or "").strip()
            tool_calls = response.get_tool_calls()
            self._emit(AgentEventType.ROUND_TOOLS_PRESENT, RoundToolsPresentPayload(has_tools=bool(tool_calls)), round_index=round_idx)

            # Append assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_text}
            sanitized_tool_calls: list[dict[str, Any]] | None = None
            if tool_calls:
                sanitized_tool_calls = _sanitize_tool_calls(tool_calls)
                assistant_msg["tool_calls"] = sanitized_tool_calls
            await self._append_message(assistant_msg, round_index=round_idx)
            # Mark pending IMMEDIATELY so a crash here leaves a recoverable state.
            if sanitized_tool_calls:
                assistant_seq = len(self.history)  # 1-based position in history
                self.sink.on_assistant_tool_calls(
                    assistant_seq=assistant_seq,
                    tool_calls=sanitized_tool_calls,
                    round_index=round_idx,
                )

            # ── No tools → completed ──
            if not tool_calls:
                self._emit(AgentEventType.ROUND_COMPLETED,
                           RoundCompletedPayload(round_index=round_idx, has_tools=False), round_index=round_idx)
                round_end = RoundEndCtx(
                    round_index=round_idx, messages=self.history,
                    has_tools=False, response_text=assistant_text,
                )
                await self.hooks.run_typed_async(HookPoint.ROUND_END, round_end)
                self.sink.on_round_ended(round_idx, usage=usage)
                await self._finalize("completed", final_text=assistant_text, rounds=round_idx + 1)
                return self._make_result("completed", final_text=assistant_text, rounds=round_idx + 1)

            # ── Hook: ROUND_DECIDE ──
            decide_ctx = RoundDecideCtx(
                round_index=round_idx, messages=self.history,
                tool_calls=tool_calls, assistant_text=assistant_text,
            )
            await self.hooks.run_typed_async(HookPoint.ROUND_DECIDE, decide_ctx)
            if decide_ctx.directive == HookDirective.BREAK:
                await self._finalize("hook_break", final_text=assistant_text, rounds=round_idx + 1)
                return self._make_result("completed", final_text=assistant_text, rounds=round_idx + 1)
            if decide_ctx.directive == HookDirective.SKIP:
                for tc in tool_calls:
                    cid = str(tc.get("id") or "")
                    tname = _tool_call_name(tc)
                    await self._append_message(
                        {"role": "tool", "tool_call_id": cid, "name": tname, "content": decide_ctx.output},
                        round_index=round_idx,
                    )
                continue

            if self.tool_registry is None:
                return self._make_result("pending_tools", final_text=assistant_text,
                                         rounds=round_idx + 1, pending_tool_calls=tool_calls)

            # ── Hook: TOOLS_BATCH_BEFORE ──
            batch_ctx = ToolsBatchBeforeCtx(
                round_index=round_idx,
                messages=self.history,
                tool_calls=tool_calls,
            )
            await self.hooks.run_typed_async(HookPoint.TOOLS_BATCH_BEFORE, batch_ctx)
            skip_batch = batch_ctx.directive == HookDirective.SKIP

            # ── Execute tools ──
            used_todo = False
            for tool_call in tool_calls:
                if _is_cancelled(self.cancel_token):
                    await self._finalize("cancelled")
                    return self._make_result("cancelled", final_text="[cancelled by user]", rounds=round_idx + 1)

                call_id = str(tool_call.get("id") or "")
                tool_name = _tool_call_name(tool_call)
                tool_args = _tool_call_args(tool_call)

                # Batch skip
                if skip_batch:
                    await self._append_message(
                        {"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": batch_ctx.output},
                        round_index=round_idx,
                    )
                    continue

                # ── Hook: TOOL_BEFORE ──
                tb_ctx = ToolBeforeCtx(
                    round_index=round_idx,
                    tool_call=tool_call,
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
                await self.hooks.run_typed_async(HookPoint.TOOL_BEFORE, tb_ctx)
                tool_name = tb_ctx.tool_name
                tool_args = tb_ctx.tool_args

                if tb_ctx.directive == HookDirective.SKIP:
                    await self._append_message(
                        {"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": tb_ctx.output},
                        round_index=round_idx,
                    )
                    continue

                self._emit(AgentEventType.TOOL_CALL_STARTED,
                           ToolCallStartedPayload(name=tool_name, tool_input=tool_args, tool_call_id=call_id),
                           round_index=round_idx)

                # ── Business logic: execute tool ──
                failed = False
                try:
                    output, failed = await self.execute_tool(tool_name, tool_args)
                except Exception as exc:
                    # ── Hook: TOOL_ERROR ──
                    err_ctx = ToolErrorCtx(
                        round_index=round_idx,
                        tool_call=tool_call,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        error=exc,
                        error_message=str(exc),
                    )
                    await self.hooks.run_typed_async(HookPoint.TOOL_ERROR, err_ctx)
                    if err_ctx.directive == HookDirective.SKIP:
                        output = err_ctx.output or f"Error: {exc}"
                    elif err_ctx.directive == HookDirective.SHORT_CIRCUIT:
                        try:
                            output, failed = await self.execute_tool(tool_name, tool_args)
                        except Exception as retry_exc:
                            output = f"Error (retry failed): {retry_exc}"
                            failed = True
                    else:
                        output = f"Error: {exc}"
                        failed = True

                # ── Hook: TOOL_AFTER ──
                ta_ctx = ToolAfterCtx(
                    round_index=round_idx,
                    tool_call=tool_call,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output=output,
                    failed=failed,
                )
                await self.hooks.run_typed_async(HookPoint.TOOL_AFTER, ta_ctx)
                output = ta_ctx.output
                failed = ta_ctx.failed

                if tool_name == "todo":
                    used_todo = True
                    self.rounds_since_todo = 0

                if failed:
                    self._emit(AgentEventType.TOOL_CALL_FAILED,
                               ToolCallFailedPayload(name=tool_name, output=output, tool_input=tool_args, tool_call_id=call_id),
                               round_index=round_idx)

                self._emit(AgentEventType.TOOL_CALL_COMPLETED,
                           ToolCallCompletedPayload(name=tool_name, output=output, tool_input=tool_args, tool_call_id=call_id),
                           round_index=round_idx)

                await self._append_message(
                    {"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": _truncate_result(output)},
                    round_index=round_idx,
                )

                # TOOL_AFTER BREAK → stop remaining tools
                if ta_ctx.directive == HookDirective.BREAK:
                    break

            # ── Hook: TOOLS_BATCH_AFTER ──
            batch_after_ctx = ToolsBatchAfterCtx(
                round_index=round_idx,
                messages=self.history,
                used_todo=used_todo,
            )
            await self.hooks.run_typed_async(HookPoint.TOOLS_BATCH_AFTER, batch_after_ctx)

            self._emit(AgentEventType.ROUND_COMPLETED,
                       RoundCompletedPayload(round_index=round_idx, has_tools=True, used_todo=used_todo),
                       round_index=round_idx)
            round_end = RoundEndCtx(
                round_index=round_idx,
                messages=self.history,
                has_tools=True,
                used_todo=used_todo,
            )
            await self.hooks.run_typed_async(HookPoint.ROUND_END, round_end)

            if not used_todo:
                self.rounds_since_todo += 1

        # ── Hit max rounds ──
        await self._append_message({
            "role": "user",
            "content": f"You have reached the maximum of {self.config.max_rounds} rounds. "
                       f"Summarize what you accomplished and what remains.",
        })
        self._emit(AgentEventType.STATUS_CHANGED, HitRoundLimitStatusPayload(max_rounds=int(self.config.max_rounds)))

        final_resp = await self.llm.complete(LLMRequest(
            messages=self.history,
            system_prompt=self.system_prompt,
            tools=self.runtime_tools,
            tool_choice="auto" if self.runtime_tools else None,
            max_tokens=int(self.config.max_tokens or 8000),
            temperature=float(self.config.temperature or 0),
        ))
        final_text = (getattr(final_resp, "raw_text", "") or getattr(final_resp, "content_text", "") or "").strip()
        self._emit(AgentEventType.USAGE_UPDATED, UsageUpdatedPayload(usage=self.ctx.update_usage(final_resp)))
        await self._finalize("hit_round_limit", final_text=f"[hit_round_limit]\n{final_text}",
                             rounds=int(self.config.max_rounds))
        return self._make_result("hit_round_limit", final_text=f"[hit_round_limit]\n{final_text}",
                                 rounds=int(self.config.max_rounds))
