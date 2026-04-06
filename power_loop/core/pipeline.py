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
from datetime import datetime
from typing import Any, Mapping

from llm_client.interface import LLMRequest, LLMResponse, LLMService

from power_loop.agent.system_prompt import DEFAULT_AGENT_SYSTEM_PROMPT
from power_loop.agent.types import AgentLoopConfig, AgentLoopResult, LoopMessage
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.hook_contexts import (
    CompactAfterCtx,
    CompactBeforeCtx,
    LlmAfterCtx,
    LlmBeforeCtx,
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


def _is_cancelled(stop_event: threading.Event | None) -> bool:
    return bool(stop_event is not None and stop_event.is_set())


def _status_payload_round_usage(*, round_index: int, max_rounds: int, usage: dict[str, Any]) -> dict[str, Any]:
    def _g(*keys: str) -> Any:
        for k in keys:
            if k in usage and usage[k] is not None:
                return usage[k]
        return None
    return {
        "kind": "round_usage",
        "time_iso": datetime.now().isoformat(timespec="seconds"),
        "round_index": round_index,
        "round_number": round_index + 1,
        "max_rounds": max_rounds,
        "prompt_tokens": _g("prompt_tokens", "input"),
        "completion_tokens": _g("completion_tokens", "output"),
        "cache_read_tokens": _g("cache_read_tokens", "cache_read"),
        "reasoning_tokens": _g("reasoning_tokens", "reasoning"),
    }


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
        stop_event: threading.Event | None = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.tool_registry = tool_registry
        self.hooks = hooks
        self.bus = bus
        self.ctx = ctx
        self.session_id = session_id
        self.stop_event = stop_event

        self.system_prompt = (config.system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT).strip()
        self.runtime_tools = tool_registry.to_openai_tools() if tool_registry is not None else None
        self.history: list[LoopMessage] = []
        self.rounds_since_todo = 0

    # ── Helper: emit event ──

    def _emit(self, event_type: AgentEventType, payload: dict[str, Any] | None = None,
              *, round_index: int | None = None, stream_id: str | None = None) -> None:
        self.bus.publish(AgentEvent(
            type=event_type,
            payload=payload or {},
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

    # ── Helper: finalize session ──

    async def _finalize(self, reason: str, *, final_text: str | None = None) -> None:
        ctx = SessionEndCtx(
            scope="main", reason=reason,
            messages=self.history, final_text=final_text,
        )
        await self.hooks.run_typed_async(HookPoint.SESSION_END, ctx)
        self._emit(AgentEventType.SESSION_ENDED, {"reason": reason})

    def _make_result(self, status: str, *, final_text: str = "", rounds: int = 0,
                     pending_tool_calls: list | None = None) -> AgentLoopResult:
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
        """Prepare a new round: todo reminders, compaction."""
        # Todo reminder
        if self.rounds_since_todo >= 5 and self.ctx.todo.has_in_progress:
            await self._append_message(
                {"role": "user", "content": "<reminder>You have an in_progress todo. Update your todos.</reminder>"},
                round_index=round_index,
            )
            self._emit(AgentEventType.USER_NOTIFICATION, {"message": "update your todos"}, round_index=round_index)
            self.rounds_since_todo = 0

        # Microcompact
        self.ctx.microcompact(self.history)

        # Auto-compact (with its own hooks, handled inline since it's conditional)
        if self.ctx.should_compact():
            compact_before = CompactBeforeCtx(
                round_index=round_index,
                messages=self.history,
                input_tokens=self.ctx.last_input_tokens,
                compact_threshold=self.ctx.compact_threshold,
            )
            await self.hooks.run_typed_async(HookPoint.COMPACT_BEFORE, compact_before)
            if compact_before.directive != HookDirective.SKIP:
                self._emit(AgentEventType.STATUS_CHANGED, {
                    "kind": "auto_compact", "phase": "started",
                    "round_index": round_index,
                    "trigger": "input_tokens_gt_threshold",
                    "input_tokens": self.ctx.last_input_tokens,
                    "compact_threshold": self.ctx.compact_threshold,
                }, round_index=round_index)
                before_len = len(self.history)
                self.history = await self.ctx.compact_async(self.llm, self.history)
                self.ctx.reset_usage()
                compact_after = CompactAfterCtx(
                    round_index=round_index,
                    messages=self.history,
                    messages_before_count=before_len,
                    messages_after_count=len(self.history),
                )
                await self.hooks.run_typed_async(HookPoint.COMPACT_AFTER, compact_after)

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
        """Call the LLM and return its response."""
        def _on_delta(text: str) -> None:
            if text:
                self._emit(AgentEventType.STREAM_DELTA,
                           {"stream_id": "main", "text": text, "is_think": False},
                           round_index=round_index, stream_id="main")

        def _on_think(text: str) -> None:
            if text:
                self._emit(AgentEventType.STREAM_THINK_DELTA,
                           {"stream_id": "main", "text": text, "is_think": True},
                           round_index=round_index, stream_id="main")

        self._emit(AgentEventType.STREAM_STARTED, {"stream_id": "main"},
                   round_index=round_index, stream_id="main")

        response = await self.llm.complete(
            LLMRequest(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            on_chunk_delta_text=_on_delta,
            on_chunk_think=_on_think,
        )

        self._emit(AgentEventType.STREAM_COMPLETED, {"stream_id": "main"},
                   round_index=round_index, stream_id="main")

        return response

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> tuple[str, bool]:
        """Execute a single tool and return ``(output_string, failed)``.

        Raises on unexpected errors — the caller handles the TOOL_ERROR hook.
        """
        validation_err = self.tool_registry.validate(tool_name, tool_args)
        if validation_err is not None:
            return (validation_err, True)

        result = await self.tool_registry.invoke_async(tool_name, tool_args)
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
        self._emit(AgentEventType.SESSION_STARTED, {"scope": "main"})

        # ── Round loop ──
        for round_idx in range(int(self.config.max_rounds)):
            if _is_cancelled(self.stop_event):
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

            self._emit(AgentEventType.ROUND_STARTED, {"round_index": round_idx}, round_index=round_idx)

            # Todo snapshot injection
            todo_snap = self.ctx.todo.snapshot_for_prompt()
            if todo_snap:
                await self._append_message({"role": "user", "content": todo_snap}, round_index=round_idx)

            # ── Hook: LLM_BEFORE ──
            llm_before = LlmBeforeCtx(
                round_index=round_idx,
                messages=self.history,
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
                # ── Business logic: call LLM ──
                response = await self.call_llm(
                    round_idx,
                    messages=llm_before.messages,
                    system_prompt=llm_before.system_prompt,
                    tools=llm_before.tools,
                    max_tokens=llm_before.max_tokens,
                    temperature=llm_before.temperature,
                )

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
                    await self._finalize("hook_break", final_text=text)
                    return self._make_result("completed", final_text=text, rounds=round_idx + 1)
                # After hook may replace the response
                if isinstance(llm_after.output, LLMResponse):
                    response = llm_after.output

            # ── Post-LLM processing ──
            usage = self.ctx.update_usage(response)
            self._emit(AgentEventType.STATUS_CHANGED, _status_payload_round_usage(
                round_index=round_idx, max_rounds=int(self.config.max_rounds), usage=usage,
            ), round_index=round_idx)
            self._emit(AgentEventType.USAGE_UPDATED, {"usage": usage}, round_index=round_idx)

            assistant_text = (getattr(response, "raw_text", "") or getattr(response, "content_text", "") or "").strip()
            tool_calls = response.get_tool_calls()
            self._emit(AgentEventType.ROUND_TOOLS_PRESENT, {"has_tools": bool(tool_calls)}, round_index=round_idx)

            # Append assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_text}
            if tool_calls:
                assistant_msg["tool_calls"] = _sanitize_tool_calls(tool_calls)
            await self._append_message(assistant_msg, round_index=round_idx)

            # Remove todo snapshot
            if todo_snap:
                idx = len(self.history) - 2
                if idx >= 0 and self.history[idx].get("content") == todo_snap:
                    self.history.pop(idx)

            # ── No tools → completed ──
            if not tool_calls:
                self._emit(AgentEventType.ROUND_COMPLETED,
                           {"round_index": round_idx, "has_tools": False}, round_index=round_idx)
                round_end = RoundEndCtx(
                    round_index=round_idx, messages=self.history,
                    has_tools=False, response_text=assistant_text,
                )
                await self.hooks.run_typed_async(HookPoint.ROUND_END, round_end)
                await self._finalize("completed", final_text=assistant_text)
                return self._make_result("completed", final_text=assistant_text, rounds=round_idx + 1)

            # ── Hook: ROUND_DECIDE ──
            decide_ctx = RoundDecideCtx(
                round_index=round_idx, messages=self.history,
                tool_calls=tool_calls, assistant_text=assistant_text,
            )
            await self.hooks.run_typed_async(HookPoint.ROUND_DECIDE, decide_ctx)
            if decide_ctx.directive == HookDirective.BREAK:
                await self._finalize("hook_break", final_text=assistant_text)
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
                if _is_cancelled(self.stop_event):
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
                           {"name": tool_name, "tool_input": tool_args, "tool_call_id": call_id},
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
                               {"name": tool_name, "output": output, "tool_input": tool_args, "tool_call_id": call_id},
                               round_index=round_idx)

                self._emit(AgentEventType.TOOL_CALL_COMPLETED,
                           {"name": tool_name, "output": output, "tool_input": tool_args, "tool_call_id": call_id},
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
                       {"round_index": round_idx, "has_tools": True, "used_todo": used_todo},
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
        self._emit(AgentEventType.STATUS_CHANGED, {"kind": "hit_round_limit", "max_rounds": int(self.config.max_rounds)})

        final_resp = await self.llm.complete(LLMRequest(
            messages=self.history,
            system_prompt=self.system_prompt,
            tools=self.runtime_tools,
            tool_choice="auto",
            max_tokens=int(self.config.max_tokens or 8000),
            temperature=float(self.config.temperature or 0),
        ))
        final_text = (getattr(final_resp, "raw_text", "") or getattr(final_resp, "content_text", "") or "").strip()
        self._emit(AgentEventType.USAGE_UPDATED, {"usage": self.ctx.update_usage(final_resp)})
        await self._finalize("hit_round_limit", final_text=f"[hit_round_limit]\n{final_text}")
        return self._make_result("hit_round_limit", final_text=f"[hit_round_limit]\n{final_text}",
                                 rounds=int(self.config.max_rounds))
