from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from llm_client.interface import LLMRequest, LLMResponse, LLMService

from power_loop.agent.system_prompt import DEFAULT_AGENT_SYSTEM_PROMPT
from power_loop.agent.types import AgentLoopConfig, AgentLoopResult, LoopMessage
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.hooks import HookContext, HookDirective, HookPoint, HookResult
from power_loop.core.agent_context import get_ctx, get_event_bus, get_hooks, get_session_id
from power_loop.tools.registry import ToolRegistry


RESULT_MAX_CHARS = 50000

def _status_payload_auto_compact(*, round_index: int, ctx: ContextManager) -> dict[str, Any]:
    return {
        "kind": "auto_compact",
        "phase": "started",
        "round_index": round_index,
        "trigger": "input_tokens_gt_threshold",
        "input_tokens": ctx.last_input_tokens,
        "compact_threshold": ctx.compact_threshold,
    }
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
def _status_payload_hit_round_limit(*, max_rounds: int) -> dict[str, Any]:
    return {"kind": "hit_round_limit", "max_rounds": max_rounds}


def _is_cancelled(stop_event: threading.Event | None) -> bool:
    return bool(stop_event is not None and stop_event.is_set())


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
        # Best-effort repair for common malformed JSON from providers.
        try:
            repaired = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            loaded = json.loads(repaired)
            return dict(loaded) if isinstance(loaded, Mapping) else {}
        except Exception:
            return {}


def _sanitize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure function.arguments is a JSON-string for history stability."""
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
                # Ensure it's parseable JSON; otherwise normalize to "{}".
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


async def agent_loop_async(
    *,
    llm: LLMService,
    config: AgentLoopConfig,
    tool_registry: ToolRegistry | None,
    messages: list[LoopMessage],
    stop_event: threading.Event | None = None,
    session_id: str | None = None,
) -> AgentLoopResult:
    bus = get_event_bus()
    hooks = get_hooks()
    ctx = get_ctx()

    runtime_tools = tool_registry.to_openai_tools() if tool_registry is not None else None
    system_prompt = (config.system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT).strip()

    history: list[LoopMessage] = [dict(m) for m in messages]

    rounds_since_todo = 0
    session_scope = "main"

    async def _append_message(msg: LoopMessage, *, round_index: int | None = None) -> None:
        """Append *msg* to history, firing MESSAGE_APPEND hook first."""
        hr = await hooks.run_async(
            HookPoint.MESSAGE_APPEND,
            context=HookContext(values={
                "message": msg,
                "round_index": round_index,
                "session_id": session_id,
            }),
        )
        # Hook may replace the message (e.g. redact, enrich).
        final_msg = hr.context.values.get("message", msg)
        history.append(final_msg)

    # Session start hook
    session_start_vals = {
        "scope": session_scope,
        "messages": history,
        "stop_event": stop_event,
    }
    session_start_hr = await hooks.run_async(HookPoint.SESSION_START, context=HookContext(values=session_start_vals))
    if isinstance(session_start_hr.context.values.get("messages"), list):
        history = session_start_hr.context.values["messages"]  # type: ignore[assignment]

    bus.publish(
        AgentEvent(
            type=AgentEventType.SESSION_STARTED,
            payload={"scope": session_scope},
            session_id=session_id,
        )
    )

    async def _finalize_session(reason: str, *, final_text: str | None = None) -> None:
        await hooks.run_async(
            HookPoint.SESSION_END,
            context=HookContext(values={"scope": session_scope, "reason": reason, "messages": history, "final_text": final_text}),
        )
        bus.publish(
            AgentEvent(
                type=AgentEventType.SESSION_ENDED,
                payload={"reason": reason},
                session_id=session_id,
            )
        )

    async def _run_llm_before(round_index: int) -> HookResult:
        llm_before_ctx: dict[str, Any] = {
            "round_index": round_index,
            "messages": history,
            "system_prompt": system_prompt,
            "tools": runtime_tools,
            "tool_choice": "auto",
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        return await hooks.run_async(HookPoint.LLM_BEFORE, context=HookContext(values=llm_before_ctx))

    async def _run_llm_after(round_index: int, response: LLMResponse) -> HookResult:
        llm_after_ctx: dict[str, Any] = {
            "round_index": round_index,
            "messages": history,
            "response": response,
        }
        return await hooks.run_async(HookPoint.LLM_AFTER, context=HookContext(values=llm_after_ctx))

    for round_idx in range(int(config.max_rounds)):
        if _is_cancelled(stop_event):
            await _finalize_session("cancelled", final_text=None)
            return AgentLoopResult(status="cancelled", final_text="[cancelled by user]", rounds=round_idx, messages=history)

        round_start_ctx = {
            "round_index": round_idx,
            "messages": history,
            "stop_event": stop_event,
        }
        round_start_hr = await hooks.run_async(HookPoint.ROUND_START, context=HookContext(values=round_start_ctx))
        if isinstance(round_start_hr.context.values.get("messages"), list):
            history = round_start_hr.context.values["messages"]  # type: ignore[assignment]

        # ROUND_START directive: BREAK -> end loop, SKIP -> skip this round
        if round_start_hr.directive == HookDirective.BREAK:
            reason = round_start_hr.context.values.get("reason", "hook_break")
            await _finalize_session(reason, final_text=None)
            return AgentLoopResult(status="completed", final_text="", rounds=round_idx, messages=history)
        if round_start_hr.directive == HookDirective.SKIP:
            continue

        bus.publish(
            AgentEvent(
                type=AgentEventType.ROUND_STARTED,
                payload={"round_index": round_idx},
                session_id=session_id,
                round_index=round_idx,
            )
        )

        if rounds_since_todo >= 5 and ctx.todo.has_in_progress:
            await _append_message(
                {"role": "user", "content": "<reminder>You have an in_progress todo. Update your todos.</reminder>"},
                round_index=round_idx,
            )
            bus.publish(
                AgentEvent(
                    type=AgentEventType.USER_NOTIFICATION,
                    payload={"message": "update your todos"},
                    session_id=session_id,
                    round_index=round_idx,
                )
            )
            rounds_since_todo = 0

        ctx.microcompact(history)

        if ctx.should_compact():
            compact_before_hr = await hooks.run_async(
                HookPoint.COMPACT_BEFORE,
                context=HookContext(values={
                    "round_index": round_idx,
                    "messages": history,
                    "input_tokens": ctx.last_input_tokens,
                    "compact_threshold": ctx.compact_threshold,
                }),
            )
            if compact_before_hr.directive != HookDirective.SKIP:
                bus.publish(
                    AgentEvent(
                        type=AgentEventType.STATUS_CHANGED,
                        payload=_status_payload_auto_compact(round_index=round_idx, ctx=ctx),
                        session_id=session_id,
                        round_index=round_idx,
                    )
                )
                history_before_len = len(history)
                history = await ctx.compact_async(llm, history)
                ctx.reset_usage()
                await hooks.run_async(
                    HookPoint.COMPACT_AFTER,
                    context=HookContext(values={
                        "round_index": round_idx,
                        "messages": history,
                        "messages_before_count": history_before_len,
                        "messages_after_count": len(history),
                    }),
                )

        todo_snap = ctx.todo.snapshot_for_prompt()
        if todo_snap:
            await _append_message({"role": "user", "content": todo_snap}, round_index=round_idx)

        # Stream lifecycle marker (best-effort; complete() might not stream).
        bus.publish(
            AgentEvent(
                type=AgentEventType.STREAM_STARTED,
                payload={"stream_id": "main"},
                session_id=session_id,
                round_index=round_idx,
                stream_id="main",
            )
        )

        def _on_delta(text: str) -> None:
            if not text:
                return
            bus.publish(
                AgentEvent(
                    type=AgentEventType.STREAM_DELTA,
                    payload={"stream_id": "main", "text": text, "is_think": False},
                    session_id=session_id,
                    round_index=round_idx,
                    stream_id="main",
                )
            )

        def _on_think(text: str) -> None:
            if not text:
                return
            bus.publish(
                AgentEvent(
                    type=AgentEventType.STREAM_THINK_DELTA,
                    payload={"stream_id": "main", "text": text, "is_think": True},
                    session_id=session_id,
                    round_index=round_idx,
                    stream_id="main",
                )
            )

        llm_before_hr = await _run_llm_before(round_idx)
        llm_vals = llm_before_hr.context.values

        # LLM_BEFORE directive: SHORT_CIRCUIT -> use values["response"] directly
        if llm_before_hr.directive == HookDirective.SHORT_CIRCUIT:
            response = llm_vals.get("response")
            if not isinstance(response, LLMResponse):
                raise ValueError("LLM_BEFORE hook returned SHORT_CIRCUIT but no valid 'response' in context values")
        else:
            response = await llm.complete(
                LLMRequest(
                    messages=llm_vals.get("messages", history),
                    system_prompt=llm_vals.get("system_prompt", system_prompt),
                    tools=llm_vals.get("tools", runtime_tools),
                    tool_choice=llm_vals.get("tool_choice", "auto"),
                    max_tokens=int(llm_vals.get("max_tokens", config.max_tokens) or config.max_tokens or 8000),
                    temperature=float(llm_vals.get("temperature", config.temperature) or 0),
                ),
                on_chunk_delta_text=_on_delta,
                on_chunk_think=_on_think,
            )

        llm_after_hr = await _run_llm_after(round_idx, response)
        # LLM_AFTER may replace the response
        new_resp = llm_after_hr.context.values.get("response")
        if isinstance(new_resp, LLMResponse):
            response = new_resp

        # LLM_AFTER directive: BREAK -> end loop immediately with current text
        if llm_after_hr.directive == HookDirective.BREAK:
            final_text = (getattr(response, "raw_text", "") or getattr(response, "content_text", "") or "").strip()
            assistant_msg_break: dict[str, Any] = {"role": "assistant", "content": final_text}
            await _append_message(assistant_msg_break, round_index=round_idx)
            await _finalize_session("hook_break", final_text=final_text)
            return AgentLoopResult(status="completed", final_text=final_text, rounds=round_idx + 1, messages=history)

        # Stream completion marker
        bus.publish(
            AgentEvent(
                type=AgentEventType.STREAM_COMPLETED,
                payload={"stream_id": "main"},
                session_id=session_id,
                round_index=round_idx,
                stream_id="main",
            )
        )

        usage = ctx.update_usage(response)
        bus.publish(
            AgentEvent(
                type=AgentEventType.STATUS_CHANGED,
                payload=_status_payload_round_usage(
                    round_index=round_idx,
                    max_rounds=int(config.max_rounds),
                    usage=usage,
                ),
                session_id=session_id,
                round_index=round_idx,
            )
        )
        bus.publish(
            AgentEvent(
                type=AgentEventType.USAGE_UPDATED,
                payload={
                    "usage": usage,
                },
                session_id=session_id,
                round_index=round_idx,
            )
        )

        assistant_text = (getattr(response, "raw_text", "") or getattr(response, "content_text", "") or "").strip()
        tool_calls = response.get_tool_calls()
        bus.publish(
            AgentEvent(
                type=AgentEventType.ROUND_TOOLS_PRESENT,
                payload={"has_tools": bool(tool_calls)},
                session_id=session_id,
                round_index=round_idx,
            )
        )

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_text}
        if tool_calls:
            assistant_msg["tool_calls"] = _sanitize_tool_calls(tool_calls)
        await _append_message(assistant_msg, round_index=round_idx)

        if todo_snap:
            # Remove the todo snapshot that was injected before the LLM call.
            # It sits right before the assistant message we just appended.
            idx = len(history) - 2
            if idx >= 0 and history[idx].get("content") == todo_snap:
                history.pop(idx)

        if not tool_calls:
            # No tools -> round end & session end.
            bus.publish(
                AgentEvent(
                    type=AgentEventType.ROUND_COMPLETED,
                    payload={"round_index": round_idx, "has_tools": False},
                    session_id=session_id,
                    round_index=round_idx,
                )
            )
            await hooks.run_async(
                HookPoint.ROUND_END,
                context=HookContext(values={"round_index": round_idx, "messages": history, "response_text": assistant_text, "has_tools": False}),
            )
            await _finalize_session("completed", final_text=assistant_text)
            return AgentLoopResult(
                status="completed",
                final_text=assistant_text,
                rounds=round_idx + 1,
                messages=history,
            )

        # ROUND_DECIDE hook: fires when tool_calls are present, before execution.
        # Supports SKIP (skip tool execution, proceed to next round) and BREAK (end loop).
        round_decide_hr = await hooks.run_async(
            HookPoint.ROUND_DECIDE,
            context=HookContext(values={
                "round_index": round_idx,
                "messages": history,
                "tool_calls": tool_calls,
                "assistant_text": assistant_text,
            }),
        )
        if round_decide_hr.directive == HookDirective.BREAK:
            await _finalize_session("hook_break", final_text=assistant_text)
            return AgentLoopResult(status="completed", final_text=assistant_text, rounds=round_idx + 1, messages=history)
        if round_decide_hr.directive == HookDirective.SKIP:
            # Skip tool execution — still need to append tool results so the
            # conversation stays valid for the next LLM call.
            for tc in tool_calls:
                cid = str(tc.get("id") or "")
                tname = _tool_call_name(tc)
                skip_output = str(round_decide_hr.context.values.get("tool_output", "[skipped by round_decide hook]"))
                await _append_message(
                    {"role": "tool", "tool_call_id": cid, "name": tname, "content": skip_output},
                    round_index=round_idx,
                )
            continue

        if tool_registry is None:
            # Tool calls are present but no registry installed.
            return AgentLoopResult(
                status="pending_tools",
                final_text=assistant_text,
                rounds=round_idx + 1,
                pending_tool_calls=tool_calls,
                messages=history,
            )

        # Tools batch before hook
        batch_before_hr = await hooks.run_async(
            HookPoint.TOOLS_BATCH_BEFORE,
            context=HookContext(values={"round_index": round_idx, "messages": history, "tool_calls": tool_calls}),
        )

        # TOOLS_BATCH_BEFORE directive: SKIP -> skip all tool execution this round
        skip_tools_batch = batch_before_hr.directive == HookDirective.SKIP

        used_todo = False
        break_after_tool = False
        for tool_call in tool_calls:
            if _is_cancelled(stop_event):
                await _finalize_session("cancelled", final_text=None)
                return AgentLoopResult(status="cancelled", final_text="[cancelled by user]", rounds=round_idx + 1, messages=history)

            call_id = str(tool_call.get("id") or "")
            tool_name = _tool_call_name(tool_call)
            tool_args = _tool_call_args(tool_call)

            # tool.before hook
            tool_before_ctx = {
                "round_index": round_idx,
                "tool_call": tool_call,
                "tool_name": tool_name,
                "tool_args": tool_args,
            }
            tool_before_hr = await hooks.run_async(HookPoint.TOOL_BEFORE, context=HookContext(values=tool_before_ctx))
            tool_name = str(tool_before_hr.context.values.get("tool_name", tool_name))
            if isinstance(tool_before_hr.context.values.get("tool_args"), Mapping):
                tool_args = dict(tool_before_hr.context.values["tool_args"])

            # TOOL_BEFORE directive: SKIP -> use values["tool_output"] or "[skipped]"
            if tool_before_hr.directive == HookDirective.SKIP:
                output = str(tool_before_hr.context.values.get("tool_output", "[skipped by hook]"))
                await _append_message(
                    {"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": output},
                    round_index=round_idx,
                )
                continue

            bus.publish(
                AgentEvent(
                    type=AgentEventType.TOOL_CALL_STARTED,
                    payload={"name": tool_name, "tool_input": tool_args, "tool_call_id": call_id},
                    session_id=session_id,
                    round_index=round_idx,
                )
            )

            # Skip actual execution when batch was skipped
            if skip_tools_batch:
                output = str(batch_before_hr.context.values.get("tool_output", "[skipped by batch hook]"))
                await _append_message(
                    {"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": output},
                    round_index=round_idx,
                )
                continue

            # Validate/execute tool
            validation_err = tool_registry.validate(tool_name, tool_args)
            output: Any
            failed = False
            if validation_err is not None:
                output = validation_err
                failed = True
            else:
                try:
                    output = await tool_registry.invoke_async(tool_name, tool_args)
                except Exception as exc:
                    # TOOL_ERROR hook: lets user handle errors (replace output, retry, etc.)
                    tool_error_hr = await hooks.run_async(
                        HookPoint.TOOL_ERROR,
                        context=HookContext(values={
                            "round_index": round_idx,
                            "tool_call": tool_call,
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "error": exc,
                            "error_message": str(exc),
                        }),
                    )
                    if tool_error_hr.directive == HookDirective.SKIP:
                        # Use hook-provided output instead of error
                        output = str(tool_error_hr.context.values.get("tool_output", f"Error: {exc}"))
                        failed = False
                    elif tool_error_hr.directive == HookDirective.SHORT_CIRCUIT:
                        # Retry once
                        try:
                            output = await tool_registry.invoke_async(tool_name, tool_args)
                        except Exception as retry_exc:
                            output = f"Error (retry failed): {retry_exc}"
                            failed = True
                    else:
                        output = f"Error: {exc}"
                        failed = True

            if tool_name == "todo":
                used_todo = True
                rounds_since_todo = 0

            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            output = str(output)

            # tool.after hook
            tool_after_hr = await hooks.run_async(
                HookPoint.TOOL_AFTER,
                context=HookContext(
                    values={
                        "round_index": round_idx,
                        "tool_call": tool_call,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_output": output,
                        "failed": failed,
                    }
                ),
            )
            # TOOL_AFTER may replace tool_output
            output = str(tool_after_hr.context.values.get("tool_output", output))

            if failed:
                bus.publish(
                    AgentEvent(
                        type=AgentEventType.TOOL_CALL_FAILED,
                        payload={"name": tool_name, "output": output, "tool_input": tool_args, "tool_call_id": call_id},
                        session_id=session_id,
                        round_index=round_idx,
                    )
                )

            bus.publish(
                AgentEvent(
                    type=AgentEventType.TOOL_CALL_COMPLETED,
                    payload={"name": tool_name, "output": output, "tool_input": tool_args, "tool_call_id": call_id},
                    session_id=session_id,
                    round_index=round_idx,
                )
            )

            await _append_message(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": _truncate_result(output),
                },
                round_index=round_idx,
            )

            # TOOL_AFTER directive: BREAK -> stop executing remaining tools
            if tool_after_hr.directive == HookDirective.BREAK:
                break_after_tool = True
                break

        # Tools batch after hook
        await hooks.run_async(
            HookPoint.TOOLS_BATCH_AFTER,
            context=HookContext(values={"round_index": round_idx, "messages": history, "used_todo": used_todo}),
        )

        bus.publish(
            AgentEvent(
                type=AgentEventType.ROUND_COMPLETED,
                payload={"round_index": round_idx, "has_tools": True, "used_todo": used_todo},
                session_id=session_id,
                round_index=round_idx,
            )
        )
        await hooks.run_async(
            HookPoint.ROUND_END,
            context=HookContext(values={"round_index": round_idx, "messages": history, "has_tools": True, "used_todo": used_todo}),
        )

        if not used_todo:
            rounds_since_todo += 1

    # Hit max rounds -> force summary.
    await _append_message(
        {
            "role": "user",
            "content": f"You have reached the maximum of {config.max_rounds} rounds. Summarize what you accomplished and what remains.",
        },
    )
    bus.publish(
        AgentEvent(
            type=AgentEventType.STATUS_CHANGED,
            payload=_status_payload_hit_round_limit(max_rounds=int(config.max_rounds)),
            session_id=session_id,
        )
    )
    final_resp = await llm.complete(
        LLMRequest(
            messages=history,
            system_prompt=system_prompt,
            tools=runtime_tools,
            tool_choice="auto",
            max_tokens=int(config.max_tokens or 8000),
            temperature=float(config.temperature or 0),
        )
    )
    final_text = (getattr(final_resp, "raw_text", "") or getattr(final_resp, "content_text", "") or "").strip()
    bus.publish(
        AgentEvent(
            type=AgentEventType.USAGE_UPDATED,
            payload={
                "usage": ctx.update_usage(final_resp),
            },
            session_id=session_id,
        )
    )
    await _finalize_session("hit_round_limit", final_text=f"[hit_round_limit]\n{final_text}")
    return AgentLoopResult(
        status="hit_round_limit",
        final_text=f"[hit_round_limit]\n{final_text}",
        rounds=int(config.max_rounds),
        messages=history,
    )

