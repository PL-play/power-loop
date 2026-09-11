"""AgentPipeline — the core agent loop refactored into discrete, hookable phases.

Phase methods (``prepare_round``, ``call_llm``, ``execute_tool``) are pure
business logic with explicit parameters and return types.  All hook
orchestration, directive checks, and event publishing live in ``run()``.

The old ``agent_loop_async`` function is preserved in ``agent.py`` as a thin
wrapper that delegates to ``AgentPipeline.run()``.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService
from power_loop.agent.sink import MessageSink, NullSink
from power_loop.agent.system_prompt import resolve_runtime_system_prompt
from power_loop.agent.types import AgentLoopConfig, AgentLoopResult, LoopMessage
from power_loop.contracts.errors import (
    CancellationRequested,
    LLMRetryExhausted,
    LLMTimeout,
    ToolNotFound,
    ToolValidationError,
)
from power_loop.contracts.event_payloads import (
    AgentErrorPayload,
    AutoCompactStatusPayload,
    BaseEventPayload,
    BudgetExceededStatusPayload,
    ContextCheckpointStatusPayload,
    HitRoundLimitStatusPayload,
    LlmCallCompletedPayload,
    LlmCallStartedPayload,
    LlmDegradedPayload,
    LlmRetryAttemptedPayload,
    LoopCancelledPayload,
    MemoryFailedPayload,
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
    CompleteDecideCtx,
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
from power_loop.runtime.budget import estimate_message_tokens, estimate_tokens
from power_loop.runtime.cancellation import CancellationLike, CancellationToken
from power_loop.runtime.compact import CompactionContext
from power_loop.runtime.human_input import HumanInputRequired
from power_loop.runtime.image_recall import drain_queued_images
from power_loop.runtime.memory import MemorySnapshot
from power_loop.runtime.retry import with_retry
from power_loop.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# How many consecutive empty LLM responses (no text, no tool call) to retry before giving up.
# An empty turn is a provider hiccup, not a completion signal; a couple of retries clears a
# transient blank, while the cap keeps a persistently-broken provider from spinning to max_rounds.
_EMPTY_RESPONSE_MAX_RETRIES = 3

# 截断（provider 因 max_tokens 硬切）与「空响应打嗝」是两回事，处置也必须不同：
# 打嗝重试一次就好了；截断**原样重试必然再次截断**——同一个 prompt、同一个模型、
# 同样写超。真实事故（conv-213）：模型一轮里写一个 25KB 的 CSS 文件，输出打到
# max_tokens=20000 被切在工具调用的 JSON 中间 → 解析不出工具调用、正文也是空的 →
# 被判成打嗝 → 重试 → 再截断。两轮各约 8 分钟、产出为零，用户看到的是 16 分钟沉默。
_TRUNCATED_MAX_RETRIES = 2
_TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens", "model_length"})
_TRUNCATION_NOTICE = (
    "[系统] 你上一轮的输出**超过了单轮长度上限，被从中间截断了**——所以那一轮什么都没生效"
    "（工具调用的 JSON 断在半路，解析不出来）。原样再写一遍只会再被截断一次。\n"
    "把它拆小再来：一次只写一个文件；单个文件很大就先写骨架、再用 edit_file/apply_patch "
    "分几次补内容；不要在一轮里同时写多个大文件。"
)


def _finish_reason(response: Any) -> str:
    """尽力从 provider 的原始响应里取 finish_reason（取不到就返回空串，绝不猜）。

    各家形状不同：OpenAI 兼容在 ``choices[0].finish_reason``，Anthropic 在
    ``stop_reason``（截断是 ``"max_tokens"``），流式聚合的放在最后一个 chunk 里。
    """
    for obj in (getattr(response, "raw_completion", None),
                getattr(response, "raw_message", None)):
        if obj is None:
            continue
        reason = getattr(obj, "stop_reason", None)
        if isinstance(reason, str) and reason:
            return reason.lower()
        choices = getattr(obj, "choices", None)
        if isinstance(choices, (list, tuple)) and choices:
            reason = getattr(choices[0], "finish_reason", None)
            if isinstance(reason, str) and reason:
                return reason.lower()
        if isinstance(obj, dict):
            reason = obj.get("stop_reason") or (
                (obj.get("choices") or [{}])[0].get("finish_reason")
                if isinstance(obj.get("choices"), list) and obj["choices"] else None
            )
            if isinstance(reason, str) and reason:
                return reason.lower()
    return ""

RESULT_MAX_CHARS = 50000


@functools.cache
def _maybe_compact_accepts_context(func: Any) -> bool:
    """Whether a compactor's ``maybe_compact`` accepts the optional ``context``
    kwarg (or ``**kwargs``). Cached per function — introspect once. The back-compat
    gate for H7 Phase 2 so old-signature compactors are never passed ``context``."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "context" in params


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


def _sanitize_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """消毒后的 tool_calls + 「有参数解析不了」的标志。

    标志走**返回值**而不是塞进 call 里：这些 dict 会原样进 assistant 消息、下一轮发回给
    供应商，多一个非标准字段可能直接把请求打挂。
    """
    out: list[dict[str, Any]] = []
    unparseable = False
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
                        # 参数解析不了，最常见的原因是**输出超长被截断**（JSON 断在半路）。
                        # 这里不做 json 修复：补全出来的 `content` 就是那半个文件，
                        # write_file 会当成功写下去、agent 继续往前走，交付一份残缺的稿子——
                        # 那是静默损坏，比报错严重得多。降成 {} 让必填校验去报，
                        # 同时留个标记，管线据此告诉模型「你是被截断了」而不是「你忘了填参数」
                        # （conv-213 实测：一条 "missing required parameter" 背后是
                        #  completion_tokens=20000 打满上限）。
                        fn2["arguments"] = "{}"
                        unparseable = True
            elif args is None:
                fn2["arguments"] = "{}"
            tc2["function"] = fn2
        out.append(tc2)
    return out, unparseable


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
        drain_follow_ups: Any | None = None,
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
        self._drain_follow_ups = drain_follow_ups

        self.runtime_tools = tool_registry.to_openai_tools() if tool_registry is not None else None
        # Auto-inject tool catalog + skill section (M1.10). Built ONCE here and
        # frozen into self.system_prompt (a plain string — NOT self.history, so
        # the compactor never touches it). The assembly is shared with
        # StatefulAgentLoop.resolve_system_prompt (the preview) via this helper
        # so the live prompt and the preview can never drift. The session-level
        # prompt override is already resolved into config.system_prompt upstream.
        self.system_prompt = resolve_runtime_system_prompt(
            config.system_prompt,
            inject_tool_descriptions=config.inject_tool_descriptions,
            tool_catalog_header=config.tool_catalog_header,
            tool_registry=tool_registry,
            skills_dir=config.skills_dir,
        )
        self.history: list[LoopMessage] = []
        self._send_start_len: int = 0          # 6.27.0 本 send 在 history 里的起点（run() 设）
        self._send_baseline_prompt: int | None = None  # 6.27.0 本 send 第一轮的真实 prompt_tokens
        self._image_rounds: dict[int, int] = {}
        # Monotonic per-session SEND index, set by the loop before run() (None when
        # unset, e.g. in tests). Stamped into each appended row's PERSISTED meta only
        # (never the in-memory/LLM message) so the transcript can delimit sends
        # authoritatively instead of heuristically. See _append_message.
        self.send_index: int | None = None
        # SCALE-4: self-invalidating running token estimate of self.history.
        # ``_tok_len`` is the history length ``_tok_total`` was computed for, or -1 when
        # dirty. The append path bumps it incrementally (O(1)/round); every wholesale
        # reassignment invalidates it; _estimate_history_tokens recomputes on any
        # length mismatch — so it is always correct, just cheaper on the common path.
        self._tok_total = 0
        self._tok_len = -1
        self.rounds_since_todo = 0
        self._completed_rounds = 0
        # Terminal-event bookkeeping (H1.5): emit SESSION_ENDED at most once, and
        # only after SESSION_STARTED, so an error-path finalize can't double-fire
        # or strand subscribers.
        self._session_started = False
        self._finalized = False

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

    # ── Helper: run a write-path sink call (offloaded for real persistence) ──

    async def _emit_sink(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        """Invoke an async write-path sink callback. The sink delegates to the async
        store, whose backend offloads blocking I/O itself (SQLite → threadpool; PG/MySQL
        → real async), so the event loop is never stalled. Awaited, so per-session
        ordering is preserved; the loop runs other sessions during the I/O."""
        await fn(*args, **kwargs)

    # ── Helper: audit ephemeral LLM_BEFORE hook injections ──

    @staticmethod
    def _summarize_hook_injection(
        final_messages: list[Any], pre_hook_ids: set[int] | None, mode: str
    ) -> dict[str, Any] | None:
        """Diff the post-LLM_BEFORE message list against the pre-hook identity snapshot to recover
        exactly the messages a hook injected this round, and summarize them for the hook_events
        audit. Returns None when auditing is off or nothing was injected. ``mode``: ``metadata``
        (no text) or ``full`` (include injected ``content``). Identity-diff (not a tail slice) so it
        captures both tail- and front-positioned injection."""
        if mode not in ("metadata", "full") or pre_hook_ids is None:
            return None
        injected_idx = [i for i, m in enumerate(final_messages) if id(m) not in pre_hook_ids]
        if not injected_idx:
            return None
        injected_set = set(injected_idx)
        orig_idx = [i for i in range(len(final_messages)) if i not in injected_set]
        # Rebind fail-safe: an LLM_BEFORE hook may REPLACE all or PART of ctx.messages with fresh
        # copies (legal per LlmBeforeCtx, though the builtin memory hook mutates in place). Those
        # copies are id-novel, so the identity diff would mislabel pre-existing turns as "injected"
        # — in `full` mode that would dump the conversation into the audit. A genuine injection is a
        # small minority of the request; so treat it as a rebind (record a small truthful marker, no
        # content) when NOTHING survived id-stable OR the id-novel messages are an implausible
        # majority. The >3 floor keeps short, legitimately memory-heavy sends from tripping it.
        rebound = (not orig_idx) or (
            len(injected_idx) > len(final_messages) / 2 and len(injected_idx) > 3
        )
        if pre_hook_ids and rebound:
            return {
                "hook_point": "LLM_BEFORE", "hook": "llm_before", "position": "unknown",
                "kind": "inject_unresolved",
                "payload": {
                    "v": 1, "items": [], "item_count": len(injected_idx), "total_chars": 0,
                    "rebound": True,
                },
            }
        items: list[dict[str, Any]] = []
        sources: set[str] = set()
        for i in injected_idx:
            m = final_messages[i]
            name = m.get("name") if isinstance(m, dict) else None
            content = m.get("content") if isinstance(m, dict) else None
            text = content if isinstance(content, str) else ("" if content is None else str(content))
            source = "builtin.memory_recall" if str(name or "").startswith("memory_") else "llm_before"
            sources.add(source)
            item: dict[str, Any] = {
                "role": (m.get("role") if isinstance(m, dict) else None),
                "name": name, "source": source, "chars": len(text),
            }
            if mode == "full":
                item["content"] = text
            items.append(item)
        # Tail when every injected item sits after every pre-existing message; else front/mixed.
        position = "tail" if (not orig_idx or min(injected_idx) > max(orig_idx)) else "front"
        hook = next(iter(sources)) if len(sources) == 1 else "llm_before"
        return {
            "hook_point": "LLM_BEFORE", "hook": hook, "position": position, "kind": "inject",
            "payload": {
                "v": 1, "items": items, "item_count": len(items),
                "total_chars": sum(int(it["chars"]) for it in items),
            },
        }

    # ── Helper: append message (with MESSAGE_APPEND hook) ──

    async def _append_message(
        self,
        msg: LoopMessage,
        *,
        round_index: int | None = None,
        hook_injected: dict[str, Any] | None = None,
    ) -> None:
        ctx = MessageAppendCtx(
            round_index=round_index or 0,
            message=dict(msg),
            session_id=self.session_id,
        )
        await self.hooks.run_typed_async(HookPoint.MESSAGE_APPEND, ctx)
        self.history.append(ctx.message)
        # 6.12.0 图片保留：记下「这一行是第几轮入的图」（旁路表，按对象身份；history 行原样发给
        # 供应商，不能夹私货字段）。
        _c = ctx.message.get("content")
        if isinstance(_c, list) and any(isinstance(b, dict) and b.get("type") == "attachment" for b in _c):
            self._image_rounds[id(ctx.message)] = int(round_index or 0)
        # SCALE-4: keep the token estimate current incrementally on the hot append path
        # — only when the cache was already in sync (else leave it for a full recompute).
        if self._tok_len == len(self.history) - 1:
            self._tok_total += estimate_message_tokens(ctx.message)
            self._tok_len = len(self.history)
        # Carry the send_index (and hook-injection audit) to the sink — but ONLY on the copy handed
        # to the sink, never on ctx.message (which lives in self.history and is sent verbatim to the
        # LLM; an unknown field would leak / break the provider). The sink persists send_index into
        # the messages.send_index column and hook_injected into the hook_events table; neither
        # reaches the LLM.
        sink_msg = ctx.message
        if self.send_index is not None:
            sink_msg = {**sink_msg, "send_index": self.send_index}
        if hook_injected is not None:
            sink_msg = {**sink_msg, "hook_injected": hook_injected}
        await self._emit_sink(self.sink.on_message_appended, sink_msg, round_index=round_index)

    async def _resolve_skipped_tool_calls(
        self, skipped: Sequence[Mapping[str, Any]], *, reason: str, round_idx: int
    ) -> None:
        """Append a synthetic ``tool`` message for each un-executed tool_call.

        When the tool loop exits early (TOOL_AFTER BREAK, or a user-input request
        batched before later tool_calls), the remaining tool_calls would otherwise
        be left without responses — a protocol-invalid sequence the provider
        rejects (and a session left wrongly 'pending'). This resolves them.
        """
        for tc in skipped:
            await self._append_message(
                {
                    "role": "tool",
                    "tool_call_id": str(tc.get("id") or ""),
                    "name": _tool_call_name(tc),
                    "content": f"[skipped: {reason}]",
                },
                round_index=round_idx,
            )

    # ── Helper: finalize session ──

    async def _finalize(self, reason: str, *, final_text: str | None = None,
                        rounds: int | None = None) -> None:
        if self._finalized:
            return  # idempotent: SESSION_ENDED fires exactly once per run
        self._finalized = True
        if rounds is not None:
            self._completed_rounds = rounds
        ctx = SessionEndCtx(
            scope="main", reason=reason,
            messages=self.history, final_text=final_text,
        )
        await self.hooks.run_typed_async(HookPoint.SESSION_END, ctx)
        self._emit(AgentEventType.SESSION_ENDED, SessionEndedPayload(reason=reason))
        await self._maybe_remember(reason=reason, final_text=final_text or "")

    async def _emit_error_terminal(self, exc: BaseException) -> None:
        """Emit the ``AGENT_ERROR`` channel + a terminal ``SESSION_ENDED`` after an
        unexpected exception escaped :meth:`run` (a raising hook, sink, prepare_round,
        store I/O …). Subscribers that saw ``SESSION_STARTED`` would otherwise be
        stranded with no terminal, and the documented ``AGENT_ERROR`` channel would
        stay dead code (H1.5).

        Best-effort and self-guarding: it must NEVER mask the original exception, so
        every step is wrapped and the caller re-raises ``exc`` after this returns.
        """
        try:
            self._emit(
                AgentEventType.AGENT_ERROR,
                AgentErrorPayload(error=str(exc), error_type=type(exc).__name__),
            )
        except Exception:  # noqa: BLE001
            logger.exception("AGENT_ERROR emit failed for session %s", self.session_id)
        if not self._session_started:
            return  # no start was observed → nothing to terminate
        try:
            await self._finalize("error")
        except Exception:  # noqa: BLE001 — a SESSION_END hook / remember may itself raise
            logger.exception("error-path finalize failed for session %s", self.session_id)
            # Guarantee a terminal event even if the SESSION_END hook raised before
            # _finalize could emit it.
            try:
                self._emit(AgentEventType.SESSION_ENDED, SessionEndedPayload(reason="error"))
            except Exception:  # noqa: BLE001
                pass

    # ── Memory: remember at end (M1.9) ──
    #
    # Recall is no longer a hardcoded step here. It is the built-in
    # ``MemoryRecallHook`` (an LLM_BEFORE hook registered by StatefulAgentLoop)
    # which injects recalled memory EPHEMERALLY at the per-call request tail —
    # never into ``self.history`` / the store — so the prompt prefix stays
    # append-only and prefix-cacheable, and there is no index↔seq realignment
    # (``sink.on_messages_inserted`` is therefore gone). See runtime.memory.

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

    def _log_send_summary(self, status: str, rounds: int) -> None:
        """一个 send 跑完记一行：轮数、上下文从多大涨到多大、这一程重发了多少、大头是谁。

        为什么值得单独记：账单和体感都由「轮数 × 每轮上下文」决定，而这两个数过去哪个都不落日志。
        真实经历：一次排查里，要回答「这一程为什么烧了 900 万 token」，只能去 docker logs 里
        grep 每一轮的 token_usage 再手工求和——而那时候日志可能已经滚掉了。
        """
        try:
            usage = dict(self.ctx.usage_totals or {})
            total_in = int(usage.get("prompt_tokens", 0) or 0)
            real = int((self.ctx.token_usage or {}).get("prompt_tokens", 0) or 0)
            base = self._send_baseline_prompt
            start_i = min(int(getattr(self, "_send_start_len", 0) or 0), len(self.history))
            logger.info(
                "send done: %s in %d rounds (send %s) | 上下文 %s → %s | 本程累计重发 %d token"
                "（均 %d/轮）| 大头: %s",
                status, rounds, self.send_index,
                f"{int(base):,}" if base else "?", f"{real:,}" if real else "?",
                total_in, total_in // max(1, rounds), self._context_profile(start_i),
            )
        except Exception:  # noqa: BLE001 — 日志永远不该弄挂一次正常收尾
            logger.debug("send summary log failed", exc_info=True)

    def _make_result(self, status: str, *, final_text: str = "", rounds: int = 0,
                     pending_tool_calls: list | None = None,
                     pending_interactions: list | None = None) -> AgentLoopResult:
        self._completed_rounds = rounds  # for MemorySnapshot
        self._log_send_summary(status, rounds)
        return AgentLoopResult(
            status=status,  # type: ignore[arg-type]
            final_text=final_text,
            rounds=rounds,
            pending_tool_calls=pending_tool_calls or [],
            pending_interactions=pending_interactions or [],
            messages=self.history,
            usage=dict(self.ctx.usage_totals),
            tool_calls=int(self.ctx.tool_calls),
        )

    async def _persist_pending_interaction(
        self,
        *,
        interaction: dict[str, Any],
        round_index: int,
    ) -> None:
        if self.store is None or self.session_id is None:
            return
        state = await self.store.get_state(self.session_id)
        pending = dict(state.pending or {}) if state is not None else {}
        interactions = list(pending.get("pending_interactions") or [])
        interactions.append(interaction)
        pending["pending_interactions"] = interactions
        pending["round_index"] = round_index
        await self.store.set_pending(self.session_id, pending)

    # ══════════════════════════════════════════════════════════════
    # Phase methods — pure business logic with explicit parameters.
    # Hook orchestration is handled entirely by run().
    # ══════════════════════════════════════════════════════════════

    _IMAGE_RETIRED_MARK = "[image retired"

    def _retire_stale_images(self, round_index: int, keep_rounds: int) -> int:
        """把当前 send 里「入上下文已超过 keep_rounds 轮」的图片附件块在内存里换成占位文字。

        图片行是 user 行（see_image / 用户发图），带 ``round_index``（``_append_message`` 记的）。
        第 r 轮入的图在第 r+keep_rounds 轮之前都以原图参与请求，之后换成
        ``[image retired: <name> — 已看过；要再看调 see_image]``。不改 pl_messages、不改行数。
        返回替换了几个块。"""
        n = 0
        rounds = getattr(self, "_image_rounds", None) or {}
        if not rounds:
            return 0
        for m in self.history:
            r = rounds.get(id(m))
            if r is None:
                continue
            c = m.get("content")
            if not isinstance(c, list) or not any(isinstance(b, dict) and b.get("type") == "attachment" for b in c):
                continue
            if round_index - int(r) < keep_rounds:
                continue
            new_blocks = []
            for b in c:
                if isinstance(b, dict) and b.get("type") == "attachment":
                    att = b.get("attachment") if isinstance(b.get("attachment"), dict) else {}
                    name = str(att.get("name") or att.get("filename") or att.get("path") or att.get("ref") or "image")
                    name = name.rsplit("/", 1)[-1]
                    new_blocks.append({"type": "text",
                                       "text": f"{self._IMAGE_RETIRED_MARK}: {name} — 已看过；要再看调 see_image]"})
                    n += 1
                else:
                    new_blocks.append(b)
            m["content"] = new_blocks
        if n:
            logger.info("image retention: retired %d image block(s) older than %d round(s) (round %d)",
                        n, keep_rounds, round_index)
        return n

    _DISTILL_MARK = "[distilled #"

    #: 参数被瘦身后留在原位的标记（值本身仍是合法 JSON 字符串，工具早已执行完，
    #: 这里只影响模型回看时读到什么）。
    _ARGS_MARK = "⟨已移出上下文"
    #: 单个参数字段超过它才瘦身：短字段（路径、开关、行号）留着才看得出这次调用干了什么。
    _ARGS_KEEP_CHARS = 200

    def _has_heavy_args(self, idx: int) -> bool:
        """这一行是 assistant，且它的某个 tool_calls 参数里有还没瘦过的大字段。"""
        m = self.history[idx]
        if m.get("role") != "assistant":
            return False
        keep = set(getattr(self.config, "insend_distill_keep_tools", ()) or ())
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            if str(fn.get("name") or "") in keep:
                continue
            raw = fn.get("arguments")
            if (
                isinstance(raw, str)
                and len(raw) > 300
                and self._ARGS_MARK not in raw
            ):
                return True
        return False

    def _distill_tool_call_args(self, idx: int, seqs: list[Any]) -> int:
        """把一行 assistant 的 tool_calls 参数里的大字段换成指针，**保持合法 JSON**。

        为什么不套用投影那套逐工具规则：那边产出的是给人读的一行散文，而这里改的是
        供应商要按 JSON 解析的 ``function.arguments``。所以规则通用——短字段原样留下
        （路径、模式、行号这些才看得出这次调用干了什么），只有超过 ``_ARGS_KEEP_CHARS``
        的长字符串换成一句带 ``recall_send`` 坐标的说明。工具早已执行完，改它不影响任何
        副作用，只影响模型回看时读到什么。返回释放的估算 token。"""
        m = self.history[idx]
        seq = seqs[idx] if idx < len(seqs) else None
        coord = (
            f"recall_send(send_index={self.send_index}, seq={seq})"
            if (self.send_index is not None and seq is not None)
            else "recall_send"
        )
        freed = 0
        keep = set(getattr(self.config, "insend_distill_keep_tools", ()) or ())
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            if str(fn.get("name") or "") in keep:
                continue   # 模型自己的发言（send_message 之类）：瘦成占位符它会照样再发一遍
            raw = fn.get("arguments")
            if not isinstance(raw, str) or len(raw) <= 300 or self._ARGS_MARK in raw:
                continue
            try:
                args = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(args, dict):
                continue
            slim = {
                k: (
                    f"{self._ARGS_MARK} {len(v)} 字符，需要时 {coord} 回取⟩"
                    if isinstance(v, str) and len(v) > self._ARGS_KEEP_CHARS
                    else v
                )
                for k, v in args.items()
            }
            new_raw = json.dumps(slim, ensure_ascii=False)
            if len(new_raw) >= len(raw):
                continue  # 没省下就别动（改了反而打断前缀缓存）
            fn["arguments"] = new_raw
            freed += (len(raw) - len(new_raw)) // 4
        return freed

    def _context_profile(self, start: int, limit: int = 4) -> str:
        """本 send 的上下文被谁占着——`工具名:来源=字符数`，最大的几个。

        排查上下文暴涨时，第一个要问的问题是「大头在哪」，而这个问题过去只能靠事后翻
        pl_messages 手工统计（真实经历：一次排查里，答案是 write_file 的**调用参数**占了
        全部参数字符的 66%，而它的回执只有一句「Wrote 4628 bytes」——光看回执永远看不出来）。
        所以这里把两块分开算：`args` 是 assistant 的 tool_calls 参数，`res` 是工具回执。

        只在保险丝触发那一轮算一次，O(本 send 行数)，不进热路径。
        """
        buckets: dict[str, int] = {}
        for m in self.history[start:]:
            role = m.get("role")
            if role == "tool":
                name = str(m.get("name") or "tool")
                buckets[f"res:{name}"] = buckets.get(f"res:{name}", 0) + len(str(m.get("content") or ""))
            elif role == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = (tc or {}).get("function") or {}
                    raw = fn.get("arguments")
                    if isinstance(raw, str):
                        k = f"args:{fn.get('name') or 'tool'}"
                        buckets[k] = buckets.get(k, 0) + len(raw)
        top = sorted(buckets.items(), key=lambda kv: -kv[1])[:limit]
        return " ".join(f"{k}={v // 1000}K" for k, v in top if v >= 1000) or "(无大项)"

    def _distill_oldest_tool_rows(self, batch: int, hot_tail: int) -> int:
        """把当前 send 里**最早的 batch 条**工具占用在内存里换成指针（6.10.0；6.25.0 起含调用参数）。

        「工具占用」有两块：结果在 tool 行、**参数在 assistant 行的 tool_calls 里**。两块按位置
        统一排队、一起算进 batch，最早的先回收。

        投影行 = 与 end-of-send 投影**同一套** ``ToolDefinition.project`` 蒸馏（找不到钩子就截断），
        带 ``recall_send(send_index, seq)`` 坐标——原文永远在 pl_messages，要用就回取。不落盘、不改
        数据库、不改行数（sink 的 index↔seq 对齐不受影响）。最近 ``hot_tail`` 条永不动。verbatim
        模式不做（那边有就地压缩器）。返回本批释放的估算 token（≈字符/4）。"""
        rep = self.config.representation
        if getattr(rep, "kind", "projection") == "verbatim":
            return 0
        project_tool = getattr(rep, "_project_tool", None)
        seqs = getattr(self.sink, "_history_seqs", None) or []
        # 6.27.0 只动**本 send** 追加的行：run 之前就在历史里的（上一个 send 留下的逐字行）归跨 send
        # 投影管，保险丝碰它们等于两套机制抢同一段历史。
        start = min(int(getattr(self, "_send_start_len", 0) or 0), len(self.history))
        tool_idx = [
            i for i, m in enumerate(self.history)
            if i >= start and m.get("role") == "tool" and isinstance(m.get("content"), str)
        ]
        # max(0, …)：本 send 的工具行少于 hot_tail 时，切片起点会是负数，Python 把它当成
        # 「倒数第 |x| 条」绕回去——5 条工具行配 hot_tail=8 时只保住最后 3 条，最早 2 条
        # 照样被蒸馏，而那 2 条正是模型手边的东西。本意是「不足 hot_tail 条就全保住」：
        # 保险丝的前提是有旧结果可回收，没有旧的就什么都不该动，让它涨到切 send 阈值
        # 优雅收尾即可（断片比切 send 贵得多）。
        protected = (
            set(tool_idx[max(0, len(tool_idx) - hot_tail):]) if hot_tail > 0 else set()
        )
        # 热尾的起点：它之后的任何一行（工具结果、assistant 的调用参数）都算「手边的东西」。
        protect_from = min(protected) if protected else len(self.history)
        result_ok = {
            i for i in tool_idx
            if i not in protected
            and not str(self.history[i].get("content")).startswith(self._DISTILL_MARK)
            and len(self.history[i].get("content") or "") > 300
        }
        # 工具在上下文里占两块：结果在 tool 行，**参数在 assistant 行的 tool_calls 里**。
        # 参数从来没被保险丝碰过，而实测它占全库上下文的 29%——write_file 一个人 12.6MB，
        # 单次最大 5.4 万字符：文件已经落盘了，正文却还在 prompt 里一轮轮重发。
        # 跨 send 的投影早就把两块一起收掉了，缺的只是 send 内这一层。
        candidates: list[tuple[str, int]] = sorted(
            [("result", i) for i in result_ok]
            + [("args", i) for i in range(start, protect_from) if self._has_heavy_args(i)],
            key=lambda x: x[1],
        )
        picked_all = candidates[: max(1, batch)]
        picked = [i for kind, i in picked_all if kind == "result"]
        picked_args = [i for kind, i in picked_all if kind == "args"]
        if not picked_all:
            return 0
        # tool_call_id → (name, args) 从前面的 assistant(tool_calls) 行反查。
        # 只解析**本批选中的**那几个 id：以前是每次触发都把全历史所有 assistant 行的所有
        # tool_calls 的 arguments 反序列化一遍建全表，只为查其中 batch 条（默认 10）。
        # 保险丝一旦启动几乎每轮都触发，于是每轮做一次 O(历史) 的 json.loads。
        wanted = {str(self.history[i].get("tool_call_id") or "") for i in picked}
        wanted.discard("")
        call_meta: dict[str, tuple[str, Any]] = {}
        for m in self.history if wanted else ():
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                cid = str((tc or {}).get("id") or "")
                if cid not in wanted or cid in call_meta:
                    continue
                fn = (tc or {}).get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (TypeError, ValueError):
                        args = {"raw": args[:200]}
                call_meta[cid] = (str(fn.get("name") or "tool"), args)
            if len(call_meta) == len(wanted):
                break
        freed = 0
        for i in picked:
            m = self.history[i]
            name, args = call_meta.get(str(m.get("tool_call_id") or ""), ("tool", {}))
            content = str(m.get("content") or "")
            if project_tool is not None:
                try:
                    d = project_tool(name, args, content, getattr(self, "tool_registry", None))
                    parts = [f"{k}={v}" for k, v in (d or {}).items() if k != "name" and v not in (None, "")]
                    line = f"{name}(" + ", ".join(str(x)[:200] for x in parts) + ")"
                except Exception:  # noqa: BLE001 — 蒸馏钩子出错就退回截断
                    line = f"{name}: {content[:200]}…"
            else:
                line = f"{name}: {content[:200]}…"
            seq = seqs[i] if i < len(seqs) else None
            coord = (f"recall_send(send_index={self.send_index}, seq={seq})"
                     if (self.send_index is not None and seq is not None) else "recall_send")
            new_content = (f"{self._DISTILL_MARK}{self.send_index} seq={seq} — 原文已从上下文移出，"
                           f"需要时 {coord} 回取] {line}")
            freed += max(0, (len(content) - len(new_content)) // 4)
            m["content"] = new_content
        for i in picked_args:
            freed += self._distill_tool_call_args(i, seqs)
        return freed

    def _estimate_history_tokens(self) -> int:
        """Current history token estimate, maintained incrementally (SCALE-4).

        Returns the cached total when it is still in sync with ``self.history``
        (the append path keeps it current); otherwise recomputes from scratch — which
        covers folds, recall front-inserts, run-start, and any hook that replaced the
        list. Correct by construction: a stale cache is detected by the length check or
        an explicit invalidation, never trusted blindly.
        """
        if self._tok_len != len(self.history):
            self._tok_total = estimate_tokens(self.history)
            self._tok_len = len(self.history)
        return self._tok_total

    def _build_compaction_context(self, round_index: int) -> CompactionContext:
        """Build the optional handle a context-aware compactor receives (H7 P2):
        the configured MemoryProvider + a read-only seq-range message accessor."""
        store = self.store
        sid = self.session_id

        async def _fetch(from_seq: int, to_seq: int) -> list[dict[str, Any]]:
            assert store is not None and sid is not None  # set only when has_store
            rows = await store.load_all_messages(sid)
            return [
                {
                    "role": r.role, "name": r.name, "content": r.content,
                    "tool_calls": r.tool_calls, "tool_call_id": r.tool_call_id,
                    "seq": r.seq, "round_index": r.round_index,
                }
                for r in rows if from_seq <= r.seq <= to_seq
            ]

        has_store = store is not None and sid is not None
        return CompactionContext(
            session_id=sid,
            memory=self.config.memory,
            round_index=round_index,
            fetch_messages=_fetch if has_store else None,
            current_tokens=self._estimate_history_tokens(),
        )

    async def prepare_round(self, round_index: int) -> None:
        """Prepare a new round: todo reminders, then run the pluggable
        compactor if one is configured on the loop config."""
        # Microcompact (dump old large tool outputs to disk + leave a short
        # pointer — orthogonal to LLM-based compaction). OFF by default as of
        # 3.1.x; opt in via config.microcompact_enabled. See AgentLoopConfig.
        # ── 6.12.0 图片看过即撤：入上下文超过 image_retention_rounds 轮的 attachment 块换成占位文字 ──
        irr = self.config.image_retention_rounds
        if irr is not None and int(irr) > 0 and self._retire_stale_images(round_index, int(irr)):
            self._tok_len = -1

        # ── 6.10.0 send 内保险丝：最早的 n 条工具结果 → 投影行（内存替换，pl_messages 不动）──
        # 触发依据 = 上一轮供应商返回的真实 prompt_tokens（就是当前上下文的真实大小；第一轮
        # 还没有就用估算）。每轮最多蒸馏一批（最早的 n 条、跳过已蒸馏的、不动最近 hot_tail 条）；
        # 下一轮真实 prompt 仍超阈值就再蒸馏下一批——逐轮递进，只蒸馏到够用为止。
        dt = self.config.insend_distill_tokens
        if dt is not None and int(dt) > 0 and round_index > 0:
            # 6.27.0：量的是**本 send 内的增长**，不是整个上下文。系统提示词 + 技能 + 工具目录 + 历史
            # 一开局就可能有几万 token，按整体判的话第一轮就超阈值、之后每轮都烧（真实事故：一个
            # 会话 44 次触发，模型自己的发言参数被瘦成占位符，再被照样发进聊天）。保险丝的本意是
            # 「这一个 send 跑得太长」才出手：第一轮的真实 prompt_tokens 是基线，之后每轮减它。
            real = int((self.ctx.token_usage or {}).get("prompt_tokens", 0) or 0)
            if real > 0:
                if self._send_baseline_prompt is None:
                    self._send_baseline_prompt = real
                growth = real - int(self._send_baseline_prompt)
            else:
                start = min(int(getattr(self, "_send_start_len", 0) or 0), len(self.history))
                growth = estimate_tokens(self.history[start:])
            if growth >= int(dt):
                freed = self._distill_oldest_tool_rows(
                    int(self.config.insend_distill_batch or 10),
                    int(self.config.insend_distill_hot_tail or 0),
                )
                if freed:
                    self._tok_len = -1  # content shrank in place; force a fresh estimate
                start_i = min(int(getattr(self, "_send_start_len", 0) or 0), len(self.history))
                # freed==0 也要记：那是「保险丝烧完了、再触发也没用」的静默状态，
                # 而它的表现和「一切正常」一模一样——上下文继续涨，日志里一个字都没有。
                logger.info(
                    "insend distill: send grew %d >= %d tokens; freed ~%d this round "
                    "(send %s, round %d, prompt %d, rows %d) | 大头: %s%s",
                    growth, int(dt), freed, self.send_index, round_index, real,
                    len(self.history) - start_i, self._context_profile(start_i),
                    "" if freed else "  ← 没有可回收的了：剩下的要么在热尾、要么已蒸馏、要么是"
                                     "模型自己的正文（保险丝只动工具结果与调用参数）",
                )
        if self.config.microcompact_enabled:
            self.ctx.microcompact(
                self.history,
                size_limit=self.config.microcompact_size_limit,
                hot_tail=self.config.microcompact_hot_tail,
                spill_dir=self.config.microcompact_spill_dir,
            )
            # microcompact mutates message CONTENT in place (shrinks it) without
            # changing len(self.history), so the SCALE-4 incremental estimate would
            # stay stale-high and over-trigger LLM-summary compaction. Invalidate it.
            self._tok_len = -1

        # 3.0: verbatim mode → an in-place Compactor mapped from config.fold_strategy; projection
        # mode → None (it folds at end-of-send in the derived layer).
        compactor = self.config.resolve_compactor()
        if compactor is None:
            return

        compact_before = CompactBeforeCtx(
            round_index=round_index,
            messages=self.history,
        )
        await self.hooks.run_typed_async(HookPoint.COMPACT_BEFORE, compact_before)
        if compact_before.directive == HookDirective.SKIP:
            return

        compact_kwargs: dict[str, Any] = dict(
            llm=self.llm,
            # Reserve headroom for the ephemeral tail-injected memory block (it
            # isn't in self.history so the compactor can't see it) — fold a bit
            # earlier so history + memory stays within the window.
            max_tokens=self.config.effective_context_budget(),
            round_index=round_index,
        )
        # Back-compat: only hand the optional CompactionContext to compactors whose
        # maybe_compact() actually accepts it, so pre-existing (old-signature) custom
        # compactors keep working unchanged (H7 Phase 2).
        if _maybe_compact_accepts_context(type(compactor).maybe_compact):
            compact_kwargs["context"] = self._build_compaction_context(round_index)
        plan = await compactor.maybe_compact(self.history, **compact_kwargs)
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
        # SCALE-4: a fold reassigns history; invalidate the incremental estimate
        # like the other reassignment sites. A single-message fold (start==end)
        # keeps len() unchanged, so without this the cache returns a stale (too-high)
        # count and the next round's trigger decision is made on it.
        self._tok_len = -1
        # Persist (no-op for NullSink). Pass the pre-fold history length so the
        # sink can refuse to persist if its index↔seq map ever drifts out of
        # alignment (H1.1 safety net), rather than mark the wrong rows.
        await self._emit_sink(
            self.sink.on_compaction,
            fold_start_idx=plan.fold_start_idx,
            fold_end_idx=plan.fold_end_idx,
            summary_text=plan.summary_text,
            before_tokens=plan.before_tokens,
            after_tokens=plan.after_tokens,
            round_index=round_index,
            expected_history_len=before_len,
        )
        compact_after = CompactAfterCtx(
            round_index=round_index,
            messages=self.history,
            messages_before_count=before_len,
            messages_after_count=len(self.history),
        )
        await self.hooks.run_typed_async(HookPoint.COMPACT_AFTER, compact_after)

    async def _runtime_messages_for_round(self, round_index: int) -> list[LoopMessage]:
        """Build transient runtime-state messages for the next LLM call.

        These messages are deliberately not appended through ``_append_message``:
        the store's runtime state is the authority, and the message is just the
        current projection the model needs for this round.
        """
        if self.store is None or self.session_id is None:
            todo_snap = self.ctx.todo.snapshot_for_prompt()
            return [{"role": "user", "content": todo_snap}] if todo_snap else []

        messages: list[LoopMessage] = []
        for projector in self.config.runtime_projectors:
            projected = await projector.project(
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
            model=self.config.model,                 # per-loop model override (None = service default)
            response_format=self.config.response_format,  # structured output when set
        )

        model_name = self.config.model or ""
        attempt_box = [0]

        async def _do_call() -> LLMResponse:
            # STREAM_STARTED/COMPLETED are emitted **per attempt** and paired in this call's own
            # finally (M-pipeline-runner-1): the terminal used to live in the OUTER finally, which
            # ran once, so N retry attempts emitted N STARTED but only 1 COMPLETED — unbalanced
            # stream events on any LLM retry. Each attempt now opens and closes exactly one stream.
            attempt_box[0] += 1
            attempt = attempt_box[0]
            call_id = f"r{round_index}.a{attempt}"
            self._emit(AgentEventType.STREAM_STARTED, StreamStartedPayload(),
                       round_index=round_index, stream_id="main")
            try:
                # Per-call observability (H4.1): pair STARTED/COMPLETED by call_id so a
                # subscriber sees per-attempt latency + per-call (not cumulative) usage,
                # making retries individually visible.
                self._emit(
                    AgentEventType.LLM_CALL_STARTED,
                    LlmCallStartedPayload(call_id=call_id, round_index=round_index,
                                          attempt=attempt, model=model_name),
                    round_index=round_index, stream_id="main",
                )
                t0 = time.perf_counter()
                try:
                    resp = await self.llm.complete(
                        request, on_chunk_delta_text=_on_delta, on_chunk_think=_on_think,
                    )
                except BaseException as exc:
                    self._emit(
                        AgentEventType.LLM_CALL_COMPLETED,
                        LlmCallCompletedPayload(
                            call_id=call_id, round_index=round_index, attempt=attempt,
                            model=model_name, duration_ms=(time.perf_counter() - t0) * 1000.0,
                            success=False, error_type=type(exc).__name__,
                        ),
                        round_index=round_index, stream_id="main",
                    )
                    raise
                usage = getattr(resp, "token_usage", None)
                self._emit(
                    AgentEventType.LLM_CALL_COMPLETED,
                    LlmCallCompletedPayload(
                        call_id=call_id, round_index=round_index, attempt=attempt,
                        model=model_name, duration_ms=(time.perf_counter() - t0) * 1000.0,
                        success=True,
                        prompt_tokens=getattr(usage, "prompt_tokens", None),
                        completion_tokens=getattr(usage, "completion_tokens", None),
                        total_tokens=getattr(usage, "total_tokens", None),
                        prompt_cached_tokens=getattr(usage, "prompt_cached_tokens", None),
                        prompt_cache_miss_tokens=getattr(usage, "prompt_cache_miss_tokens", None),
                    ),
                    round_index=round_index, stream_id="main",
                )
                return resp
            finally:
                self._emit(AgentEventType.STREAM_COMPLETED, StreamCompletedPayload(),
                           round_index=round_index, stream_id="main")

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

        return response

    def _is_async_capable(self, tool_name: str, tool_args: Mapping[str, Any] | None = None) -> bool:
        # 6.15.0：async_capable 可以是一组 action 名，所以判定要带上这次调用的参数——
        # 同一个工具的 get 只读、freeze 要写，不能一刀切。
        from power_loop.tools.registry import async_capable_for

        reg = self.tool_registry
        rt = reg.get(tool_name) if reg is not None else None
        return bool(rt is not None and async_capable_for(rt.definition, tool_args))

    def _spawn_tool_task(
        self, tool_name: str, tool_args: dict[str, Any], sem: asyncio.Semaphore
    ) -> asyncio.Future:
        """同轮并发的一个成员：受信号量限流地跑 execute_tool。contextvars 随 task 自动拷贝
        （PEP 567），计费 / 活动打标 / 运行时上下文零处理。异常原样留在 task 里，轮到它时
        再由串行路径的 HumanInputRequired / TOOL_ERROR 分支处理。"""

        async def _run() -> tuple[str, bool]:
            async with sem:
                return await self.execute_tool(tool_name, tool_args)

        return asyncio.ensure_future(_run())

    @staticmethod
    def _cancel_pre_tasks(pre: dict[int, tuple[asyncio.Future, Any]]) -> None:
        for task, _ in pre.values():
            if not task.done():
                task.cancel()
        pre.clear()

    async def execute_tool(
        self, tool_name: str, tool_args: dict[str, Any], *, count: bool = True
    ) -> tuple[str, bool]:
        """Execute a single tool and return ``(output_string, failed)``.

        Catches :class:`ToolNotFound` / :class:`ToolValidationError` from
        the registry and returns them as error strings (failed=True), making
        them visible to the LLM so it can self-correct.

        ``count`` increments ``ctx.tool_calls`` only for a real, validated
        dispatch. A rejected (unknown/invalid) call never ran, and the
        SHORT_CIRCUIT retry passes ``count=False`` so one logical tool call is
        not counted twice.
        """
        if self.tool_registry is None:
            return (f"Error: tool '{tool_name}' requested but no tool registry configured", True)
        try:
            validation_err = self.tool_registry.validate(tool_name, tool_args)
            if validation_err is not None:
                return (validation_err, True)

            if count:
                self.ctx.tool_calls += 1
            result = await self.tool_registry.invoke_async(tool_name, tool_args)
        except (ToolNotFound, ToolValidationError) as exc:
            return (str(exc), True)
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        return (str(result), False)

    # ══════════════════════════════════════════════════════════════
    # Main orchestrator — loop, hooks, directive checks, events
    # ══════════════════════════════════════════════════════════════

    async def _complete_decide(
        self, *, reason: str, round_idx: int, final_text: str, next_round: int,
    ) -> bool:
        """Consult COMPLETE_DECIDE hooks at a send's terminal boundary.

        Returns True when a hook SHORT_CIRCUITs with a non-empty ``inject``:
        the text is appended as a durable user message (same send) and the
        round budget is extended so at least ``extra_rounds`` more rounds run
        starting at ``next_round``. Returns False to end the send normally.
        """
        ctx = CompleteDecideCtx(
            round_index=round_idx, reason=reason, final_text=final_text,
            fire_count=self._complete_decide_fires,
        )
        await self.hooks.run_typed_async(HookPoint.COMPLETE_DECIDE, ctx)
        inject = str(ctx.inject or "").strip()
        if ctx.directive != HookDirective.SHORT_CIRCUIT or not inject:
            return False
        self._complete_decide_fires += 1
        await self._append_message({"role": "user", "content": inject}, round_index=round_idx)
        try:
            extra = max(1, int(ctx.extra_rounds))
        except (TypeError, ValueError):
            extra = 1
        # A terminal injection owns a fresh, exact allowance. In particular,
        # budget-exceeded finalizers must not inherit all unused main-loop
        # rounds (which could turn a 4-round finalizer into dozens of unbudgeted
        # calls), while round-limit finalizers still extend past the old limit.
        self._round_limit = next_round + extra
        # The normal token budget is already a terminal condition. Once a hook
        # elects to continue, let its explicitly bounded allowance run; checking
        # the same exhausted budget on the next boundary would immediately stop
        # before the injected prompt ever reached the model.
        self._terminal_grace_active = True
        return True

    async def run(self, messages: list[LoopMessage]) -> AgentLoopResult:
        """Run the full agent loop. Returns when done, cancelled, or hit round limit."""
        self.history = [dict(m) for m in messages]
        self._image_rounds = {}   # 6.12.0：本 send 新入的图片行 id(row) → round_index
        self._tok_len = -1  # SCALE-4: wholesale (re)assignment invalidates the estimate
        # 6.27.0 send 内保险丝只管**本 send**：起点 = 进入 run 时的历史长度（之前的行归跨 send 投影管）；
        # 触发量 = 本 send 内的增长（第一轮真实 prompt_tokens 是基线）。
        self._send_start_len = len(self.history)
        self._send_baseline_prompt = None

        # ── Session start ──
        session_ctx = SessionStartCtx(
            scope="main", messages=self.history, stop_event=self.stop_event,
        )
        await self.hooks.run_typed_async(HookPoint.SESSION_START, session_ctx)
        if isinstance(session_ctx.messages, list):
            self.history = session_ctx.messages
            self._tok_len = -1  # a hook may have replaced history (even same-length)
            self._send_start_len = len(self.history)
        self._emit(AgentEventType.SESSION_STARTED, SessionStartedPayload(scope="main"))
        self._session_started = True

        # Memory recall happens per-round in the built-in MemoryRecallHook
        # (LLM_BEFORE), injected ephemerally into the request tail — see run()'s
        # LLM_BEFORE block and runtime.memory.MemoryRecallHook.

        # ── Round loop ──
        # The budget is DYNAMIC: a COMPLETE_DECIDE hook can extend it
        # (same-send injection — see _complete_decide), so this is a while
        # over an explicit limit rather than a range().
        self._round_limit = int(self.config.max_rounds)
        self._complete_decide_fires = 0
        self._terminal_grace_active = False
        # A provider hiccup — a round with NO text and NO tool call — is not a
        # completion signal (see the no-tools block below). Count consecutive
        # empties so we retry a bounded number of times before giving up.
        self._empty_response_streak = 0
        self._truncated_streak = 0
        round_idx = -1
        while True:
            round_idx += 1
            if round_idx >= self._round_limit:
                # ── Hook: COMPLETE_DECIDE (round budget exhausted) ──
                # A SHORT_CIRCUIT with inject extends the budget and keeps
                # looping in the same send; otherwise fall through to the
                # forced wrap-up below.
                if await self._complete_decide(
                    reason="hit_round_limit", round_idx=round_idx, final_text="",
                    next_round=round_idx,
                ):
                    pass  # limit extended; run this round normally
                else:
                    break
            # Track for MemorySnapshot: how many round attempts we made.
            self._completed_rounds = round_idx
            if _is_cancelled(self.cancel_token):
                await self._finalize("cancelled")
                return self._make_result("cancelled", final_text="[cancelled by user]", rounds=round_idx)

            # ── Per-run token budget (round boundary: the round that crossed
            # the budget already finished cleanly, so no dangling tool_calls;
            # we stop before paying for the next LLM call). ──
            budget = self.config.max_tokens_per_run
            if (
                budget is not None
                and int(budget) > 0
                and round_idx > 0
                and not self._terminal_grace_active
            ):
                totals = self.ctx.usage_totals
                spent = int(totals.get("prompt_tokens", 0)) + int(totals.get("completion_tokens", 0))
                if spent >= int(budget):
                    self._emit(
                        AgentEventType.STATUS_CHANGED,
                        BudgetExceededStatusPayload(
                            budget_tokens=int(budget), spent_tokens=spent, rounds=round_idx,
                        ),
                        round_index=round_idx,
                    )
                    # COMPLETE_DECIDE gets first refusal at the same clean
                    # boundary as the terminal result. A hook injection grants
                    # a bounded finalization window in this send; with no hook,
                    # preserve the historical budget_exceeded behavior.
                    if not await self._complete_decide(
                        reason="budget_exceeded", round_idx=round_idx, final_text="",
                        next_round=round_idx,
                    ):
                        await self._finalize("budget_exceeded", rounds=round_idx)
                        return self._make_result(
                            "budget_exceeded",
                            final_text="[budget_exceeded]",
                            rounds=round_idx,
                        )

            # ── 6.10.0 上下文检查点（轮边界）：上一轮供应商返回的真实 prompt_tokens 就是当前上下文的
            # 真实大小。达到阈值 → 宿主 COMPLETE_DECIDE 先拿收尾窗口（总结做了什么/还剩什么），否则
            # 以 context_checkpoint 结束本 send → 正常投影 → 宿主续接机制在新 send 里接着干。
            # 与 max_tokens_per_run 不同：那是累计费用（Σ prompt+completion，随轮数平方增长），
            # 量不出「上下文有多大」；这里量的正是它。
            ckpt = self.config.context_checkpoint_tokens
            if (
                ckpt is not None
                and int(ckpt) > 0
                and round_idx > 0
                and not self._terminal_grace_active
            ):
                last_prompt = int((self.ctx.token_usage or {}).get("prompt_tokens", 0) or 0)
                if last_prompt >= int(ckpt):
                    self._emit(
                        AgentEventType.STATUS_CHANGED,
                        ContextCheckpointStatusPayload(
                            budget_tokens=int(ckpt), spent_tokens=last_prompt, rounds=round_idx,
                        ),
                        round_index=round_idx,
                    )
                    if not await self._complete_decide(
                        reason="context_checkpoint", round_idx=round_idx, final_text="",
                        next_round=round_idx,
                    ):
                        await self._finalize("context_checkpoint", rounds=round_idx)
                        return self._make_result(
                            "context_checkpoint",
                            final_text="[context_checkpoint]",
                            rounds=round_idx,
                        )

            # ── In-flight steering: drain follow-up queue at round boundary ──
            # Runs BEFORE the ROUND_START hooks so (a) a break-deciding hook (host
            # pass_turn hard-stop) sees `drained_follow_ups` and can reconsider a
            # stale silence decision, and (b) a hook BREAK can never strand queued
            # steering that arrived during the previous round (the old ordering
            # silently dropped it — the drain sat after the break check).
            drained_count = 0
            if self._drain_follow_ups is not None:
                drained = await self._drain_follow_ups()
                drained_count = len(drained)
                for msg in drained:
                    await self._append_message(msg, round_index=round_idx)

            # ── Hook: ROUND_START ──
            round_ctx = RoundStartCtx(
                round_index=round_idx, messages=self.history,
                stop_event=self.stop_event,
                drained_follow_ups=drained_count,
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
                self._tok_len = -1  # SCALE-4: a ROUND_START hook may have replaced history

            # ── Business logic: prepare round ──
            await self.prepare_round(round_idx)

            await self._emit_sink(self.sink.on_round_started, round_idx)
            self._emit(AgentEventType.ROUND_STARTED, RoundStartedPayload(round_index=round_idx), round_index=round_idx)

            runtime_messages = await self._runtime_messages_for_round(round_idx)
            llm_messages = [*self.history, *runtime_messages]

            # Audit (opt-in): snapshot the message identities BEFORE LLM_BEFORE hooks run, so we can
            # later diff out exactly what they ephemerally injected (e.g. recalled memory) for this
            # round's LLM call — by identity, so it works regardless of inject position (tail/front).
            _hook_audit_mode = self.config.record_hook_events
            _pre_hook_ids = (
                {id(m) for m in llm_messages}
                if _hook_audit_mode in ("metadata", "full")
                else None
            )

            # ── Hook: LLM_BEFORE ──
            llm_before = LlmBeforeCtx(
                round_index=round_idx,
                messages=llm_messages,
                system_prompt=self.system_prompt,
                tools=self.runtime_tools,
                max_tokens=int(self.config.max_tokens or 8000),
                temperature=float(self.config.temperature or 0),
                session_id=self.session_id,
            )
            await self.hooks.run_typed_async(HookPoint.LLM_BEFORE, llm_before)
            hook_audit = self._summarize_hook_injection(
                llm_before.messages, _pre_hook_ids, _hook_audit_mode
            )

            # Durable LLM_BEFORE injections (persist_messages): unlike the ephemeral, request-only
            # edits to `messages` (captured by hook_audit above), each of these becomes a REAL turn —
            # persisted to history + store with the round's send_index via the loop's own append path
            # — and is added to this round's request tail. Used for injected turns that must survive
            # the send (e.g. a periodic "you haven't called X in N rounds" reminder). Computed AFTER
            # hook_audit so these durable rows don't get counted as ephemeral injections.
            for _pm in llm_before.persist_messages:
                await self._append_message(_pm, round_index=round_idx)
                llm_before.messages.append(dict(_pm))

            # Image recall: a tool (see_image / recall_send / an image generator) can put
            # pictures in front of the model. DURABLE ones become real `user` rows — the image
            # stays in view for the rest of the send (cheap: it sits in the provider's cached
            # prefix) and is distilled to `[image: … · file_uuid=…]` across sends, so nothing
            # accumulates without bound. EPHEMERAL ones go into this request only.
            # Appended AFTER the hook's own persist_messages so a tool's picture lands below
            # the tool result that announced it — which is exactly where the model expects it.
            _durable_imgs, _ephemeral_imgs = drain_queued_images(self.session_id)
            for _img in _durable_imgs:
                await self._append_message(_img, round_index=round_idx)
                llm_before.messages.append(dict(_img))
            for _img in _ephemeral_imgs:
                llm_before.messages.append(_img)

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
                    await self._append_message(
                        {"role": "assistant", "content": msg},
                        round_index=round_idx, hook_injected=hook_audit,
                    )
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
                    await self._append_message(
                        {"role": "assistant", "content": text},
                        round_index=round_idx, hook_injected=hook_audit,
                    )
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

            # Any productive round (said something OR called a tool) clears the empty streak.
            if assistant_text or tool_calls:
                self._empty_response_streak = 0
            if tool_calls:
                self._truncated_streak = 0

            # Append assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_text}
            sanitized_tool_calls: list[dict[str, Any]] | None = None
            _args_cut = False
            if tool_calls:
                sanitized_tool_calls, _args_cut = _sanitize_tool_calls(tool_calls)
                assistant_msg["tool_calls"] = sanitized_tool_calls
            await self._append_message(assistant_msg, round_index=round_idx, hook_injected=hook_audit)
            # 截断的第二种表现：工具调用本身在，但它的 arguments JSON 断在半路 → 降成 {} →
            # 必填校验报「缺参数」，模型据此以为自己忘了填，于是原样再写一遍、再被截断。
            # 真实事故（conv-213）：一条 "missing required parameter" 背后是 20000 token 打满。
            # 提示要补，但**不能在这里补**——见工具循环之后那一处。
            if sanitized_tool_calls and _args_cut:
                logger.warning(
                    "tool-call arguments were unparseable at round %d (likely truncation; "
                    "finish_reason=%r, completion_tokens=%s) — will tell the model to write smaller "
                    "after this round's tool results (session=%s)",
                    round_idx, _finish_reason(response),
                    (usage or {}).get("completion_tokens"), self.session_id,
                )
            # Mark pending IMMEDIATELY so a crash here leaves a recoverable state.
            if sanitized_tool_calls:
                assistant_seq = len(self.history)  # 1-based position in history
                await self._emit_sink(
                    self.sink.on_assistant_tool_calls,
                    assistant_seq=assistant_seq,
                    tool_calls=sanitized_tool_calls,
                    round_index=round_idx,
                )

            # ── 截断优先判定：它和「空响应打嗝」长得像，但重试方式必须相反 ──
            # provider 因 max_tokens 硬切时，工具调用的 JSON 断在半路 → 解析不出 tool_calls，
            # 正文往往也是空的（内容全在那段 JSON 里）。**原样重试必然再次截断**，
            # 每次烧满一个 max_tokens（conv-213：两轮各约 8 分钟、产出为零）。
            # 所以这里不重试，而是把「你被截断了，拆小再来」作为一条 user 消息落进历史——
            # 改变了输入，模型才有可能给出不一样的输出。
            if not tool_calls:
                reason = _finish_reason(response)
                cap = self.config.max_tokens
                out_tokens = int((usage or {}).get("completion_tokens") or 0)
                truncated = (reason in _TRUNCATION_FINISH_REASONS
                             or (bool(cap) and out_tokens >= int(cap)))
                if truncated and self._truncated_streak < _TRUNCATED_MAX_RETRIES:
                    self._truncated_streak += 1
                    logger.warning(
                        "truncated LLM response at round %d (finish_reason=%r, "
                        "completion_tokens=%d/%s) streak=%d/%d — nudging to write smaller "
                        "(session=%s)",
                        round_idx, reason, out_tokens, cap, self._truncated_streak,
                        _TRUNCATED_MAX_RETRIES, self.session_id,
                    )
                    await self._append_message(
                        {"role": "user", "content": _TRUNCATION_NOTICE}, round_index=round_idx)
                    self._emit(AgentEventType.ROUND_COMPLETED,
                               RoundCompletedPayload(round_index=round_idx, has_tools=False),
                               round_index=round_idx)
                    await self.hooks.run_typed_async(HookPoint.ROUND_END, RoundEndCtx(
                        round_index=round_idx, messages=self.history,
                        has_tools=False, response_text=assistant_text))
                    await self._emit_sink(self.sink.on_round_ended, round_idx, usage=usage)
                    continue

            # ── No tools → completed (unless a follow-up is waiting) ──
            if not tool_calls:
                # A round with NO text AND NO tool call is a provider hiccup, not a
                # completion signal. A finishing agent either says something (non-empty
                # text, handled below) or calls a terminal tool (pass_turn → tool_calls).
                # A truly empty turn means the model produced nothing; treating it as
                # "done" silently discards the send — observed in production as 29 rounds
                # of real work collapsing to a blank final answer. Retry a bounded number
                # of times (each retry advances round_idx, so max_rounds still caps it),
                # then fall through to close the send rather than spin.
                if not assistant_text:
                    self._empty_response_streak += 1
                    if self._empty_response_streak <= _EMPTY_RESPONSE_MAX_RETRIES:
                        logger.warning(
                            "empty LLM response (no text, no tool call) at round %d "
                            "streak=%d/%d — retrying (session=%s)",
                            round_idx, self._empty_response_streak,
                            _EMPTY_RESPONSE_MAX_RETRIES, self.session_id,
                        )
                        # Close this round's stream/usage bookkeeping before looping again,
                        # mirroring the follow-up drain path — else the round is left
                        # unterminated and its per-round usage row is never written.
                        self._emit(AgentEventType.ROUND_COMPLETED,
                                   RoundCompletedPayload(round_index=round_idx, has_tools=False),
                                   round_index=round_idx)
                        await self.hooks.run_typed_async(HookPoint.ROUND_END, RoundEndCtx(
                            round_index=round_idx, messages=self.history,
                            has_tools=False, response_text=""))
                        await self._emit_sink(self.sink.on_round_ended, round_idx, usage=usage)
                        continue
                    # Streak exhausted: stop retrying and let the normal completion path
                    # below close the send. final_text stays "" — but now it's an explicit
                    # give-up after N retries, not a silent first-empty completion.
                    logger.warning(
                        "empty LLM response persisted %d rounds — closing send (session=%s)",
                        self._empty_response_streak, self.session_id,
                    )
                # In-flight steering arriving during this (otherwise terminal)
                # round would never be drained at a later round-start, so it
                # would be silently dropped. Drain here: if anything is queued,
                # run another round to address it instead of completing.
                #
                # Residual micro-race (accepted-follow_up vs. terminal drain),
                # handled BEST-EFFORT by design: a follow_up enqueued in the
                # window *after* this drain returns empty but *before* the
                # caller releases the session lock is still accepted into the
                # in-process queue — within this process it is not lost (the
                # now-idle session's next send() drains it at round start). The
                # only loss is if the host process restarts in that window
                # before the next send drains it: the in-process queue is not
                # persisted, so the just-accepted item is dropped on restart.
                # We accept this rather than add cross-restart durability here:
                # the queue is intentionally in-process, and DeepTalk dispatch
                # is best-effort (restart seeks to end, no replay) with
                # consistency reestablished by api persistence + client seq
                # sync and the next ambient/explicit wake. See agent
                # session_folder.py ("in-process only; lost on restart").
                if self._drain_follow_ups is not None:
                    drained = await self._drain_follow_ups()
                    if drained:
                        # Close THIS (no-tools) round in the event stream + usage before steering
                        # reopens the loop — else the round is left unterminated and its per-round
                        # usage row is never written (pipeline-runner-3). Mirrors the terminal block.
                        self._emit(AgentEventType.ROUND_COMPLETED,
                                   RoundCompletedPayload(round_index=round_idx, has_tools=False),
                                   round_index=round_idx)
                        await self.hooks.run_typed_async(HookPoint.ROUND_END, RoundEndCtx(
                            round_index=round_idx, messages=self.history,
                            has_tools=False, response_text=assistant_text))
                        await self._emit_sink(self.sink.on_round_ended, round_idx, usage=usage)
                        for msg in drained:
                            await self._append_message(msg, round_index=round_idx)
                        continue
                self._emit(AgentEventType.ROUND_COMPLETED,
                           RoundCompletedPayload(round_index=round_idx, has_tools=False), round_index=round_idx)
                round_end = RoundEndCtx(
                    round_index=round_idx, messages=self.history,
                    has_tools=False, response_text=assistant_text,
                )
                await self.hooks.run_typed_async(HookPoint.ROUND_END, round_end)
                await self._emit_sink(self.sink.on_round_ended, round_idx, usage=usage)
                # ── Hook: COMPLETE_DECIDE (natural completion) ──
                if await self._complete_decide(
                    reason="completed", round_idx=round_idx, final_text=assistant_text,
                    next_round=round_idx + 1,
                ):
                    continue
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
                # Drive the terminal lifecycle before returning: every other terminal
                # return funnels through _finalize (SESSION_ENDED after SESSION_STARTED
                # + the end-of-run memory snapshot). Skipping it here stranded any
                # event subscriber and dropped the memory snapshot. _finalize is
                # idempotent, so this is safe.
                await self._finalize("pending_tools", final_text=assistant_text,
                                     rounds=round_idx + 1)
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
            # 6.11.0 同轮并发（design/86 修订）：模型面对「三张图」的直觉是同一轮批量发调用，
            # 而不是先起 background_run 再回头查（conv-224：三次 design_pack_image 同轮发出、
            # 逐个排队 3 分钟）。所以顺着它：同一轮 ≥2 个 async_capable 调用并发执行。
            # 不变量：TOOL_BEFORE 仍按原顺序先跑（闸类 hook 语义不变）；结果按原顺序回填；
            # TOOL_AFTER / 事件 / 落库全部串行；非 async_capable 工具永远串行。
            used_todo = False
            _conc = int(self.config.tool_batch_concurrency or 0)
            _eligible: list[int] = (
                [j for j, tc in enumerate(tool_calls)
                 if self._is_async_capable(_tool_call_name(tc), _tool_call_args(tc))]
                if (_conc > 1 and not skip_batch) else []
            )
            if len(_eligible) < 2:
                _eligible = []
            _pre: dict[int, tuple[asyncio.Future, Any]] = {}   # 已起任务：index → (task, tb_ctx)
            _pre_skip: dict[int, Any] = {}                       # TOOL_BEFORE 判 SKIP 的 hoisted 调用
            _pre_sem: asyncio.Semaphore | None = None
            for i, tool_call in enumerate(tool_calls):
                if _is_cancelled(self.cancel_token):
                    self._cancel_pre_tasks(_pre)
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

                pre_task: asyncio.Future | None = None
                if i in _pre_skip:
                    tb_ctx = _pre_skip.pop(i)
                    await self._append_message(
                        {"role": "tool", "tool_call_id": call_id, "name": tb_ctx.tool_name,
                         "content": tb_ctx.output},
                        round_index=round_idx,
                    )
                    continue
                if i in _pre:
                    pre_task, tb_ctx = _pre.pop(i)
                    tool_name = tb_ctx.tool_name
                    tool_args = tb_ctx.tool_args
                else:
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

                    if i in _eligible and any(j > i for j in _eligible):
                        # 起批：本调用 + 后面所有 eligible 调用一起起任务。后者的 TOOL_BEFORE 在这里
                        # 按序先跑（保序），判 SKIP 的不起任务，轮到它时直接落 SKIP 结果。
                        _pre_sem = _pre_sem or asyncio.Semaphore(_conc)
                        pre_task = self._spawn_tool_task(tool_name, tool_args, _pre_sem)
                        for j in _eligible:
                            if j <= i or j in _pre or j in _pre_skip:
                                continue
                            tc_j = tool_calls[j]
                            tb_j = ToolBeforeCtx(
                                round_index=round_idx, tool_call=tc_j,
                                tool_name=_tool_call_name(tc_j), tool_args=_tool_call_args(tc_j),
                            )
                            await self.hooks.run_typed_async(HookPoint.TOOL_BEFORE, tb_j)
                            if tb_j.directive == HookDirective.SKIP:
                                _pre_skip[j] = tb_j
                                continue
                            self._emit(AgentEventType.TOOL_CALL_STARTED,
                                       ToolCallStartedPayload(name=tb_j.tool_name, tool_input=tb_j.tool_args,
                                                              tool_call_id=str(tc_j.get("id") or "")),
                                       round_index=round_idx)
                            _pre[j] = (self._spawn_tool_task(tb_j.tool_name, tb_j.tool_args, _pre_sem), tb_j)

                # ── Business logic: execute tool ──
                failed = False
                try:
                    if pre_task is not None:
                        output, failed = await pre_task
                    else:
                        output, failed = await self.execute_tool(tool_name, tool_args)
                except HumanInputRequired as exc:
                    self._cancel_pre_tasks(_pre)
                    interaction = exc.to_pending(tool_call_id=call_id, tool_name=tool_name)
                    await self._persist_pending_interaction(interaction=interaction, round_index=round_idx)
                    # The model batched request_user_input with later tool_calls;
                    # those won't run. Resolve them so the resumed turn isn't an
                    # invalid sequence (assistant.tool_calls with no matching tool).
                    await self._resolve_skipped_tool_calls(
                        tool_calls[i + 1 :], reason="superseded by a user-input request",
                        round_idx=round_idx,
                    )
                    await self._finalize("waiting_for_input", final_text=assistant_text, rounds=round_idx + 1)
                    return self._make_result(
                        "waiting_for_input",
                        final_text=assistant_text,
                        rounds=round_idx + 1,
                        pending_tool_calls=tool_calls,
                        pending_interactions=[interaction],
                    )
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
                            # count=False: this is a retry of the SAME logical tool
                            # call, already counted on the first dispatch.
                            output, failed = await self.execute_tool(
                                tool_name, tool_args, count=False
                            )
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
                           ToolCallCompletedPayload(name=tool_name, output=output, tool_input=tool_args,
                                                    tool_call_id=call_id, failed=failed),
                           round_index=round_idx)

                await self._append_message(
                    {"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": _truncate_result(output)},
                    round_index=round_idx,
                )

                # TOOL_AFTER BREAK → stop remaining tools. Still resolve them so
                # the next round's request isn't an invalid sequence (assistant
                # with tool_calls that have no matching tool responses).
                if ta_ctx.directive == HookDirective.BREAK:
                    self._cancel_pre_tasks(_pre)
                    await self._resolve_skipped_tool_calls(
                        tool_calls[i + 1 :], reason="tool.after hook stopped the batch",
                        round_idx=round_idx,
                    )
                    break

            # 🔴 截断提示只能补在**所有 tool 结果都落完之后**。
            # 补在 assistant(tool_calls) 与它的 tool 结果之间，会造出
            # `assistant(tool_calls) → user → tool` 这种非法序列，下一次请求直接 400：
            # "An assistant message with 'tool_calls' must be followed by tool messages
            #  responding to each 'tool_call_id'" → 重试耗尽 → 整个 run 降级。
            # 真实事故 conv-215：第一次发卡片就死在这（提示本身是对的，位置错了）。
            # 同一条不变量在上面 TOOL_AFTER BREAK 处也写着——那里守住了，这里当初漏了。
            if _args_cut:
                await self._append_message(
                    {"role": "user", "content": _TRUNCATION_NOTICE}, round_index=round_idx)

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
            # Per-round usage row for TOOL rounds too. This was only emitted on the
            # no-tools paths above, so an agent session (where nearly every round calls
            # tools) persisted usage for just its final round — a 34-round leaf showed
            # exactly one usage_rounds row. Totals were never wrong (session_stats bumps
            # from the run's in-memory aggregate at end-of-send); what was missing is the
            # per-round breakdown this table exists for.
            await self._emit_sink(self.sink.on_round_ended, round_idx, usage=usage)

            if not used_todo:
                self.rounds_since_todo += 1

        # ── Hit max rounds ──
        await self._append_message({
            "role": "user",
            "content": f"You have reached the maximum of {self.config.max_rounds} rounds. "
                       f"Summarize what you accomplished and what remains.",
        })
        self._emit(AgentEventType.STATUS_CHANGED, HitRoundLimitStatusPayload(max_rounds=int(self.config.max_rounds)))

        max_rounds = int(self.config.max_rounds)
        # Honor cancellation before the (billable) summary call.
        if _is_cancelled(self.cancel_token):
            await self._finalize("cancelled")
            return self._make_result("cancelled", final_text="[cancelled by user]", rounds=max_rounds)
        # Route through call_llm so the final summary gets the same retry / timeout /
        # cancellation handling, per-loop model, and stream events as every other
        # call — instead of an unguarded llm.complete that could hang, ignore a
        # cancel, or crash the run on a transient error.
        try:
            final_resp = await self.call_llm(
                max_rounds,
                messages=self.history,
                system_prompt=self.system_prompt,
                tools=self.runtime_tools,
                max_tokens=int(self.config.max_tokens or 8000),
                temperature=float(self.config.temperature or 0),
            )
        except CancellationRequested as exc:
            self._emit(AgentEventType.LOOP_CANCELLED,
                       LoopCancelledPayload(reason=exc.reason, round_index=max_rounds))
            await self._finalize("cancelled")
            return self._make_result("cancelled", final_text=f"[cancelled: {exc.reason}]", rounds=max_rounds)
        except (LLMRetryExhausted, LLMTimeout) as exc:
            reason = "timeout" if isinstance(exc, LLMTimeout) else "retry_exhausted"
            inner = getattr(exc, "last_error", exc)
            self._emit(AgentEventType.LLM_DEGRADED, LlmDegradedPayload(
                reason=reason, attempts=getattr(exc, "attempts", 0),
                error_type=type(inner).__name__, error_message=str(inner)[:500]),
                round_index=max_rounds)
            msg = f"[degraded: LLM {reason} — {type(inner).__name__}: {str(inner)[:200]}]"
            await self._append_message({"role": "assistant", "content": msg}, round_index=max_rounds)
            await self._finalize("degraded", final_text=msg, rounds=max_rounds)
            return self._make_result("degraded", final_text=msg, rounds=max_rounds)
        final_text = (getattr(final_resp, "raw_text", "") or getattr(final_resp, "content_text", "") or "").strip()
        self._emit(AgentEventType.USAGE_UPDATED, UsageUpdatedPayload(usage=self.ctx.update_usage(final_resp)))
        # Persist the wrap-up summary as the assistant turn before finalizing (M-stateful-loop-2):
        # the success branch returned it to the caller but never recorded it, so the next send's
        # history had a dangling "summarize…" user prompt with no answer. Mirrors the degraded branch.
        if final_text:
            await self._append_message({"role": "assistant", "content": final_text}, round_index=max_rounds)
        await self._finalize("hit_round_limit", final_text=f"[hit_round_limit]\n{final_text}",
                             rounds=int(self.config.max_rounds))
        return self._make_result("hit_round_limit", final_text=f"[hit_round_limit]\n{final_text}",
                                 rounds=int(self.config.max_rounds))
