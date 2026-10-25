"""AgentPipeline — the core agent loop refactored into discrete, hookable phases.

Phase methods (``prepare_round``, ``call_llm``, ``execute_tool``) are pure
business logic with explicit parameters and return types.  All hook
orchestration, directive checks, and event publishing live in ``run()``.

The old ``agent_loop_async`` function is preserved in ``agent.py`` as a thin
wrapper that delegates to ``AgentPipeline.run()``.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
)
from power_loop.agent.sink import MessageSink, NullSink
from power_loop.agent.system_prompt import resolve_runtime_system_prompt
from power_loop.agent.types import AgentLoopConfig, AgentLoopResult, LoopMessage
from power_loop.contracts.errors import (
    CancellationRequested,
    InboxClaimChanged,
    LLMNonRetryable,
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
    SteerInterruptedPayload,
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
from power_loop.core.agent_context import (
    reset_current_cancel_token,
    set_current_cancel_token,
)
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.state import ContextManager
from power_loop.runtime.budget import estimate_message_tokens, estimate_tokens
from power_loop.runtime.cancellation import CancellationLike, CancellationToken
from power_loop.runtime.compact import CompactionContext
from power_loop.runtime.human_input import HumanInputRequired
from power_loop.runtime.image_recall import drain_queued_images
from power_loop.runtime.memory import MemorySnapshot
from power_loop.runtime.retry import _status_of, with_retry
from power_loop.runtime.usage_estimate import (
    estimate_completion_tokens,
    estimate_prompt_tokens,
)
from power_loop.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _persist_key(msg: Mapping[str, Any]) -> str:
    """Identity of a durable LLM_BEFORE injection (role + name + content), for Z8 dedupe."""
    try:
        body = json.dumps(msg.get("content"), ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        body = str(msg.get("content"))
    return f"{msg.get('role')}|{msg.get('name') or ''}|{body}"


def _failure_outcome(exc: BaseException) -> str:
    """How an LLM attempt ended, for the usage record: ``aborted`` (the call's task was
    cancelled — a retry deadline, steering, a stop), ``timeout`` (the transport gave up), or
    ``error``."""
    if isinstance(exc, asyncio.CancelledError):
        return "aborted"
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "timeout"
    return "error"

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
#: 只有思考、没有正文也没有工具调用时的提示。与上面那条分开：这里不是「写太长」，
#: 而是「想太久，预算全花在思考上了」——让它直接给结论，别再展开想。
_THINK_ONLY_NOTICE = (
    "[系统] 你上一轮**只产出了思考、没有任何正文和工具调用**——那一轮对外什么都没发生。"
    "思考内容也算在单轮长度上限里，想得太久就没有余量写答案了。\n"
    "这一轮直接给结论或直接调工具：先写要做的那一步，别再展开推演；"
    "确实需要很多步的，就只做第一步，做完再想下一步。"
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

RESULT_MAX_CHARS = 20000


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

def _llm_endpoint(llm: Any) -> str:
    """The base URL that serves this LLM (unwrapping retry / proxy wrappers), or ""."""
    for _ in range(5):
        if llm is None:
            break
        cfg = getattr(llm, "_cfg", None) or getattr(llm, "config", None)
        url = getattr(cfg, "base_url", None)
        if url:
            return str(url)
        llm = getattr(llm, "inner", None) or getattr(llm, "_inner", None) or getattr(llm, "llm", None)
    return ""


def _truncate_result(output: Any, limit: int | None = RESULT_MAX_CHARS) -> str:
    """Cap a tool result before it enters history and the audit log (None = no cap). Keeps the
    head and the tail: errors and summaries are usually at the end."""
    s = str(output)
    if limit is None or limit <= 0 or len(s) <= limit:
        return s
    note = f"\n\n... ({len(s) - limit} of {len(s)} chars omitted) ...\n\n"
    head = max(0, (limit - len(note)) // 2)
    tail = max(0, limit - len(note) - head)
    return s[:head] + note + (s[-tail:] if tail else "")


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
        steer_event: asyncio.Event | None = None,
    ) -> None:
        self.llm = llm
        self.config = config
        # design/188：pass_turn 永远在（宿主可以用同名工具覆盖说明和处理）。
        # 没有工具表 = 工具全由宿主执行（调用原样交回，pending_tools）：那种模式不补，宿主自己处理
        from power_loop.core.pass_turn import ensure_pass_turn
        self.tool_registry = ensure_pass_turn(tool_registry) if tool_registry is not None else None
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
        # design/124 §7: set while a steer-mode inbox item waits undelivered. The model call and
        # interruptible tools race it; the next round's drain delivers the item (and clears it).
        self._steer_event = steer_event
        self._model_policy_due = asyncio.Event()
        self._steer_aborts = 0                     # consecutive steer-aborted model calls
        # Z8: durable LLM_BEFORE injections already written by a round whose model call a steer
        # aborted — the re-run round's hooks may produce them again; they're in history already.
        self._aborted_persist_keys: set[str] = set()
        # design/124 §8.2: each running tool call's own stop token (child of the run's).
        self._tool_tokens: dict[asyncio.Future, CancellationToken] = {}

        self.runtime_tools = tool_registry.to_openai_tools() if tool_registry is not None else None
        from power_loop.runtime.tool_context import active_engine
        effective_context = getattr(active_engine.get(), 'config', config.tool_context)
        # Registered implementations own any model-facing context schema changes.
        if effective_context is not None:
            import copy
            self.runtime_tools = copy.deepcopy(self.runtime_tools)
            from power_loop.runtime.tool_contracts import context_spec
            for spec in self.runtime_tools or []:
                fn = spec.get('function', {})
                registered = self.tool_registry.get(fn.get('name'))
                if registered is not None:
                    spec.update(context_spec(registered, spec))
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
    ) -> bool:
        # An inbox delivery (design/124 §6) arrives with a private ``_inbox`` marker naming the
        # claimed inbox rows it carries. Strip it BEFORE hooks/history — the history row is sent
        # verbatim to the provider — and hand it only to the sink, which marks those rows
        # delivered in the same transaction as this transcript row.
        message = dict(msg)
        inbox = message.pop("_inbox", None)
        ctx = MessageAppendCtx(
            round_index=round_index or 0,
            message=message,
            session_id=self.session_id,
        )
        await self.hooks.run_typed_async(HookPoint.MESSAGE_APPEND, ctx)
        sink_msg = ctx.message
        if self.send_index is not None:
            sink_msg = {**sink_msg, "send_index": self.send_index}
        if hook_injected is not None:
            sink_msg = {**sink_msg, "hook_injected": hook_injected}
        if inbox is not None:
            sink_msg = {**sink_msg, "inbox": inbox}
            # Persist the claim before admitting any text/images/token estimates
            # to the in-memory request. Only a proven stale claim is retryable.
            try:
                await self._emit_sink(self.sink.on_message_appended, sink_msg, round_index=round_index)
            except InboxClaimChanged:
                return False
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
        if inbox is None:
            await self._emit_sink(self.sink.on_message_appended, sink_msg, round_index=round_index)
        return True

    async def _append_drained(self, drained: list[LoopMessage], *, round_index: int) -> int:
        """Reclaim stale batches at most three times, without dispatching a model."""
        committed = 0
        for attempt in range(3):
            stale = False
            for msg in drained:
                if await self._append_message(msg, round_index=round_index):
                    committed += 1
                else:
                    stale = True
            if not stale or attempt == 2 or self._drain_follow_ups is None:
                break
            drained = await self._drain_follow_ups()
        return committed

    # ── Steering (design/124 §7) ──

    #: Safety valve: more consecutive steer-aborts than this in one send and the model call
    #: runs to completion regardless (a steer flag that can never be cleared must not spin).
    MAX_STEER_ABORTS = 5
    STEER_SKIP_REASON = "为先处理用户的新消息，这个调用没有执行；需要的话重新调用"
    STEER_ABORT_TEXT = ("[interrupted: 为先处理用户的新消息，这个调用已中断，结果没有产生。"
                        "不是用户拒绝了它，需要的话可以重新执行。]")
    STEER_BACKGROUND_TEXT = ("[moved to background: 为先处理用户的新消息，这个调用已转到后台继续运行，"
                             "task_id={task_id}。完成后结果会自动送到你面前；不是用户拒绝了它，"
                             "也不要重新调用。]")

    def _steer_pending(self) -> bool:
        ev = self._steer_event
        return ev is not None and ev.is_set()

    def _interrupt_mode(self, tool_name: str) -> str:
        reg = self.tool_registry
        rt = reg.get(tool_name) if reg is not None else None
        definition = getattr(rt, "definition", None) if rt is not None else None
        return str(getattr(definition, "interrupt", "finish") or "finish")

    async def _race(self, work: asyncio.Future, *, steer: bool, policy_deadline: bool = False) -> tuple[Any, str]:
        """Await ``work`` unless steering (when ``steer``) or a stop arrives first.
        ``(result, "done")`` when it finished (its exception propagates); ``(None, "stop")`` /
        ``(None, "steer")`` otherwise — ``work`` is left RUNNING, the caller decides. A stop wins
        over a steer that arrives in the same instant."""
        waiters: dict[str, asyncio.Future] = {}
        if steer and self._steer_event is not None:
            waiters["steer"] = asyncio.ensure_future(self._steer_event.wait())
        if not self.cancel_token.is_never:
            waiters["stop"] = asyncio.ensure_future(self.cancel_token.wait())
        if policy_deadline:
            waiters["policy_deadline"] = asyncio.ensure_future(self._model_policy_due.wait())
        if not waiters:
            return await work, "done"
        try:
            done, _ = await asyncio.wait({work, *waiters.values()},
                                         return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # We are being cancelled (a stop, shutdown): take the work down WITH us and let
            # it finish its own cleanup first — a model call records its (estimated) usage
            # and closes its stream in its cancel path; leaving that to run after we are gone
            # reorders events and can lose the accounting entirely.
            for w in waiters.values():
                w.cancel()
            work.cancel()
            try:
                await asyncio.wait({work}, timeout=.1 if policy_deadline else 5)
            finally:
                if policy_deadline and not work.done():
                    from power_loop.runtime.task_runtime import _SEND
                    binding = _SEND.get()
                    binding.runtime.track_task(work, job_id=binding.lease.job_id)
            raise
        for w in waiters.values():
            if w not in done:
                w.cancel()
        if work in done:
            return work.result(), "done"
        if "stop" in waiters and waiters["stop"] in done:
            return None, "stop"
        if "policy_deadline" in waiters and waiters["policy_deadline"] in done:
            return None, "policy_deadline"
        return None, "steer"

    def _start_tool(
        self, tool_name: str, tool_args: dict[str, Any], sem: asyncio.Semaphore | None = None,
        *, tool_call_id: str | None = None,
    ) -> asyncio.Future:
        """Run one tool call as its own task with its OWN stop token (a child of the run's —
        design/124 §8.2), exposed to the handler as the current cancel token. Stopping the run
        reaches it; stopping just this call (e.g. once it was moved to the background) leaves
        the run alone."""
        tok = self.cancel_token.child()
        assistant_seq = getattr(self.sink, "persisted_assistant_seq", None)

        async def _run() -> tuple[str, bool]:
            if sem is None:
                return await self.execute_tool(tool_name, tool_args, cancel_token=tok,
                                               tool_call_id=tool_call_id, assistant_seq=assistant_seq)
            async with sem:
                return await self.execute_tool(tool_name, tool_args, cancel_token=tok,
                                               tool_call_id=tool_call_id, assistant_seq=assistant_seq)

        fut = asyncio.ensure_future(_run())
        self._tool_tokens[fut] = tok
        fut.add_done_callback(lambda f: self._tool_tokens.pop(f, None))
        return fut

    async def _settle_stopped_tool(
        self, work: asyncio.Future, mode: str,
    ) -> tuple[str, bool]:
        """The run was stopped while this tool ran (design/124 §8.4). Mechanical, bounded:
        ``abort`` → cancel now; a sub-task (``background``: its child token is already flipped,
        it winds itself down) → up to ``subtask_s``; ``finish`` → up to ``finish_tool_s`` for its
        real result. Past the limit the call is cancelled and the model is told so."""
        pol = self.config.stop_policy
        reason = self.cancel_token.reason
        if mode == "abort":
            work.cancel()
            await asyncio.wait({work}, timeout=pol.abort_tool_s)
            return (f"[stopped: 用户要求停止（{reason}），这个调用已中止，结果没有产生。]", False)
        grace = pol.subtask_s if mode == "background" else pol.finish_tool_s
        done, _ = await asyncio.wait({work}, timeout=grace)
        if work in done and not work.cancelled():
            try:
                out, failed = work.result()
            except Exception as exc:  # noqa: BLE001
                out, failed = f"Error: {exc}", True
            return (f"[stopped: 用户要求停止（{reason}）；这个调用在停止前收了尾，下面是它交回的结果]\n"
                    f"{out}", failed)
        work.cancel()
        await asyncio.wait({work}, timeout=pol.abort_tool_s)
        return (f"[stopped: 用户要求停止（{reason}）；这个调用 {grace:.0f} 秒内没有结束，已强制中止，"
                "结果不完整或没有。]", False)

    async def _settle_steered_tool(
        self, work: asyncio.Future, mode: str, tool_name: str, tool_args: dict[str, Any],
        call_id: str, round_idx: int,
    ) -> str:
        """A steer arrived while this tool ran: ``abort`` cancels it, ``background`` hands the
        running call to the background task table. Returns the synthetic result for the model."""
        task_id = ""
        if mode == "background":
            from power_loop.tools.default_tools import BG

            task_id = await BG.adopt(tool_name, tool_args, work,
                                     stop_token=self._tool_tokens.get(work))
            output = self.STEER_BACKGROUND_TEXT.format(task_id=task_id)
        else:
            work.cancel()
            with contextlib.suppress(BaseException):
                await work
            output = self.STEER_ABORT_TEXT
        self._emit(AgentEventType.STEER_INTERRUPTED, SteerInterruptedPayload(
            where="tool", action=mode, round_index=round_idx, tool_name=tool_name,
            tool_call_id=call_id, task_id=task_id), round_index=round_idx)
        return output

    async def _restart_round_for_steer(self, round_idx: int, llm_before: Any) -> None:
        """Close a round whose model call a steer aborted, so the next round (whose drain
        delivers the steer) starts clean: paired round events, no usage row (the aborted call's
        estimate is already in the run totals), the round not counted against the limit, and the
        round's durable LLM_BEFORE injections remembered so they aren't written twice (Z8)."""
        self._steer_aborts += 1
        self._aborted_persist_keys.update(
            _persist_key(m) for m in getattr(llm_before, "persist_messages", ()) or ())
        self._emit(AgentEventType.STEER_INTERRUPTED, SteerInterruptedPayload(
            where="llm", action="restart", round_index=round_idx), round_index=round_idx)
        self._emit(AgentEventType.ROUND_COMPLETED,
                   RoundCompletedPayload(round_index=round_idx, has_tools=False),
                   round_index=round_idx)
        await self.hooks.run_typed_async(HookPoint.ROUND_END, RoundEndCtx(
            round_index=round_idx, messages=self.history, has_tools=False, response_text=""))
        await self._emit_sink(self.sink.on_round_ended, round_idx, usage=None)
        self._round_limit += 1

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
        st = getattr(self, "_pass_state", None) or {}
        texts = list(getattr(self, "_step_texts", []) or [])
        # Recover an earlier answer only within the latest input. A queued
        # correction followed by an empty pass must not reuse stale text.
        if st.get("passed") and not final_text and texts:
            for message in reversed(self.history):
                if message.get("role") == "user":
                    break
                content = message.get("content")
                if message.get("role") == "assistant" and isinstance(content, str) and content.strip():
                    final_text = content.strip()
                    break
        return AgentLoopResult(
            status=status,  # type: ignore[arg-type]
            final_text=final_text,
            rounds=rounds,
            pending_tool_calls=pending_tool_calls or [],
            pending_interactions=pending_interactions or [],
            messages=self.history,
            usage=dict(self.ctx.usage_totals),
            tool_calls=int(self.ctx.tool_calls),
            passed=bool(st.get("passed")),
            pass_reason=st.get("reason"),
            new_input_after_pass=bool(st.get("new_input_after_pass")),
            texts=texts,
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
        # A model that cannot see images was shown a "you did not see it" placeholder, not the
        # picture — retiring it as "已看过" would tell it the opposite. Unknown → the old wording.
        caps = getattr(self.llm, "capabilities", None)
        if caps is not None and hasattr(caps, "for_model"):
            caps = caps.for_model(self.config.model)
        blind = caps is not None and getattr(caps, "supports_image_input", None) is not True
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
                    # keep the host's recall coordinate (file_uuid=…) — it is how the image is fetched again
                    coord = f" · {att['ref']}" if att.get("ref") else ""
                    if blind and att.get("kind") == "image":
                        tail = "当前模型看不了图片，没看到过它的内容"
                    else:
                        tail = "已看过；要再看调 see_image"
                    new_blocks.append({"type": "text",
                                       "text": f"{self._IMAGE_RETIRED_MARK}: {name}{coord} — {tail}]"})
                    n += 1
                else:
                    new_blocks.append(b)
            m["content"] = new_blocks
        if n:
            logger.info("image retention: retired %d image block(s) older than %d round(s) (round %d)",
                        n, keep_rounds, round_index)
        return n

    _DISTILL_MARK = "[distilled #"

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

    def _distill_oldest_tool_rows(self, batch: int, hot_tail: int, *, target_tokens: int | None = None) -> int:
        """Distill complete calls oldest first; ``batch`` is ignored for compatibility.

        Protect invocation IDs, not result rows: parallel results may arrive out of order.
        Only replace in-memory copies, keeping persisted audit records and seq alignment intact.
        """
        rep = self.config.representation
        distill = getattr(rep, "distill_tool_call", None)
        if getattr(rep, "kind", "projection") == "verbatim" or distill is None:
            return 0
        seqs = getattr(self.sink, "_history_seqs", None) or []
        start = min(int(getattr(self, "_send_start_len", 0) or 0), len(self.history))
        calls = []
        results = {}
        for i in range(start, len(self.history)):
            row = self.history[i]
            if row.get("role") == "assistant":
                for j, tc in enumerate(row.get("tool_calls") or []):
                    if tc.get("id"):
                        calls.append((str(tc["id"]), i, j))
            elif row.get("role") == "tool":
                results[str(row.get("tool_call_id") or "")] = i
        protected = {cid for cid, _, _ in calls[-hot_tail:]} if hot_tail > 0 else set()
        # Optional host protection for individual calls (e.g. bash loading persistent skills).
        protected.update(getattr(self, "_context_protected_call_ids", ()))
        keep = set(self.config.insend_distill_keep_tools or ())
        freed_chars = 0
        for cid, i, j in calls:
            if cid in protected or cid not in results:
                continue
            ri = results[cid]
            content = self.history[ri].get("content")
            tc = self.history[i]["tool_calls"][j]
            fn = tc.get("function") or {}
            raw = fn.get("arguments")
            name = str(fn.get("name") or "tool")
            if (name in keep or not isinstance(content, str) or not isinstance(raw, str)
                    or content.startswith(self._DISTILL_MARK)):
                continue
            try:
                args = json.loads(raw)
                line = distill(name, args, content, getattr(self, "tool_registry", None))
            except Exception:  # a broken projector must not stop the run or starve later calls
                logger.warning("in-send distillation failed for %s", name, exc_info=True)
                continue
            # recall_send needs the result row; it includes the matching call arguments.
            seq = seqs[ri] if ri < len(seqs) else None
            coord = (f"recall_send(send_index={self.send_index}, seq={seq})"
                     if self.send_index is not None and seq is not None else "recall_send")
            new_content = f"{self._DISTILL_MARK}{self.send_index} seq={seq}; {coord}] {line}"
            new_raw = json.dumps({"_distilled": coord}, ensure_ascii=False)
            saved = len(raw) + len(content) - len(new_raw) - len(new_content)
            if saved <= 0:
                continue
            new_calls = list(self.history[i]["tool_calls"])
            new_calls[j] = {**tc, "function": {**fn, "arguments": new_raw}}
            self.history[i] = {**self.history[i], "tool_calls": new_calls}
            self.history[ri] = {**self.history[ri], "content": new_content}
            freed_chars += saved
            if target_tokens is not None and freed_chars // 4 >= target_tokens:
                break
        return freed_chars // 4

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

        from power_loop.runtime.tool_context import active_engine
        context_engine = active_engine.get()
        if context_engine is not None:
            retired = await context_engine.prepare_history(self, "age_retirement")
            if retired:
                self._tok_len = -1
            anchor = getattr(self, "_tool_context_prompt_anchor", None)
            if anchor is not None:
                self._context_prompt_tokens_estimate = max(
                    0, anchor[0] + self._estimate_history_tokens() - anchor[1])

        if context_engine is not None:
            # The in-send fuse (trigger/target, two levels). The only fuse when tool context is on.
            from power_loop.runtime.tool_archive import manage_capacity
            await manage_capacity(context_engine, self)

        # Without tool context: the plain fuse over total prompt size (fixed prompt + history).
        dt = self.config.insend_distill_tokens
        if context_engine is None and dt is not None and int(dt) > 0 and (
            round_index > 0 or getattr(self, "_context_prompt_tokens_estimate", None) is not None
        ):
            real = int((self.ctx.token_usage or {}).get("prompt_tokens", 0) or 0)
            if real and self._send_baseline_prompt is None:
                self._send_baseline_prompt = real  # diagnostics only; never subtracted from trigger
            # Optional host estimate after context retirement. Never mutate provider
            # usage: the latter is billing evidence for the previous request.
            current_estimate = getattr(self, "_context_prompt_tokens_estimate", None)
            total = (max(0, int(current_estimate)) if current_estimate is not None
                     else real or self._estimate_history_tokens())
            if total >= int(dt):
                freed = self._distill_oldest_tool_rows(
                    0, int(self.config.insend_distill_hot_tail or 0),
                    target_tokens=max(1, total - int(dt) + 1),
                )
                if freed:
                    self._tok_len = -1
                logger.info("insend distill: total prompt %d >= %d; freed ~%d tokens",
                            total, int(dt), freed)
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

        Current state projections are ephemeral. Completed background tool results
        opt into durable history with ``_persist`` so later sends retain their
        original-tool summary after the notification is acknowledged.
        """
        if self.store is None or self.session_id is None:
            todo_snap = self.ctx.todo.snapshot_for_prompt()
            return [{"role": "user", "content": todo_snap}] if todo_snap else []

        messages: list[LoopMessage] = []
        # A projection is only "consumed" once the model saw it: projectors hand back a private
        # ``_ack`` marker, stripped here and settled by _acknowledge_runtime() after a SUCCESSFUL
        # LLM call. A failed call leaves nothing acknowledged, so the next round re-projects.
        self._pending_acks = []
        for projector in self.config.runtime_projectors:
            projected = await projector.project(
                store=self.store,
                session_id=self.session_id,
                round_index=round_index,
                context=self.ctx,
            )
            for msg in projected:
                m = dict(msg)
                ack = m.pop("_ack", None)
                if ack is not None and hasattr(projector, "acknowledge"):
                    self._pending_acks.append((projector, ack))
                if m.pop("_persist", False):
                    if not any(_persist_key(old) == _persist_key(m) for old in self.history):
                        await self._append_message(m, round_index=round_index)
                else:
                    messages.append(m)
        return messages

    async def _acknowledge_runtime(self) -> None:
        """Settle this round's projection acks (the model saw the projected state)."""
        pending, self._pending_acks = getattr(self, "_pending_acks", []), []
        for projector, ack in pending:
            try:
                await projector.acknowledge(store=self.store, session_id=self.session_id, ack=ack)
            except Exception:  # an ack failure only means the state is shown once more
                logger.warning("runtime projection ack failed", exc_info=True)

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
        # What THIS attempt has streamed so far — the only record of billed output if the call
        # dies before its final (usage-carrying) chunk (design/124 §10). Reset per attempt.
        streamed: dict[str, list[str]] = {"text": [], "think": []}

        def _on_delta(text: str) -> None:
            if text:
                streamed["text"].append(text)
                self._emit(AgentEventType.STREAM_DELTA,
                           StreamDeltaPayload(text=text, is_think=False),
                           round_index=round_index, stream_id="main")

        def _on_think(text: str) -> None:
            if text:
                streamed["think"].append(text)
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

        # The model that actually served the call: the per-loop override if set, else the
        # service's own configured model (design/124 U6 — "" left 61% of usage rows unattributed).
        model_name = (self.config.model
                      or getattr(getattr(self.llm, "capabilities", None), "model", None) or "")
        attempt_box = [0]

        async def _do_call() -> LLMResponse:
            from power_loop.workflow.admission import request_slot
            async with request_slot(self.cancel_token):
                return await _perform_call()

        async def _perform_call() -> LLMResponse:
            from power_loop.runtime.task_runtime import check_dispatch
            # STREAM_STARTED/COMPLETED are emitted **per attempt** and paired in this call's own
            # finally (M-pipeline-runner-1): the terminal used to live in the OUTER finally, which
            # ran once, so N retry attempts emitted N STARTED but only 1 COMPLETED — unbalanced
            # stream events on any LLM retry. Each attempt now opens and closes exactly one stream.
            attempt_box[0] += 1
            attempt = attempt_box[0]
            call_id = f"r{round_index}.a{attempt}"
            from power_loop.runtime.model_attempts import (
                event_identity,
                observe,
                provider_deadline,
                response_observation,
                start_call,
            )
            try:
                await check_dispatch(self.session_id)
                model_attempt = await start_call(self, request, round_index=round_index, retry_index=attempt, model=model_name)
            except Exception as exc:
                # Durable admission failure is not permission for the retry
                # policy to allocate a replacement request/identity.
                raise LLMNonRetryable(attempts=attempt, last_error=exc) from exc
            durable_identity = event_identity(model_attempt)
            streamed["text"].clear()
            streamed["think"].clear()
            self._emit(AgentEventType.STREAM_STARTED, StreamStartedPayload(),
                       round_index=round_index, stream_id="main")
            policy_alarm = None
            try:
                # Per-call observability (H4.1): pair STARTED/COMPLETED by call_id so a
                # subscriber sees per-attempt latency + per-call (not cumulative) usage,
                # making retries individually visible.
                self._emit(
                    AgentEventType.LLM_CALL_STARTED,
                    LlmCallStartedPayload(call_id=call_id, round_index=round_index,
                                          attempt=attempt, model=model_name, **durable_identity),
                    round_index=round_index, stream_id="main",
                )
                t0 = time.perf_counter()
                policy_clock = None
                provider_started = False
                try:
                    from power_loop._vendor.llm_client.attempt_scope import single_attempt
                    attempt_token = single_attempt.set(True) if model_attempt is not None else None
                    try:
                        deadline = await provider_deadline(model_attempt)
                        if deadline is not None and deadline <= asyncio.get_running_loop().time():
                            from power_loop.runtime.jobs import JobError
                            raise JobError("owner_wait_deadline_exceeded")
                        policy_clock = asyncio.timeout_at(deadline)
                        async with policy_clock:
                            if deadline is not None:
                                policy_alarm = asyncio.get_running_loop().call_at(deadline, self._model_policy_due.set)
                            provider_started = True
                            resp = await self.llm.complete(
                                request, on_chunk_delta_text=_on_delta, on_chunk_think=_on_think,
                            )
                    finally:
                        if attempt_token is not None:
                            single_attempt.reset(attempt_token)
                except BaseException as exc:
                    if isinstance(exc, TimeoutError) and policy_clock is not None and policy_clock.expired():
                        from power_loop.runtime.jobs import JobError
                        exc = JobError("owner_wait_deadline_exceeded")
                    # No usage chunk will ever come for this attempt, but it was billed: prompt
                    # in full once the provider started on it, plus what it streamed before the
                    # failure (design/124 §11.2). A request REJECTED with an HTTP status before any
                    # output (402, 401, 400…) was never processed → prompt 0.
                    outcome = _failure_outcome(exc)
                    produced = "".join(streamed["text"]), "".join(streamed["think"])
                    # Managed attempts lack a provider rejection receipt: an
                    # HTTP status alone cannot release their unknown risk.
                    rejected = model_attempt is None and _status_of(exc) is not None and not any(produced)
                    p_est = 0 if rejected or not provider_started else estimate_prompt_tokens(
                        messages=messages, system_prompt=system_prompt, tools=tools)
                    c_est = estimate_completion_tokens(text=produced[0], think=produced[1])
                    self.ctx.add_unreported_usage(p_est, c_est)
                    await observe(model_attempt, usage={"prompt_tokens": p_est, "completion_tokens": c_est,
                        "total_tokens": p_est + c_est}, estimated=True, error=exc)
                    self._emit(
                        AgentEventType.LLM_CALL_COMPLETED,
                        LlmCallCompletedPayload(
                            call_id=call_id, round_index=round_index, attempt=attempt,
                            model=model_name, duration_ms=(time.perf_counter() - t0) * 1000.0,
                            success=False, error_type=type(exc).__name__,
                            prompt_tokens=p_est, completion_tokens=c_est,
                            total_tokens=p_est + c_est, estimated=True, outcome=outcome,
                            endpoint=_llm_endpoint(self.llm),
                            **durable_identity,
                        ),
                        round_index=round_index, stream_id="main",
                    )
                    if model_attempt is not None and isinstance(exc, Exception):
                        raise LLMNonRetryable(attempts=attempt, last_error=exc) from exc
                    raise
                usage = getattr(resp, "token_usage", None)
                estimated = usage is None or getattr(usage, "prompt_tokens", None) is None
                if estimated:
                    # The provider sent no usage for a call that completed (U1: we no longer
                    # borrow the service's previous call's numbers) — estimate and say so.
                    p_est = estimate_prompt_tokens(
                        messages=messages, system_prompt=system_prompt, tools=tools)
                    c_est = estimate_completion_tokens(
                        text=getattr(resp, "raw_text", "") or "",
                        think=getattr(resp, "think", "") or "",
                        tool_calls=getattr(resp, "tool_calls", None) or ())
                    usage = LLMTokenUsage(prompt_tokens=p_est, completion_tokens=c_est,
                                          total_tokens=p_est + c_est)
                    resp.token_usage = usage
                    self.ctx.note_estimated_call(p_est + c_est)
                if model_attempt is not None:
                    await observe(model_attempt, **response_observation(usage, resp), estimated=estimated)
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
                        estimated=estimated,
                        endpoint=_llm_endpoint(self.llm),
                        provider_cost=getattr(usage, "cost", None),
                        prompt_image_tokens=getattr(usage, "prompt_image_tokens", None),
                        **durable_identity,
                    ),
                    round_index=round_index, stream_id="main",
                )
                if policy_clock is not None and policy_clock.expired():
                    # A provider may suppress cancellation and return late. Keep
                    # its actual usage above, but never use it as an on-time answer.
                    from power_loop.runtime.jobs import JobError
                    raise LLMNonRetryable(attempts=attempt, last_error=JobError("owner_wait_deadline_exceeded"))
                return resp
            finally:
                if policy_alarm is not None:
                    policy_alarm.cancel()
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
        self, tool_name: str, tool_args: dict[str, Any], sem: asyncio.Semaphore,
        *, tool_call_id: str | None = None,
    ) -> asyncio.Future:
        """同轮并发的一个成员：受信号量限流地跑 execute_tool（自带子停止标记，见 _start_tool）。
        contextvars 随 task 自动拷贝（PEP 567），计费 / 活动打标 / 运行时上下文零处理。异常原样
        留在 task 里，轮到它时再由串行路径的 HumanInputRequired / TOOL_ERROR 分支处理。"""
        return self._start_tool(tool_name, tool_args, sem, tool_call_id=tool_call_id)

    @staticmethod
    def _cancel_pre_tasks(pre: dict[int, tuple[asyncio.Future, Any]]) -> None:
        for task, _ in pre.values():
            if not task.done():
                task.cancel()
        pre.clear()

    async def execute_tool(
        self, tool_name: str, tool_args: dict[str, Any], *, count: bool = True,
        cancel_token: CancellationToken | None = None,
        tool_call_id: str | None = None, assistant_seq: int | None = None,
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
            # design/124 §8.2: the handler sees this call's stop token (a sub-run it starts
            # derives from it). Default: a fresh child of the run's token.
            _ctok = set_current_cancel_token(cancel_token or self.cancel_token.child())
            try:
                from power_loop.runtime.invocation import invocation_context
                with invocation_context(self.session_id,
                        assistant_seq if assistant_seq is not None else getattr(self.sink, "persisted_assistant_seq", None),
                        tool_call_id):
                    from power_loop.runtime.task_runtime import check_tool_dispatch
                    registered = await check_tool_dispatch(self.session_id, tool_name, self.tool_registry, tool_args)
                    from power_loop.runtime.tool_contracts import validate_registration
                    validate_registration(self.tool_registry, tool_name, registered)
                    result = await self.tool_registry.invoke_async(tool_name, tool_args)
            finally:
                reset_current_cancel_token(_ctok)
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
        from power_loop.core import pass_turn as _pass

        token = _pass.begin()          # pass_turn 的状态按 send 隔离（子 agent 内联跑也不串）
        self._pass_state = _pass.current()
        self._step_texts: list[dict[str, Any]] = []
        try:
            return await self._run(messages)
        finally:
            try:
                from power_loop.runtime.task_runtime import preserve_model_usage
                preserve_model_usage(self.session_id, self.ctx.usage_totals)
            finally:
                _pass.end(token)

    async def _run(self, messages: list[LoopMessage]) -> AgentLoopResult:
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
            from power_loop.agent.stateful_loop import take_round_bonus
            _bonus = take_round_bonus(self.session_id)
            if _bonus:
                self._round_limit += _bonus   # 宿主在运行中加了轮数（extend_rounds）
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
            # Fuse and cut are exclusive: a running fuse (tool-context capacity or the plain
            # fuse) turns cut-send off.
            from power_loop.runtime.tool_context import active_engine as _cut_engine
            _fuse_engine = _cut_engine.get()
            ckpt = self.config.context_checkpoint_tokens
            if (
                ckpt is not None
                and not (_fuse_engine is not None and _fuse_engine.config.capacity_enabled)
                and (self.config.insend_distill_tokens or 0) <= 0
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
                drained_count = await self._append_drained(drained, round_index=round_idx)

            # ── pass_turn：上一轮调了 → 这一轮开始前硬停（design/188，不再调模型）──
            if self._pass_state and self._pass_state.get("passed"):
                if drained_count:
                    # 停的这一刻有刚折进来的新输入：不撤销停，留痕交给宿主（另起一轮处理）
                    self._pass_state["new_input_after_pass"] = True
                await self._finalize("pass_turn")
                return self._make_result("completed", rounds=round_idx)

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
            if getattr(self, '_capacity_blocked', None):
                await self._finalize('capacity_blocked', rounds=round_idx)
                return self._make_result('degraded', final_text=self._capacity_blocked, rounds=round_idx)

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
                _pk = _persist_key(_pm)
                if _pk in self._aborted_persist_keys:
                    # Z8: written by the round a steer aborted — already in history (and in
                    # llm_before.messages via self.history); don't write it a second time.
                    self._aborted_persist_keys.discard(_pk)
                    continue
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

            from power_loop.runtime.tool_context import active_engine
            _admission_engine = active_engine.get()
            if (_admission_engine is not None
                    and llm_before.directive not in (HookDirective.SHORT_CIRCUIT, HookDirective.BREAK)):
                from power_loop.runtime.tool_archive import admit_request
                if not await admit_request(_admission_engine, self, llm_before.messages,
                        llm_before.system_prompt, llm_before.tools, llm_before.max_tokens):
                    await self._finalize('capacity_blocked', rounds=round_idx)
                    return self._make_result('degraded', final_text=self._capacity_blocked, rounds=round_idx)

            from power_loop.runtime.task_runtime import prepare_model_processing
            await prepare_model_processing(self, llm_before.messages, round_idx,
                provider_call=llm_before.directive not in (HookDirective.SHORT_CIRCUIT, HookDirective.BREAK))

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
                    _llm_call = self.call_llm(
                        round_idx,
                        messages=llm_before.messages,
                        system_prompt=llm_before.system_prompt,
                        tools=llm_before.tools,
                        max_tokens=llm_before.max_tokens,
                        temperature=llm_before.temperature,
                    )
                    # design/124 §7.2: the model call races incoming steering. A steer flag
                    # already up at this point survived the round-start drain (store blip) —
                    # racing it would abort at once, forever; so only a steer that arrives
                    # DURING the call interrupts it, and only a bounded number of times.
                    # design/124 §8.4: a STOP interrupts the model call too (immediately — its
                    # output is discarded like a steered one, the run ends cancelled).
                    from power_loop.runtime.model_attempts import is_managed
                    _managed_call = is_managed(self.session_id)
                    _steer_ok = (not _managed_call and self._steer_event is not None and not self._steer_event.is_set()
                                 and self._steer_aborts < self.MAX_STEER_ABORTS)
                    if _steer_ok or _managed_call or not self.cancel_token.is_never:
                        self._model_policy_due.clear()
                        _llm_work = asyncio.ensure_future(_llm_call)
                        response, _why = await self._race(_llm_work, steer=_steer_ok, policy_deadline=_managed_call)
                        if _why == "policy_deadline":
                            from power_loop.runtime.task_runtime import _SEND
                            _binding = _SEND.get()
                            _binding.runtime.track_task(_llm_work, job_id=_binding.lease.job_id)
                            # The timeout already signalled the exact request once.
                            # Keep an uncooperative call reachable, with its durable
                            # attempt unresolved, instead of awaiting it indefinitely
                            # or cancelling its usage cleanup a second time.
                            await asyncio.wait({_llm_work}, timeout=.1)
                            msg = "[timed out: original continuation policy deadline reached]"
                            await self._finalize("timed_out", final_text=msg, rounds=round_idx + 1)
                            return self._make_result("timed_out", final_text=msg, rounds=round_idx + 1)
                        if _why != "done":
                            # Discard the call: nothing it produced reaches history (the
                            # response is only assembled at its end), its partial usage is
                            # recorded as an estimate by call_llm's own failure path.
                            _llm_work.cancel()
                            if _managed_call:
                                try:
                                    await asyncio.wait({_llm_work}, timeout=.1)
                                finally:
                                    if not _llm_work.done():
                                        from power_loop.runtime.task_runtime import _SEND
                                        _binding = _SEND.get()
                                        _binding.runtime.track_task(_llm_work, job_id=_binding.lease.job_id)
                            if _llm_work.done() or not _managed_call:
                                with contextlib.suppress(BaseException):
                                    await _llm_work
                            if _why == "stop":
                                raise CancellationRequested(self.cancel_token.reason)
                            await self._restart_round_for_steer(round_idx, llm_before)
                            continue
                    else:
                        response = await _llm_call
                    self._steer_aborts = 0
                    await self._acknowledge_runtime()
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
                    if getattr(getattr(exc, "last_error", exc), "code", None) == "owner_wait_deadline_exceeded":
                        msg = "[timed out: original continuation policy deadline reached]"
                        await self._finalize("timed_out", final_text=msg, rounds=round_idx + 1)
                        return self._make_result("timed_out", final_text=msg, rounds=round_idx + 1)
                    reason = (
                        "timeout" if isinstance(exc, LLMTimeout)
                        else "non_retryable" if isinstance(exc, LLMNonRetryable)
                        else "retry_exhausted"
                    )
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
            if self.config.tool_context is not None:
                prompt = int((self.ctx.token_usage or {}).get("prompt_tokens", 0) or 0)
                self._tool_context_prompt_anchor = (prompt, self._estimate_history_tokens()) if prompt else None
                self._context_prompt_tokens_estimate = None
            self._emit(AgentEventType.STATUS_CHANGED, _round_usage_payload(
                round_index=round_idx, max_rounds=int(self.config.max_rounds), usage=usage,
            ), round_index=round_idx)
            self._emit(AgentEventType.USAGE_UPDATED, UsageUpdatedPayload(usage=usage), round_index=round_idx)

            assistant_text = (getattr(response, "raw_text", "") or getattr(response, "content_text", "") or "").strip()
            tool_calls = response.get_tool_calls()
            self._emit(AgentEventType.ROUND_TOOLS_PRESENT, RoundToolsPresentPayload(has_tools=bool(tool_calls)), round_index=round_idx)

            # 每一步的文字（design/188）：模型每一轮都可能写字，全部留着交给宿主（不只是最后一段）
            if assistant_text:
                self._step_texts.append({"step": round_idx + 1, "text": assistant_text,
                                         "tools": [_tool_call_name(tc) for tc in (tool_calls or [])]})
            from power_loop.core import pass_turn as _pass_state
            _pass_state.note_step(bool(assistant_text), [_tool_call_name(tc) for tc in (tool_calls or [])])

            # Any productive round (said something OR called a tool) clears the empty streak.
            if assistant_text or tool_calls:
                self._empty_response_streak = 0
            if tool_calls:
                self._truncated_streak = 0

            # Append assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_text}
            # 🔴 思考内容要原样带回去。DeepSeek 的思考模式（deepseek-flash 等）明确要求：
            # 带过 reasoning_content 的 assistant 消息，下一次请求必须把它一起回传，否则 400
            # "The `reasoning_content` in the thinking mode must be passed back to the API."
            # ——那是个硬错，重试同样的历史只会再错一遍，最后 retry_exhausted 整个 run 降级
            # （真实事故：会话刚发完第一张卡片就死在这里）。
            #
            # 只在**这次回复真的带了**思考内容时才回传：产生它的供应商必然收得回去；
            # 没产生的（OpenAI 正统、Anthropic 走自己的 thinking 块）什么都不加，行为不变。
            _think = (getattr(response, "think", "") or "").strip()
            if _think:
                assistant_msg["reasoning_content"] = _think
            # 第二道：正文、工具调用、思考**三样都没有**的那一轮，历史里不要留这条空 assistant 行。
            # 它一个字的信息都不带，却会让下一次请求以一条空 assistant 消息结尾——在 DeepSeek 的
            # 思考模式下那是非法的（见上），别的供应商也只是白占一行。下面的空回复分支会重试这一轮。
            _skip_empty = not assistant_text and not tool_calls and not _think
            sanitized_tool_calls: list[dict[str, Any]] | None = None
            _args_cut = False
            if tool_calls:
                sanitized_tool_calls, _args_cut = _sanitize_tool_calls(tool_calls)
                assistant_msg["tool_calls"] = sanitized_tool_calls
            if not _skip_empty:
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
                # 🔴 第三条判据：**有思考、却没正文也没工具调用**。
                # 开了思考的模型把整轮预算花在思考上就是这个样子，而流式响应里 finish_reason 与
                # usage 常常都取不到（真实事故：子 agent cap=3000，上面两条判据双双落空，
                # 于是被当成「打嗝」原样重试——重试当然又是一轮只有思考的空回复）。
                # 无论是不是真被 max_tokens 切的，结论一样：**输入不变，输出不会变**，
                # 所以走「改变输入」这条路，而不是重试。
                think_only = bool(_think) and not assistant_text
                truncated = (reason in _TRUNCATION_FINISH_REASONS
                             or (bool(cap) and out_tokens >= int(cap))
                             or think_only)
                if truncated and self._truncated_streak < _TRUNCATED_MAX_RETRIES:
                    self._truncated_streak += 1
                    logger.warning(
                        "%s at round %d (finish_reason=%r, "
                        "completion_tokens=%d/%s) streak=%d/%d — nudging to write smaller "
                        "(session=%s)",
                        "think-only response (no text, no tool call)" if think_only
                        else "truncated LLM response",
                        round_idx, reason, out_tokens, cap, self._truncated_streak,
                        _TRUNCATED_MAX_RETRIES, self.session_id,
                    )
                    await self._append_message(
                        {"role": "user",
                         "content": _THINK_ONLY_NOTICE if think_only else _TRUNCATION_NOTICE},
                        round_index=round_idx)
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
                # A terminal response still consumes freshly committed follow-ups. Items
                # arriving after this drain stay in the durable inbox for a later flush;
                # a withdrawn claim must not create a new paid round or a duplicate end.
                round_ended = False
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
                        round_ended = True
                        if await self._append_drained(drained, round_index=round_idx):
                            continue
                if not round_ended:
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

                # design/124 §7.3: steering is waiting → don't START anything new this batch.
                # (Calls already running concurrently go through the normal path below, where
                # their own interrupt mode decides.) The next round's drain delivers the steer.
                if self._steer_pending() and i not in _pre and i not in _pre_skip and not skip_batch:
                    await self._resolve_skipped_tool_calls(
                        [tool_call], reason=self.STEER_SKIP_REASON, round_idx=round_idx)
                    self._emit(AgentEventType.STEER_INTERRUPTED, SteerInterruptedPayload(
                        where="between_tools", action="skip", round_index=round_idx,
                        tool_name=tool_name, tool_call_id=call_id, skipped=1),
                        round_index=round_idx)
                    continue

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
                        pre_task = self._spawn_tool_task(tool_name, tool_args, _pre_sem, tool_call_id=call_id)
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
                            _pre[j] = (self._spawn_tool_task(tb_j.tool_name, tb_j.tool_args, _pre_sem,
                                                          tool_call_id=str(tc_j.get("id") or "")), tb_j)

                # ── Business logic: execute tool ──
                failed = False
                _stopped = False
                try:
                    _mode = self._interrupt_mode(tool_name)
                    _steerable = self._steer_event is not None and _mode != "finish"
                    if _steerable or not self.cancel_token.is_never:
                        # design/124 §7.3 / §8.4: race the call against steering (if its mode
                        # allows) and against a stop (always).
                        _work = (pre_task if pre_task is not None
                                 else self._start_tool(tool_name, tool_args, tool_call_id=call_id))
                        _res, _why = await self._race(_work, steer=_steerable)
                        if _why == "done":
                            output, failed = _res
                        elif _why == "steer":
                            output = await self._settle_steered_tool(
                                _work, _mode, tool_name, tool_args, call_id, round_idx)
                        else:
                            output, failed = await self._settle_stopped_tool(_work, _mode)
                            _stopped = True
                    elif pre_task is not None:
                        output, failed = await pre_task
                    else:
                        output, failed = await self.execute_tool(tool_name, tool_args, tool_call_id=call_id)
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
                                tool_name, tool_args, count=False, tool_call_id=call_id
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

                from power_loop.runtime.tool_contracts import has_capability
                if not failed and has_capability(self.tool_registry.get(tool_name), "task.progress.v1"):
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

                stored_output = _truncate_result(output, self.config.tool_result_max_chars)
                from power_loop.runtime.tool_contracts import format_result
                stored_output = format_result(self.tool_registry.get(tool_name), str(output),
                    self.config.tool_result_max_chars or 20000) or stored_output
                await self._append_message(
                    {"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": stored_output},
                    round_index=round_idx,
                )

                from power_loop.runtime.tool_context import ToolSource, active_engine
                context_engine = active_engine.get()
                if context_engine is not None:
                    seqs = getattr(self.sink, "_history_seqs", None) or []
                    if seqs:
                        await context_engine.submit(ToolSource(
                            self.session_id, self.send_index, seqs[-1], call_id,
                            tool_name, tool_args, str(self.history[-1].get("content") or "")))

                if _stopped:
                    # design/124 §8.4: the run was stopped mid-tool — this call is settled; the
                    # rest of the batch never starts (the host's abort_pending resolves them),
                    # concurrent siblings are cancelled, the run ends cancelled.
                    self._cancel_pre_tasks(_pre)
                    self._emit(AgentEventType.LOOP_CANCELLED,
                               LoopCancelledPayload(reason=self.cancel_token.reason,
                                                    round_index=round_idx),
                               round_index=round_idx)
                    await self._finalize("cancelled")
                    return self._make_result(
                        "cancelled", final_text=f"[cancelled: {self.cancel_token.reason}]",
                        rounds=round_idx + 1)

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
