"""Typed hook context dataclasses — one per hook point.

Each dataclass declares the exact fields a hook handler receives and can
modify, replacing the untyped ``HookContext.values`` dict.  Handlers
mutate the context in place and set ``directive`` when needed.

Usage::

    def my_handler(ctx: ToolBeforeCtx) -> None:
        if "rm -rf" in str(ctx.tool_args):
            ctx.output = "[blocked by policy]"
            ctx.directive = HookDirective.SKIP
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from power_loop.contracts.hooks import HookDirective

if TYPE_CHECKING:
    from power_loop._vendor.llm_client.interface import LLMResponse
    from power_loop.agent.types import LoopMessage


# ── Base ──


@dataclass
class BaseHookCtx:
    """Common fields shared by all hook contexts."""

    round_index: int = 0
    directive: HookDirective = HookDirective.CONTINUE


# ── Session ──


@dataclass
class SessionStartCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.SESSION_START`.

    Handler may modify ``messages``.
    """

    scope: str = "main"
    messages: list[LoopMessage] = field(default_factory=list)
    stop_event: threading.Event | None = None


@dataclass
class SessionEndCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.SESSION_END` (read-only)."""

    scope: str = "main"
    reason: str = ""
    messages: list[LoopMessage] = field(default_factory=list)
    final_text: str | None = None


# ── Timer ──


@dataclass
class TimerFireCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.TIMER_FIRE` — runs when a due timer is
    about to be delivered to its session. The orchestrator's veto point:
    check busy state, dedupe after at-least-once re-fires, audit, reroute.

    Directives: CONTINUE delivers (default); SKIP marks the firing as skipped
    (no delivery, timer done); BREAK cancels the timer. Set ``postpone_s`` > 0
    (with CONTINUE) to re-arm at now + postpone_s instead of delivering.
    Handler may rewrite ``message`` — the text that will be injected.
    """

    session_id: str = ""
    timer_id: int = 0
    note: str = ""
    due_at: int = 0  # epoch ms
    message: str = ""
    # Handler output
    postpone_s: float = 0.0


# ── Round ──


@dataclass
class RoundStartCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.ROUND_START`.

    Directives: BREAK (set ``reason``), SKIP.
    Handler may modify ``messages``.
    """

    messages: list[LoopMessage] = field(default_factory=list)
    stop_event: threading.Event | None = None
    # Handler output
    reason: str = ""


@dataclass
class RoundEndCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.ROUND_END` (read-only).

    Both ``response_text`` and ``used_todo`` are always present.
    """

    messages: list[LoopMessage] = field(default_factory=list)
    has_tools: bool = False
    response_text: str = ""
    used_todo: bool = False


# ── LLM ──


@dataclass
class LlmBeforeCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.LLM_BEFORE`.

    Directives: SHORT_CIRCUIT (set ``output`` to an ``LLMResponse``), BREAK.
    Handler may modify any input field. ``messages`` is the fresh per-call list
    actually sent to the LLM — mutating it (e.g. appending an ephemeral memory
    block) never touches the loop's persisted history.

    ``persist_messages`` is the durable counterpart: any message a handler appends
    here becomes a REAL history/store row (stamped with the round's send_index, via
    the same append path as the loop's own turns) AND is added to this round's
    request. Use it for injected turns that must survive the send — e.g. a periodic
    "you haven't called X in N rounds" reminder — as opposed to the ephemeral,
    request-only edits to ``messages``. Appended in order, at the tail of the request.
    """

    messages: list[LoopMessage] = field(default_factory=list)
    system_prompt: str = ""
    tools: list[dict[str, Any]] | None = None
    max_tokens: int = 8000
    temperature: float = 0.0
    session_id: str | None = None
    # Durable injections: persisted as real turns after LLM_BEFORE, then added to the request.
    persist_messages: list[LoopMessage] = field(default_factory=list)
    # Handler output (for SHORT_CIRCUIT)
    output: LLMResponse | None = None


@dataclass
class LlmAfterCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.LLM_AFTER`.

    Directives: BREAK.
    Handler may replace ``output``.
    """

    messages: list[LoopMessage] = field(default_factory=list)
    output: LLMResponse | None = None


# ── Round decide ──


@dataclass
class RoundDecideCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.ROUND_DECIDE`.

    Directives: SKIP (set ``output`` as skip message), BREAK.
    """

    messages: list[LoopMessage] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    assistant_text: str = ""
    # Handler output (for SKIP)
    output: str = "[skipped by round_decide hook]"


# ── Tools batch ──


@dataclass
class ToolsBatchBeforeCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.TOOLS_BATCH_BEFORE`.

    Directives: SKIP (set ``output`` as placeholder result for all tools).
    """

    messages: list[LoopMessage] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Handler output (for SKIP)
    output: str = "[skipped by batch hook]"


@dataclass
class ToolsBatchAfterCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.TOOLS_BATCH_AFTER` (read-only)."""

    messages: list[LoopMessage] = field(default_factory=list)
    used_todo: bool = False


# ── Individual tool ──


@dataclass
class ToolBeforeCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.TOOL_BEFORE`.

    Directives: SKIP (set ``output``).
    Handler may modify ``tool_name`` and ``tool_args``.
    """

    tool_call: dict[str, Any] = field(default_factory=dict)
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    # Handler output (for SKIP)
    output: str = "[skipped by hook]"


@dataclass
class ToolAfterCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.TOOL_AFTER`.

    Directives: BREAK.
    Handler may replace ``output`` and ``failed``.
    """

    tool_call: dict[str, Any] = field(default_factory=dict)
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    failed: bool = False


@dataclass
class ToolErrorCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.TOOL_ERROR`.

    Directives: SKIP (use ``output`` as fallback), SHORT_CIRCUIT (retry).
    """

    tool_call: dict[str, Any] = field(default_factory=dict)
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    error: Exception | None = None
    error_message: str = ""
    # Handler output (fallback for SKIP)
    output: str = ""


# ── Compact ──


@dataclass
class CompactBeforeCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.COMPACT_BEFORE`.

    Directives: SKIP (skip compaction this round).
    """

    messages: list[LoopMessage] = field(default_factory=list)


@dataclass
class CompactAfterCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.COMPACT_AFTER` (read-only)."""

    messages: list[LoopMessage] = field(default_factory=list)
    messages_before_count: int = 0
    messages_after_count: int = 0


# ── Message ──


@dataclass
class MessageAppendCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.MESSAGE_APPEND`.

    Handler may modify ``message``.
    """

    message: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None


# ── Memory (M1.9) ──


@dataclass
class MemoryRecalledCtx(BaseHookCtx):
    """Context for :pyattr:`HookPoint.MEMORY_RECALLED`.

    Fired after :meth:`MemoryProvider.recall` returns, before injection.
    Handler may mutate ``recalled`` (filter, redact, reorder) or set
    ``directive=SKIP`` to drop everything and inject nothing.
    """

    recalled: list[dict[str, Any]] = field(default_factory=list)
    session_id: str | None = None
    budget_tokens: int = 1500
