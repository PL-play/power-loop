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
    from llm_client.interface import LLMResponse

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
    Handler may modify any input field.
    """

    messages: list[LoopMessage] = field(default_factory=list)
    system_prompt: str = ""
    tools: list[dict[str, Any]] | None = None
    max_tokens: int = 8000
    temperature: float = 0.0
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
    input_tokens: int = 0
    compact_threshold: int = 0


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
