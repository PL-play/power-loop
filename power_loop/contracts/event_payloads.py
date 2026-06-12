"""Typed event payload dataclasses — one per AgentEventType.

Each dataclass declares the exact fields an event subscriber receives,
replacing the untyped ``AgentEvent.payload`` dict.  Subscribers can
access fields directly with IDE auto-completion and type checking.

Legacy dict-based access is preserved via :meth:`BaseEventPayload.to_dict`
and the ``AgentEvent.payload`` property for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Base ──


@dataclass
class BaseEventPayload:
    """Common base for all event payloads.

    Provides :meth:`to_dict` for backward-compatible dict serialization.
    """

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                result[k] = v
        return result


# ── Session ──


@dataclass
class LlmRetryAttemptedPayload(BaseEventPayload):
    """Emitted after each *failed* LLM attempt that will be retried.

    ``attempt`` is the 0-based index of the attempt that **just failed**;
    ``next_sleep_seconds`` is what the policy will sleep before the next
    attempt (capped by remaining deadline).
    """
    attempt: int = 0
    max_attempts: int = 0
    error_type: str = ""
    error_message: str = ""
    next_sleep_seconds: float = 0.0


@dataclass
class LlmDegradedPayload(BaseEventPayload):
    """Emitted when the loop gives up on the LLM (retry exhausted or
    total_timeout) and proceeds with status=``degraded``."""
    reason: str = ""           # "retry_exhausted" | "timeout"
    attempts: int = 0
    error_type: str = ""
    error_message: str = ""


@dataclass
class LoopCancelledPayload(BaseEventPayload):
    """Emitted when the loop terminates because a CancellationToken fired."""
    reason: str = "cancelled"
    round_index: int | None = None


@dataclass
class MemoryRecalledPayload(BaseEventPayload):
    """Emitted after a MemoryProvider.recall() call.

    ``injected`` is the number actually placed into ``messages`` (after
    the MEMORY_RECALLED hook had a chance to filter/skip). ``returned``
    is what the provider gave us before hook filtering.
    """
    returned: int = 0
    injected: int = 0
    budget_tokens: int = 0


@dataclass
class MemoryFailedPayload(BaseEventPayload):
    """Emitted when recall() or remember() raises.

    The loop continues regardless — memory is best-effort. ``phase`` is
    ``"recall"`` or ``"remember"`` so subscribers know which side broke.
    """
    phase: str = ""
    error_type: str = ""
    error_message: str = ""


@dataclass
class SessionStartedPayload(BaseEventPayload):
    scope: str = "main"


@dataclass
class SessionEndedPayload(BaseEventPayload):
    reason: str = ""


# ── Round ──


@dataclass
class RoundStartedPayload(BaseEventPayload):
    round_index: int = 0


@dataclass
class RoundCompletedPayload(BaseEventPayload):
    round_index: int = 0
    has_tools: bool = False
    used_todo: bool = False


@dataclass
class RoundToolsPresentPayload(BaseEventPayload):
    has_tools: bool = False


# ── Stream ──


@dataclass
class StreamStartedPayload(BaseEventPayload):
    stream_id: str = "main"


@dataclass
class StreamDeltaPayload(BaseEventPayload):
    stream_id: str = "main"
    text: str = ""
    is_think: bool = False


@dataclass
class StreamCompletedPayload(BaseEventPayload):
    stream_id: str = "main"


# ── Tool ──


@dataclass
class ToolCallStartedPayload(BaseEventPayload):
    name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""


@dataclass
class ToolCallCompletedPayload(BaseEventPayload):
    name: str = ""
    output: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""


@dataclass
class ToolCallFailedPayload(BaseEventPayload):
    name: str = ""
    output: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""


# ── Status ──


@dataclass
class StatusChangedPayload(BaseEventPayload):
    """Polymorphic status payload, discriminated by ``kind``."""
    kind: str = ""


@dataclass
class AutoCompactStatusPayload(StatusChangedPayload):
    kind: str = "auto_compact"
    phase: str = ""
    round_index: int = 0
    trigger: str = ""
    before_tokens: int = 0
    after_tokens: int = 0


@dataclass
class RoundUsageStatusPayload(StatusChangedPayload):
    kind: str = "round_usage"
    time_iso: str = ""
    round_index: int = 0
    round_number: int = 0
    max_rounds: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_read_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass
class HitRoundLimitStatusPayload(StatusChangedPayload):
    kind: str = "hit_round_limit"
    max_rounds: int = 0


@dataclass
class BudgetExceededStatusPayload(StatusChangedPayload):
    kind: str = "budget_exceeded"
    budget_tokens: int = 0
    spent_tokens: int = 0
    rounds: int = 0


# ── Usage ──


@dataclass
class UsageUpdatedPayload(BaseEventPayload):
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimerFiredPayload(BaseEventPayload):
    """Outcome of one due-timer firing: delivered / queued (session was
    mid-run, injected as follow_up) / skipped / cancelled / postponed /
    error."""

    timer_id: int = 0
    note: str = ""
    due_at: int = 0
    outcome: str = "delivered"


@dataclass
class PhaseEventPayload(BaseEventPayload):
    """Generic payload for events published by the ``@phase`` decorator
    (user-defined pipelines). Built-in pipeline events all have their own
    typed payload class; this wrapper guarantees ``event.data`` is never
    None even on the generic path."""

    values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


# ── Todo ──


@dataclass
class TodoUpdatedPayload(BaseEventPayload):
    kind: str = "todo_snapshot"
    items: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    rendered: str = ""
    text: str = ""


# ── Notification / Log ──


@dataclass
class UserNotificationPayload(BaseEventPayload):
    message: str = ""


@dataclass
class AgentErrorPayload(BaseEventPayload):
    error: str = ""
    error_type: str = ""
    details: str = ""


@dataclass
class SystemLogPayload(BaseEventPayload):
    message: str = ""
    level: str = "info"


# ── Subagent ──


@dataclass
class SubagentTaskStartPayload(BaseEventPayload):
    task: str = ""
    preset: str = "core"
    sub_session_id: str = ""
    depth: int = 0


@dataclass
class SubagentTextPayload(BaseEventPayload):
    sub_session_id: str = ""
    status: str = ""
    rounds: int = 0
    final_text: str = ""


@dataclass
class SubagentLimitPayload(BaseEventPayload):
    sub_session_id: str = ""
    max_rounds: int = 0


@dataclass
class SubagentCompletedPayload(BaseEventPayload):
    sub_session_id: str = ""
    status: str = ""
    rounds: int = 0
    final_text: str = ""
