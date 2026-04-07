"""Typed event payload dataclasses — one per AgentEventType.

Each dataclass declares the exact fields an event subscriber receives,
replacing the untyped ``AgentEvent.payload`` dict.  Subscribers can
access fields directly with IDE auto-completion and type checking.

Legacy dict-based access is preserved via :meth:`BaseEventPayload.to_dict`
and the ``AgentEvent.payload`` property for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ── Base ──


@dataclass
class BaseEventPayload:
    """Common base for all event payloads.

    Provides :meth:`to_dict` for backward-compatible dict serialization.
    """

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                result[k] = v
        return result


# ── Session ──


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
    tool_input: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""


@dataclass
class ToolCallCompletedPayload(BaseEventPayload):
    name: str = ""
    output: str = ""
    tool_input: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""


@dataclass
class ToolCallFailedPayload(BaseEventPayload):
    name: str = ""
    output: str = ""
    tool_input: Dict[str, Any] = field(default_factory=dict)
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
    input_tokens: int = 0
    compact_threshold: int = 0


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


# ── Usage ──


@dataclass
class UsageUpdatedPayload(BaseEventPayload):
    usage: Dict[str, Any] = field(default_factory=dict)


# ── Todo ──


@dataclass
class TodoUpdatedPayload(BaseEventPayload):
    kind: str = "todo_snapshot"
    items: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
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
