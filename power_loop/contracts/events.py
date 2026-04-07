from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from power_loop.contracts.event_payloads import BaseEventPayload


class AgentEventType(str, Enum):
    # Session lifecycle
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"

    # Round lifecycle
    ROUND_STARTED = "round_started"
    ROUND_COMPLETED = "round_completed"
    ROUND_TOOLS_PRESENT = "round_tools_present"

    # Streaming lifecycle
    STREAM_STARTED = "stream_started"
    STREAM_DELTA = "stream_delta"
    STREAM_THINK_DELTA = "stream_think_delta"
    STREAM_COMPLETED = "stream_completed"

    # Tool lifecycle
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"

    # Status and usage
    STATUS_CHANGED = "status_changed"
    USAGE_UPDATED = "usage_updated"

    # Task list / planner (optional feature, used when todo tool is enabled)
    TODO_UPDATED = "todo_updated"

    # System and user-facing notifications
    USER_NOTIFICATION = "user_notification"
    AGENT_ERROR = "agent_error"
    SYSTEM_LOG = "system_log"

    # Subagent events
    SUBAGENT_TASK_START = "subagent_task_start"
    SUBAGENT_TEXT = "subagent_text"
    SUBAGENT_LIMIT = "subagent_limit"
    SUBAGENT_COMPLETED = "subagent_completed"


@dataclass
class AgentEvent:
    """Agent lifecycle event with typed payload.

    The ``data`` field holds a strongly-typed payload dataclass (e.g.
    ``StreamDeltaPayload``, ``ToolCallStartedPayload``).  Subscribers can
    access fields directly with IDE auto-completion::

        def on_delta(event: AgentEvent) -> None:
            delta: StreamDeltaPayload = event.data
            print(delta.text)

    The legacy ``payload`` dict is auto-generated from ``data.to_dict()``
    for backward compatibility.  If ``data`` is not set, ``payload`` dict
    is used directly (legacy path).
    """
    type: AgentEventType
    payload: Dict[str, Any] = field(default_factory=dict)
    data: BaseEventPayload | None = field(default=None, repr=False)
    session_id: str | None = None
    round_index: int | None = None
    stream_id: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.data is not None and not self.payload:
            self.payload = self.data.to_dict()
