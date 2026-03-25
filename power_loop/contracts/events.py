from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


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


@dataclass
class AgentEvent:
    """Agent lifecycle event.
    Common payload conventions (JSON-serializable dicts):
    - ``STATUS_CHANGED``: always includes ``kind`` (discriminator). Known kinds:
      ``auto_compact`` (phase, round_index, trigger, input_tokens, compact_threshold),
      ``round_usage`` (time_iso, round_index, round_number, max_rounds, token fields),
      ``hit_round_limit`` (max_rounds).
   - ``USAGE_UPDATED``: ``usage`` includes both **completion_*** (single last LLM response) and
      **session_*** (cumulative for this agent session); legacy keys ``input`` / ``total_in`` remain
      aliases. ``session`` mirrors :class:`~power_loop.core.state.ContextManager` counters (with
      legacy field names duplicated). Optional ``summary`` is human-readable; prefer structured keys.
 
      and optional ``summary`` (human line; prefer ``usage`` / ``session`` for logic).
    - ``TODO_UPDATED``: ``kind`` = ``todo_snapshot``, ``items``, ``counts``, ``rendered``;
      ``text`` is kept as an alias of ``rendered`` for older subscribers.
    """
    type: AgentEventType
    payload: Dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    round_index: int | None = None
    stream_id: str | None = None
    source: str | None = None
