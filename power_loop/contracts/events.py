from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from power_loop.contracts.event_payloads import BaseEventPayload

# Process-wide monotonic event counter. ``itertools.count.__next__`` is atomic in
# CPython, so this is a thread-safe total order across every bus/session/sub-agent
# in the process — the basis for ordering interleaved streams (H4.2).
_event_seq = count(1)


class AgentEventType(str, Enum):
    # Session lifecycle
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"

    # Round lifecycle
    ROUND_STARTED = "round_started"
    ROUND_COMPLETED = "round_completed"
    ROUND_TOOLS_PRESENT = "round_tools_present"

    # LLM call lifecycle (per attempt — latency / usage / retry attribution)
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"

    # LLM retry / cancellation lifecycle
    LLM_RETRY_ATTEMPTED = "llm_retry_attempted"
    LLM_DEGRADED = "llm_degraded"
    LOOP_CANCELLED = "loop_cancelled"

    # Memory lifecycle (M1.9)
    MEMORY_RECALLED = "memory_recalled"
    MEMORY_FAILED = "memory_failed"

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
    TIMER_FIRED = "timer_fired"

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
    payload: dict[str, Any] = field(default_factory=dict)
    data: BaseEventPayload | None = field(default=None, repr=False)
    session_id: str | None = None
    round_index: int | None = None
    stream_id: str | None = None
    source: str | None = None
    # Envelope timing/ordering (H4.2): wall-clock creation time (epoch seconds) and a
    # process-wide monotonic sequence. Together they timestamp and totally-order every
    # event — the keystone for an OpenTelemetry span bridge and for reconstructing
    # interleaved sub-agent/workflow streams. Auto-populated when left at their
    # defaults; a deserializer may pass explicit values to preserve the original
    # ordering. Excluded from equality so they never define event identity.
    ts: float = field(default=0.0, compare=False)
    seq: int = field(default=0, compare=False)

    def __post_init__(self) -> None:
        if self.data is not None and not self.payload:
            self.payload = self.data.to_dict()
        if not self.ts:
            self.ts = time.time()
        if not self.seq:
            self.seq = next(_event_seq)
