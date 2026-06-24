"""Backend-neutral row/enum types for the storage layer.

Every storage backend (SQLite / PostgreSQL / MySQL) reads and writes through the same
``SessionStore`` method surface and returns these exact dataclasses + enums, so callers
(``SQLiteSink``, ``StatefulAgentLoop``, timers, notes, workflow journal, …) are
backend-agnostic. These deliberately carry NO SQLite coupling — they are plain frozen
shapes plus four ``str`` enums.

(Extracted verbatim from the original ``session_store.py``; that module re-exports them
for backward-compatible imports.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "SessionKind",
    "SessionStatus",
    "SubagentLifecycle",
    "MessageState",
    "TimerRow",
    "SessionStatsRow",
    "SessionRow",
    "MessageRow",
    "SessionStateRow",
    "CompactionRow",
    "BackgroundTaskRow",
    "NoteRow",
    "ProjectMessageRow",
]


class SessionKind(str, Enum):
    ROOT = "root"
    SUBAGENT = "subagent"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class SubagentLifecycle(str, Enum):
    """How long a subagent's session persists relative to its parent."""

    EPHEMERAL = "ephemeral"  # delete child immediately after it returns
    LINKED = "linked"        # keep, cascade-delete when parent is closed
    DETACHED = "detached"    # keep, independent of parent's lifecycle


class MessageState(str, Enum):
    ACTIVE = "active"
    COMPACTED_OUT = "compacted_out"
    #: Deactivated by the history-repair backstop (an orphan tool-result row that would make the
    #: provider reject the prompt). Excluded from the active history like ``compacted_out``, but
    #: distinct so it is auditable as a repair, not a compaction.
    DROPPED = "dropped"


@dataclass
class TimerRow:
    """A durable per-session wake-up. One-shot (interval_s is None):
    armed -> firing -> fired | cancelled. Recurring (interval_s set):
    armed -> firing -> armed again at fire-time + interval (fixed-delay,
    missed periods while down collapse into one) until cancelled.
    The row is the source of truth; in-process scheduling is just an
    accelerator over it."""

    session_id: str
    timer_id: int
    due_at: int  # epoch ms
    note: str
    status: str
    interval_s: int | None
    fire_count: int
    last_fired_at: int | None
    created_at: int
    updated_at: int


@dataclass
class SessionStatsRow:
    """Cumulative per-session accounting (bumped once per finished send)."""

    session_id: str
    sends: int
    rounds: int
    llm_calls: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    first_send_at: int | None
    last_send_at: int | None
    updated_at: int


@dataclass
class SessionRow:
    session_id: str
    created_at: int
    updated_at: int
    system_prompt: str | None
    model: str | None
    config: dict[str, Any]
    status: SessionStatus
    kind: SessionKind
    parent_session_id: str | None
    spawn_tool_call_id: str | None
    spawn_depth: int
    lifecycle: SubagentLifecycle
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageRow:
    session_id: str
    seq: int
    role: str
    name: str | None
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    tool_call_id: str | None
    round_index: int | None
    state: MessageState
    meta: dict[str, Any]
    created_at: int
    #: Authoritative, monotonic per-session SEND index (one value per loop.send(); never
    #: resets). NULL on rows written before schema v2 / outside a send. Default keeps legacy
    #: constructors (and the SQLite-only store) working.
    send_index: int | None = None


@dataclass
class SessionStateRow:
    session_id: str
    next_seq: int
    round_index: int
    last_compact_seq: int
    pending: dict[str, Any] | None


@dataclass
class CompactionRow:
    session_id: str
    compact_seq: int
    note_seq: int
    from_seq: int
    to_seq: int
    before_tokens: int | None
    after_tokens: int | None
    round_index: int | None
    created_at: int


@dataclass
class BackgroundTaskRow:
    session_id: str
    task_id: str
    command: str
    status: str
    return_code: int | None
    output_tail: str | None
    output_path: str | None
    last_seen_at: int
    created_at: int
    updated_at: int


@dataclass
class NoteRow:
    """One agent-authored note. ``note_id`` is a short per-session integer so
    the model can reference notes naturally ("update note 3")."""

    session_id: str
    note_id: int
    content: str
    pinned: bool
    created_at: int
    updated_at: int


@dataclass
class ProjectMessageRow:
    """One row of the send-context PROJECTION layer (``{prefix}project_messages``), the
    derived per-send view fed to the LLM — distinct from ``MessageRow`` (the immutable
    ``pl_messages`` loop-internal audit). ``kind`` is ``user`` / ``project`` / ``compact``;
    ``content`` is the parsed ``content_json`` (a structured, re-renderable summary).
    For ``compact`` rows, ``compact_from_send``/``compact_to_send`` bound the folded send
    range (``compact_to_send`` doubles as the read cursor). ``source_seq_lo``/``_hi`` point
    back to the ``pl_messages`` rows this projection covers (traceability + recall)."""

    session_id: str
    send_index: int
    kind: str
    content: Any
    rendered_text: str | None
    source_seq_lo: int | None
    source_seq_hi: int | None
    compact_from_send: int | None
    compact_to_send: int | None
    projector_version: int
    token_estimate: int | None
    created_at: int


@dataclass
class HookEventRow:
    """One audit row of an EPHEMERAL hook augmentation (``{prefix}hook_events``) — e.g. the memory
    block an ``LLM_BEFORE`` hook injected into a single LLM call's request. It is observability ONLY:
    written alongside the message it fed into, NEVER read back into history or the LLM request. So
    the injected context (which today vanishes after the call) is recoverable for audit.

    ``event_id`` is a per-session monotonic id. ``message_seq`` links to the ``messages`` row this
    augmentation fed into (the assistant response of that round); ``round_index``/``send_index``
    locate the round/send. ``hook_point`` is the hook phase (``LLM_BEFORE``); ``hook`` a coarse
    source label (``builtin.memory_recall`` / ``llm_before``); ``position`` is where the items landed
    in the request (``tail``/``front``); ``kind`` the effect (``inject``). ``payload`` is the parsed
    ``payload_json`` — ``{v, items:[{role,name,source,chars,content?}], item_count, total_chars}``
    (``content`` present only when captured in ``full`` mode)."""

    session_id: str
    event_id: int
    message_seq: int | None
    round_index: int | None
    send_index: int | None
    hook_point: str
    hook: str | None
    position: str | None
    kind: str
    payload: Any
    created_at: int
