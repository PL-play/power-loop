from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class HookPoint(str, Enum):
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    ROUND_START = "round.start"
    ROUND_END = "round.end"
    LLM_BEFORE = "llm.before"
    LLM_AFTER = "llm.after"
    TOOLS_BATCH_BEFORE = "tools.batch.before"
    TOOLS_BATCH_AFTER = "tools.batch.after"
    TOOL_BEFORE = "tool.before"
    TOOL_AFTER = "tool.after"
    MESSAGE_APPEND = "message.append"


@dataclass
class HookContext:
    """Mutable context passed through each hook chain."""

    values: Dict[str, Any] = field(default_factory=dict)
