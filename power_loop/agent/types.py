from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

LoopStatus = Literal["completed", "pending_tools", "cancelled", "hit_round_limit"]
LoopMessage = dict[str, Any]


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop."""

    system_prompt: str | None = None
    max_rounds: int = 24
    temperature: float | None = 0.0
    max_tokens: int | None = 8000


@dataclass
class AgentLoopResult:
    status: LoopStatus
    final_text: str = ""
    rounds: int = 0
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[LoopMessage] = field(default_factory=list)
