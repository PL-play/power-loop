from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

LoopStatus = Literal["completed", "pending_tools", "cancelled", "hit_round_limit"]
LoopMessage = Dict[str, Any]


@dataclass
class AgentLoopConfig:
    """Config for Agent Loop v1.

    v1 scope:
    - pure round loop
    - no tool execution
    - no hooks/events
    """

    system_prompt: str | None = None
    max_rounds: int = 24
    temperature: float | None = 0.0
    max_tokens: int | None = 8000


@dataclass
class AgentLoopResult:
    status: LoopStatus
    final_text: str = ""
    rounds: int = 0
    pending_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[LoopMessage] = field(default_factory=list)
