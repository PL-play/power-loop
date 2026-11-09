"""Internal control signal for task review; request_user_input tool is removed."""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HumanInputRequired(Exception):
    """Control signal used internally by wait_review for finite child task reviews.

    Note: The public request_user_input tool has been removed in 12.0.0.
    This exception remains for internal use by wait_review only.
    """

    kind: str
    prompt: str
    options: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    interaction_id: str = field(default_factory=lambda: "input_" + secrets.token_hex(8))

    def __post_init__(self) -> None:
        Exception.__init__(self, self.prompt)

    def to_pending(self, *, tool_call_id: str, tool_name: str = "wait_review") -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "kind": self.kind,
            "prompt": self.prompt,
            "options": list(self.options),
            "metadata": dict(self.metadata),
        }
