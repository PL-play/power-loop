from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HumanInputRequired(Exception):
    """Control signal raised by request_user_input to pause the loop."""

    kind: str
    prompt: str
    options: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    interaction_id: str = field(default_factory=lambda: "input_" + secrets.token_hex(8))

    def __post_init__(self) -> None:
        Exception.__init__(self, self.prompt)

    def to_pending(self, *, tool_call_id: str, tool_name: str = "request_user_input") -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "kind": self.kind,
            "prompt": self.prompt,
            "options": list(self.options),
            "metadata": dict(self.metadata),
        }


def request_user_input(
    *,
    kind: str = "text",
    prompt: str,
    options: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Special tool handler: pause the loop and ask the caller for input."""
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        raise ValueError("prompt is required")
    raise HumanInputRequired(
        kind=(kind or "text").strip() or "text",
        prompt=clean_prompt,
        options=list(options or []),
        metadata=dict(metadata or {}),
    )


__all__ = ["HumanInputRequired", "request_user_input"]
