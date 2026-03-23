from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool call contract emitted by assistant messages."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_openai_tool_call(self) -> Dict[str, Any]:
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class AgentMessage:
    """Unified runtime message contract used inside AgentLoop."""

    role: MessageRole
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai_message(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.role == "assistant" and self.tool_calls:
            payload["tool_calls"] = [call.to_openai_tool_call() for call in self.tool_calls]
        if self.role == "tool" and self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


def SystemMessage(content: str, **metadata: Any) -> AgentMessage:
    return AgentMessage(role="system", content=content, metadata=dict(metadata))


def UserMessage(content: str, **metadata: Any) -> AgentMessage:
    return AgentMessage(role="user", content=content, metadata=dict(metadata))


def AssistantMessage(
    content: str,
    *,
    tool_calls: list[ToolCall] | None = None,
    **metadata: Any,
) -> AgentMessage:
    return AgentMessage(
        role="assistant",
        content=content,
        tool_calls=list(tool_calls or []),
        metadata=dict(metadata),
    )


def ToolResultMessage(
    content: str,
    *,
    tool_call_id: str,
    name: str | None = None,
    **metadata: Any,
) -> AgentMessage:
    return AgentMessage(
        role="tool",
        content=content,
        tool_call_id=tool_call_id,
        name=name,
        metadata=dict(metadata),
    )
