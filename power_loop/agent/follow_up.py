"""Follow-up / steering queue for in-flight agent loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from power_loop.agent.types import LoopMessage

FOLLOW_UP_MESSAGE_NAME = "follow_up"


@dataclass(frozen=True)
class FollowUpQueued:
    """Returned when :meth:`StatefulAgentLoop.follow_up` enqueues steering input."""

    session_id: str
    queue_depth: int


def format_follow_up_user_message(text: str) -> LoopMessage:
    """Wrap steering text as a user message for the LLM transcript."""
    body = text.strip()
    return {
        "role": "user",
        "name": FOLLOW_UP_MESSAGE_NAME,
        "content": f"<follow_up>\n{body}\n</follow_up>",
    }


def merge_follow_up_inputs(items: list[str | LoopMessage]) -> LoopMessage | None:
    """Merge queued follow-up payloads into one user message."""
    parts: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        else:
            text = _content_as_text(item.get("content")).strip()
        if text:
            parts.append(text)
    if not parts:
        return None
    return format_follow_up_user_message("\n\n".join(parts))


def follow_up_text(item: str | LoopMessage) -> str:
    """Flatten one queued payload to plain text.

    Needed because the cross-process queue is a TEXT column: a LoopMessage cannot be stored as-is,
    and ``merge_follow_up_inputs`` accepts strings anyway, so the round trip is lossless for the
    only field that reaches the model.
    """
    return item if isinstance(item, str) else _content_as_text(item.get("content"))


def _content_as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    import json

    return json.dumps(content, ensure_ascii=False)


__all__ = [
    "FOLLOW_UP_MESSAGE_NAME",
    "FollowUpQueued",
    "follow_up_text",
    "format_follow_up_user_message",
    "merge_follow_up_inputs",
]
