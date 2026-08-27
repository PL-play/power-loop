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
    """Merge queued follow-up payloads into one user message.

    Text is merged into the ``<follow_up>`` envelope. NON-TEXT blocks (images) are carried
    through as their own content blocks rather than flattened away: steering an in-flight loop
    used to silently DROP any image in it, so the same user photo was visible when the session
    happened to be idle and invisible when it happened to be busy.
    """
    parts: list[str] = []
    blocks: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        else:
            content = item.get("content")
            text = _content_as_text(content).strip()
            blocks.extend(_non_text_blocks(content))
        if text:
            parts.append(text)
    if not parts and not blocks:
        return None
    merged = format_follow_up_user_message("\n\n".join(parts))
    if blocks:
        # Keep the envelope as the leading text block so the transcript still reads as steering.
        merged = {**merged, "content": [{"type": "text", "text": merged["content"]}, *blocks]}
    return merged


def _non_text_blocks(content: Any) -> list[dict[str, Any]]:
    """Content blocks that carry something other than plain text (images/attachments)."""
    if not isinstance(content, list):
        return []
    return [
        b for b in content
        if isinstance(b, dict) and b.get("type") not in (None, "text")
    ]


def follow_up_text(item: str | LoopMessage) -> str:
    """Flatten one queued payload to plain text.

    Needed because the cross-process queue is a TEXT column: a LoopMessage cannot be stored as-is,
    and ``merge_follow_up_inputs`` accepts strings anyway, so the round trip is lossless for the
    only field that reaches the model.
    """
    return item if isinstance(item, str) else _content_as_text(item.get("content"))


def _content_as_text(content: Any) -> str:
    """Flatten content to text: real text from text blocks, a short marker for anything else.

    ``json.dumps``-ing the whole list (the previous behaviour) put a serialized image block
    into the steering text — with an inlined data URL that is the entire base64 payload,
    unreadable to the model and unbounded in size. A marker keeps the fact that an image was
    there without pasting its bytes.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(str(block.get("text") or ""))
                else:
                    out.append(f"[{block.get('type') or 'block'}]")
            else:
                out.append(str(block))
        return "\n".join(x for x in out if x)
    import json

    return json.dumps(content, ensure_ascii=False)


__all__ = [
    "FOLLOW_UP_MESSAGE_NAME",
    "FollowUpQueued",
    "follow_up_text",
    "format_follow_up_user_message",
    "merge_follow_up_inputs",
]
