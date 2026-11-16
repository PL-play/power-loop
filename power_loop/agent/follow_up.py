"""Follow-up / steering: the session inbox and how its items become transcript messages.

design/124 §6. Anything that should reach a session's model while it can't be written into the
transcript right now waits in the durable per-session inbox (store table ``follow_up_queue``).
The pipeline claims what is waiting at each round boundary and appends it; an idle session gets
it as the input of a fresh send. Each item has a ``kind`` (who/what it is from) and a ``mode``
(``queue`` — at the next round boundary; ``steer`` — as soon as the loop can take it; the
``steer`` interrupt points arrive in design/124 §7, until then both are delivered alike).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from power_loop.agent.types import LoopMessage

FOLLOW_UP_MESSAGE_NAME = "follow_up"

#: Well-known item kinds. Hosts may use their own strings; kinds only group and label.
INBOX_KIND_USER = "user"            # something a human said
INBOX_KIND_TASK_DONE = "task_done"  # a background task / workflow the agent started settled
INBOX_KIND_REMINDER = "reminder"    # a timer / system nudge
INBOX_KIND_RELAY = "relay"          # another agent handed something over
INBOX_MODES = ("queue", "steer")


@dataclass(frozen=True)
class InboxItem:
    """One thing to put in front of a session's model.

    ``item_id`` is the dedupe key: the same id is accepted at most once per session, ever (a
    re-sent item is a silent no-op). Omit it and a random one is generated — then nothing
    dedupes. ``content`` is text or a user :data:`LoopMessage` (multimodal blocks survive).
    ``meta`` is host data stored with the item and echoed on the transcript row's
    ``meta["inbox"]``."""

    content: str | LoopMessage
    kind: str = INBOX_KIND_USER
    mode: str = "queue"
    item_id: str | None = None
    meta: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mode not in INBOX_MODES:
            raise ValueError(f"InboxItem.mode must be one of {INBOX_MODES}, got {self.mode!r}")






def session_batch_content(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One deterministic user-message body for the whole atomic input snapshot."""
    return {
        "version": 1,
        "items": [
            {
                "message_id": row["item_id"],
                "priority": row["priority"],
                "kind": row["kind"],
                "content": json.loads(row["content_json"]),
            }
            for row in rows
        ],
    }


def _batch_item_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        inner = content.get("content")
        if inner is not None:
            return _batch_item_text(inner)
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                else:
                    parts.append(f"[{block.get('type') or 'block'}]")
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return json.dumps(content, ensure_ascii=False)


def _batch_header(item: Mapping[str, Any]) -> str:
    return (f"[{item.get('kind') or 'user'} · {item.get('message_id')} · "
            f"{item.get('priority') or 'normal'}]")


def render_batch_text(merged: Mapping[str, Any]) -> str:
    """Deterministic text rendering of a merged batch for text providers (design 215 §7.3).

    Every item keeps its ID, kind, priority and boundary; a child result is never
    disguised as a human turn. The canonical JSON body stays in the raw message for
    audit; this is the equivalent text the model actually reads.

    A single ordinary user input renders as plain text (the common chat case, unchanged).
    A single urgent or non-user item carries the same deterministic header as a multi-item
    batch, so its ID/source/priority stay visible to the model (design 215 §7.4 urgent mark).
    """
    items = merged.get("items") or []
    if len(items) == 1:
        only = items[0]
        kind = only.get("kind") or "user"
        priority = only.get("priority") or "normal"
        body = _batch_item_text(only.get("content"))
        if kind == "user" and priority == "normal":
            return body
        return _batch_header(only) + "\n\n" + body
    lines = []
    for item in items:
        lines.append(_batch_header(item))
        lines.append(_batch_item_text(item.get("content")))
    return "\n\n".join(lines)


def is_batch_content(content: Any) -> bool:
    return isinstance(content, Mapping) and content.get("version") == 1 and isinstance(content.get("items"), list)


def render_batch_content(merged: Mapping[str, Any]) -> str | list[dict[str, Any]]:
    """Keep image blocks as model inputs while retaining batch source markers."""
    items = merged["items"]
    bodies = [item["content"].get("content") if isinstance(item["content"], Mapping)
              else item["content"] for item in items]
    if not any(_non_text_blocks(body) for body in bodies):
        return render_batch_text(merged)
    if len(items) == 1 and items[0]["kind"] == "user" and items[0]["priority"] == "normal":
        return bodies[0]
    blocks = []
    for item, body in zip(items, bodies, strict=True):
        blocks.append({"type": "text", "text": _batch_header(item)})
        if isinstance(body, list):
            blocks.extend(body)
        else:
            blocks.append({"type": "text", "text": _batch_item_text(item["content"])})
    return blocks


def _merge_plain(payloads: list[str | LoopMessage]) -> LoopMessage | None:
    """Like :func:`merge_follow_up_inputs` but WITHOUT the ``<follow_up>`` envelope — the input
    of a fresh send is the send's own input, not steering into someone else's run."""
    if len(payloads) == 1:
        only = payloads[0]
        return {"role": "user", "content": only} if isinstance(only, str) else dict(only)
    parts: list[str] = []
    blocks: list[dict[str, Any]] = []
    for p in payloads:
        if isinstance(p, str):
            text = p.strip()
        else:
            text = _content_as_text(p.get("content")).strip()
            blocks.extend(_non_text_blocks(p.get("content")))
        if text:
            parts.append(text)
    if not parts and not blocks:
        return None
    body = "\n\n".join(parts)
    if blocks:
        return {"role": "user", "content": [{"type": "text", "text": body}, *blocks]}
    return {"role": "user", "content": body}




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
    "INBOX_KIND_REMINDER",
    "INBOX_KIND_RELAY",
    "INBOX_KIND_TASK_DONE",
    "INBOX_KIND_USER",
    "InboxItem",
    "session_batch_content",
    "follow_up_text",
    "format_follow_up_user_message",
    "merge_follow_up_inputs",
]
