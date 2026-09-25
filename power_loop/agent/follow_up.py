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
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class FollowUpQueued:
    """Returned when :meth:`StatefulAgentLoop.deliver` / ``follow_up`` parked input in the inbox
    instead of running it (a run is in flight, another process holds the session, or every item
    was a duplicate).

    ``queue_depth`` — undelivered items now waiting; ``accepted`` — how many of THIS call's items
    were new; ``duplicates`` — how many were already known (same ``item_id``) and dropped."""

    session_id: str
    queue_depth: int
    accepted: int = 1
    duplicates: int = 0
    accepted_ids: tuple[str, ...] = field(default_factory=tuple)


def inbox_row(item: InboxItem) -> dict[str, Any]:
    """The store payload for ``item`` (``SessionStore.inbox_put``). Structured content is JSON-
    encoded and flagged so :func:`render_inbox_rows` rebuilds it exactly (images included — the
    old text-only cross-process queue flattened them away)."""
    content: Any = item.content
    if isinstance(content, Mapping):
        content = content.get("content")
    structured = not (content is None or isinstance(content, str))
    text = json.dumps(content, ensure_ascii=False) if structured else str(content or "")
    meta: dict[str, Any] = {}
    if structured:
        meta["structured"] = True
    if item.meta:
        meta["host"] = dict(item.meta)
    return {
        "item_id": item.item_id or f"auto:{uuid.uuid4().hex}",
        "kind": item.kind,
        "mode": item.mode,
        "content": text,
        "meta": meta or None,
    }


def _row_payload(row: Mapping[str, Any]) -> str | LoopMessage:
    meta = row.get("meta") or {}
    content = row.get("content") or ""
    if meta.get("structured"):
        try:
            return {"role": "user", "content": json.loads(content)}
        except ValueError:
            return str(content)
    return str(content)


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


def render_inbox_rows(
    rows: Sequence[Mapping[str, Any]], *, wrap: bool
) -> list[LoopMessage]:
    """Turn claimed inbox rows (oldest first) into transcript user messages.

    Consecutive rows of the same ``kind`` merge into one message; a change of kind starts a new
    one — what a person said never shares a message with a system wake-up (design/124 Z2). With
    ``wrap`` (steering into a run in flight) each message gets the ``<follow_up>`` envelope and
    ``name="follow_up"``; without it (the input of a fresh send) the content goes in as-is.

    Every message carries a private ``_inbox`` marker (claim token + row ids) that the pipeline
    strips before the provider sees it and hands to the store, which marks exactly those rows
    delivered in the same transaction as the transcript row."""
    groups: list[list[Mapping[str, Any]]] = []
    for r in rows:
        if groups and groups[-1][0].get("kind") == r.get("kind"):
            groups[-1].append(r)
        else:
            groups.append([r])
    out: list[LoopMessage] = []
    for g in groups:
        payloads = [_row_payload(r) for r in g]
        msg = merge_follow_up_inputs(payloads) if wrap else _merge_plain(payloads)
        if msg is None:
            # Empty content: still deliver (and mark) so an empty item can't wedge the inbox.
            msg = {"role": "user", "content": ""}
        msg = dict(msg)
        msg["_inbox"] = {
            "claim_token": g[0].get("claim_token"),
            "ids": [int(r["id"]) for r in g],
            "item_ids": [str(r["item_id"]) for r in g],
            "kinds": [str(g[0].get("kind") or INBOX_KIND_USER)],
        }
        out.append(msg)
    return out


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
    "FollowUpQueued",
    "InboxItem",
    "inbox_row",
    "render_inbox_rows",
    "follow_up_text",
    "format_follow_up_user_message",
    "merge_follow_up_inputs",
]
