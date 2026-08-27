"""Put an image back in front of the model for ONE round.

Why this exists: a projection deliberately distils a past image down to a text reference
(``[image: shot.png]``) — keeping the picture would re-send it on every later send and the
token bill grows without bound. But an agent that later wants to LOOK at that image again has
nothing to look at.

Why it is not just a tool return value: the OpenAI-compatible protocol does not allow images
in a ``tool`` message — ``ChatCompletionToolMessageParam.content`` is typed as
``str | Iterable[ChatCompletionContentPartTextParam]``, text parts only. (Anthropic's
``tool_result`` does allow image blocks, but building on that would make history shape depend
on which vendor the definition happens to point at, and definitions switch models at runtime.)
So the image travels as its own **user** message instead, and the tool returns only a line
saying it is there.

Why EPHEMERAL: the queued message is appended to the round's request, never to the store. A
durable injection would write the image into history and the projection, so an agent that
recalled three times would carry three images forever — exactly the unbounded growth the
projection exists to prevent. One round, then gone; recall again if you need another look.
"""

from __future__ import annotations

import threading
from typing import Any

LoopMessage = dict[str, Any]

# session_id -> messages queued for the NEXT round of that session. Module-level rather than a
# contextvar because a tool handler may run in a worker thread (``asyncio.to_thread``), which
# does not inherit the caller's context.
_pending: dict[str, list[LoopMessage]] = {}
_lock = threading.Lock()

#: A queue is drained every round, so anything left is a tool that queued without a round
#: following. Cap it so a pathological loop cannot grow the dict without bound.
MAX_PENDING_PER_SESSION = 8


def queue_image_for_next_round(
    session_id: str | None, *, path: str, note: str = ""
) -> bool:
    """Queue one image (by local path) to be shown to the model on the next round.

    Returns False when there is no session to queue against, or the queue is full — the caller
    should then say so in its own return value rather than pretend the image was delivered.
    """
    if not session_id or not path:
        return False
    blocks: list[dict[str, Any]] = []
    if note:
        blocks.append({"type": "text", "text": note})
    # An `attachment` block (path, not bytes) — the renderer reads the file, applies the
    # model's declared size limit and encodes it at request time.
    blocks.append({
        "type": "attachment",
        "attachment": _attachment_ref(path),
    })
    with _lock:
        queue = _pending.setdefault(session_id, [])
        if len(queue) >= MAX_PENDING_PER_SESSION:
            return False
        queue.append({"role": "user", "content": blocks})
    return True


def drain_queued_images(session_id: str | None) -> list[LoopMessage]:
    """Take everything queued for this session (and clear it)."""
    if not session_id:
        return []
    with _lock:
        return _pending.pop(session_id, [])


def discard_queued_images(session_id: str | None) -> None:
    """Drop anything queued — used when a session ends so a killed run leaves nothing behind."""
    if session_id:
        with _lock:
            _pending.pop(session_id, None)


def _attachment_ref(path: str) -> dict[str, Any]:
    from power_loop._vendor.llm_client.multimodal import create_attachment_ref

    return create_attachment_ref(path)
