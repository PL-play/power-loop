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

DURABLE by default, and that default was reversed on purpose. The first cut was ephemeral —
request-only, gone next round — out of a fear that "recall three times and you carry three
images forever". Measurement killed that fear: the provider's prefix cache serves a stable
history at ~99% hit rate, so an image sitting in the prefix costs about a tenth of its face
value every later round. Meanwhile the SEMANTICS clearly favour durable: an agent that looked
at a UI screenshot and then spends fifteen rounds coding against it should still have the
picture in front of it, not have to ask for it again. Across sends the projection distils the
row down to `[image: shot.png · file_uuid=…]`, so nothing accumulates without bound.

Pass ``durable=False`` for the look-once case (request-only, never stored).
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
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
    session_id: str | None, *, path: str, note: str = "", ref: str = "",
    durable: bool = True,
) -> bool:
    """Queue one image (by local path) to be shown to the model from the next round on.

    ``durable=True`` (default) stores it as a real ``user`` row, so it stays in front of the
    model for the rest of the send and is distilled to a text reference across sends.
    ``durable=False`` puts it in the request only — visible for exactly one round.

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
        # Carry the host's recall coordinate through: if the definition has meanwhile switched to
        # a model that cannot see images, this is what the placeholder text will point at.
        "attachment": _attachment_ref(path, ref=ref),
    })
    with _lock:
        queue = _pending.setdefault(session_id, [])
        if len(queue) >= MAX_PENDING_PER_SESSION:
            return False
        queue.append({"role": "user", "content": blocks, "__durable__": durable})
    return True


def queue_images_for_next_round(
    session_id: str | None,
    images: Sequence[tuple[str, str]],
    *,
    note: str = "",
    durable: bool = True,
) -> int:
    """Queue a BATCH as ONE user message: a single note + one attachment block per image.

    Queuing them one by one produces N separate user turns, each repeating the note — three
    screenshots asked about with one question became three copies of that question in the
    transcript. A batch is one turn, which is also what the provider APIs expect.

    ``images`` is ``[(path, ref)]``. Returns how many were accepted (0 = nothing queued, so the
    caller must say so rather than claim the pictures were delivered).
    """
    if not session_id:
        return 0
    blocks: list[dict[str, Any]] = []
    if note:
        blocks.append({"type": "text", "text": note})
    accepted = 0
    for path, ref in images:
        if not path:
            continue
        blocks.append({"type": "attachment", "attachment": _attachment_ref(path, ref=ref)})
        accepted += 1
    if not accepted:
        return 0
    with _lock:
        queue = _pending.setdefault(session_id, [])
        if len(queue) >= MAX_PENDING_PER_SESSION:
            return 0
        queue.append({"role": "user", "content": blocks, "__durable__": durable})
    return accepted


def drain_queued_images(session_id: str | None) -> tuple[list[LoopMessage], list[LoopMessage]]:
    """Take everything queued for this session (and clear it).

    Returns ``(durable, ephemeral)`` — the caller persists the first group and appends the
    second to this round's request only. The marker key is stripped so neither ever reaches
    a provider payload.
    """
    if not session_id:
        return [], []
    with _lock:
        queued = _pending.pop(session_id, [])
    durable: list[LoopMessage] = []
    ephemeral: list[LoopMessage] = []
    for item in queued:
        keep = bool(item.pop("__durable__", True))
        (durable if keep else ephemeral).append(item)
    return durable, ephemeral


def discard_queued_images(session_id: str | None) -> None:
    """Drop anything queued — used when a session ends so a killed run leaves nothing behind."""
    if session_id:
        with _lock:
            _pending.pop(session_id, None)


def _attachment_ref(path: str, *, ref: str = "") -> dict[str, Any]:
    from power_loop._vendor.llm_client.multimodal import create_attachment_ref

    return create_attachment_ref(path, ref=ref)
