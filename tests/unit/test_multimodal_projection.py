"""Multimodal content across the storage → projection → replay boundary.

Two defects are covered here, both of which fail SILENTLY (the model gets prose where an image
was meant, and nobody sees an error):

1. ``_row_to_loop_dict`` (the VERBATIM replay path) did not decode structured content, despite a
   docstring claiming it mirrors ``stateful_loop._row_to_loop_message``. A multimodal turn
   replayed as the literal JSON string it was stored as.
2. ``project_send`` keeps a send's input verbatim — fine for text, ruinous for an inlined
   ``data:`` URL, which is unreadable as text yet re-billed on every later send.
"""

from __future__ import annotations

import json

from power_loop.agent.sink import CONTENT_ENCODING_JSON, CONTENT_ENCODING_META_KEY, _encode_content
from power_loop.runtime.representation import (
    ProjectedRepresentation,
    VerbatimRepresentation,
    distill_multimodal_text,
)
from power_loop.runtime.store.types import MessageRow, ProjectMessageRow, decode_row_content

_JSON_META = {CONTENT_ENCODING_META_KEY: CONTENT_ENCODING_JSON}
_DATA_URL = "data:image/png;base64," + "iVBORw0KGgoAAAANSUhEUg" * 8


def _row(seq: int, role: str, content: str | None, meta: dict | None = None) -> MessageRow:
    return MessageRow(
        session_id="s", seq=seq, role=role, name=None, content=content, tool_calls=None,
        tool_call_id=None, round_index=0, state="active", meta=meta or {}, created_at=0,
        send_index=1,
    )


def _project_rows(rep, rows) -> list[ProjectMessageRow]:
    ps = rep.project_send(rows, send_index=1, tool_registry=None)
    return [
        ProjectMessageRow(
            session_id="s", send_index=1, kind=r.kind, content=r.content, rendered_text=None,
            source_seq_lo=1, source_seq_hi=len(rows), compact_from_send=None,
            compact_to_send=None, projector_version=1, token_estimate=None, created_at=0,
        )
        for r in ps.rows
    ]


# ── encode/decode round-trip ─────────────────────────────────────────────────


def test_encode_decode_round_trip_is_lossless() -> None:
    content = [{"type": "text", "text": "看这个"},
               {"type": "attachment", "attachment": {"path": "/w/a.png", "kind": "image"}}]
    text, structured = _encode_content(content)
    assert structured is True
    assert decode_row_content(text, _JSON_META) == content


def test_plain_text_is_not_flagged_or_decoded() -> None:
    text, structured = _encode_content("just words")
    assert (text, structured) == ("just words", False)
    # A user string that merely LOOKS like JSON must not be decoded — that is why the marker
    # lives in meta rather than being sniffed from the content column.
    assert decode_row_content('[{"type": "text"}]', {}) == '[{"type": "text"}]'


# ── defect 1: verbatim replay dropped the decode ─────────────────────────────


def test_verbatim_replay_restores_structured_content() -> None:
    content = [{"type": "text", "text": "看这个"},
               {"type": "image_url", "image_url": {"url": _DATA_URL}}]
    stored, _ = _encode_content(content)
    rows = _project_rows(VerbatimRepresentation(), [_row(1, "user", stored, _JSON_META)])
    replayed = VerbatimRepresentation().render(rows)
    # A list, not the JSON string it was stored as.
    assert replayed[0]["content"] == content
    assert not isinstance(replayed[0]["content"], str)


def test_verbatim_replay_leaves_plain_text_alone() -> None:
    rows = _project_rows(VerbatimRepresentation(), [_row(1, "user", "hello")])
    assert VerbatimRepresentation().render(rows)[0]["content"] == "hello"


def test_corrupt_payload_degrades_to_raw_text_rather_than_raising() -> None:
    rows = _project_rows(VerbatimRepresentation(), [_row(1, "user", "{not json", _JSON_META)])
    assert VerbatimRepresentation().render(rows)[0]["content"] == "{not json"


# ── defect 2: projection kept base64 verbatim ────────────────────────────────


def test_attachment_block_distills_to_a_usable_reference() -> None:
    out = distill_multimodal_text(json.dumps(
        [{"type": "text", "text": "配色如何？"},
         {"type": "attachment", "attachment": {"path": "/w/shot.png", "filename": "shot.png",
                                               "kind": "image"}}]), _JSON_META)
    # The filename SURVIVES: that is the handle a host uses to fetch the image back later.
    assert out == "配色如何？\n[image: shot.png]"


def test_inlined_data_url_leaves_no_base64_in_the_projection() -> None:
    out = distill_multimodal_text(json.dumps(
        [{"type": "text", "text": "配色如何？"},
         {"type": "image_url", "image_url": {"url": _DATA_URL}}]), _JSON_META)
    assert out == "配色如何？\n[image]"
    assert "base64" not in out


def test_http_image_url_keeps_its_reference() -> None:
    out = distill_multimodal_text(json.dumps(
        [{"type": "image_url", "image_url": {"url": "https://x/a.png"}}]), _JSON_META)
    assert out == "[image: https://x/a.png]"


def test_data_url_is_stripped_even_from_unflagged_plain_text() -> None:
    # Belt and braces: whatever route puts a data URL into a message, it does not reach a
    # projection row. The marker can be missing (older rows, a custom sink).
    assert "base64" not in (distill_multimodal_text(f"see {_DATA_URL} ok", {}) or "")


def test_projection_of_a_multimodal_send_is_small_and_readable() -> None:
    stored, _ = _encode_content([{"type": "text", "text": "配色如何？"},
                                 {"type": "image_url", "image_url": {"url": _DATA_URL}}])
    rep = ProjectedRepresentation()
    rows = _project_rows(rep, [_row(1, "user", stored, _JSON_META), _row(2, "assistant", "还行")])
    user_text = [m for m in rep.render(rows) if m["role"] == "user"][0]["content"]
    assert "base64" not in user_text
    assert "[image]" in user_text
    assert len(user_text) < len(stored) // 4


def test_mid_send_injected_multimodal_row_is_distilled_too() -> None:
    # A user row AFTER the first assistant turn is a mid-send injection (recorded as __user__),
    # which is exactly the shape an image-injection hook produces.
    stored, _ = _encode_content([{"type": "image_url", "image_url": {"url": _DATA_URL}}])
    rep = ProjectedRepresentation()
    rows = _project_rows(rep, [
        _row(1, "user", "开始"), _row(2, "assistant", "好"),
        _row(3, "user", stored, _JSON_META),
    ])
    blob = json.dumps([r.content for r in rows], ensure_ascii=False)
    assert "base64" not in blob
    assert "__user__" in blob
