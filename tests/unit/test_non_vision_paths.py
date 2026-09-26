"""A model that cannot see images must be told so — never shown a placeholder under a line
claiming the picture is in front of it, never sent an image block it cannot take.

Most configured models are text-only; the one the real suite runs on happens to see images.
These pin the text-only side of every path that puts a picture in front of the model.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from power_loop._vendor.llm_client.capabilities import ModelCapabilities, coerce_capabilities
from power_loop._vendor.llm_client.interface import LLMRequest
from power_loop._vendor.llm_client.multimodal import create_attachment_ref, render_message_content
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.core.pipeline import AgentPipeline
from power_loop.runtime import image_recall
from power_loop.runtime.store.types import CONTENT_ENCODING_JSON, CONTENT_ENCODING_META_KEY
from power_loop.tools.default_tools import _render_recall_row

pytestmark = pytest.mark.unit

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
VISION = {"supports_image_input": True}


def _loop(caps: dict | None, *, model: str = "main-model", loop_model: str | None = None):
    llm = SimpleNamespace(capabilities=coerce_capabilities(caps, model=model))
    return SimpleNamespace(llm=llm, config=SimpleNamespace(model=loop_model))


class _InLoop:
    def __init__(self, loop, sid: str = "s1") -> None:
        self.loop, self.sid = loop, sid

    def __enter__(self):
        self._lt, self._st = set_current_loop(self.loop), set_session_id(self.sid)
        return self

    def __exit__(self, *exc):
        reset_session_id(self._st)
        reset_current_loop(self._lt)
        image_recall.discard_queued_images(self.sid)


# ── a declaration belongs to one model ────────────────────────────────────


def test_declaration_applies_only_to_the_declared_model() -> None:
    caps = coerce_capabilities(VISION, model="vision-model")
    assert caps.for_model(None) is caps and caps.for_model("vision-model") is caps
    other = caps.for_model("text-only-model")
    assert other.supports_image_input is None and other.model == "text-only-model"
    assert not other.sees_images and caps.sees_images


def test_sub_agent_on_another_model_does_not_inherit_vision(tmp_path) -> None:
    """A child that overrides ``model`` on the parent's client used to be rendered with the
    PARENT's declaration — images went to a model that cannot take them."""
    img = tmp_path / "shot.png"
    img.write_bytes(_PNG)
    content = [{"type": "text", "text": "what?"},
               {"type": "attachment", "attachment": create_attachment_ref(str(img))}]
    caps = coerce_capabilities(VISION, model="vision-model")
    req = LLMRequest(messages=[{"role": "user", "content": content}], model="text-only-model")
    rendered = req.to_messages(caps.for_model(req.model))[0]["content"]
    assert isinstance(rendered, str) and "当前模型看不了图片" in rendered
    same = LLMRequest(messages=[{"role": "user", "content": content}], model="vision-model")
    parts = same.to_messages(caps.for_model(same.model))[0]["content"]
    assert any(p.get("type") == "image_url" for p in parts)


# ── ready-made image blocks go through the same gate as attachments ───────


@pytest.mark.parametrize("block", [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    {"type": "input_image", "image_url": "https://x/y.png"},
    {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
])
def test_raw_image_block_to_a_blind_model_becomes_a_placeholder(block) -> None:
    out = render_message_content([{"type": "text", "text": "look"}, block], "user",
                                 ModelCapabilities(model="m"))
    assert isinstance(out, str) and "look" in out and "当前模型看不了图片" in out
    kept = render_message_content([block], "user", ModelCapabilities(model="m", supports_image_input=True))
    assert kept == [block]


# ── queueing a picture for the next round ─────────────────────────────────


def test_queue_refuses_when_the_current_model_cannot_see(tmp_path) -> None:
    with _InLoop(_loop(None)):
        assert image_recall.current_model_sees_images() is False
        assert image_recall.queue_image_for_next_round("s1", path=str(tmp_path / "a.png")) is False
        assert image_recall.queue_images_for_next_round("s1", [(str(tmp_path / "a.png"), "")]) == 0
        assert image_recall.drain_queued_images("s1") == ([], [])


def test_queue_accepts_for_a_vision_model_and_when_it_cannot_be_told(tmp_path) -> None:
    with _InLoop(_loop(VISION)):
        assert image_recall.current_model_sees_images() is True
        assert image_recall.queue_image_for_next_round("s1", path=str(tmp_path / "a.png"))
    # a child loop overriding the model on a vision client: nothing declared for it
    with _InLoop(_loop(VISION, loop_model="other-model")):
        assert image_recall.current_model_sees_images() is False
    # no loop in context / a client wrapper without `capabilities`: unknown → old behaviour
    assert image_recall.current_model_sees_images() is None
    with _InLoop(SimpleNamespace(llm=object(), config=None)):
        assert image_recall.current_model_sees_images() is None
        assert image_recall.queue_image_for_next_round("s1", path=str(tmp_path / "a.png"))


# ── recall_send of a row that held images ─────────────────────────────────


def _image_row(tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(_PNG)
    blocks = [{"type": "text", "text": "看这张"},
              {"type": "attachment", "attachment": create_attachment_ref(str(img), ref="file_uuid=abc")}]
    return SimpleNamespace(role="user", seq=7, name=None, tool_call_id=None,
                           content=json.dumps(blocks),
                           meta={CONTENT_ENCODING_META_KEY: CONTENT_ENCODING_JSON})


def test_recall_row_with_images_on_a_blind_model_says_so(tmp_path) -> None:
    row = _image_row(tmp_path)
    with _InLoop(_loop(None)):
        out = _render_recall_row(row, [row], cap=4000, head="send #1")
        assert image_recall.drain_queued_images("s1") == ([], [])
    assert "放到你眼前" not in out
    assert "当前模型看不了图片" in out and "shot.png · file_uuid=abc" in out


def test_recall_row_with_images_on_a_vision_model_puts_them_back(tmp_path) -> None:
    row = _image_row(tmp_path)
    with _InLoop(_loop(VISION)):
        out = _render_recall_row(row, [row], cap=4000, head="send #1")
        durable, _ = image_recall.drain_queued_images("s1")
    assert "1 张图已放到你眼前" in out and len(durable) == 1


# ── retiring a picture after its rounds ───────────────────────────────────


def _retire(caps: dict | None, *, loop_model: str | None = None) -> str:
    msg = {"role": "user", "content": [
        {"type": "attachment", "attachment": {"kind": "image", "path": "/w/shot.png",
                                              "filename": "shot.png", "ref": "file_uuid=abc"}},
    ]}
    fake = SimpleNamespace(
        _image_rounds={id(msg): 0}, history=[msg], _IMAGE_RETIRED_MARK="[image retired",
        llm=SimpleNamespace(capabilities=coerce_capabilities(caps, model="main-model")),
        config=SimpleNamespace(model=loop_model),
    )
    assert AgentPipeline._retire_stale_images(fake, round_index=3, keep_rounds=1) == 1
    return msg["content"][0]["text"]


def test_retired_image_on_a_blind_model_is_not_called_seen() -> None:
    text = _retire(None)
    assert "已看过" not in text and "没看到过" in text and "file_uuid=abc" in text


def test_retired_image_on_a_vision_model_keeps_its_recall_coordinate() -> None:
    text = _retire(VISION)
    assert "已看过" in text and "shot.png · file_uuid=abc" in text
    assert "没看到过" in _retire(VISION, loop_model="other-model")
