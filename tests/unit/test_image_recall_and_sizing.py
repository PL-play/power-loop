"""按需图片回取 + 尺寸裁剪（design/75 P2）。

两件事在这里保证：

1. **尺寸裁剪在渲染的唯一汇点生效**，所以「新发的图」和「recall 回来的图」都被覆盖，不需要
   每条入口各记得裁一次。图片按**像素**计费而非字节，所以这是唯一真正省钱的旋钮。
2. **recall 一行多模态记录不会把 base64 吐给模型**：文本侧蒸馏，图片侧另起一条 user 消息
   放回眼前，且只活一轮。
"""

from __future__ import annotations

import base64
import io
import struct
import zlib
from pathlib import Path

import pytest

from power_loop._vendor.llm_client.capabilities import coerce_capabilities
from power_loop._vendor.llm_client.multimodal import create_attachment_ref, render_message_content
from power_loop.runtime.image_recall import (
    MAX_PENDING_PER_SESSION,
    discard_queued_images,
    drain_queued_images,
    queue_image_for_next_round,
)


def _png(w: int, h: int) -> bytes:
    raw = b"".join(b"\x00" + bytes((200, 40, 40)) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return struct.pack(">I", len(data)) + payload + struct.pack(
            ">I", zlib.crc32(payload) & 0xFFFFFFFF
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _render(path: Path, max_edge: int | None):
    caps = coerce_capabilities(
        {"supports_image_input": True, "max_image_edge": max_edge}, model="m"
    )
    out = render_message_content(
        [{"type": "attachment", "attachment": create_attachment_ref(str(path))}],
        role="user", capabilities=caps,
    )
    url = out[0]["image_url"]["url"]
    header, payload = url.split(",", 1)
    return header, base64.b64decode(payload)


def _dims(data: bytes):
    Image = pytest.importorskip("PIL.Image")
    with Image.open(io.BytesIO(data)) as img:
        return img.size


# ── 尺寸裁剪 ────────────────────────────────────────────────────────────────


def test_oversized_image_is_downscaled_at_the_render_point(tmp_path) -> None:
    p = tmp_path / "big.png"
    p.write_bytes(_png(1600, 1200))
    _, data = _render(p, 768)
    assert _dims(data) == (768, 576)


def test_image_already_within_budget_passes_through_untouched(tmp_path) -> None:
    # 未超限就不解码不重编码——常见情况（宿主已经缩过）不该付任何代价。
    p = tmp_path / "small.png"
    p.write_bytes(_png(320, 240))
    header, data = _render(p, 768)
    assert header == "data:image/png;base64"  # 仍是原始 PNG，没被转成 JPEG
    assert len(data) == p.stat().st_size


def test_no_limit_means_send_as_is(tmp_path) -> None:
    p = tmp_path / "big.png"
    p.write_bytes(_png(1600, 1200))
    header, data = _render(p, None)
    assert header == "data:image/png;base64"
    assert _dims(data) == (1600, 1200)


def test_limit_applies_to_a_recalled_image_too(tmp_path) -> None:
    # 关键：裁剪挂在渲染汇点，所以 recall 放回来的图同样被裁——不靠每条入口各自记得。
    p = tmp_path / "recalled.png"
    p.write_bytes(_png(2000, 1000))
    discard_queued_images("s-size")
    assert queue_image_for_next_round("s-size", path=str(p), note="回取的图")
    queued = drain_queued_images("s-size")
    caps = coerce_capabilities(
        {"supports_image_input": True, "max_image_edge": 512}, model="m"
    )
    out = render_message_content(queued[0]["content"], role="user", capabilities=caps)
    payload = base64.b64decode(out[-1]["image_url"]["url"].split(",", 1)[1])
    assert _dims(payload) == (512, 256)


def test_recalled_image_degrades_when_the_model_cannot_see(tmp_path) -> None:
    # 回取不是绕过声明的后门；但定义换到看不了图的模型后也不能因此炸掉——降级并带出路。
    p = tmp_path / "r.png"
    p.write_bytes(_png(64, 64))
    discard_queued_images("s-caps")
    queue_image_for_next_round("s-caps", path=str(p), ref="file_uuid=abc-1")
    queued = drain_queued_images("s-caps")
    out = render_message_content(
        queued[0]["content"], role="user", capabilities=coerce_capabilities(None, model="m")
    )
    assert isinstance(out, str)
    assert "没有看到" in out and "file_uuid=abc-1" in out


# ── 回取队列：只活一轮 ──────────────────────────────────────────────────────


def test_queue_is_drained_once_and_then_empty(tmp_path) -> None:
    p = tmp_path / "a.png"
    p.write_bytes(_png(8, 8))
    discard_queued_images("s1")
    assert queue_image_for_next_round("s1", path=str(p), note="看这个")
    first = drain_queued_images("s1")
    assert [b["type"] for b in first[0]["content"]] == ["text", "attachment"]
    # 只活一轮：下一轮就没了，要再看得再 recall 一次。
    assert drain_queued_images("s1") == []


def test_queue_refuses_without_a_session(tmp_path) -> None:
    p = tmp_path / "a.png"
    p.write_bytes(_png(8, 8))
    assert queue_image_for_next_round(None, path=str(p)) is False
    assert queue_image_for_next_round("s", path="") is False


def test_queue_is_bounded(tmp_path) -> None:
    p = tmp_path / "a.png"
    p.write_bytes(_png(8, 8))
    discard_queued_images("s-cap")
    accepted = sum(
        bool(queue_image_for_next_round("s-cap", path=str(p)))
        for _ in range(MAX_PENDING_PER_SESSION + 4)
    )
    assert accepted == MAX_PENDING_PER_SESSION
    discard_queued_images("s-cap")


def test_sessions_do_not_leak_into_each_other(tmp_path) -> None:
    p = tmp_path / "a.png"
    p.write_bytes(_png(8, 8))
    discard_queued_images("s-a")
    discard_queued_images("s-b")
    queue_image_for_next_round("s-a", path=str(p))
    assert drain_queued_images("s-b") == []
    assert len(drain_queued_images("s-a")) == 1


# ── recall 一行：文本蒸馏，两种记录都要对 ──────────────────────────────────


class _Row:
    def __init__(self, content, meta=None, *, seq=1, role="user", name=None):
        self.content, self.meta, self.seq, self.role, self.name = content, meta or {}, seq, role, name
        self.tool_calls = None
        self.tool_call_id = None
        self.round_index = 0


def test_recall_body_leaves_plain_text_alone() -> None:
    # 非多模态记录必须原样——蒸馏不能把普通文本改坏。
    from power_loop.tools.default_tools import _recall_row_body

    text, images = _recall_row_body(_Row("普通的一段文字，含 [方括号] 和 {花括号}"))
    assert text == "普通的一段文字，含 [方括号] 和 {花括号}"
    assert images == []


def test_recall_body_distils_multimodal_and_reports_images(tmp_path) -> None:
    import json

    p = tmp_path / "shot.png"
    p.write_bytes(_png(8, 8))
    content = json.dumps([
        {"type": "text", "text": "配色如何？"},
        {"type": "attachment", "attachment": create_attachment_ref(str(p))},
    ], ensure_ascii=False)
    from power_loop.tools.default_tools import _recall_row_body

    text, images = _recall_row_body(_Row(content, {"content_encoding": "json"}))
    assert text == "配色如何？\n[image: shot.png]"
    assert images == [(str(p), "")]


def test_recall_body_never_returns_an_inlined_base64_payload() -> None:
    import json

    data_url = "data:image/png;base64," + "A" * 5000
    content = json.dumps([{"type": "image_url", "image_url": {"url": data_url}}])
    from power_loop.tools.default_tools import _recall_row_body

    text, images = _recall_row_body(_Row(content, {"content_encoding": "json"}))
    assert "base64" not in text
    assert text == "[image]"
    assert images == []  # data URL 没有可回取的路径


def test_recall_body_survives_a_corrupt_payload() -> None:
    from power_loop.tools.default_tools import _recall_row_body

    text, images = _recall_row_body(_Row("{not json", {"content_encoding": "json"}))
    assert text == "{not json"
    assert images == []

def test_recall_carries_the_hosts_ref_through(tmp_path) -> None:
    import json

    p = tmp_path / "shot.png"
    p.write_bytes(_png(8, 8))
    content = json.dumps([{
        "type": "attachment",
        "attachment": create_attachment_ref(str(p), ref="file_uuid=491b-abc"),
    }], ensure_ascii=False)
    from power_loop.tools.default_tools import _recall_row_body

    text, images = _recall_row_body(_Row(content, {"content_encoding": "json"}))
    # 蒸馏行与回取坐标都带着 ref：换到看不了图的模型时，这是唯一的出路。
    assert text == "[image: shot.png · file_uuid=491b-abc]"
    assert images == [(str(p), "file_uuid=491b-abc")]


def test_switching_to_a_text_only_model_does_not_break_a_verbatim_replay(tmp_path) -> None:
    """回归：换模型后 verbatim 重放历史里的图，曾让整个 send 抛异常、会话彻底不可用。"""
    import json

    from power_loop.agent.sink import _encode_content
    from power_loop.runtime.representation import ProjectMessageRow, VerbatimRepresentation
    from power_loop.runtime.store.types import MessageRow

    p = tmp_path / "hist.png"
    p.write_bytes(_png(16, 16))
    stored, _ = _encode_content([
        {"type": "text", "text": "看这张图"},
        {"type": "attachment", "attachment": create_attachment_ref(str(p), ref="file_uuid=u-9")},
    ])
    row = MessageRow(session_id="s", seq=1, role="user", name=None, content=stored,
                     tool_calls=None, tool_call_id=None, round_index=0, state="active",
                     meta={"content_encoding": "json"}, created_at=0, send_index=1)
    rep = VerbatimRepresentation()
    ps = rep.project_send([row], send_index=1, tool_registry=None)
    stored_rows = [ProjectMessageRow(
        session_id="s", send_index=1, kind=r.kind, content=r.content, rendered_text=None,
        source_seq_lo=1, source_seq_hi=1, compact_from_send=None, compact_to_send=None,
        projector_version=1, token_estimate=None, created_at=0) for r in ps.rows]

    from power_loop._vendor.llm_client.interface import LLMRequest

    replayed = rep.render(stored_rows)
    out = LLMRequest(messages=replayed, model="m").to_messages(
        coerce_capabilities(None, model="deepseek-v4-flash")  # 换成看不了图的模型
    )
    blob = json.dumps(out, ensure_ascii=False)
    assert "image_url" not in blob          # 没有图片被硬发出去
    assert "file_uuid=u-9" in blob          # 但找回原图的坐标还在
    assert "没有看到" in blob                # 且模型知道自己没看到
