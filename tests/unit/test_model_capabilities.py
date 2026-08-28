"""Declared-capability contract: nothing is inferred, nothing is silently downgraded.

Regression cover for the failure this replaced — a model whose NAME was not in a vendor
regex table (``deepseek-v4-flash-vision-exp``) was judged image-blind, its images were
swapped for an apology sentence, and the caller got a confident answer produced without
ever seeing the picture.
"""

from __future__ import annotations

import base64
import json

import pytest

from power_loop._vendor.llm_client.capabilities import (
    ModelCapabilities,
    ModelCapabilityError,
    coerce_capabilities,
)
from power_loop._vendor.llm_client.interface import LLMRequest
from power_loop._vendor.llm_client.multimodal import create_attachment_ref, render_message_content

# 1x1 PNG — keeps the test free of a Pillow dependency.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def image_path(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(_PNG)
    return p


def _image_content(path):
    return [
        {"type": "text", "text": "what colour?"},
        {"type": "attachment", "attachment": create_attachment_ref(str(path))},
    ]


# ── tri-state ────────────────────────────────────────────────────────────────


def test_undeclared_is_the_default_and_is_not_false() -> None:
    caps = ModelCapabilities(model="m")
    assert caps.supports_image_input is None  # undeclared, NOT False


def test_coerce_accepts_dict_instance_and_none() -> None:
    assert coerce_capabilities({"supports_image_input": True}, model="m").supports_image_input is True
    assert coerce_capabilities(None, model="m").supports_image_input is None
    inst = ModelCapabilities(model="m", supports_image_input=False)
    assert coerce_capabilities(inst).supports_image_input is False


def test_retired_capability_keys_are_rejected_not_ignored() -> None:
    # supports_tools/stream/api_family/provider/pdf were read by nothing; accepting them
    # silently would let a config claim a capability that is never honoured.
    for dead in ("supports_tools", "supports_stream", "api_family", "supports_pdf_input_chat"):
        with pytest.raises(ValueError, match="Unknown model capability"):
            coerce_capabilities({dead: True})


# ── images: declared, or it raises ───────────────────────────────────────────


def test_declared_image_support_renders_native_data_url(image_path) -> None:
    caps = coerce_capabilities({"supports_image_input": True}, model="vision-model")
    out = render_message_content(_image_content(image_path), role="user", capabilities=caps)
    assert [b["type"] for b in out] == ["text", "image_url"]
    assert out[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_undeclared_model_gets_a_placeholder_that_says_it_did_not_see_the_image(image_path) -> None:
    """未声明能力时降级，但降级必须是**响亮**的。

    这里不抛异常，是因为同一次 render 同时渲染历史与本轮输入：一个定义换到看不了图的模型
    之后，历史里的图会让整个 send 抛异常、会话彻底不可用——而历史是既成事实。降级之所以不
    等于回到老毛病，全在文案：必须说清楚模型**没有看到**，并给出把图找回来的坐标。
    """
    caps = coerce_capabilities(None, model="deepseek-v4-flash")
    text = render_message_content(_image_content(image_path), role="user", capabilities=caps)
    # 全是文本块时渲染器合并成一个字符串——也就是说没有任何图片块被发出去。
    assert isinstance(text, str)
    assert "shot.png" in text                     # 指明是哪张图
    assert "看不了图片" in text and "没有看到" in text  # 模型不会以为自己看过
    assert "see_image" in text                    # 给出路


def test_placeholder_carries_the_hosts_recall_coordinate(tmp_path) -> None:
    # 换模型之后，这行文本是模型唯一能据以把原图找回来的东西。
    p = tmp_path / "shot.png"
    p.write_bytes(_PNG)
    content = [{"type": "attachment",
                "attachment": create_attachment_ref(str(p), ref="file_uuid=491b-abc")}]
    caps = coerce_capabilities({"supports_image_input": False}, model="text-only")
    assert "file_uuid=491b-abc" in render_message_content(
        content, role="user", capabilities=caps
    )


def test_explicitly_unsupported_also_degrades(image_path) -> None:
    caps = coerce_capabilities({"supports_image_input": False}, model="text-only")
    out = render_message_content(_image_content(image_path), role="user", capabilities=caps)
    assert isinstance(out, str) and "看不了图片" in out


def test_model_name_no_longer_grants_capability(image_path) -> None:
    # 旧正则表会白送 qwen-vl-plus 图片能力。现在名字完全不作数——没声明就是没声明。
    caps = coerce_capabilities(None, model="qwen-vl-plus")
    out = render_message_content(_image_content(image_path), role="user", capabilities=caps)
    assert isinstance(out, str) and "看不了图片" in out


def test_callers_can_still_assert_strictly(image_path) -> None:
    # 渲染宽容，但「发图给看不了图的模型」仍可当场判死：宿主自查用 require_image_input()。
    caps = coerce_capabilities(None, model="m")
    with pytest.raises(ModelCapabilityError, match="has not declared image support"):
        caps.require_image_input(what="image 'shot.png'")


def test_to_messages_without_capabilities_does_not_leak_raw_blocks(image_path) -> None:
    # 以前 to_messages(None) 整个跳过渲染，把 {"type": "attachment"} 原样发给 provider。
    req = LLMRequest(messages=[{"role": "user", "content": _image_content(image_path)}], model="m")
    rendered = req.to_messages()
    blob = json.dumps(rendered, ensure_ascii=False)
    assert "attachment" not in blob or "看不了图片" in blob
    assert "image_url" not in blob


def test_assistant_content_is_untouched(image_path) -> None:
    # Only user turns carry attachments; an assistant turn must not be re-rendered.
    caps = coerce_capabilities(None, model="m")
    assert render_message_content("plain reply", role="assistant", capabilities=caps) == "plain reply"


# ── PDFs: extracted text is a faithful path, an unreadable PDF is not ────────


def test_unreadable_pdf_raises(tmp_path) -> None:
    p = tmp_path / "scan.pdf"
    p.write_bytes(b"%PDF-1.4\n% not a real pdf body\n")
    caps = coerce_capabilities({"supports_image_input": True}, model="m")
    content = [{"type": "attachment", "attachment": create_attachment_ref(str(p))}]
    with pytest.raises(ModelCapabilityError, match="no text could be extracted"):
        render_message_content(content, role="user", capabilities=caps)


def test_unsupported_attachment_type_degrades(tmp_path) -> None:
    """认不出的类型 → 明说没读到的占位，**不抛**。

    原先抛 ModelCapabilityError，理由是「占位读起来像文件被读过」——那对含糊的占位成立，
    对这句不成立（它明说模型没拿到内容）。而抛出去的代价是整个会话不可用：历史里躺着一个
    认不出类型的附件，之后每一次 send 都失败。
    """
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"\x00\x01")
    caps = coerce_capabilities({"supports_image_input": True}, model="m")
    content = [{"type": "attachment", "attachment": create_attachment_ref(str(p))}]
    out = render_message_content(content, role="user", capabilities=caps)
    assert isinstance(out, str)
    assert "clip.mp3" in out and "没有读到" in out


def test_webp_is_recognised_even_when_the_system_mime_table_is_not(tmp_path, monkeypatch) -> None:
    """回归 DeepTalk conv-201：容器里 `guess_type('.webp')` 返回 None。

    宿主 Python 认得 webp，生产镜像不认——于是 MIME 落到 application/octet-stream，
    渲染层判定「这不是图片」，agent 刚把配图处理成 webp、正要看一眼，会话就死在那里。
    本地全绿、生产炸掉，只能靠内置兜底表挡住。
    """
    import mimetypes as _mt

    from power_loop._vendor.llm_client.multimodal import create_attachment_ref as _mk

    monkeypatch.setattr(_mt, "guess_type", lambda *a, **k: (None, None))
    p = tmp_path / "shot.webp"
    p.write_bytes(_PNG)                       # 字节是什么不重要，这里验的是类型判定
    ref = _mk(str(p))                          # create_attachment_ref 返回的是 dict 负载
    assert ref["mime_type"] == "image/webp" and ref["kind"] == "image"


def test_missing_file_degrades_instead_of_killing_the_send(tmp_path) -> None:
    """图片文件读不到 → 占位文本，**不抛**。

    回归自 DeepTalk conv-198：宿主把一个相对路径交给渲染器，它按进程 cwd 解析，两张刚生成
    的图找不到。当时这里让 FileNotFoundError 穿了出去，于是**每一次** send 都失败——图在
    历史里，重试三次全抛，整个 run 降级终止。agent 停在半路，图既没被模型看到，也没发给用户。

    宿主的路径 bug 该修（那是根因），但渲染器不能把「一个附件读不到」放大成「会话不可用」。
    降级文案同样必须说清模型没有看到，并留下坐标。
    """
    content = [{"type": "attachment",
                "attachment": create_attachment_ref(str(tmp_path / "gone.png"),
                                                    ref="file_uuid=491b-abc")}]
    caps = coerce_capabilities({"supports_image_input": True}, model="vision-model")
    out = render_message_content(content, role="user", capabilities=caps)
    assert isinstance(out, str)                 # 没有任何 image_url 块发出去
    assert "读不到" in out and "没有看到" in out  # 模型不会以为自己看过
    assert "file_uuid=491b-abc" in out          # 还能把图找回来
