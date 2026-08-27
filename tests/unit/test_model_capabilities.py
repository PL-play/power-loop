"""Declared-capability contract: nothing is inferred, nothing is silently downgraded.

Regression cover for the failure this replaced — a model whose NAME was not in a vendor
regex table (``deepseek-v4-flash-vision-exp``) was judged image-blind, its images were
swapped for an apology sentence, and the caller got a confident answer produced without
ever seeing the picture.
"""

from __future__ import annotations

import base64

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


def test_undeclared_model_raises_instead_of_downgrading(image_path) -> None:
    caps = coerce_capabilities(None, model="deepseek-v4-flash-vision-exp")
    with pytest.raises(ModelCapabilityError) as excinfo:
        render_message_content(_image_content(image_path), role="user", capabilities=caps)
    msg = str(excinfo.value)
    assert "shot.png" in msg  # names the offending file, not just the config field
    assert "deepseek-v4-flash-vision-exp" in msg
    assert "supports_image_input" in msg  # tells you how to fix it


def test_explicitly_unsupported_raises_too(image_path) -> None:
    caps = coerce_capabilities({"supports_image_input": False}, model="text-only")
    with pytest.raises(ModelCapabilityError, match="declared NOT to support"):
        render_message_content(_image_content(image_path), role="user", capabilities=caps)


def test_model_name_no_longer_grants_capability(image_path) -> None:
    # The old regex table handed qwen-vl-plus image support for free. Now the name is inert.
    caps = coerce_capabilities(None, model="qwen-vl-plus")
    with pytest.raises(ModelCapabilityError):
        render_message_content(_image_content(image_path), role="user", capabilities=caps)


def test_to_messages_without_capabilities_raises_rather_than_passing_blocks_through(image_path) -> None:
    # Previously to_messages(None) skipped rendering entirely and shipped the raw
    # {"type": "attachment"} block to the provider.
    req = LLMRequest(messages=[{"role": "user", "content": _image_content(image_path)}], model="m")
    with pytest.raises(ModelCapabilityError):
        req.to_messages()


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


def test_unsupported_attachment_type_raises(tmp_path) -> None:
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"\x00\x01")
    caps = coerce_capabilities({"supports_image_input": True}, model="m")
    content = [{"type": "attachment", "attachment": create_attachment_ref(str(p))}]
    with pytest.raises(ModelCapabilityError, match="unsupported type"):
        render_message_content(content, role="user", capabilities=caps)
