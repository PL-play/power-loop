"""Real-model cover for the declared-capability contract (images + PDFs).

Runs against the endpoint in ``.env`` — currently ``deepseek-v4-flash-vision-exp``, which
is exactly the model the retired name-guessing table got WRONG: no vendor regex matched it,
so it was classified image-blind and every picture sent to it was silently swapped for an
apology sentence. These tests fail if that ever comes back, because they assert on content
only a model that actually saw the bytes can produce.

Fixtures are built with the stdlib (zlib/struct) on purpose: Pillow and reportlab are not
power-loop dependencies, and a test for "did the image really arrive" must not itself
depend on an image library being installed.
"""

from __future__ import annotations

import asyncio
import struct
import zlib

import pytest

from power_loop._vendor.llm_client.capabilities import ModelCapabilityError
from power_loop._vendor.llm_client.interface import LLMRequest
from power_loop._vendor.llm_client.multimodal import create_attachment_ref
from tests.real._llm import make_llm

VISION = {"supports_image_input": True}


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """A solid-colour PNG, stdlib only."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _text_pdf(lines: list[str]) -> bytes:
    """A minimal one-page text PDF with a valid xref table, stdlib only."""
    text = " ".join(f"({line}) Tj 0 -30 Td" for line in lines)
    stream = f"BT /F1 24 Tf 40 700 Td {text} ET".encode()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj".encode() + body + b"endobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n{xref_at}\n".encode() + b"%%EOF\n"
    )
    return bytes(out)


def _ask_about(path, question: str, *, capabilities=VISION, max_tokens: int = 200) -> str:
    svc = make_llm(max_tokens=max_tokens, temperature=0.0, capabilities=capabilities)
    request = LLMRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "attachment", "attachment": create_attachment_ref(str(path))},
                ],
            }
        ]
    )
    response = asyncio.run(svc.complete(request))
    return (response.content_text or "").strip()


# ── images ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [((220, 20, 20), ("红", "red")), ((20, 90, 220), ("蓝", "blue"))],
)
def test_declared_vision_model_really_sees_the_image(tmp_path, rgb, expected) -> None:
    # The filename is deliberately neutral: an earlier throwaway probe called its fixture
    # "probe_blue.png" and the model answered "blue" WITHOUT the image, from the filename
    # alone — a false green that looked exactly like a pass.
    path = tmp_path / "attachment.png"
    path.write_bytes(_solid_png(64, 64, rgb))
    answer = _ask_about(path, "这张图是什么纯色？只回颜色名，不要解释。")
    assert any(token in answer.lower() for token in expected), answer


def test_same_model_undeclared_raises_instead_of_answering(tmp_path) -> None:
    # Same model, same image — the ONLY difference is that nothing was declared. Before this
    # rework the library shipped the request anyway (minus the image) and returned prose.
    path = tmp_path / "attachment.png"
    path.write_bytes(_solid_png(64, 64, (220, 20, 20)))
    with pytest.raises(ModelCapabilityError, match="has not declared image support"):
        _ask_about(path, "这张图是什么纯色？", capabilities=None)


# ── PDFs ─────────────────────────────────────────────────────────────────────


def test_pdf_text_actually_reaches_the_model(tmp_path) -> None:
    # The passphrase is unguessable, so a correct answer proves the extracted text really
    # travelled — not that the model produced something plausible about "a PDF".
    path = tmp_path / "attachment.pdf"
    path.write_bytes(_text_pdf(["ARTICHOKE-7742", "is the passphrase"]))
    answer = _ask_about(path, "文档里的口令(passphrase)是什么？原样回答，不要解释。")
    assert "ARTICHOKE-7742" in answer.upper(), answer


def test_pdf_needs_no_image_capability(tmp_path) -> None:
    # PDFs are delivered as extracted text on every transport, so a text-only model handles
    # them — declaring image support is neither required nor consulted.
    path = tmp_path / "attachment.pdf"
    path.write_bytes(_text_pdf(["ARTICHOKE-7742", "is the passphrase"]))
    answer = _ask_about(
        path, "文档里的口令(passphrase)是什么？原样回答。", capabilities={"supports_image_input": False}
    )
    assert "ARTICHOKE-7742" in answer.upper(), answer


def test_unreadable_pdf_raises_rather_than_sending_a_placeholder(tmp_path) -> None:
    path = tmp_path / "attachment.pdf"
    path.write_bytes(b"%PDF-1.4\n% image-only export, no extractable text\n")
    with pytest.raises(ModelCapabilityError, match="no text could be extracted"):
        _ask_about(path, "口令是什么？")
