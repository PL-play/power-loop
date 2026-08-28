from __future__ import annotations

import base64
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capabilities import ModelCapabilities, ModelCapabilityError

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)

MAX_PDF_TEXT_CHARS = 24_000


@dataclass(frozen=True)
class AttachmentRef:
    path: str
    filename: str
    mime_type: str
    kind: str
    #: 宿主给的「怎么把这张图找回来」坐标，原样带进蒸馏与降级文案（DeepTalk 放
    #: ``file_uuid=…``，模型可直接拿去 see_image）。空字符串 = 没有。
    ref: str = ""


@dataclass(frozen=True)
class PreparedAttachment:
    ref: AttachmentRef
    text_fallback: str = ""
    rendered_parts: tuple[dict[str, Any], ...] = ()
    strategy: str = "text"


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def create_attachment_ref(path: str | Path, *, ref: str = "") -> dict[str, Any]:
    """``ref`` 是可选的回取坐标，会跟着这张图走到蒸馏与降级文案里——**换到看不了图的模型
    之后**，那行文本就是模型唯一能据以把原图找回来的东西。"""
    file_path = Path(path).expanduser().resolve()
    mime_type = _guess_mime_type(file_path)
    kind = "other"
    if mime_type.startswith("image/"):
        kind = "image"
    elif mime_type == "application/pdf":
        kind = "pdf"
    attachment = AttachmentRef(
        path=str(file_path),
        filename=file_path.name,
        mime_type=mime_type,
        kind=kind,
        ref=ref,
    )
    return attachment.__dict__.copy()


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(str(block))
            continue

        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text") or ""))
            continue
        if block_type == "attachment":
            attachment = block.get("attachment") or {}
            filename = attachment.get("filename") or Path(str(attachment.get("path") or "attachment")).name
            parts.append(f"[Attached file: {filename}]")
            continue
        if "text" in block:
            parts.append(str(block.get("text") or ""))
    return "\n\n".join(part for part in parts if part).strip()


def _file_to_data_url(path: Path, mime_type: str) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _extract_pdf_text(path: Path) -> str:
    if PdfReader is None:
        return ""

    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""

    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = ""
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append(f"[Page {index}]\n{text}")

    return "\n\n".join(pages)[:MAX_PDF_TEXT_CHARS]


def _downscale_to_data_url(path: Path, mime_type: str, max_edge: int | None) -> str:
    """Data URL for ``path``, downscaled so its longest edge is at most ``max_edge``.

    A file already within the limit is streamed through UNTOUCHED — no decode, no re-encode —
    so the common case (a host that already sized the image) costs nothing. Pillow is optional;
    without it the image is sent at its original size rather than not at all, with one warning.
    """
    if not max_edge or max_edge <= 0:
        return _file_to_data_url(path, mime_type)
    try:
        import io

        from PIL import Image
    except ImportError:
        logger.warning(
            "max_image_edge=%s requested but Pillow is not installed; sending %s at its "
            "original size. Install power-loop[images] (or Pillow) to enforce the limit.",
            max_edge, path.name,
        )
        return _file_to_data_url(path, mime_type)
    try:
        with Image.open(path) as img:
            if max(img.width, img.height) <= max_edge:
                return _file_to_data_url(path, mime_type)  # already within budget
            scale = max_edge / float(max(img.width, img.height))
            resized = img.convert("RGB").resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            )
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=85, optimize=True)
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
    except OSError:
        # 文件读不到不是「缩放失败」——报 "sending it at original size" 会把调用方引到
        # 图像解码上，而真正的毛病在路径。交给上层按「读不到」降级。
        raise
    except Exception:  # noqa: BLE001 — a decode failure must not lose the image entirely
        logger.warning("could not downscale %s; sending it at original size", path.name,
                       exc_info=True)
        return _file_to_data_url(path, mime_type)


def _render_image_attachment(ref: AttachmentRef, path: Path, capabilities: ModelCapabilities) -> PreparedAttachment:
    """图片 → 原生 image 块；模型没声明看图能力时 → **明确的**文本占位。

    为什么不抛：一次 render 同时渲染**历史**和本轮输入。一个定义换到看不了图的模型之后，
    历史里的图会让整个 send 抛异常、会话彻底不可用——而历史是既成事实，不是调用方的错。

    为什么降级不等于回到老毛病：老实现塞的是一句含糊的
    "The current model does not support image input"，混在附件描述里，模型读过去照样按
    「我看过这张图」的语气编答案。这里的占位必须做到两件事——**说清楚模型没有看到**，并且
    **给出把图找回来的坐标**（``ref``，DeepTalk 放 file_uuid，可直接喂给 see_image）。
    调用方想要「发图给看不了图的模型就报错」，用 ``capabilities.require_image_input()`` 自查。
    """
    if capabilities.supports_image_input is True:
        try:
            url = _downscale_to_data_url(path, ref.mime_type, capabilities.max_image_edge)
        except OSError as exc:
            # 文件读不到（路径解析错、被清理、权限）——**降级，不抛**。抛出去的代价与
            # 能力不匹配那一路完全一样：一次 render 同时渲染历史与本轮，一个读不到的
            # 附件会让每一次 send 都失败，重试耗尽后整个 run 终止。DeepTalk conv-198
            # 真实发生过：两张刚生成的图因相对路径被按进程 cwd 解析而找不到，agent 就此
            # 停在半路，图既没被看到也没发出去。占位文本同样必须说清「你没有看到」。
            logger.warning("image %s could not be read (%s); surfaced as a text placeholder",
                           ref.filename, exc)
            text = (
                f"[图片 {describe_attachment_ref(ref)}——**这个文件读不到，你没有看到它的内容**。"
                "不要凭空描述这张图；需要看就按上面的坐标用视觉工具重新取它。]"
            )
            return PreparedAttachment(
                ref=ref,
                text_fallback=text,
                rendered_parts=({"type": "text", "text": text},),
                strategy="image-unreadable",
            )
        part = {"type": "image_url", "image_url": {"url": url}}
        return PreparedAttachment(ref=ref, rendered_parts=(part,), strategy="native-image")

    logger.warning(
        "image %s not sent: model %r has not declared image support (supports_image_input=%r); "
        "surfaced as a text placeholder instead",
        ref.filename, capabilities.model or "<unnamed>", capabilities.supports_image_input,
    )
    text = (
        f"[图片 {describe_attachment_ref(ref)}——**当前模型看不了图片，你没有看到它的内容**。"
        "不要凭空描述这张图；需要看就用 see_image 之类的视觉工具，按上面的坐标取它。]"
    )
    return PreparedAttachment(
        ref=ref,
        text_fallback=text,
        rendered_parts=({"type": "text", "text": text},),
        strategy="image-unsupported",
    )


def describe_attachment_ref(ref: AttachmentRef) -> str:
    """``shot.png · file_uuid=491b…`` —— 蒸馏与降级共用的一行标识。"""
    return f"{ref.filename} · {ref.ref}" if ref.ref else ref.filename


def _render_pdf_attachment(ref: AttachmentRef, path: Path, capabilities: ModelCapabilities) -> PreparedAttachment:
    """PDFs are always delivered as EXTRACTED TEXT.

    Native PDF transmission is not implemented by this library on any transport, so a
    ``supports_pdf_input_*`` capability field would be a promise nothing keeps — it was
    removed rather than left as decoration. Text extraction is a faithful path for a text
    PDF (the content really does reach the model), so it is not a silent downgrade and does
    not raise.

    What DOES raise: a PDF no text can be extracted from (a scan, an image-only export, an
    encrypted file). Feeding the model "[Attached PDF: x.pdf] no readable text" invites the
    same unfounded-answer failure the image path just eliminated.
    """
    extracted_text = _extract_pdf_text(path)
    if not extracted_text:
        raise ModelCapabilityError(
            f"Cannot send PDF {ref.filename!r}: no text could be extracted from it "
            "(scanned/image-only or encrypted PDF), and native PDF input is not implemented "
            "on any transport. Convert its pages to images and send those to a model that "
            "declares supports_image_input, or extract the text yourself."
        )
    text = f"[Attached PDF: {ref.filename}]\n\n{extracted_text}"
    return PreparedAttachment(
        ref=ref,
        text_fallback=text,
        rendered_parts=({"type": "text", "text": text},),
        strategy="pdf-extracted-text",
    )


def prepare_attachment(ref_payload: dict[str, Any], capabilities: ModelCapabilities) -> PreparedAttachment:
    ref = AttachmentRef(
        path=str(ref_payload.get("path") or ""),
        filename=str(ref_payload.get("filename") or Path(str(ref_payload.get("path") or "attachment")).name),
        mime_type=str(ref_payload.get("mime_type") or "application/octet-stream"),
        kind=str(ref_payload.get("kind") or "other"),
        ref=str(ref_payload.get("ref") or ""),
    )
    path = Path(ref.path)

    if ref.kind == "image":
        return _render_image_attachment(ref, path, capabilities)
    if ref.kind == "pdf":
        return _render_pdf_attachment(ref, path, capabilities)

    raise ModelCapabilityError(
        f"Cannot send attachment {ref.filename!r}: unsupported type {ref.mime_type!r}. "
        "This library renders images (declared models only) and text-extractable PDFs; "
        "anything else must be converted by the caller. It is not summarised into a "
        "placeholder line — a placeholder reads to the model as if the file had been read."
    )


def render_message_content(content: Any, role: str, capabilities: ModelCapabilities) -> Any:
    if not isinstance(content, list) or role != "user":
        return content

    rendered: list[dict[str, Any]] = []
    debug_strategies: list[str] = []
    for block in content:
        if isinstance(block, str):
            rendered.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            rendered.append({"type": "text", "text": str(block)})
            continue

        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("text") or "")
            if text:
                rendered.append({"type": "text", "text": text})
            continue

        if block_type == "attachment":
            prepared = prepare_attachment(block.get("attachment") or {}, capabilities)
            debug_strategies.append(prepared.strategy)
            rendered.extend(prepared.rendered_parts)
            continue

        rendered.append(block)

    text_parts = [part.get("text", "") for part in rendered if isinstance(part, dict) and part.get("type") == "text"]
    non_text_parts = [part for part in rendered if not (isinstance(part, dict) and part.get("type") == "text")]

    if not non_text_parts:
        return "\n\n".join(text for text in text_parts if text).strip()

    if text_parts:
        merged: list[dict[str, Any]] = [{"type": "text", "text": "\n\n".join(text for text in text_parts if text).strip()}]
        merged.extend(non_text_parts)
        return [part for part in merged if not (part.get("type") == "text" and not part.get("text"))]

    return non_text_parts