from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .capabilities import ModelCapabilities

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment]


MAX_PDF_TEXT_CHARS = 24_000


@dataclass(frozen=True)
class AttachmentRef:
    path: str
    filename: str
    mime_type: str
    kind: str


@dataclass(frozen=True)
class PreparedAttachment:
    ref: AttachmentRef
    text_fallback: str = ""
    rendered_parts: tuple[Dict[str, Any], ...] = ()
    strategy: str = "text"


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def create_attachment_ref(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    mime_type = _guess_mime_type(file_path)
    kind = "other"
    if mime_type.startswith("image/"):
        kind = "image"
    elif mime_type == "application/pdf":
        kind = "pdf"
    ref = AttachmentRef(
        path=str(file_path),
        filename=file_path.name,
        mime_type=mime_type,
        kind=kind,
    )
    return ref.__dict__.copy()


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    parts: List[str] = []
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

    pages: List[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = ""
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append(f"[Page {index}]\n{text}")

    return "\n\n".join(pages)[:MAX_PDF_TEXT_CHARS]


def _render_image_attachment(ref: AttachmentRef, path: Path, capabilities: ModelCapabilities) -> PreparedAttachment:
    if capabilities.supports_image_input and capabilities.supports_data_url:
        part = {
            "type": "image_url",
            "image_url": {
                "url": _file_to_data_url(path, ref.mime_type),
            },
        }
        return PreparedAttachment(ref=ref, rendered_parts=(part,), strategy="native-image")

    text = (
        f"[Attached image: {ref.filename}]\n"
        "The current model does not support image input, so the image was not sent natively."
    )
    return PreparedAttachment(
        ref=ref,
        text_fallback=text,
        rendered_parts=({"type": "text", "text": text},),
        strategy="image-unsupported",
    )


def _render_pdf_attachment(ref: AttachmentRef, path: Path, capabilities: ModelCapabilities) -> PreparedAttachment:
    extracted_text = _extract_pdf_text(path)

    if capabilities.supports_pdf_input_chat:
        text = (
            f"[Attached PDF: {ref.filename}]\n"
            "Native PDF chat transmission is not enabled in this build, so extracted text fallback was used instead."
        )
        if extracted_text:
            text = f"{text}\n\n{extracted_text}"
        return PreparedAttachment(
            ref=ref,
            text_fallback=text,
            rendered_parts=({"type": "text", "text": text},),
            strategy="pdf-fallback-text",
        )

    if extracted_text:
        text = f"[Attached PDF: {ref.filename}]\n\n{extracted_text}"
        return PreparedAttachment(
            ref=ref,
            text_fallback=text,
            rendered_parts=({"type": "text", "text": text},),
            strategy="pdf-extracted-text",
        )

    text = (
        f"[Attached PDF: {ref.filename}]\n"
        "No readable text could be extracted from this PDF, and the current model/path does not support native PDF input."
    )
    return PreparedAttachment(
        ref=ref,
        text_fallback=text,
        rendered_parts=({"type": "text", "text": text},),
        strategy="pdf-unreadable",
    )


def prepare_attachment(ref_payload: Dict[str, Any], capabilities: ModelCapabilities) -> PreparedAttachment:
    ref = AttachmentRef(
        path=str(ref_payload.get("path") or ""),
        filename=str(ref_payload.get("filename") or Path(str(ref_payload.get("path") or "attachment")).name),
        mime_type=str(ref_payload.get("mime_type") or "application/octet-stream"),
        kind=str(ref_payload.get("kind") or "other"),
    )
    path = Path(ref.path)

    if ref.kind == "image":
        return _render_image_attachment(ref, path, capabilities)
    if ref.kind == "pdf":
        return _render_pdf_attachment(ref, path, capabilities)

    text = f"[Attached file: {ref.filename}] Unsupported attachment type: {ref.mime_type}"
    return PreparedAttachment(
        ref=ref,
        text_fallback=text,
        rendered_parts=({"type": "text", "text": text},),
        strategy="unsupported-file",
    )


def render_message_content(content: Any, role: str, capabilities: ModelCapabilities) -> Any:
    if not isinstance(content, list) or role != "user":
        return content

    rendered: List[Dict[str, Any]] = []
    debug_strategies: List[str] = []
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
        merged: List[Dict[str, Any]] = [{"type": "text", "text": "\n\n".join(text for text in text_parts if text).strip()}]
        merged.extend(non_text_parts)
        return [part for part in merged if not (part.get("type") == "text" and not part.get("text"))]

    return non_text_parts