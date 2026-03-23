from __future__ import annotations

import base64
import json
import mimetypes
import os
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener

import certifi


DEFAULT_DASHSCOPE_IMAGE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_DASHSCOPE_IMAGE_OUTPUT_DIR = "outputs/generated-images"
DEFAULT_DASHSCOPE_IMAGE_EDIT_OUTPUT_DIR = "outputs/edited-images"
SYNC_GENERATION_PATH = "/services/aigc/multimodal-generation/generation"


@dataclass(frozen=True)
class QwenImageConfig:
    api_key: str
    model: str
    base_url: str = DEFAULT_DASHSCOPE_IMAGE_BASE_URL
    default_size: str | None = None
    prompt_extend: bool = True
    watermark: bool = False
    use_proxy: bool = False
    output_dir: str = DEFAULT_DASHSCOPE_IMAGE_OUTPUT_DIR
    timeout_s: float = 180.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + SYNC_GENERATION_PATH


@dataclass(frozen=True)
class QwenImageEditConfig:
    api_key: str
    model: str
    base_url: str = DEFAULT_DASHSCOPE_IMAGE_BASE_URL
    default_size: str | None = None
    prompt_extend: bool = True
    watermark: bool = False
    use_proxy: bool = False
    output_dir: str = DEFAULT_DASHSCOPE_IMAGE_EDIT_OUTPUT_DIR
    timeout_s: float = 180.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + SYNC_GENERATION_PATH


class QwenImageError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool = False,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code


def _parse_optional_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def qwen_image_config_from_env(env: Mapping[str, Any]) -> QwenImageConfig | None:
    model = str(env.get("DASHSCOPE_IMAGE_MODEL") or "").strip()
    api_key = str(env.get("DASHSCOPE_IMAGE_API_KEY") or env.get("DASHSCOPE_API_KEY") or "").strip()
    if not (model and api_key):
        return None

    base_url = str(env.get("DASHSCOPE_IMAGE_BASE_URL") or DEFAULT_DASHSCOPE_IMAGE_BASE_URL).strip()
    default_size = str(env.get("DASHSCOPE_IMAGE_DEFAULT_SIZE") or "").strip() or None
    output_dir = str(env.get("DASHSCOPE_IMAGE_OUTPUT_DIR") or DEFAULT_DASHSCOPE_IMAGE_OUTPUT_DIR).strip()

    try:
        timeout_s = float(env.get("DASHSCOPE_IMAGE_TIMEOUT_S") or 180.0)
    except Exception:
        timeout_s = 180.0

    return QwenImageConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        default_size=default_size,
        prompt_extend=_parse_optional_bool(env.get("DASHSCOPE_IMAGE_PROMPT_EXTEND"), True),
        watermark=_parse_optional_bool(env.get("DASHSCOPE_IMAGE_WATERMARK"), False),
        use_proxy=_parse_optional_bool(env.get("DASHSCOPE_IMAGE_USE_PROXY"), False),
        output_dir=output_dir,
        timeout_s=max(1.0, timeout_s),
    )


def qwen_image_edit_config_from_env(env: Mapping[str, Any]) -> QwenImageEditConfig | None:
    model = str(env.get("DASHSCOPE_IMAGE_EDIT_MODEL") or "").strip()
    api_key = str(env.get("DASHSCOPE_IMAGE_EDIT_API_KEY") or env.get("DASHSCOPE_API_KEY") or "").strip()
    if not (model and api_key):
        return None

    base_url = str(
        env.get("DASHSCOPE_IMAGE_EDIT_BASE_URL") or env.get("DASHSCOPE_IMAGE_BASE_URL") or DEFAULT_DASHSCOPE_IMAGE_BASE_URL
    ).strip()
    default_size = str(env.get("DASHSCOPE_IMAGE_EDIT_DEFAULT_SIZE") or "").strip() or None
    output_dir = str(env.get("DASHSCOPE_IMAGE_EDIT_OUTPUT_DIR") or DEFAULT_DASHSCOPE_IMAGE_EDIT_OUTPUT_DIR).strip()

    try:
        timeout_s = float(env.get("DASHSCOPE_IMAGE_EDIT_TIMEOUT_S") or 180.0)
    except Exception:
        timeout_s = 180.0

    return QwenImageEditConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        default_size=default_size,
        prompt_extend=_parse_optional_bool(env.get("DASHSCOPE_IMAGE_EDIT_PROMPT_EXTEND"), True),
        watermark=_parse_optional_bool(env.get("DASHSCOPE_IMAGE_EDIT_WATERMARK"), False),
        use_proxy=_parse_optional_bool(env.get("DASHSCOPE_IMAGE_EDIT_USE_PROXY"), False),
        output_dir=output_dir,
        timeout_s=max(1.0, timeout_s),
    )


def _create_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def _build_url_opener(use_proxy: bool):
    handlers = [HTTPSHandler(context=_create_ssl_context())]
    if not use_proxy:
        handlers.insert(0, ProxyHandler({}))
    return build_opener(*handlers)


def build_qwen_image_payload(
    *,
    model: str,
    prompt: str,
    negative_prompt: str | None = None,
    size: str | None = None,
    prompt_extend: bool = True,
    watermark: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ]
        },
        "parameters": {
            "prompt_extend": bool(prompt_extend),
            "watermark": bool(watermark),
        },
    }

    parameters = payload["parameters"]
    if negative_prompt:
        parameters["negative_prompt"] = negative_prompt
    if size:
        parameters["size"] = size
    return payload


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def _file_to_data_url(path: Path) -> str:
    mime_type = _guess_mime_type(path)
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def build_qwen_image_edit_payload(
    *,
    model: str,
    prompt: str,
    image_sources: list[str],
    negative_prompt: str | None = None,
    size: str | None = None,
    n: int | None = None,
    prompt_extend: bool = True,
    watermark: bool = False,
) -> dict[str, Any]:
    if not image_sources:
        raise ValueError("At least one image source is required")

    content: list[dict[str, Any]] = [{"image": source} for source in image_sources]
    content.append({"text": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
        "parameters": {
            "prompt_extend": bool(prompt_extend),
            "watermark": bool(watermark),
        },
    }
    parameters = payload["parameters"]
    if negative_prompt:
        parameters["negative_prompt"] = negative_prompt
    if size:
        parameters["size"] = size
    if n is not None:
        parameters["n"] = int(n)
    return payload


def _post_json(url: str, api_key: str, payload: dict[str, Any], timeout_s: float, *, use_proxy: bool = False) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        opener = _build_url_opener(use_proxy)
        with opener.open(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            details = exc.read().decode("utf-8")
        except Exception:
            details = str(exc)
        raise QwenImageError(
            f"DashScope image request failed with HTTP {exc.code}: {details}",
            category="provider_http_error" if exc.code < 500 else "provider_server_error",
            retryable=exc.code >= 500 or exc.code == 429,
            status_code=exc.code,
        ) from exc
    except URLError as exc:
        raise QwenImageError(
            f"DashScope image request failed: {exc}",
            category="network_error",
            retryable=True,
        ) from exc


def _download_binary(url: str, timeout_s: float, *, use_proxy: bool = False) -> bytes:
    request = Request(url, method="GET")
    try:
        opener = _build_url_opener(use_proxy)
        with opener.open(request, timeout=timeout_s) as response:
            return response.read()
    except HTTPError as exc:
        raise QwenImageError(
            f"Image download failed with HTTP {exc.code}: {url}",
            category="download_http_error" if exc.code < 500 else "download_server_error",
            retryable=exc.code >= 500 or exc.code == 429,
            status_code=exc.code,
        ) from exc
    except URLError as exc:
        raise QwenImageError(
            f"Image download failed: {exc}",
            category="download_network_error",
            retryable=True,
        ) from exc


def _extract_image_urls(response_payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    output = response_payload.get("output") or {}
    choices = output.get("choices") or []
    for choice in choices:
        message = (choice or {}).get("message") or {}
        for item in message.get("content") or []:
            image_url = (item or {}).get("image")
            if image_url:
                urls.append(str(image_url))

    results = output.get("results") or []
    for item in results:
        image_url = (item or {}).get("url")
        if image_url:
            urls.append(str(image_url))

    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _resolve_image_input_sources(image_paths: list[str | Path]) -> list[str]:
    sources: list[str] = []
    for raw_path in image_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise QwenImageError(f"Image input not found: {raw_path}", category="input_not_found")
        mime_type = _guess_mime_type(path)
        if not mime_type.startswith("image/"):
            raise QwenImageError(
                f"Unsupported image input type: {raw_path} ({mime_type})",
                category="input_type_error",
            )
        sources.append(_file_to_data_url(path))
    if not (1 <= len(sources) <= 3):
        raise QwenImageError("Qwen image edit requires 1 to 3 input images", category="input_count_error")
    return sources


def _suggest_filename(image_url: str, request_id: str, index: int, prefix: str | None) -> str:
    parsed = urlparse(image_url)
    basename = Path(unquote(parsed.path)).name or f"image-{index}.png"
    stem = Path(basename).stem or f"image-{index}"
    suffix = Path(basename).suffix or ".png"
    safe_prefix = (prefix or request_id or "generated-image").strip().replace(" ", "-")
    safe_prefix = "".join(ch for ch in safe_prefix if ch.isalnum() or ch in {"-", "_"}).strip("-_") or "generated-image"
    return f"{safe_prefix}-{index}{suffix}" if stem == safe_prefix else f"{safe_prefix}-{stem}{suffix}"


def summarize_image_operation_result(
    result: Mapping[str, Any], *, operation: str, input_paths: list[str] | None = None
) -> dict[str, Any]:
    images = result.get("images") or []
    image_items: list[dict[str, Any]] = []
    output_paths: list[str] = []
    for item in images:
        if not isinstance(item, Mapping):
            continue
        path_value = item.get("path")
        source_url = item.get("source_url")
        image_info: dict[str, Any] = {}
        if path_value:
            image_info["path"] = str(path_value)
            output_paths.append(str(path_value))
        if source_url:
            image_info["source_url"] = str(source_url)
        if image_info:
            image_items.append(image_info)

    summary: dict[str, Any] = {
        "ok": True,
        "operation": operation,
        "provider": result.get("provider"),
        "model": result.get("model"),
        "request_id": result.get("request_id"),
        "image_count": len(image_items),
        "paths": output_paths,
        "images": image_items,
    }
    if output_paths:
        summary["primary_path"] = output_paths[0]
    if input_paths is not None:
        summary["input_paths"] = [str(path) for path in input_paths]
    if result.get("width") is not None:
        summary["width"] = result.get("width")
    if result.get("height") is not None:
        summary["height"] = result.get("height")

    usage = result.get("usage")
    if isinstance(usage, Mapping) and usage:
        summary["usage"] = dict(usage)

    return summary


def summarize_image_operation_error(
    exc: Exception,
    *,
    operation: str,
    input_paths: list[str] | None = None,
) -> dict[str, Any]:
    category = "unknown_error"
    retryable = False
    status_code: int | None = None

    if isinstance(exc, QwenImageError):
        category = exc.category
        retryable = exc.retryable
        status_code = exc.status_code
    elif isinstance(exc, FileNotFoundError):
        category = "path_not_found"
    elif isinstance(exc, ValueError):
        category = "invalid_input"

    summary: dict[str, Any] = {
        "ok": False,
        "operation": operation,
        "error": {
            "category": category,
            "message": str(exc),
            "retryable": retryable,
        },
    }
    if input_paths is not None:
        summary["input_paths"] = [str(path) for path in input_paths]
    if status_code is not None:
        summary["error"]["status_code"] = status_code
    return summary


def generate_image_with_qwen(
    config: QwenImageConfig,
    *,
    prompt: str,
    output_dir: Path,
    negative_prompt: str | None = None,
    size: str | None = None,
    prompt_extend: bool | None = None,
    watermark: bool | None = None,
    filename_prefix: str | None = None,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    if not config.enabled:
        raise QwenImageError("Qwen image generation is not configured", category="configuration_error")
    if not prompt or not prompt.strip():
        raise QwenImageError("prompt is required", category="invalid_input")

    payload = build_qwen_image_payload(
        model=config.model,
        prompt=prompt.strip(),
        negative_prompt=(negative_prompt or "").strip() or None,
        size=(size or config.default_size or "").strip() or None,
        prompt_extend=config.prompt_extend if prompt_extend is None else bool(prompt_extend),
        watermark=config.watermark if watermark is None else bool(watermark),
    )
    response_payload = _post_json(config.endpoint, config.api_key, payload, config.timeout_s, use_proxy=config.use_proxy)
    image_urls = _extract_image_urls(response_payload)
    if not image_urls:
        raise QwenImageError(
            f"DashScope returned no image URL: {json.dumps(response_payload, ensure_ascii=False)}",
            category="empty_result",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = str(response_payload.get("request_id") or int(time.time()))
    saved_images: list[dict[str, Any]] = []
    for index, image_url in enumerate(image_urls, start=1):
        file_name = _suggest_filename(image_url, request_id, index, filename_prefix)
        destination = output_dir / file_name
        destination.write_bytes(_download_binary(image_url, config.timeout_s, use_proxy=config.use_proxy))

        path_value = str(destination)
        if workspace_root is not None:
            try:
                path_value = destination.resolve().relative_to(workspace_root.resolve()).as_posix()
            except Exception:
                path_value = str(destination)

        saved_images.append(
            {
                "path": path_value,
                "source_url": image_url,
            }
        )

    usage = response_payload.get("usage") or {}
    return {
        "provider": "dashscope",
        "model": config.model,
        "request_id": response_payload.get("request_id"),
        "images": saved_images,
        "usage": usage,
        "width": usage.get("width"),
        "height": usage.get("height"),
        "raw_response": response_payload,
    }


def edit_image_with_qwen(
    config: QwenImageEditConfig,
    *,
    prompt: str,
    image_paths: list[str | Path],
    output_dir: Path,
    negative_prompt: str | None = None,
    size: str | None = None,
    n: int | None = None,
    prompt_extend: bool | None = None,
    watermark: bool | None = None,
    filename_prefix: str | None = None,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    if not config.enabled:
        raise QwenImageError("Qwen image edit is not configured", category="configuration_error")
    if not prompt or not prompt.strip():
        raise QwenImageError("prompt is required", category="invalid_input")

    image_sources = _resolve_image_input_sources(image_paths)
    payload = build_qwen_image_edit_payload(
        model=config.model,
        prompt=prompt.strip(),
        image_sources=image_sources,
        negative_prompt=(negative_prompt or "").strip() or None,
        size=(size or config.default_size or "").strip() or None,
        n=n,
        prompt_extend=config.prompt_extend if prompt_extend is None else bool(prompt_extend),
        watermark=config.watermark if watermark is None else bool(watermark),
    )
    response_payload = _post_json(config.endpoint, config.api_key, payload, config.timeout_s, use_proxy=config.use_proxy)
    image_urls = _extract_image_urls(response_payload)
    if not image_urls:
        raise QwenImageError(
            f"DashScope returned no edited image URL: {json.dumps(response_payload, ensure_ascii=False)}",
            category="empty_result",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = str(response_payload.get("request_id") or int(time.time()))
    saved_images: list[dict[str, Any]] = []
    for index, image_url in enumerate(image_urls, start=1):
        file_name = _suggest_filename(image_url, request_id, index, filename_prefix)
        destination = output_dir / file_name
        destination.write_bytes(_download_binary(image_url, config.timeout_s, use_proxy=config.use_proxy))

        path_value = str(destination)
        if workspace_root is not None:
            try:
                path_value = destination.resolve().relative_to(workspace_root.resolve()).as_posix()
            except Exception:
                path_value = str(destination)

        saved_images.append({"path": path_value, "source_url": image_url})

    usage = response_payload.get("usage") or {}
    return {
        "provider": "dashscope",
        "model": config.model,
        "request_id": response_payload.get("request_id"),
        "images": saved_images,
        "usage": usage,
        "width": usage.get("width"),
        "height": usage.get("height"),
        "raw_response": response_payload,
    }