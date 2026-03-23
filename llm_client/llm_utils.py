"""
Lightweight LLM utilities (no LangChain dependency).

Goals:
- Provide a unified return object (raw_text / json_data / token_usage / errors)
- Provide robust JSON extraction similar to agent-psychology (strip fences, extract substring, fallback)
- Keep call sites (LangGraph nodes) minimal and consistent
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from .interface import LLMResponse


def log_llm_json_result(
    logger: logging.Logger,
    result: "LLMResponse",
    *,
    level: str = "info",
    prefix: str = "",
    include_debug: bool = False,
    max_raw_chars: int = 600,
    max_content_chars: int = 600,
) -> None:
    """
    Convenience helper to log a result with a consistent format.
    """
    msg = (prefix + " " if prefix else "") + result.to_log_str(
        max_raw_chars=max_raw_chars,
        max_content_chars=max_content_chars,
        include_debug=include_debug,
    )
    fn = getattr(logger, level, None)
    if not callable(fn):
        fn = logger.info
    fn(msg)


def strip_code_fences(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped

    # drop first fence line (``` or ```json etc)
    lines = lines[1:]
    # drop trailing fences
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_substring(text: str) -> Optional[str]:
    """
    Extract the first top-level JSON object substring by brace matching.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _try_json_loads(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, None
        return None, "parsed_json_is_not_object"
    except Exception as e:
        return None, f"json_decode_error:{e}"


def parse_json_from_model_output(raw_text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Parse JSON from model output with multi-step fallback.
    Returns (json_data, error_message).
    """
    text = (raw_text or "").strip()
    if not text:
        return {}, "empty_response"

    # Strategy 1: direct parse
    obj, err = _try_json_loads(text)
    if obj is not None:
        return obj, None

    # Strategy 2: strip code fences
    cleaned = strip_code_fences(text)
    if cleaned != text:
        obj2, err2 = _try_json_loads(cleaned)
        if obj2 is not None:
            return obj2, None

    # Strategy 3: extract substring by braces
    sub = extract_json_substring(cleaned)
    if sub:
        obj3, err3 = _try_json_loads(sub)
        if obj3 is not None:
            return obj3, None

    # Strategy 4: simple repairs (trailing commas)
    repaired = re.sub(r",\s*([}\]])", r"\1", sub or cleaned)
    if repaired and repaired != (sub or cleaned):
        obj4, err4 = _try_json_loads(repaired)
        if obj4 is not None:
            return obj4, None

    return {}, err


def parse_json_from_model_output_detailed(raw_text: str) -> "LLMResponse":
    """
    Detailed parser that returns a rich `LLMResponse` with intermediate artifacts.
    This is the recommended API for new code.
    """
    raw = (raw_text or "")
    text = raw.strip()
    res = LLMResponse(raw_text=raw)

    if not text:
        res.parse_error = "empty_response"
        res.debug = {"strategy": "empty"}
        return res

    # Attempt raw direct parse (raw_json_data)
    raw_obj, raw_err = _try_json_loads(text)
    res.raw_json_data = raw_obj or {}
    res.raw_json_error = raw_err

    if raw_obj is not None:
        # In this case, business json is the same as raw json.
        res.content_text = text
        res.json_data = raw_obj
        res.parse_error = None
        res.debug = {"strategy": "direct"}
        return res

    cleaned = strip_code_fences(text)
    res.debug["cleaned_text"] = cleaned

    # Try cleaned direct parse
    if cleaned != text:
        obj2, err2 = _try_json_loads(cleaned)
        if obj2 is not None:
            res.content_text = cleaned
            res.json_data = obj2
            res.parse_error = None
            res.debug["strategy"] = "strip_code_fences"
            return res

    # Extract substring
    sub = extract_json_substring(cleaned)
    res.debug["json_substring"] = sub or ""
    if sub:
        obj3, err3 = _try_json_loads(sub)
        if obj3 is not None:
            res.content_text = sub
            res.json_data = obj3
            res.parse_error = None
            res.debug["strategy"] = "extract_substring"
            return res

    # Repairs: trailing commas
    candidate = sub or cleaned
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    res.debug["repaired_text"] = repaired
    if repaired and repaired != candidate:
        obj4, err4 = _try_json_loads(repaired)
        if obj4 is not None:
            res.content_text = repaired
            res.json_data = obj4
            res.parse_error = None
            res.debug["strategy"] = "repair_trailing_commas"
            return res

    # All failed
    res.content_text = candidate
    res.json_data = {}
    res.parse_error = raw_err or "unable_to_parse_json"
    res.debug["strategy"] = "failed"
    return res


# Note: do not add extra aliases here; keep naming consistent at the interfaces layer.


