"""Measure what a configured model can actually do (design/125). PROVISIONAL.

Capabilities are declared, never guessed from the model NAME (design/75) — but a declaration is
a human's tick box, and nothing corrected a wrong one. Measured on a real model library
(2026-09-26): one model sees images though nobody ticked it; another was ticked and answered a
blue picture with "off-white". So this sends a handful of tiny real requests and records what
came back, with the model's own words as evidence.

Four checks, one call each, run concurrently, judged independently:

``image_input``   a 64×64 picture, left half one colour, right half another (two of six, random
                  each time); asked which is which. Both right, in order → ``yes`` (a blind
                  guess lands 1 in 30). Wrong colours / "I can't see" / a 4xx about images → ``no``.
``json_schema``   a native ``response_format`` json_schema; the prompt does NOT mention JSON, so
                  only a server that really enforces the schema returns a matching object → ``yes``.
                  A 4xx about response_format, or an answer that ignores the schema → ``no``.
``tools``         one no-arg tool the model is asked to call → called ``yes``; a 4xx about tools
                  → ``no``; plain text instead → ``inconclusive`` (it may just have chosen not to).
``thinking``      does a plain answer come with reasoning content → ``yes`` / ``no``.

``inconclusive`` is for anything that is not evidence about the capability: 429, 5xx, timeouts,
network errors, empty replies, auth/model errors. A host must never overwrite a conclusion with
it — one rate-limited probe must not turn a vision model blind.

Every check goes through this library's real transport (rendering, downscaling, protocol
translation), with the capability under test temporarily declared, so a pass here means the
path an agent will actually use works — not merely that some HTTP call did.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import random
import re
import struct
import tempfile
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ALL_CHECKS",
    "CapabilityReport",
    "CheckResult",
    "probe_capabilities",
]

ALL_CHECKS: tuple[str, ...] = ("image_input", "json_schema", "tools", "thinking")

#: Reasoning models spend hundreds of tokens before the first word; a small cap reads as "no answer".
PROBE_MAX_TOKENS = 2048
EVIDENCE_CHARS = 60

YES, NO, INCONCLUSIVE = "yes", "no", "inconclusive"

#: (Chinese word, English words, RGB). No white / grey / orange: blank or washed-out renderings
#: read as "white"/"off-white", and orange sits between red and yellow.
_COLOURS: tuple[tuple[str, tuple[str, ...], tuple[int, int, int]], ...] = (
    ("红", ("red",), (220, 30, 30)),
    ("绿", ("green",), (30, 160, 60)),
    ("蓝", ("blue",), (30, 90, 220)),
    ("黄", ("yellow",), (240, 200, 20)),
    ("黑", ("black",), (15, 15, 15)),
    ("紫", ("purple", "violet"), (130, 40, 170)),
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # "yes" | "no" | "inconclusive"
    evidence: str = ""  # the model's own words (trimmed) or an error summary — never a secret
    latency_ms: int | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CapabilityReport:
    model: str
    checks: dict[str, CheckResult]

    def capabilities(self) -> dict[str, bool]:
        """Conclusive results as ``ModelCapabilities`` keys. Inconclusive checks are ABSENT —
        they must leave whatever the host already knew untouched."""
        out: dict[str, bool] = {}
        for check, key in (("image_input", "supports_image_input"),
                           ("json_schema", "supports_json_schema")):
            r = self.checks.get(check)
            if r is not None and r.status in (YES, NO):
                out[key] = r.status == YES
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model,
                "checks": {k: v.to_dict() for k, v in self.checks.items()}}


# ── helpers ──────────────────────────────────────────────────────────────


def _two_colour_png(left: tuple[int, int, int], right: tuple[int, int, int], size: int = 64) -> bytes:
    """A PNG, left half one colour and right half another. Stdlib only."""
    half = size // 2
    row = b"\x00" + bytes(left) * half + bytes(right) * (size - half)
    raw = row * size

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def _first_hit(text: str, words: Sequence[str]) -> int:
    hits = [i for i in (text.find(w) for w in words) if i >= 0]
    return min(hits) if hits else -1


def _image_verdict(answer: str, left: tuple, right: tuple) -> str:
    """Both colours named, left one first → yes. Anything else that IS an answer → no."""
    text = (answer or "").strip().lower()
    if not text:
        return INCONCLUSIVE
    li = _first_hit(text, (left[0], *left[1]))
    ri = _first_hit(text, (right[0], *right[1]))
    return YES if 0 <= li < ri else NO


_SECRETISH = re.compile(r"(sk|key|token)[-_][A-Za-z0-9_\-]{6,}", re.IGNORECASE)


def _evidence(text: str, *, secret: str = "") -> str:
    out = " ".join((text or "").split())
    if secret:
        out = out.replace(secret, "***")
    return _SECRETISH.sub(r"\1-***", out)[:EVIDENCE_CHARS]


def _usage(resp: Any) -> dict[str, int]:
    u = getattr(resp, "token_usage", None)
    out: dict[str, int] = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(u, k, None)
        if isinstance(v, int):
            out[k] = v
    return out


def _provider_message(exc: BaseException) -> str:
    """The provider's own error sentence (SDK errors carry it in ``body``), not the SDK's
    "Error code: 400 - {'error': {..." wrapper that eats the evidence budget."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error") if isinstance(body.get("error"), dict) else body
        m = inner.get("message") if isinstance(inner, dict) else None
        if isinstance(m, str) and m.strip():
            return m.strip()
    m = getattr(exc, "message", None)
    return m.strip() if isinstance(m, str) and m.strip() else str(exc)


def _classify_error(exc: BaseException, *, about: Sequence[str], secret: str) -> tuple[str, str]:
    """(status, evidence) for a failed call. Only a client error that NAMES the capability is a
    ``no``; rate limits, server errors, timeouts, auth or unknown-model errors say nothing about
    it."""
    code = getattr(exc, "status_code", None)
    msg = _provider_message(exc)
    ev = _evidence(f"{code or type(exc).__name__}: {msg}", secret=secret)
    if isinstance(code, int) and 400 <= code < 500 and code not in (401, 403, 404, 408, 409, 429):
        low = msg.lower()
        if any(w in low for w in about):
            return NO, ev
    return INCONCLUSIVE, ev


# ── the checks ───────────────────────────────────────────────────────────


async def _call(make: Callable[[dict[str, Any]], Any], caps: dict[str, Any], request: Any,
                timeout_s: float) -> tuple[Any, float]:
    svc = make(caps)
    t0 = time.monotonic()
    try:
        resp = await asyncio.wait_for(svc.complete(request), timeout_s)
        return resp, time.monotonic() - t0
    finally:
        close = getattr(svc, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001 — closing a probe client must not mask its result
                pass


def _ms(dt: float) -> int:
    return int(dt * 1000)


async def _check_image(make, timeout_s: float, rng: random.Random, secret: str) -> CheckResult:
    from power_loop._vendor.llm_client.interface import LLMRequest
    from power_loop._vendor.llm_client.multimodal import create_attachment_ref

    left, right = rng.sample(_COLOURS, 2)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="pl_probe_")
    t0 = time.monotonic()
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(_two_colour_png(left[2], right[2]))
        request = LLMRequest(messages=[{"role": "user", "content": [
            {"type": "text", "text": "这张图的左半边和右半边各是什么颜色？按「左：颜色，右：颜色」回答，只写颜色名。"},
            {"type": "attachment", "attachment": create_attachment_ref(path)},
        ]}], max_tokens=PROBE_MAX_TOKENS, temperature=0.0)
        try:
            resp, dt = await _call(make, {"supports_image_input": True}, request, timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            return CheckResult("image_input", INCONCLUSIVE, "timeout", _ms(time.monotonic() - t0))
        except Exception as exc:  # noqa: BLE001
            status, ev = _classify_error(exc, about=("image", "vision", "multimodal", "image_url"),
                                         secret=secret)
            return CheckResult("image_input", status, ev, _ms(time.monotonic() - t0))
        answer = (getattr(resp, "content_text", "") or getattr(resp, "raw_text", "") or "").strip()
        verdict = _image_verdict(answer, left, right)
        expected = f"（应为 左{left[0]} 右{right[0]}）"
        return CheckResult("image_input", verdict, _evidence(answer, secret=secret) + expected,
                           _ms(dt), _usage(resp))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


async def _check_json_schema(make, timeout_s: float, protocol: str, secret: str) -> CheckResult:
    from power_loop._vendor.llm_client.interface import LLMRequest
    from power_loop.runtime.structured import StructuredOutputSpec, parse_structured

    if protocol == "anthropic":
        # This library has no native json_schema on the Anthropic transport — it always writes the
        # schema into the prompt. A "yes" here would only measure that fallback.
        return CheckResult("json_schema", NO, "anthropic 协议没有原生 json_schema（一律写进提示词）")
    schema = {"type": "object", "properties": {"n": {"type": "integer"}},
              "required": ["n"], "additionalProperties": False}
    request = LLMRequest(
        messages=[{"role": "user", "content": "What is 2 + 3? Put the result in n."}],
        response_format=StructuredOutputSpec(name="Probe", schema=schema).to_openai_response_format(),
        max_tokens=PROBE_MAX_TOKENS, temperature=0.0,
    )
    t0 = time.monotonic()
    try:
        resp, dt = await _call(make, {"supports_json_schema": True}, request, timeout_s)
    except (asyncio.TimeoutError, TimeoutError):
        return CheckResult("json_schema", INCONCLUSIVE, "timeout", _ms(time.monotonic() - t0))
    except Exception as exc:  # noqa: BLE001
        status, ev = _classify_error(exc, about=("response_format", "json_schema", "json schema",
                                                 "structured"), secret=secret)
        return CheckResult("json_schema", status, ev, _ms(time.monotonic() - t0))
    text = (getattr(resp, "content_text", "") or getattr(resp, "raw_text", "") or "").strip()
    if not text:
        return CheckResult("json_schema", INCONCLUSIVE, "empty reply", _ms(dt), _usage(resp))
    try:
        obj = parse_structured(text, schema=schema)
        ok = isinstance(obj.get("n"), int)
    except Exception:  # noqa: BLE001 — the server took the parameter but did not honour it
        ok = False
    return CheckResult("json_schema", YES if ok else NO, _evidence(text, secret=secret), _ms(dt),
                       _usage(resp))


async def _check_tools(make, timeout_s: float, secret: str) -> CheckResult:
    from power_loop._vendor.llm_client.interface import LLMRequest

    tools = [{"type": "function", "function": {
        "name": "get_probe_code",
        "description": "Return the probe code. Call it whenever the probe code is requested.",
        "parameters": {"type": "object", "properties": {}},
    }}]
    request = LLMRequest(messages=[{"role": "user", "content": "Call get_probe_code to fetch the probe code."}],
                         tools=tools, max_tokens=PROBE_MAX_TOKENS, temperature=0.0)
    t0 = time.monotonic()
    try:
        resp, dt = await _call(make, {}, request, timeout_s)
    except (asyncio.TimeoutError, TimeoutError):
        return CheckResult("tools", INCONCLUSIVE, "timeout", _ms(time.monotonic() - t0))
    except Exception as exc:  # noqa: BLE001
        status, ev = _classify_error(exc, about=("tool", "function"), secret=secret)
        return CheckResult("tools", status, ev, _ms(time.monotonic() - t0))
    calls = resp.get_tool_calls() if hasattr(resp, "get_tool_calls") else []
    names = [((c.get("function") or {}).get("name") or c.get("name")) for c in calls or []]
    if names:
        return CheckResult("tools", YES, _evidence("called " + ", ".join(map(str, names))), _ms(dt),
                           _usage(resp))
    text = (getattr(resp, "content_text", "") or getattr(resp, "raw_text", "") or "").strip()
    return CheckResult("tools", INCONCLUSIVE, _evidence("no call: " + text, secret=secret), _ms(dt),
                       _usage(resp))


async def _check_thinking(make, timeout_s: float, secret: str) -> CheckResult:
    from power_loop._vendor.llm_client.interface import LLMRequest

    request = LLMRequest(messages=[{"role": "user", "content": "What is 17 × 3? Reply with just the number."}],
                         max_tokens=PROBE_MAX_TOKENS, temperature=0.0)
    t0 = time.monotonic()
    try:
        resp, dt = await _call(make, {}, request, timeout_s)
    except (asyncio.TimeoutError, TimeoutError):
        return CheckResult("thinking", INCONCLUSIVE, "timeout", _ms(time.monotonic() - t0))
    except Exception as exc:  # noqa: BLE001
        _status, ev = _classify_error(exc, about=(), secret=secret)
        return CheckResult("thinking", INCONCLUSIVE, ev, _ms(time.monotonic() - t0))
    think = (getattr(resp, "think", "") or "").strip()
    text = (getattr(resp, "content_text", "") or getattr(resp, "raw_text", "") or "").strip()
    if think:
        return CheckResult("thinking", YES, _evidence(f"{len(think)} chars of reasoning"), _ms(dt),
                           _usage(resp))
    if not text:
        return CheckResult("thinking", INCONCLUSIVE, "empty reply", _ms(dt), _usage(resp))
    return CheckResult("thinking", NO, _evidence(text, secret=secret), _ms(dt), _usage(resp))


# ── entry point ──────────────────────────────────────────────────────────


async def probe_capabilities(
    config: Any,
    *,
    checks: Sequence[str] = ALL_CHECKS,
    timeout_s: float = 30.0,
    rng: random.Random | None = None,
    service_factory: Callable[[Any], Any] | None = None,
) -> CapabilityReport:
    """Probe ``config`` (an ``LLMProviderConfig``) and report per-check results.

    Checks run concurrently; each builds its own client from a copy of ``config`` with only the
    capability under test declared, no transport retries (a retry would hide the 429 that makes
    a result inconclusive) and a thinking-safe ``max_tokens``. ``service_factory`` builds the
    client from such a copy (default: ``create_llm_service_from_config``) — the test seam, and a
    host hook for wrapping clients (e.g. usage metering).
    """
    from power_loop.runtime.provider import create_llm_service_from_config

    unknown = set(checks) - set(ALL_CHECKS)
    if unknown:
        raise ValueError(f"unknown probe checks: {sorted(unknown)}; choose from {ALL_CHECKS}")
    factory = service_factory or create_llm_service_from_config
    secret = str(getattr(config, "api_key", "") or "")
    protocol = str(getattr(config, "provider", "") or "openai").lower()

    def make(caps: dict[str, Any]) -> Any:
        cfg = dataclasses.replace(config, capabilities=dict(caps), max_retries=0,
                                  max_tokens=PROBE_MAX_TOKENS, temperature=0.0)
        return factory(cfg)

    rng = rng or random.Random()
    jobs: dict[str, Any] = {}
    if "image_input" in checks:
        jobs["image_input"] = _check_image(make, timeout_s, rng, secret)
    if "json_schema" in checks:
        jobs["json_schema"] = _check_json_schema(make, timeout_s, protocol, secret)
    if "tools" in checks:
        jobs["tools"] = _check_tools(make, timeout_s, secret)
    if "thinking" in checks:
        jobs["thinking"] = _check_thinking(make, timeout_s, secret)
    results = await asyncio.gather(*jobs.values())
    return CapabilityReport(model=str(getattr(config, "model", "") or ""),
                            checks=dict(zip(jobs, results, strict=True)))
