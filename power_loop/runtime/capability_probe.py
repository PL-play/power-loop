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

When a check is inconclusive because something FAILED, ``CheckResult.error_kind`` says what (one
of ``ERROR_KINDS``), so a host can tell "the account ran out of money" from "try again later"
without parsing evidence. The first four (``BLOCKING_ERROR_KINDS``: balance, auth, forbidden,
model_missing) will not go away by retrying — someone has to fix the configuration or the
account. Their WORDS (out of balance, bad key, access denied, no such model) and a bare 402 / 401
win over the "a 4xx that names the capability is a no" rule: a 402 that mentions images says
nothing about images. A bare 403 / 404 does not: OpenRouter answers a model without image input
with 404 "No endpoints found that support image input", which is a ``no``. The configured model's
name is taken out of the message before looking for the capability: a model called
"llama-3.2-11b-vision-instruct" or "xxx-tools" must not make its own not-found error read as
"does not support vision / tools". The text patterns started from DeepTalk's
``agent/app/llm_incidents.py``.

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
import socket
import struct
import tempfile
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ALL_CHECKS",
    "BLOCKING_ERROR_KINDS",
    "ERROR_KINDS",
    "CapabilityReport",
    "CheckResult",
    "probe_capabilities",
]

ALL_CHECKS: tuple[str, ...] = ("image_input", "json_schema", "tools", "thinking")

#: Reasoning models spend hundreds of tokens before the first word; a small cap reads as "no answer".
PROBE_MAX_TOKENS = 2048
EVIDENCE_CHARS = 60

YES, NO, INCONCLUSIVE = "yes", "no", "inconclusive"

#: Why a check failed to conclude (``CheckResult.error_kind``), most in need of a human first.
ERROR_KINDS: tuple[str, ...] = ("balance", "auth", "forbidden", "model_missing", "rate_limited",
                                "server", "timeout", "network", "empty", "unknown")
#: Configuration / account problems: retrying will not help.
BLOCKING_ERROR_KINDS: frozenset[str] = frozenset({"balance", "auth", "forbidden", "model_missing"})

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
    #: set only when an ``inconclusive`` result comes from a failure — one of ``ERROR_KINDS``.
    #: Last field, so ``CheckResult(name, status, evidence, latency_ms, usage)`` still works.
    error_kind: str | None = None

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


#: (kind, message pattern), in priority order — balance first: OpenAI reports an exhausted quota
#: as a 429 and an inactive account as a 429 "… check your billing details", Moonshot a suspended
#: one as a 429 exceeded_current_quota_error "… check your plan and billing details", DashScope an
#: arrears as a 400 "Access denied, please make sure your account is in good standing" (code
#: Arrearage), Anthropic as a 400 "Your credit balance is too low". No bare "billing": OpenAI's
#: free-tier 429 rate-limit message links to the …/account/billing page. Gemini says a bad key
#: with a 400 "API key not valid". "does not exist" only with "model" before it in the same
#: sentence (a dot inside a name like gpt-4.1 does not end it): "function does not exist" is about
#: tools. The model pattern also knows DeepSeek's wrong-name 400 ("The supported API model names
#: are …, but you passed …") — both halves, so "the schema you passed is invalid" is not a missing
#: model — and its older "Model Not Exist". Started from DeepTalk agent/app/llm_incidents.py.
_BLOCKING_TEXT: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("balance", re.compile(
        r"insufficient[ _]balance|insufficient[ _]quota|exceeded your current quota|"
        r"exceeded[ _]current[ _]quota|billing details|billing[ _]not[ _]active|credit balance|"
        r"payment required|arrearage|account is in good standing|余额|欠费|billing hard limit",
        re.I)),
    ("auth", re.compile(
        r"invalid[ _]api[ _]key|incorrect api key|api[ _]key not valid|authentication|unauthorized|"
        r"invalid token", re.I)),
    ("forbidden", re.compile(r"permission denied|forbidden|access denied", re.I)),
    ("model_missing", re.compile(
        r"model[ _]not[ _]found|model\b(?:[^.\n]|\.(?=\S)){0,80}?does not exist|"
        r"model[ _]not[ _]exist|no such model|unknown model|model names are[\s\S]*you passed", re.I)),
)
#: Bare codes that are never about a capability, whatever the message says.
_BLOCKING_CODES: dict[int, str] = {402: "balance", 401: "auth"}
#: Bare codes that lose to a message naming the capability: OpenRouter answers a model without
#: image input / tool use with 404 "No endpoints found that support image input".
_FALLBACK_CODES: dict[int, str] = {403: "forbidden", 404: "model_missing"}
# SDK-free on purpose (the transports are optional extras): matched by class name along the MRO.
# openai / anthropic ``APITimeoutError`` subclasses ``APIConnectionError``, so timeouts go first.
_TIMEOUT_TYPES = frozenset({"APITimeoutError", "TimeoutException"})  # SDKs; httpx
_NETWORK_TYPES = frozenset({"APIConnectionError", "TransportError"})  # SDKs; httpx (connect/read…)


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if not isinstance(code, int):
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def _blocking_kind(code: int | None, text: str) -> str | None:
    """A configuration / account problem that outranks any capability named in the message:
    its words first, then a bare 402 / 401."""
    for kind, pattern in _BLOCKING_TEXT:
        if pattern.search(text):
            return kind
    return _BLOCKING_CODES.get(code) if code is not None else None


def _without_model(text: str, model: str) -> str:
    """``text`` with the configured model's name taken out (any case) — the full id and, for an
    OpenRouter-style "org/name" id, the name alone — so the name cannot count as naming a
    capability."""
    for name in (model, model.rsplit("/", 1)[-1]):
        if name:
            text = re.sub(re.escape(name), " ", text, flags=re.I)
    return text


def _error_kind(exc: BaseException, code: int | None) -> str:
    """Why a call failed, once neither blocking words nor a capability named in the message
    decided it."""
    fallback = _FALLBACK_CODES.get(code) if code is not None else None
    if fallback is not None:
        return fallback
    if code == 429:
        return "rate_limited"
    if code is not None and 500 <= code < 600:
        return "server"
    names = {c.__name__ for c in type(exc).__mro__}
    if code == 408 or isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or names & _TIMEOUT_TYPES:
        return "timeout"
    if isinstance(exc, (ConnectionError, socket.gaierror)) or names & _NETWORK_TYPES:
        return "network"
    return "unknown"


def _classify_error(exc: BaseException, *, about: Sequence[str], secret: str,
                    model: str = "") -> tuple[str, str, str | None]:
    """(status, evidence, error_kind) for a failed call, in this order:

    1. balance / auth / forbidden / model_missing WORDS, or a bare 402 / 401 → inconclusive with
       that kind, even when the message names the capability (a 402 about images says nothing
       about images);
    2. any other 4xx (not 408 / 409 / 429) whose message names the capability → ``no``,
       error_kind ``None`` — an answer, not a failure. This includes 403 / 404: OpenRouter says
       "No endpoints found that support image input" with a 404. The configured ``model`` name is
       taken out first: "No endpoints found for …-vision-instruct." names the model, not vision;
    3. otherwise inconclusive: bare 403 forbidden, 404 model_missing, 429, 5xx, timeout,
       network, unknown."""
    code = _status_code(exc)
    msg = _provider_message(exc)
    ev = _evidence(f"{code or type(exc).__name__}: {msg}", secret=secret)
    kind = _blocking_kind(code, f"{msg}\n{exc}")
    if kind is not None:
        return INCONCLUSIVE, ev, kind
    if code is not None and 400 <= code < 500 and code not in (408, 409, 429):
        low = _without_model(msg, model).lower()
        if any(w in low for w in about):
            return NO, ev, None
    return INCONCLUSIVE, ev, _error_kind(exc, code)


def _failure(name: str, exc: BaseException, t0: float, *, about: Sequence[str],
             secret: str, model: str) -> CheckResult:
    ms = _ms(time.monotonic() - t0)
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):  # incl. the probe's own wait_for
        return CheckResult(name, INCONCLUSIVE, "timeout", ms, error_kind="timeout")
    status, ev, kind = _classify_error(exc, about=about, secret=secret, model=model)
    return CheckResult(name, status, ev, ms, error_kind=kind)


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


async def _check_image(make, timeout_s: float, rng: random.Random, secret: str,
                       model: str) -> CheckResult:
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
        except Exception as exc:  # noqa: BLE001
            return _failure("image_input", exc, t0, secret=secret, model=model,
                            about=("image", "vision", "multimodal", "image_url"))
        answer = (getattr(resp, "content_text", "") or getattr(resp, "raw_text", "") or "").strip()
        verdict = _image_verdict(answer, left, right)
        expected = f"（应为 左{left[0]} 右{right[0]}）"
        if verdict == INCONCLUSIVE:
            return CheckResult("image_input", INCONCLUSIVE, "empty reply" + expected, _ms(dt),
                               _usage(resp), error_kind="empty")
        return CheckResult("image_input", verdict, _evidence(answer, secret=secret) + expected,
                           _ms(dt), _usage(resp))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


async def _check_json_schema(make, timeout_s: float, protocol: str, secret: str,
                             model: str) -> CheckResult:
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
    except Exception as exc:  # noqa: BLE001
        return _failure("json_schema", exc, t0, secret=secret, model=model,
                        about=("response_format", "json_schema", "json schema", "structured"))
    text = (getattr(resp, "content_text", "") or getattr(resp, "raw_text", "") or "").strip()
    if not text:
        return CheckResult("json_schema", INCONCLUSIVE, "empty reply", _ms(dt), _usage(resp),
                           error_kind="empty")
    try:
        obj = parse_structured(text, schema=schema)
        ok = isinstance(obj.get("n"), int)
    except Exception:  # noqa: BLE001 — the server took the parameter but did not honour it
        ok = False
    return CheckResult("json_schema", YES if ok else NO, _evidence(text, secret=secret), _ms(dt),
                       _usage(resp))


async def _check_tools(make, timeout_s: float, secret: str, model: str) -> CheckResult:
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
    except Exception as exc:  # noqa: BLE001
        return _failure("tools", exc, t0, about=("tool", "function"), secret=secret, model=model)
    calls = resp.get_tool_calls() if hasattr(resp, "get_tool_calls") else []
    names = [((c.get("function") or {}).get("name") or c.get("name")) for c in calls or []]
    if names:
        return CheckResult("tools", YES, _evidence("called " + ", ".join(map(str, names))), _ms(dt),
                           _usage(resp))
    text = (getattr(resp, "content_text", "") or getattr(resp, "raw_text", "") or "").strip()
    if not text:
        return CheckResult("tools", INCONCLUSIVE, "empty reply", _ms(dt), _usage(resp),
                           error_kind="empty")
    # Answered without calling: it may just have chosen not to — not a failure, no error_kind.
    return CheckResult("tools", INCONCLUSIVE, _evidence("no call: " + text, secret=secret), _ms(dt),
                       _usage(resp))


async def _check_thinking(make, timeout_s: float, secret: str, model: str) -> CheckResult:
    from power_loop._vendor.llm_client.interface import LLMRequest

    request = LLMRequest(messages=[{"role": "user", "content": "What is 17 × 3? Reply with just the number."}],
                         max_tokens=PROBE_MAX_TOKENS, temperature=0.0)
    t0 = time.monotonic()
    try:
        resp, dt = await _call(make, {}, request, timeout_s)
    except Exception as exc:  # noqa: BLE001 — about=(): no error is ever a "no" for thinking
        return _failure("thinking", exc, t0, about=(), secret=secret, model=model)
    think = (getattr(resp, "think", "") or "").strip()
    text = (getattr(resp, "content_text", "") or getattr(resp, "raw_text", "") or "").strip()
    if think:
        return CheckResult("thinking", YES, _evidence(f"{len(think)} chars of reasoning"), _ms(dt),
                           _usage(resp))
    if not text:
        return CheckResult("thinking", INCONCLUSIVE, "empty reply", _ms(dt), _usage(resp),
                           error_kind="empty")
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
    model = str(getattr(config, "model", "") or "")
    protocol = str(getattr(config, "provider", "") or "openai").lower()

    def make(caps: dict[str, Any]) -> Any:
        cfg = dataclasses.replace(config, capabilities=dict(caps), max_retries=0,
                                  max_tokens=PROBE_MAX_TOKENS, temperature=0.0)
        return factory(cfg)

    rng = rng or random.Random()
    jobs: dict[str, Any] = {}
    if "image_input" in checks:
        jobs["image_input"] = _check_image(make, timeout_s, rng, secret, model)
    if "json_schema" in checks:
        jobs["json_schema"] = _check_json_schema(make, timeout_s, protocol, secret, model)
    if "tools" in checks:
        jobs["tools"] = _check_tools(make, timeout_s, secret, model)
    if "thinking" in checks:
        jobs["thinking"] = _check_thinking(make, timeout_s, secret, model)
    results = await asyncio.gather(*jobs.values())
    return CapabilityReport(model=model, checks=dict(zip(jobs, results, strict=True)))
