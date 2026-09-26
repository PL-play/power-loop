"""design/125: measuring what a configured model can do.

The judgement rules are what matter — above all, that anything which is not evidence about a
capability comes back ``inconclusive`` (so a host never overwrites a conclusion with it), and
that a "yes" needs an answer only a model that really saw / really enforced could give.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

from power_loop import LLMProviderConfig
from power_loop._vendor.llm_client.interface import LLMResponse, LLMTokenUsage
from power_loop.runtime import capability_probe as cp

pytestmark = pytest.mark.unit

KEY = "sk-live-abcdef1234567890"


def _cfg(provider: str = "openai", model: str = "m1") -> LLMProviderConfig:
    return LLMProviderConfig(base_url="https://llm.example/v1", api_key=KEY, model=model,
                             provider=provider, max_tokens=64, max_retries=3)


class _HTTPError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"Error code: {code} - {{'error': {{'message': '{message}'}}}}")
        self.status_code = code
        self.body = {"error": {"message": message}}


def _resp(text: str = "", *, think: str = "", tool: str | None = None) -> LLMResponse:
    r = LLMResponse(raw_text=text, content_text=text, think=think,
                    token_usage=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    if tool:
        r.tool_calls = [{"id": "c1", "type": "function", "function": {"name": tool, "arguments": "{}"}}]
    return r


class _Factory:
    """Builds fake clients; ``script[check]`` is a response, an exception, or a callable."""

    def __init__(self, script: dict[str, Any]) -> None:
        self.script = script
        self.configs: list[LLMProviderConfig] = []

    def __call__(self, cfg: LLMProviderConfig) -> Any:
        self.configs.append(cfg)
        factory = self

        class _Svc:
            async def complete(self, request: Any) -> LLMResponse:
                if request.tools:
                    key = "tools"
                elif request.response_format:
                    key = "json_schema"
                elif isinstance(request.messages[0]["content"], list):
                    key = "image_input"
                else:
                    key = "thinking"
                out = factory.script[key]
                if callable(out) and not isinstance(out, LLMResponse):
                    out = await out(request)
                if isinstance(out, BaseException):
                    raise out
                return out

            async def close(self) -> None:
                return None

        return _Svc()


def _colours(seed: int) -> tuple[str, str]:
    left, right = random.Random(seed).sample(cp._COLOURS, 2)
    return left[0], right[0]


async def _probe(script: dict[str, Any], *, seed: int = 7, provider: str = "openai",
                 model: str = "m1", **kw):
    f = _Factory(script)
    rep = await cp.probe_capabilities(_cfg(provider, model), rng=random.Random(seed),
                                      service_factory=f, **kw)
    return rep, f


# ── what each check declares on its client ──────────────────────────────────


@pytest.mark.asyncio
async def test_each_check_declares_only_what_it_tests_with_no_retries_and_room_to_think() -> None:
    lft, rgt = _colours(7)
    _rep, f = await _probe({
        "image_input": _resp(f"左：{lft}，右：{rgt}"), "json_schema": _resp('{"n": 5}'),
        "tools": _resp(tool="get_probe_code"), "thinking": _resp("51", think="17*3=51"),
    })
    caps = sorted((tuple(sorted(c.capabilities.items())) for c in f.configs))
    assert caps == [(), (), (("supports_image_input", True),), (("supports_json_schema", True),)]
    assert all(c.max_retries == 0 and c.max_tokens == cp.PROBE_MAX_TOKENS for c in f.configs)


# ── image: both colours, in order ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_image_needs_both_colours_in_order() -> None:
    lft, rgt = _colours(7)
    rep, _ = await _probe({"image_input": _resp(f"左：{lft}色，右：{rgt}色")}, checks=["image_input"])
    assert rep.checks["image_input"].status == "yes"
    assert rep.capabilities() == {"supports_image_input": True}
    swapped, _ = await _probe({"image_input": _resp(f"左：{rgt}，右：{lft}")}, checks=["image_input"])
    assert swapped.checks["image_input"].status == "no"
    blind, _ = await _probe({"image_input": _resp("左：米白色，右：米白色")}, checks=["image_input"])
    assert blind.checks["image_input"].status == "no"
    assert "应为" in blind.checks["image_input"].evidence  # the evidence says what it should have been
    cant, _ = await _probe({"image_input": _resp("抱歉，我无法查看图片。")}, checks=["image_input"])
    assert cant.checks["image_input"].status == "no"


def test_image_verdict_reads_english_and_empty_answers() -> None:
    red, blue = cp._COLOURS[0], cp._COLOURS[2]
    assert cp._image_verdict("Left: RED, right: Blue", red, blue) == "yes"
    assert cp._image_verdict("  ", red, blue) == "inconclusive"


@pytest.mark.asyncio
async def test_an_image_the_server_refuses_is_no_but_a_rate_limit_is_not_evidence() -> None:
    no, _ = await _probe({"image_input": _HTTPError(400, "image_url is not supported by this model")},
                         checks=["image_input"])
    assert no.checks["image_input"].status == "no"
    for exc in (_HTTPError(429, "rate limited"), _HTTPError(503, "overloaded"),
                _HTTPError(401, "bad image key"), _HTTPError(400, "messages too long"),
                ConnectionError("reset")):
        rep, _ = await _probe({"image_input": exc}, checks=["image_input"])
        assert rep.checks["image_input"].status == "inconclusive", exc
        assert rep.capabilities() == {}  # nothing a host could overwrite with


@pytest.mark.asyncio
async def test_a_hung_call_is_inconclusive_within_the_timeout() -> None:
    async def _hang(_req):
        await asyncio.sleep(5)

    rep, _ = await _probe({"image_input": _hang}, checks=["image_input"], timeout_s=0.05)
    assert rep.checks["image_input"].status == "inconclusive"
    assert rep.checks["image_input"].evidence == "timeout"


# ── json_schema: enforced, not merely accepted ──────────────────────────────


@pytest.mark.asyncio
async def test_json_schema_yes_only_when_the_server_enforces_it() -> None:
    yes, _ = await _probe({"json_schema": _resp('```json\n{"n": 5}\n```')}, checks=["json_schema"])
    assert yes.checks["json_schema"].status == "yes"
    ignored, _ = await _probe({"json_schema": _resp("The answer is 5.")}, checks=["json_schema"])
    assert ignored.checks["json_schema"].status == "no"  # took the parameter, did not honour it
    refused, _ = await _probe(
        {"json_schema": _HTTPError(400, "This response_format type is unavailable now")},
        checks=["json_schema"])
    assert refused.checks["json_schema"].status == "no"
    assert "response_format type is unavailable" in refused.checks["json_schema"].evidence
    assert refused.capabilities() == {"supports_json_schema": False}


@pytest.mark.asyncio
async def test_anthropic_has_no_native_json_schema_and_is_not_called() -> None:
    rep, f = await _probe({}, provider="anthropic", checks=["json_schema"])
    assert rep.checks["json_schema"].status == "no" and not f.configs


# ── tools / thinking: recorded, not corrected ───────────────────────────────


@pytest.mark.asyncio
async def test_tools_and_thinking() -> None:
    rep, _ = await _probe({"tools": _resp("I would rather not."), "thinking": _resp("51")},
                          checks=["tools", "thinking"])
    assert rep.checks["tools"].status == "inconclusive"  # it may just have chosen not to call
    assert rep.checks["thinking"].status == "no"
    rep2, _ = await _probe({"tools": _HTTPError(400, "tools are not supported"),
                            "thinking": _resp("51", think="17×3")}, checks=["tools", "thinking"])
    assert rep2.checks["tools"].status == "no" and rep2.checks["thinking"].status == "yes"
    assert rep2.capabilities() == {}  # neither maps to a declared capability


# ── evidence never carries the key; unknown checks are refused ──────────────


@pytest.mark.asyncio
async def test_evidence_never_carries_the_api_key() -> None:
    rep, _ = await _probe({"image_input": _HTTPError(400, f"image rejected for key {KEY}")},
                          checks=["image_input"])
    assert KEY not in rep.checks["image_input"].evidence
    assert "sk-live" not in rep.to_dict()["checks"]["image_input"]["evidence"]


@pytest.mark.asyncio
async def test_unknown_check_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown probe checks"):
        await cp.probe_capabilities(_cfg(), checks=["context_window"], service_factory=_Factory({}))


# ── error_kind: why an inconclusive check failed ────────────────────────────


def _sdk_status(status: int, message: str, **extra: Any) -> Exception:
    """A real openai SDK status error, shaped the way the SDK builds it from a response."""
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    req = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    body = {"message": message, **extra}
    cls = {400: openai.BadRequestError, 401: openai.AuthenticationError,
           403: openai.PermissionDeniedError, 404: openai.NotFoundError,
           429: openai.RateLimitError}.get(status, openai.APIStatusError)
    return cls(f"Error code: {status} - {{'error': {body!r}}}",
               response=httpx.Response(status, request=req), body=body)


def _transport_errors() -> list[tuple[Exception, str]]:
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    req = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    out: list[tuple[Exception, str]] = [
        (openai.APITimeoutError(request=req), "timeout"),
        (httpx.ReadTimeout("read timed out", request=req), "timeout"),
        (openai.APIConnectionError(request=req), "network"),
        (httpx.ConnectError("connection refused", request=req), "network"),
        (httpx.ReadError("peer closed", request=req), "network"),
        (ConnectionResetError("reset by peer"), "network"),
    ]
    try:
        import anthropic
    except ImportError:
        return out
    return out + [(anthropic.APITimeoutError(request=req), "timeout"),
                  (anthropic.APIConnectionError(request=req), "network")]


_KINDS: list[tuple[str, Exception, str]] = [
    ("402 bare", _HTTPError(402, "Insufficient Balance"), "balance"),
    # 402 whose message names the capability (and matches no balance words): still not evidence
    ("402 naming images", _HTTPError(402, "image_url input is a paid feature"), "balance"),
    ("400 about images, but out of money",
     _HTTPError(400, "image_url rejected: insufficient balance"), "balance"),
    ("chinese balance", _HTTPError(400, "账户余额不足，请充值"), "balance"),
    ("anthropic credit balance", _HTTPError(
        400, "Your credit balance is too low to access the Anthropic API. Please go to Plans & "
             "Billing to upgrade or purchase credits."), "balance"),
    ("openai inactive account", _HTTPError(
        429, "Your account is not active, please check your billing details on our website."),
     "balance"),
    ("401", _HTTPError(401, "Authentication Fails, Your api key: ****7890 is invalid"), "auth"),
    ("400 bad key text", _HTTPError(400, "Incorrect API key provided"), "auth"),
    ("gemini bad key", _HTTPError(400, "API key not valid. Please pass a valid API key."), "auth"),
    # a bare 403 / 404 whose message does not name the capability (one that does is a "no")
    ("403", _HTTPError(403, "Access to this model is not allowed"), "forbidden"),
    ("403 text", _HTTPError(400, "permission denied for image input"), "forbidden"),
    ("404", _HTTPError(404, "Not Found"), "model_missing"),
    ("404 wrong path", _HTTPError(404, "404 page not found"), "model_missing"),
    ("model does not exist", _HTTPError(400, "The model `m1` does not exist"), "model_missing"),
    # a dot inside a model name does not end the sentence
    ("openai dotted model", _HTTPError(
        400, "The model `gpt-4.1-nano` does not exist or you do not have access to it."),
     "model_missing"),
    ("old deepseek wrong name", _HTTPError(400, "Model Not Exist"), "model_missing"),
    ("deepseek wrong name", _HTTPError(
        400, "The supported API model names are deepseek-v4-flash or deepseek-v4-pro, "
             "but you passed deepseek-no-such-model."), "model_missing"),
    ("429", _HTTPError(429, "Rate limit reached, please retry later"), "rate_limited"),
    ("500", _HTTPError(500, "internal error"), "server"),
    ("503", _HTTPError(503, "overloaded"), "server"),
    ("408", _HTTPError(408, "request timeout"), "timeout"),
    ("builtin timeout", TimeoutError("timed out"), "timeout"),
    ("builtin network", ConnectionError("reset"), "network"),
    ("400 other", _HTTPError(400, "messages too long"), "unknown"),
    ("anything else", ValueError("weird"), "unknown"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("label", "exc", "kind"), _KINDS, ids=[k[0] for k in _KINDS])
async def test_each_failure_is_inconclusive_with_its_kind(label: str, exc: Exception, kind: str) -> None:
    rep, _ = await _probe({"image_input": exc}, checks=["image_input"])
    r = rep.checks["image_input"]
    assert (r.status, r.error_kind) == ("inconclusive", kind), (label, r)
    assert rep.capabilities() == {}  # nothing a host could overwrite with
    assert r.to_dict()["error_kind"] == kind
    assert KEY not in r.evidence and "sk-live" not in r.evidence


@pytest.mark.asyncio
async def test_real_sdk_errors_are_classified() -> None:
    quota = _sdk_status(429, "You exceeded your current quota, please check your plan and billing "
                             "details.", type="insufficient_quota", code="insufficient_quota")
    cases = [
        (quota, "balance"),  # OpenAI reports an exhausted quota as a 429
        (_sdk_status(429, "Rate limit reached for requests"), "rate_limited"),
        # a rate limit that points at the billing page is still a rate limit, not an empty account
        (_sdk_status(429, "Rate limit reached for gpt-4o-mini in organization org-abc on requests "
                          "per min (RPM): Limit 3, Used 3, Requested 1. Please try again in 20s. "
                          "Visit https://platform.openai.com/account/rate-limits to learn more. You "
                          "can increase your rate limit by adding a payment method to your account "
                          "at https://platform.openai.com/account/billing.",
                     type="requests", code="rate_limit_exceeded"), "rate_limited"),
        # DashScope (Aliyun) arrears: a 400 that says "Access denied" — it is money, not permission
        (_sdk_status(400, "Access denied, please make sure your account is in good standing. For "
                          "details, see: https://help.aliyun.com/zh/model-studio/error-code"
                          "#overdue-payment", type="Arrearage", code="Arrearage"), "balance"),
        (_sdk_status(402, "Insufficient Balance", type="unknown_error"), "balance"),
        # OpenAI: an account without billing set up
        (_sdk_status(429, "Your account is not active, please check your billing details on our "
                          "website.", type="billing_not_active", code="billing_not_active"),
         "balance"),
        # Moonshot: a suspended account, as a 429
        (_sdk_status(429, "Your account org-abc<ak-xyz> is suspended, please check your plan and "
                          "billing details", type="exceeded_current_quota_error"), "balance"),
        # the type alone says it when the sentence does not
        (_sdk_status(429, "Your account is suspended", type="exceeded_current_quota_error"),
         "balance"),
        (_sdk_status(401, "Authentication Fails, Your api key: ****7890 is invalid",
                     type="authentication_error"), "auth"),
        (_sdk_status(400, "The supported API model names are deepseek-v4-flash or deepseek-v4-pro, "
                          "but you passed nope.", type="invalid_request_error"), "model_missing"),
        (_sdk_status(502, "Bad gateway"), "server"),
        *_transport_errors(),
    ]
    for exc, kind in cases:
        rep, _ = await _probe({"json_schema": exc}, checks=["json_schema"])
        r = rep.checks["json_schema"]
        assert (r.status, r.error_kind) == ("inconclusive", kind), (type(exc).__name__, exc, r)


@pytest.mark.asyncio
async def test_blocking_kinds_win_over_a_capability_named_in_the_message_on_every_check() -> None:
    rep, _ = await _probe({
        "image_input": _HTTPError(402, "image input requires balance"),
        "json_schema": _HTTPError(401, "response_format: invalid api key"),
        "tools": _HTTPError(403, "tool calling is forbidden for this key"),
        "thinking": _HTTPError(404, "model not found"),
    })
    got = {k: (v.status, v.error_kind) for k, v in rep.checks.items()}
    assert got == {"image_input": ("inconclusive", "balance"), "json_schema": ("inconclusive", "auth"),
                   "tools": ("inconclusive", "forbidden"), "thinking": ("inconclusive", "model_missing")}
    assert rep.capabilities() == {}


@pytest.mark.asyncio
async def test_a_refusal_that_names_the_capability_is_an_answer_not_an_error() -> None:
    rep, _ = await _probe({
        "image_input": _HTTPError(400, "image_url is not supported by this model"),
        "json_schema": _sdk_status(400, "This response_format type is unavailable now",
                                   type="invalid_request_error"),
        "tools": _HTTPError(422, "tools are not supported"),
        # thinking has no refusal: a 4xx there is never a "no"
        "thinking": _HTTPError(400, "thinking is not supported"),
    })
    got = {k: (v.status, v.error_kind) for k, v in rep.checks.items()}
    assert got == {"image_input": ("no", None), "json_schema": ("no", None), "tools": ("no", None),
                   "thinking": ("inconclusive", "unknown")}


@pytest.mark.asyncio
async def test_a_403_or_404_that_names_the_capability_is_a_no() -> None:
    """OpenRouter routes by capability: when no endpoint of the model takes images / tools it
    answers 404 "No endpoints found that support …". That is the answer, not a missing model."""
    rep, _ = await _probe({
        "image_input": _sdk_status(404, "No endpoints found that support image input", code=404),
        "tools": _sdk_status(404, "No endpoints found that support tool use. To learn more about "
                                  "provider routing, visit: https://openrouter.ai/docs/"
                                  "provider-routing", code=404),
        # "you passed" alone is not DeepSeek's wrong-model-name error
        "json_schema": _sdk_status(400, "Invalid schema for response_format 'probe': the schema "
                                        "you passed is invalid.", type="invalid_request_error"),
    }, checks=["image_input", "tools", "json_schema"])
    got = {k: (v.status, v.error_kind) for k, v in rep.checks.items()}
    assert got == {"image_input": ("no", None), "tools": ("no", None), "json_schema": ("no", None)}
    assert "support image input" in rep.checks["image_input"].evidence
    assert rep.capabilities() == {"supports_image_input": False, "supports_json_schema": False}
    forbidden, _ = await _probe({"image_input": _HTTPError(403, "image input is not enabled for "
                                                                "this model")},
                                checks=["image_input"])
    r = forbidden.checks["image_input"]
    assert (r.status, r.error_kind) == ("no", None)


@pytest.mark.asyncio
async def test_does_not_exist_is_a_missing_model_only_when_it_is_about_the_model() -> None:
    """A bare "does not exist" is not a missing model: a tools 400 "function does not exist" is
    about tools, and "model" in an earlier sentence does not make it about the model."""
    for msg in ("Invalid 'tools[0].function.name': function does not exist",
                "Model m1 rejected the request. Function get_probe_code does not exist."):
        rep, _ = await _probe({"tools": _HTTPError(400, msg)}, checks=["tools"])
        r = rep.checks["tools"]
        assert (r.status, r.error_kind) == ("no", None), (msg, r)


# ── the model's own name does not name a capability ─────────────────────────


@pytest.mark.asyncio
async def test_a_model_named_after_a_capability_is_still_missing_not_a_no() -> None:
    """OpenRouter answers an id it has no endpoint for with 404 "No endpoints found for <id>." —
    when the id says "vision", that is still a missing model, not "cannot see"."""
    llama = "meta-llama/llama-3.2-11b-vision-instruct"
    cases = [
        (llama, "image_input",
         _sdk_status(404, f"No endpoints found for {llama}.", code=404)),
        # the configured name is matched in any case
        ("Acme/Vision-Pro", "image_input", _HTTPError(404, "acme/vision-pro is not available")),
        # the "org/" prefix may be left out of the message
        (llama, "image_input", _HTTPError(404, "llama-3.2-11b-vision-instruct is not available")),
        ("qwen-vl-max", "image_input", _HTTPError(400, "model qwen-vl-max does not exist")),
        ("acme/xxx-tools", "tools", _HTTPError(404, "No endpoints found for acme/xxx-tools.")),
    ]
    for model, check, exc in cases:
        rep, _ = await _probe({check: exc}, checks=[check], model=model)
        r = rep.checks[check]
        assert (r.status, r.error_kind) == ("inconclusive", "model_missing"), (model, exc, r)
        assert rep.capabilities() == {}
    # a refusal that names the capability besides the model is still a "no"
    rep, _ = await _probe({"image_input": _HTTPError(
        404, f"No endpoints found for {llama} that support image input")},
        checks=["image_input"], model=llama)
    assert (rep.checks["image_input"].status, rep.checks["image_input"].error_kind) == ("no", None)


@pytest.mark.asyncio
async def test_the_probes_own_timeout_is_kind_timeout() -> None:
    async def _hang(_req):
        await asyncio.sleep(5)

    rep, _ = await _probe({"image_input": _hang, "json_schema": _hang, "tools": _hang,
                           "thinking": _hang}, timeout_s=0.05)
    assert {k: (v.status, v.evidence, v.error_kind) for k, v in rep.checks.items()} == {
        k: ("inconclusive", "timeout", "timeout") for k in cp.ALL_CHECKS}


@pytest.mark.asyncio
async def test_empty_replies_are_kind_empty_but_declining_to_call_is_not_an_error() -> None:
    rep, _ = await _probe({"image_input": _resp(""), "json_schema": _resp("  "), "tools": _resp(""),
                           "thinking": _resp("")})
    assert {k: (v.status, v.error_kind) for k, v in rep.checks.items()} == {
        k: ("inconclusive", "empty") for k in cp.ALL_CHECKS}
    assert "应为" in rep.checks["image_input"].evidence
    declined, _ = await _probe({"tools": _resp("I would rather not.")}, checks=["tools"])
    assert (declined.checks["tools"].status, declined.checks["tools"].error_kind) == ("inconclusive", None)


@pytest.mark.asyncio
async def test_conclusive_results_carry_no_error_kind() -> None:
    lft, rgt = _colours(7)
    rep, _ = await _probe({
        "image_input": _resp(f"左：{lft}，右：{rgt}"), "json_schema": _resp("The answer is 5."),
        "tools": _resp(tool="get_probe_code"), "thinking": _resp("51", think="17*3=51"),
    })
    assert {k: v.error_kind for k, v in rep.checks.items()} == dict.fromkeys(cp.ALL_CHECKS)
    assert rep.to_dict()["checks"]["tools"]["error_kind"] is None


def test_check_result_positional_construction_still_works() -> None:
    r = cp.CheckResult("tools", "yes", "called x", 12, {"total_tokens": 3})
    assert r.error_kind is None
    assert list(r.to_dict()) == ["name", "status", "evidence", "latency_ms", "usage", "error_kind"]
    assert set(cp.BLOCKING_ERROR_KINDS) < set(cp.ERROR_KINDS)
