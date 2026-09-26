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


def _cfg(provider: str = "openai") -> LLMProviderConfig:
    return LLMProviderConfig(base_url="https://llm.example/v1", api_key=KEY, model="m1",
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


async def _probe(script: dict[str, Any], *, seed: int = 7, provider: str = "openai", **kw):
    f = _Factory(script)
    rep = await cp.probe_capabilities(_cfg(provider), rng=random.Random(seed), service_factory=f, **kw)
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
