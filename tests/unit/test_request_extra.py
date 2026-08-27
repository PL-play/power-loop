"""`OpenAICompatibleChatConfig.request_extra` — config-level request kwargs under each request's
`extra` (request wins; `extra_body` merged key-by-key), and the LLMProviderConfig passthrough."""

from __future__ import annotations

import pytest

from power_loop._vendor.llm_client.llm_factory import merge_request_extra
from power_loop.runtime.provider import LLMProviderConfig

pytestmark = pytest.mark.unit


def test_merge_request_extra_request_wins_and_extra_body_merges():
    base = {"extra_body": {"enable_thinking": True, "x": 1}, "top_p": 0.9}
    out = merge_request_extra(base, {"extra_body": {"x": 2}, "seed": 7})
    assert out == {"extra_body": {"enable_thinking": True, "x": 2}, "top_p": 0.9, "seed": 7}
    assert merge_request_extra(None, None) == {}
    assert merge_request_extra(base, None)["extra_body"] is not base["extra_body"]  # copied, not aliased


def test_provider_config_passes_request_extra_through():
    cfg = LLMProviderConfig(base_url="http://x", api_key="k", model="m",
                            extra={"request_extra": {"extra_body": {"enable_thinking": False}}})
    assert cfg.to_openai_compatible().request_extra == {"extra_body": {"enable_thinking": False}}
    assert LLMProviderConfig(base_url="http://x", api_key="k", model="m").to_openai_compatible().request_extra == {}
