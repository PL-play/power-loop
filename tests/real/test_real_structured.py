"""Real-LLM smoke for M1.3 structured output.

We hit the real provider with ``response_format`` set to a small
json_schema and assert that the parsed dict satisfies the schema. This
proves three wires at once:

1. ``LLMRequest.response_format`` reaches ``chat.completions.create``
   via ``_request_kwargs``.
2. The provider returns valid JSON (or near-valid that our repair
   handles).
3. ``parse_structured`` round-trips real provider output into a typed
   dict.
"""

from __future__ import annotations

import pytest

from power_loop import StructuredOutputSpec, parse_structured
from power_loop._vendor.llm_client.interface import LLMRequest

from ._llm import make_llm

CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "favorite_number": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "favorite_number"],
    "additionalProperties": False,
}


@pytest.mark.asyncio
async def test_real_structured_card_extraction() -> None:
    spec = StructuredOutputSpec(
        name="UserCard",
        schema=CARD_SCHEMA,
        description="Extract user profile facts mentioned in the message.",
    )
    llm = make_llm(max_tokens=200, temperature=0.0)
    try:
        req = LLMRequest(
            messages=[{
                "role": "user",
                "content": "我叫阿岚，住上海，最喜欢的数字是 37。请只输出 JSON。",
            }],
            system_prompt="Extract the user's profile into a JSON object that matches the schema.",
            response_format=spec.to_openai_response_format(),
            max_tokens=200, temperature=0.0,
        )
        resp = await llm.complete(req)
        card = parse_structured(resp, schema=CARD_SCHEMA)
        assert card["name"]
        # Provider should respect json_schema or our parser should rescue it
        assert isinstance(card["favorite_number"], int)
        assert card["favorite_number"] == 37
    finally:
        await llm.close()
