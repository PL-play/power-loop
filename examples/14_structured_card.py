"""14 · 结构化输出 / Structured output: LLM emits JSON cards directly

What you learn / 你将学到
--------------------------
- ``StructuredOutputSpec(name, schema, strict=True)`` 一次性描述「我要
  什么 JSON」；``.to_openai_response_format()`` 渲染成 OpenAI 兼容的
  ``response_format`` 字典直接灌进 ``LLMRequest``。
  / describes "what JSON I want" in one shot; ``.to_openai_response_format()``
  renders an OpenAI-compatible ``response_format`` dict for ``LLMRequest``
- ``parse_structured(response, schema=...)`` 把 LLM 输出转成 dict：
  自动剥 markdown 围栏、抓第一段 ``{...}``、修补尾逗号；schema 缺字段
  报 ``StructuredOutputError(reason="missing_required:<field>")``。
  / converts LLM output to dict: auto-strips markdown fences, extracts first
  ``{...}`` block, repairs trailing commas; missing schema fields raise
  ``StructuredOutputError(reason="missing_required:<field>")``
- 解析失败给的是**带原文**的可调试异常 (`raw_text` / `reason`)，不会 silent 吞掉。
  / parse failures produce **debuggable** exceptions with ``raw_text`` / ``reason`` — never silently swallowed

适用场景 / Use cases
---------------------
- Agent 卡片输出（关系洞察 / 会话纪要 / 引导提示）
  / Agent card output (relationship insights / session summaries / guided prompts)
- 一切「我要 JSON、希望它对得上 schema」的回合
  / any scenario where you want JSON that conforms to a schema

Run / 运行
----------
    python examples/14_structured_card.py
"""

from __future__ import annotations

import asyncio
import json

from _helpers import make_llm

from power_loop import StructuredOutputError, StructuredOutputSpec, parse_structured
from power_loop._vendor.llm_client.interface import LLMRequest

# ── 1. 卡片 schema —— 关系洞察周报里可能用到的一小段 ─────────────────────
#    Card schema — a small slice that might appear in a relationship insight report

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "user's name"},
        "favorite_number": {"type": "integer"},
        "city": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "favorite_number"],
    "additionalProperties": False,
}

SPEC = StructuredOutputSpec(
    name="UserCard",
    schema=SCHEMA,
    description="Extract the user's profile mentioned in the message.",
)


async def extract_card(user_text: str) -> dict:
    llm = make_llm(max_tokens=200, temperature=0.0)
    try:
        req = LLMRequest(
            messages=[{"role": "user", "content": user_text}],
            system_prompt=(
                "Extract the user's profile into a JSON object that matches "
                "the schema. Output JSON only — no prose."
            ),
            response_format=SPEC.to_openai_response_format(),
            max_tokens=200, temperature=0.0,
        )
        resp = await llm.complete(req)
        return parse_structured(resp, schema=SCHEMA)
    finally:
        await llm.close()


async def main() -> None:
    # ── 1. 走真实 LLM 的正常路径 / Normal path via real LLM ──────────────
    card = await extract_card(
        "My name is Alan, I live in Shanghai, my favorite number is 37. "
        "Hobbies: hiking, coding, cooking."
    )
    print("[ok] card =", json.dumps(card, ensure_ascii=False, indent=2))

    # ── 2. 演示 parse_structured 的修复能力（不走 LLM）────────────────
    #    Demo parse_structured's repair capabilities (no LLM)
    noisy = (
        "Sure, here is the card:\n"
        "```json\n"
        "{\n"
        '  "name": "Xiao Ming",\n'
        '  "favorite_number": 7,\n'   # trailing comma below intentionally
        "}\n"
        "```\n"
        "Hope this helps!"
    )
    print("[repair] parsed =", parse_structured(noisy, schema=SCHEMA))

    # ── 3. Schema mismatch：原文里就没数字，靠 parse_structured 报错 ─
    #    Schema mismatch: no number in source — parse_structured catches it
    bad = '{"name": "Xiao Ming"}'
    try:
        parse_structured(bad, schema=SCHEMA)
    except StructuredOutputError as exc:
        print(f"[caught] reason={exc.reason!r} raw_text={exc.raw_text!r}")


if __name__ == "__main__":
    asyncio.run(main())
