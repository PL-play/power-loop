"""14 · 结构化输出：让 LLM 直接吐 JSON 卡片（M1.3）

What you learn
--------------
- ``StructuredOutputSpec(name, schema, strict=True)`` 一次性描述「我要
  什么 JSON」；``.to_openai_response_format()`` 渲染成 OpenAI 兼容的
  ``response_format`` 字典直接灌进 ``LLMRequest``。
- ``parse_structured(response, schema=...)`` 把 LLM 输出转成 dict：
  自动剥 markdown 围栏、抓第一段 ``{...}``、修补尾逗号；schema 缺字段
  报 ``StructuredOutputError(reason="missing_required:<field>")``。
- 解析失败给的是**带原文**的可调试异常 (`raw_text` / `reason`)，不会
  silent 吞掉。

适用场景
--------
- Agent 卡片输出（DeepTalk 关系洞察 / 会话纪要 / 引导提示）
- 一切「我要 JSON、希望它对得上 schema」的回合

Run
---
    python examples/14_structured_card.py
"""

from __future__ import annotations

import asyncio
import json

from _helpers import make_llm

from llm_client.interface import LLMRequest
from power_loop import StructuredOutputError, StructuredOutputSpec, parse_structured

# ── 1. 卡片 schema —— 关系洞察周报里可能用到的一小段 ─────────────────────

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "user 的称呼"},
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
    # ── 1. 走真实 LLM 的正常路径 ──────────────────────────────────────
    card = await extract_card("我叫阿岚，住上海，最喜欢的数字是 37。爱好：徒步、写代码、做饭。")
    print("[ok] card =", json.dumps(card, ensure_ascii=False, indent=2))

    # ── 2. 演示 parse_structured 的修复能力（不走 LLM）────────────────
    noisy = (
        "Sure, here is the card:\n"
        "```json\n"
        "{\n"
        '  "name": "小明",\n'
        '  "favorite_number": 7,\n'   # trailing comma below intentionally
        "}\n"
        "```\n"
        "Hope this helps!"
    )
    print("[repair] parsed =", parse_structured(noisy, schema=SCHEMA))

    # ── 3. Schema mismatch：原文里就没数字，靠 parse_structured 报错 ─
    bad = '{"name": "小明"}'
    try:
        parse_structured(bad, schema=SCHEMA)
    except StructuredOutputError as exc:
        print(f"[caught] reason={exc.reason!r} raw_text={exc.raw_text!r}")


if __name__ == "__main__":
    asyncio.run(main())
