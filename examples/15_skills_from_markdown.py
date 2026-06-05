"""15 · Skills：从 SKILL.md 文件加载领域知识注入 system prompt

## What you'll learn
- 加载 ``SKILL.md`` 格式的 frontmatter 文件并解析出 name / description / instructions
- 把 skill 内容拼接进 ``AgentLoopConfig.system_prompt`` 作为领域知识
- Agent 能按照 skill 里定义的规则回答问题

## Prerequisites
- 需要 ``.env`` 配置 ``POWER_LOOP_*``（或 ``OPENAI_COMPAT_*``）

## Run
    python examples/15_skills_from_markdown.py

## Key concepts
- **SKILL.md**: Markdown 文件，YAML frontmatter 声明 ``name`` / ``description``，
  正文是给 Agent 的指令。这是 power-loop 里最轻量的「给 Agent 注入领域知识」的方式。
- **System prompt 组装**: 多个 skill 可以拼接成一个 system prompt，业务方自由组合。
- 本例不依赖 ``runtime/skills.py``（那是内部实现），直接演示外部如何加载文件。

## Next
看看 `16_custom_compactor.py` — 实现自定义压缩策略
"""

from __future__ import annotations

import asyncio
import re
import textwrap

from _helpers import make_llm

from power_loop import AgentLoopConfig, StatefulAgentLoop

# ── 1. 两个简单的 SKILL.md 文件（inline，不写盘）───────────────────────

SKILL_PYTHON = textwrap.dedent("""\
    ---
    name: python-expert
    description: Answer Python questions with best practices
    ---
    # Python Expert

    When answering Python questions:
    1. Always provide a runnable example.
    2. Mention the Python version if relevant (3.10+).
    3. Prefer stdlib over third-party packages when possible.
    4. Use type hints in all code examples.
    """)

SKILL_SECURITY = textwrap.dedent("""\
    ---
    name: security-reviewer
    description: Flag security concerns in code
    ---
    # Security Reviewer

    When reviewing code:
    1. Point out SQL injection, XSS, path traversal, and command injection risks.
    2. Suggest fixes with code examples.
    3. If the code is safe, explicitly say so.
    """)


# ── 2. 简易 frontmatter 解析器 ──────────────────────────────────────────


def parse_skill_md(text: str) -> dict[str, str]:
    """Parse a SKILL.md string. Returns ``{name, description, instructions}``."""
    name = description = ""
    # Extract YAML frontmatter
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1)
        body = fm_match.group(2).strip()
        for line in fm_text.splitlines():
            line = line.strip()
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                description = line.split(":", 1)[1].strip()
    else:
        body = text.strip()
    return {"name": name, "description": description, "instructions": body}


def build_system_prompt(*skills: str) -> str:
    """Combine multiple SKILL.md texts into a single system prompt."""
    parts = [parse_skill_md(s) for s in skills]
    prompt = "You are a helpful assistant with these skills:\n\n"
    for p in parts:
        prompt += f"## {p['name']}\n{p['instructions']}\n\n"
    prompt += "Use the skills above to answer the user's question."
    return prompt


# ── 3. 跑两个场景 ────────────────────────────────────────────────────────


async def main() -> None:
    llm = make_llm(max_tokens=200, temperature=0.0)

    # Scenario A: Python expert
    prompt_a = build_system_prompt(SKILL_PYTHON)
    loop_a = StatefulAgentLoop(
        llm=llm,
        config=AgentLoopConfig(system_prompt=prompt_a, max_rounds=1, compactor=None),
    )
    try:
        r = await loop_a.send("How do I read a JSON file?")
        print(f"[Python Expert]\n{r.final_text}\n")
    finally:
        loop_a.close()

    # Scenario B: Python expert + Security reviewer
    prompt_b = build_system_prompt(SKILL_PYTHON, SKILL_SECURITY)
    loop_b = StatefulAgentLoop(
        llm=llm,
        config=AgentLoopConfig(system_prompt=prompt_b, max_rounds=1, compactor=None),
    )
    try:
        r = await loop_b.send(
            "Review this code: `query = f\"SELECT * FROM users WHERE name='{name}'\"`"
        )
        print(f"[Python Expert + Security]\n{r.final_text}")
    finally:
        loop_b.close()


if __name__ == "__main__":
    asyncio.run(main())