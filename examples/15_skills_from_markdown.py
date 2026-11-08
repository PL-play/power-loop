"""15 · Parse skill instructions and supply the resulting config in the factory.

This inline demonstration has no external resource dependency. File-backed
skills need a stable, authorized resource profile resolved on every worker.
"""
from __future__ import annotations

import asyncio
import re
import textwrap
import time

from _runtime import demo_runtime

from power_loop import AgentLoopConfig, LLMResponse, LLMService

# ── 1. 两个简单的 SKILL.md 文件（inline，不写盘）───────────────────────
#    Two simple SKILL.md files (inline, not written to disk)

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


# ── 2. 简易 frontmatter 解析器 / Simple frontmatter parser ──────────────


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


# ── 3. 跑两个场景 / Run two scenarios ────────────────────────────────────


class SkillsModel(LLMService):
    async def complete(self, request, **kwargs):
        assert 'Use type hints' in request.system_prompt
        return LLMResponse(raw_text='def greet(name: str) -> str: return "Hello " + name')

    async def close(self):
        pass


async def main():
    for skills in [(SKILL_PYTHON,), (SKILL_PYTHON, SKILL_SECURITY)]:
        config = AgentLoopConfig(system_prompt=build_system_prompt(*skills), max_rounds=1, compactor=None)
        async with demo_runtime(SkillsModel(), config=config) as (runtime, loop, target, actor):
            result = await loop.send(target, 'Write a typed greeting function', message_id='skill', actor=actor,
                                     deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
            print(result.final_text)


if __name__ == '__main__':
    asyncio.run(main())
