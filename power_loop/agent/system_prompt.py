from __future__ import annotations

from dataclasses import dataclass


DEFAULT_AGENT_SYSTEM_PROMPT = """You are an interactive CLI coding agent.

Core behavior:
- Read code before proposing code changes.
- Keep changes minimal, reversible, and testable.
- For multi-step tasks, operate in small validated increments.
- Prefer explicitness over hidden side effects.

Execution discipline:
- If tool calls are needed, emit structured tool calls.
- After tool results are returned, continue reasoning based on observations.
- Stop when user goal is satisfied or when blocked with a clear reason.
"""


DEFAULT_SUBAGENT_SYSTEM_PROMPT = """You are a coding subagent.

Execution mode:
- Complete the assigned task autonomously.
- Keep outputs concise and evidence-based.
- Return key findings with exact file paths and line anchors when possible.
"""


DEFAULT_EXPLORE_SUBAGENT_SYSTEM_PROMPT = """You are a read-only exploration subagent.

Execution mode:
- Search and read files only; do not modify code.
- Summarize findings with concrete references.
- Flag uncertainties explicitly.
"""


@dataclass(frozen=True)
class SystemPromptContext:
    model: str
    workspace_dir: str
    agent_dir: str | None = None
    skills_dir: str | None = None


def _runtime_suffix(ctx: SystemPromptContext, extra: str | None = None) -> str:
    lines = [
        "Runtime context:",
        f"- model: {ctx.model}",
        f"- workspace: {ctx.workspace_dir}",
    ]
    if ctx.agent_dir:
        lines.append(f"- agent_home: {ctx.agent_dir}")
    if ctx.skills_dir:
        lines.append(f"- skills_dir: {ctx.skills_dir}")
    if extra and extra.strip():
        lines.append("")
        lines.append("Extra instructions:")
        lines.append(extra.strip())
    return "\n".join(lines)


def build_agent_system_prompt(ctx: SystemPromptContext, extra: str | None = None) -> str:
    return DEFAULT_AGENT_SYSTEM_PROMPT.strip() + "\n\n" + _runtime_suffix(ctx, extra=extra).strip() + "\n"


def build_subagent_system_prompt(ctx: SystemPromptContext, extra: str | None = None) -> str:
    return DEFAULT_SUBAGENT_SYSTEM_PROMPT.strip() + "\n\n" + _runtime_suffix(ctx, extra=extra).strip() + "\n"


def build_explore_subagent_system_prompt(ctx: SystemPromptContext, extra: str | None = None) -> str:
    return DEFAULT_EXPLORE_SUBAGENT_SYSTEM_PROMPT.strip() + "\n\n" + _runtime_suffix(ctx, extra=extra).strip() + "\n"
