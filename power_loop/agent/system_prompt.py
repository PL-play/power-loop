"""Composable system prompt builder for power-loop agents.

Design goals
------------
* **Library-friendly**: nothing is hard-coded.  Users can fully replace the
  prompt, cherry-pick built-in sections, or append custom sections.
* **Runtime-aware**: ``SystemPromptContext`` carries workspace paths, tool
  list, skill descriptions, model name, etc.  Sections that need this info
  receive it automatically.
* **Backward-compatible**: the old ``build_agent_system_prompt(ctx)`` API
  still works.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

# ---------------------------------------------------------------------------
# Context that sections can reference
# ---------------------------------------------------------------------------

@dataclass
class SystemPromptContext:
    """Runtime context injected into prompt sections.

    All fields are optional so callers only supply what they have.
    """

    model: str = ""
    workspace_dir: str = ""
    agent_dir: str = ""
    skills_dir: str = ""
    skill_descriptions: str = ""
    tool_names: Sequence[str] = ()
    extra: str = ""


# ---------------------------------------------------------------------------
# Built-in prompt sections  (each is a plain function: ctx -> str | None)
# ---------------------------------------------------------------------------
SectionRenderer = Callable[[SystemPromptContext], str | None]


def section_identity(ctx: SystemPromptContext) -> str:
    parts = ["You are an interactive coding agent."]
    if ctx.model:
        parts.append(f"Model: {ctx.model}")
    return "\n".join(parts)


def section_style(_ctx: SystemPromptContext) -> str:
    return (
        "# Tone and Style\n"
        "- Be concise and direct; use markdown sparingly.\n"
        "- Prioritize technical accuracy. Respectful correction is more valuable than false agreement.\n"
        "- Never propose changes to code you haven't read. Always read first, then modify.\n"
        "- Avoid over-engineering. Only make changes directly requested or clearly necessary.\n"
        "  - Don't add features, refactoring, comments, or type annotations beyond what was asked.\n"
        "  - Don't add error handling for scenarios that can't happen.\n"
        "  - Don't create abstractions for one-time operations."
    )


def section_workflow(_ctx: SystemPromptContext) -> str:
    return (
        "# Workflow\n"
        "1. Understand: read relevant files before acting. Use glob/grep to find files, not bash.\n"
        "2. Plan: for multi-step tasks (3+ steps), create a todo list FIRST.\n"
        "3. Execute: work through items one at a time. Mark each in_progress before starting, completed when done.\n"
        "4. Verify: after edits, run tests or linters when appropriate."
    )


def section_security(_ctx: SystemPromptContext) -> str:
    return (
        "# Security\n"
        "- Never request passwords, tokens, API keys, or any secrets via chat.\n"
        "- Do NOT simulate interactive password prompts. If a command needs sudo, explain what the user should run manually."
    )


def section_tool_guide(ctx: SystemPromptContext) -> str | None:
    """Per-tool usage hints.  Only emits tools that are actually registered."""
    catalog: Dict[str, str] = {
        "bash": (
            "persistent session — cwd and env vars survive across calls. "
            "Use restart=true to reset. Avoid dangerous commands (rm -rf /, sudo)."
        ),
        "read_file": (
            'returns numbered lines ("  1|code"). Use offset/limit for large files. '
            "Pass a directory path to list contents. Always read before editing."
        ),
        "write_file": (
            "create or overwrite a file. Both 'path' and 'content' are REQUIRED in one JSON object. "
            "Prefer edit_file or apply_patch for existing files."
        ),
        "edit_file": (
            "str_replace (old_text -> new_text). old_text must be unique — include more context if ambiguous. "
            "Set replace_all=true to replace every occurrence. You MUST read_file before editing."
        ),
        "apply_patch": (
            "apply a patch using @@ context lines for positioning and +/- for line changes. "
            "You MUST read_file before patching. Best for large edits."
        ),
        "glob": "find files by pattern (e.g. '*.py', '**/*.ts'). Prefer over bash find.",
        "grep": "search file contents by regex. Prefer over bash grep/rg. Supports include filter.",
        "load_skill": "load specialized knowledge before tackling unfamiliar domains.",
        "todo": (
            "track multi-step tasks. Keep exactly one item in_progress at a time. "
            "Mark in_progress BEFORE starting, completed IMMEDIATELY when done."
        ),
        "background_run": "run a shell command asynchronously in a background worker.",
        "check_background": "inspect status/output of background tasks by task_id or list all.",
    }

    names = ctx.tool_names if ctx.tool_names else list(catalog.keys())
    lines = ["# Tool Usage"]
    lines.append(
        "CRITICAL — File path fidelity: When passing file paths to ANY tool, "
        "use the EXACT path string as given. NEVER rename, re-space, or re-punctuate file names."
    )
    for name in names:
        hint = catalog.get(name)
        if hint:
            lines.append(f"- {name}: {hint}")
    return "\n".join(lines)


def section_paths(ctx: SystemPromptContext) -> str | None:
    if not ctx.workspace_dir:
        return None
    lines = [
        "# Paths — CRITICAL",
        f"- **Workspace (working directory)**: {ctx.workspace_dir}",
        "  All file operations default to this directory. Relative paths resolve here.",
    ]
    if ctx.agent_dir:
        lines.append(
            f"- Agent home (library installation): {ctx.agent_dir}\n"
            "  Agent-home internals are restricted. "
            "Allowlisted paths: .cache, logs, and skills directory."
        )
        lines.append(
            f"- IMPORTANT: Do NOT confuse workspace ({ctx.workspace_dir}) with agent home ({ctx.agent_dir})."
        )
    if ctx.skills_dir:
        lines.append(f"- Skills directory: {ctx.skills_dir}")
    lines.append(
        "- Use @workspace/<path> or @agent/<path> when an explicit root is needed, "
        "but prefer plain relative paths (they default to workspace)."
    )
    return "\n".join(lines)


def section_todo_discipline(_ctx: SystemPromptContext) -> str:
    return (
        "# Todo Discipline\n"
        "- Use the todo tool for any task with 3+ steps.\n"
        "- Mark a todo in_progress BEFORE you start working on it.\n"
        "- Mark it completed IMMEDIATELY when done — do not batch.\n"
        "- Only one item should be in_progress at any time."
    )


def section_skills(ctx: SystemPromptContext) -> str | None:
    if not ctx.skill_descriptions or ctx.skill_descriptions == "(no skills available)":
        return None
    lines = [
        "# Skills",
        f"Skills directory: {ctx.skills_dir or '(default)'}",
        "Use load_skill(name) to load a skill by name.",
        ctx.skill_descriptions,
    ]
    return "\n".join(lines)


# Ordered registry of all built-in sections.
BUILTIN_SECTIONS: Dict[str, SectionRenderer] = OrderedDict([
    ("identity", section_identity),
    ("style", section_style),
    ("workflow", section_workflow),
    ("security", section_security),
    ("tool_guide", section_tool_guide),
    ("paths", section_paths),
    ("todo_discipline", section_todo_discipline),
    ("skills", section_skills),
])


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class SystemPromptBuilder:
    """Composable system prompt builder.

    Examples::

        # Use all built-in sections with runtime context
        ctx = SystemPromptContext(model="gpt-4o", workspace_dir="/app")
        prompt = SystemPromptBuilder().build(ctx)

        # Cherry-pick sections
        prompt = (
            SystemPromptBuilder()
            .use("identity", "tool_guide", "paths")
            .build(ctx)
        )

        # Add custom section
        prompt = (
            SystemPromptBuilder()
            .add("my_rules", lambda ctx: "Always respond in Chinese.")
            .build(ctx)
        )

        # Fully custom — ignore built-ins
        prompt = (
            SystemPromptBuilder(use_defaults=False)
            .add("custom", lambda ctx: f"You work in {ctx.workspace_dir}")
            .build(ctx)
        )
    """

    def __init__(self, *, use_defaults: bool = True) -> None:
        self._sections: OrderedDict[str, SectionRenderer] = OrderedDict()
        if use_defaults:
            self._sections.update(BUILTIN_SECTIONS)

    def use(self, *names: str) -> "SystemPromptBuilder":
        """Keep only the named built-in sections (in given order)."""
        selected = OrderedDict()
        for name in names:
            renderer = BUILTIN_SECTIONS.get(name) or self._sections.get(name)
            if renderer is None:
                raise ValueError(
                    f"Unknown section '{name}'. "
                    f"Built-in: {', '.join(BUILTIN_SECTIONS)}"
                )
            selected[name] = renderer
        self._sections = selected
        return self

    def add(
        self,
        name: str,
        renderer: SectionRenderer | str,
        *,
        before: str | None = None,
        after: str | None = None,
    ) -> "SystemPromptBuilder":
        """Add or replace a section.

        *renderer* can be a callable ``(ctx) -> str`` or a plain string.
        Use *before*/*after* to control position relative to an existing section.
        """
        fn: SectionRenderer = (lambda _c, _s=renderer: _s) if isinstance(renderer, str) else renderer  # type: ignore[assignment]

        if before or after:
            anchor = before or after
            new = OrderedDict()
            inserted = False
            for k, v in self._sections.items():
                if k == anchor and before:
                    new[name] = fn
                    inserted = True
                new[k] = v
                if k == anchor and after:
                    new[name] = fn
                    inserted = True
            if not inserted:
                new[name] = fn
            self._sections = new
        else:
            self._sections[name] = fn
        return self

    def remove(self, *names: str) -> "SystemPromptBuilder":
        """Remove sections by name."""
        for name in names:
            self._sections.pop(name, None)
        return self

    def build(self, ctx: SystemPromptContext) -> str:
        """Render all sections into a single prompt string."""
        parts: list[str] = []
        for _name, renderer in self._sections.items():
            result = renderer(ctx)
            if result and result.strip():
                parts.append(result.strip())
        return "\n\n".join(parts) + "\n"

    def section_names(self) -> list[str]:
        """Return current section names in order."""
        return list(self._sections.keys())


# ---------------------------------------------------------------------------
# Convenience defaults (backward-compatible)
# ---------------------------------------------------------------------------

DEFAULT_AGENT_SYSTEM_PROMPT = SystemPromptBuilder().build(SystemPromptContext())

DEFAULT_SUBAGENT_SYSTEM_PROMPT = (
    SystemPromptBuilder(use_defaults=False)
    .add("identity", lambda _: (
        "You are a coding subagent.\n\n"
        "Execution mode:\n"
        "- Complete the assigned task autonomously.\n"
        "- Keep outputs concise and evidence-based.\n"
        "- Return key findings with exact file paths and line anchors when possible."
    ))
    .add("paths", section_paths)
).build(SystemPromptContext())

DEFAULT_EXPLORE_SUBAGENT_SYSTEM_PROMPT = (
    SystemPromptBuilder(use_defaults=False)
    .add("identity", lambda _: (
        "You are a read-only exploration subagent.\n\n"
        "Execution mode:\n"
        "- Search and read files only; do not modify code.\n"
        "- Summarize findings with concrete references.\n"
        "- Flag uncertainties explicitly."
    ))
    .add("paths", section_paths)
).build(SystemPromptContext())


def build_agent_system_prompt(
    ctx: SystemPromptContext,
    extra: str | None = None,
    *,
    builder: SystemPromptBuilder | None = None,
) -> str:
    """Build a full agent system prompt with runtime context.

    If *builder* is ``None`` a default builder with all sections is used.
    *extra* is appended at the end.
    """
    b = builder if builder is not None else SystemPromptBuilder()
    if extra and extra.strip():
        b = SystemPromptBuilder(use_defaults=False)
        # Copy sections from source builder
        for name in (builder or SystemPromptBuilder()).section_names():
            section = BUILTIN_SECTIONS.get(name)
            if section:
                b.add(name, section)
        b.add("extra", extra.strip())
    return b.build(ctx)


def build_subagent_system_prompt(
    ctx: SystemPromptContext,
    extra: str | None = None,
) -> str:
    b = (
        SystemPromptBuilder(use_defaults=False)
        .add("identity", lambda _: (
            "You are a coding subagent.\n\n"
            "- Complete the assigned task autonomously.\n"
            "- Keep outputs concise and evidence-based.\n"
            "- Return key findings with exact file paths and line anchors."
        ))
        .add("paths", section_paths)
    )
    if extra and extra.strip():
        b.add("extra", extra.strip())
    return b.build(ctx)


def build_explore_subagent_system_prompt(
    ctx: SystemPromptContext,
    extra: str | None = None,
) -> str:
    b = (
        SystemPromptBuilder(use_defaults=False)
        .add("identity", lambda _: (
            "You are a read-only exploration subagent.\n\n"
            "- Search and read files only; do not modify code.\n"
            "- Summarize findings with concrete references.\n"
            "- Flag uncertainties explicitly."
        ))
        .add("tool_guide", section_tool_guide)
        .add("paths", section_paths)
    )
    if extra and extra.strip():
        b.add("extra", extra.strip())
    return b.build(ctx)
