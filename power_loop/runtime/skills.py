"""Skill loading from ``SKILL.md`` files.

Skills are Markdown files with YAML frontmatter. Each skill lives in its own
directory under ``skills_dir``::

    skills_dir/
    ├── python-expert/
    │   └── SKILL.md      # name: python-expert, description: ...
    └── security-reviewer/
        └── SKILL.md

The library ships with a :class:`SkillLoader` that parses these files and
provides ``get_descriptions()`` (for system prompts) and ``get_content()``
(for detailed instructions). A ``load_skill`` tool can be registered in
:class:`ToolRegistry` so the LLM can dynamically fetch skill content.
"""

from __future__ import annotations

import re
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.env import get_runtime_env

_yaml = import_module("yaml") if find_spec("yaml") else None

LOAD_SKILL_DEFINITION = ToolDefinition(
    name="load_skill",
    description=(
        "Load the full content of a skill by name. "
        "Use this when you need detailed instructions for a specific domain. "
        "Param: name (string) — the skill name (e.g. 'python-expert')."
    ),
    input_schema={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Skill name to load"}},
        "required": ["name"],
    },
    required_params=("name",),
)


def _resolve_skills_dir(explicit: str | Path | None) -> Path:
    """Resolve the skills directory from config or POWER_LOOP_SKILLS_DIR."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        path = path.resolve()
        if path.exists() and path.is_dir():
            return path
        raise FileNotFoundError(f"Skills directory does not exist: {path}")
    return get_runtime_env().require_skills_dir()


class SkillLoader:
    """Loads and caches ``SKILL.md`` files from a directory tree.

    Args:
        skills_dir: Path to the directory containing skill subdirectories.
            If ``None``, uses the env-var / default resolution chain.

    The loader scans for ``*/SKILL.md`` files and parses their YAML
    frontmatter. Each skill's ``name`` comes from its parent directory name.
    """

    def __init__(self, skills_dir: str | Path | None = None):
        self.skills_dir: Path = _resolve_skills_dir(skills_dir)
        self.skills: dict[str, dict[str, Any]] = {}
        self._load_all()

    def reload(self) -> None:
        """Rescan the skills directory. Call after adding/removing skills."""
        self.skills.clear()
        self._load_all()

    # ── Internal ────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.glob("*/SKILL.md")):
            name = f.parent.name
            text = f.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f.resolve())}

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text.strip()
        body = match.group(2).strip()
        if _yaml is None:
            return {}, body
        try:
            meta = _yaml.safe_load(match.group(1))
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}
        return meta, body

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def names(self) -> list[str]:
        """Sorted list of available skill names."""
        return sorted(self.skills.keys())

    def get_descriptions(self) -> str:
        """One-line-per-skill summary suitable for system prompts."""
        if not self.skills:
            return "(no skills available)"
        lines = ["Available skills:"]
        for name in sorted(self.skills):
            s = self.skills[name]
            desc = s["meta"].get("description", "No description")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """Full instructions for a skill, including execution context."""
        skill = self.skills.get(name)
        if not skill:
            available = ", ".join(sorted(self.skills))
            return f"Error: Unknown skill '{name}'. Available: {available}"
        skill_path = Path(skill["path"]).resolve()
        skill_root = skill_path.parent
        runtime_env = get_runtime_env()
        workspace = runtime_env.workspace_dir or "(not configured)"
        home = runtime_env.home_dir or "(not configured)"
        return (
            f'<skill name="{name}" path="{skill["path"]}">\n'
            f"Source: {skill['path']}\n\n"
            "[Execution Context]\n"
            f"- Workspace (user project): {workspace}\n"
            f"- Runtime home: {home}\n"
            f"- Skill root: {skill_root}\n"
            "- Rules:\n"
            f"  1) Run skill-relative commands from skill root: {skill_root}\n"
            f"  2) Prefer absolute paths when applicable.\n"
            "  3) Output files go to workspace.\n"
            "  4) Do not search outside workspace/skill root unless requested.\n\n"
            f"{skill['body']}\n"
            "</skill>"
        )

    def build_system_prompt_section(self) -> str:
        """A section to append to ``AgentLoopConfig.system_prompt``.

        Includes skill descriptions in a compact format the LLM can parse.
        """
        if not self.skills:
            return ""
        parts = ["## Available Skills"]
        parts.append(
            "You have a `load_skill` tool. Call `load_skill(name=\"...\")` "
            "to get detailed instructions for any skill below.\n"
        )
        parts.append(self.get_descriptions())
        return "\n".join(parts)

    # ── Tool handler ────────────────────────────────────────────────────

    def handle_load_skill(self, name: str) -> str:
        """Handler for the ``load_skill`` tool. Returns the skill's full content."""
        return self.get_content(name)


# Module-level cache for backward compatibility.
_default_loaders: dict[Path, SkillLoader] = {}


def get_default_loader() -> SkillLoader:
    """Return a cached :class:`SkillLoader` for the current runtime env."""
    skills_dir = _resolve_skills_dir(None)
    loader = _default_loaders.get(skills_dir)
    if loader is None:
        loader = SkillLoader(skills_dir)
        _default_loaders[skills_dir] = loader
    return loader


def register_skill_tools(registry, *, skills_dir: str | Path | None = None) -> SkillLoader:
    """Register the ``load_skill`` tool in the given registry.

    Returns the :class:`SkillLoader` instance so callers can inspect or
    inject the skill descriptions into the system prompt.
    """
    loader = SkillLoader(skills_dir)
    registry.register(LOAD_SKILL_DEFINITION, loader.handle_load_skill)
    return loader


__all__ = [
    "LOAD_SKILL_DEFINITION",
    "SkillLoader",
    "get_default_loader",
    "register_skill_tools",
]
