from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
import re
from pathlib import Path
from typing import Any

yaml = import_module("yaml") if find_spec("yaml") else None

from power_loop.runtime.env import AGENT_DIR, SKILLS_DIR, WORKSPACE_DIR


class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: dict[str, dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.glob("*/SKILL.md")):
            name = f.parent.name
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        match = re.match(r"^---\\n(.*?)\\n---\\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        if yaml is None:
            return {}, match.group(2).strip()
        try:
            meta = yaml.safe_load(match.group(1))
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc} (path: {skill['path']})"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        skill_path = Path(skill["path"]).resolve()
        skill_root = skill_path.parent
        return (
            f"<skill name=\"{name}\" path=\"{skill['path']}\">\\n"
            f"Source: {skill['path']}\\n\\n"
            "[Execution Context]\\n"
            f"- Workspace (user project): {WORKSPACE_DIR}\\n"
            f"- Agent home: {AGENT_DIR}\\n"
            f"- Skill root: {skill_root}\\n"
            "- Rules:\\n"
            "  1) For relative commands in this skill (for example `python scripts/cli.py ...`), run from Skill root.\\n"
            f"  2) Prefer absolute command form: `python {skill_root}/scripts/cli.py ...` when applicable.\\n"
            "  3) Output files from skill execution should go to WORKSPACE.\\n"
            "  4) Do NOT search outside workspace/skill root unless explicitly requested.\\n\\n"
            f"{skill['body']}\\n"
            "</skill>"
        )


SKILL_LOADER = SkillLoader(SKILLS_DIR)
