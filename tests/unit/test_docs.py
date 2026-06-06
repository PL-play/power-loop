from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
MERMAID_BLOCK_RE = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL)

def _published_markdown_files() -> list[Path]:
    return [
        path
        for path in sorted(ROOT.rglob("*.md"))
        if ".skills" not in path.parts
    ]


def test_published_markdown_local_links_exist() -> None:
    missing: list[str] = []
    for path in _published_markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(text):
            raw_target = match.group(1).strip()
            target_path = raw_target.split("#", 1)[0].strip()
            if (
                not target_path
                or target_path.startswith("#")
                or re.match(r"^(https?|mailto):", target_path)
            ):
                continue
            if target_path.startswith("<") and target_path.endswith(">"):
                target_path = target_path[1:-1]

            resolved = (path.parent / target_path).resolve()
            if not resolved.exists():
                line = text.count("\n", 0, match.start()) + 1
                rel_path = path.relative_to(ROOT)
                missing.append(f"{rel_path}:{line} -> {raw_target}")

    assert missing == []


def test_mermaid_blocks_avoid_fragile_inline_html() -> None:
    failures: list[str] = []
    for path in _published_markdown_files():
        text = path.read_text(encoding="utf-8")
        for index, block in enumerate(MERMAID_BLOCK_RE.findall(text), start=1):
            if "<br" in block or "<aborted" in block:
                rel_path = path.relative_to(ROOT)
                failures.append(f"{rel_path} mermaid block {index}")

    assert failures == []
