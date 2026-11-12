"""Find every test surface that still references old execution identities."""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent
TOKENS = ("JobRef", "JobManager", "TaskRuntime")
EXPLICIT = (
    "tests/unit/test_task_message_identity_192.py",
    "tests/unit/test_request_shutdown_handoff_192.py",
    "tests/real/test_real_workflow_task_runtime_192.py",
    "tests/real/test_real_shell_control_192.py",
    "tests/real/test_real_workflow_control_192.py",
    "tests/real/test_real_workflow_control.py",
    "tests/real/test_real_pending_resume.py",
    "tests/unit/test_node_control_192.py",
    "tests/unit/test_node_nudge_runtime_192.py",
    "tests/unit/test_node_resume_192.py",
)


def has_test(path):
    tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
               for node in ast.walk(tree))


def entries():
    result = []
    for path in sorted(TESTS.rglob("*.py")):
        text = path.read_text(errors="ignore")
        tokens = [token for token in TOKENS if re.search(rf"\b{token}\b", text)]
        rel = path.relative_to(WS / "power-loop").as_posix()
        if tokens or rel in EXPLICIT:
            result.append({
                "path": rel,
                "tokens": tokens,
                "has_test": has_test(path),
                "kind": "test" if has_test(path) else "helper",
            })
    return result


def manifest(mapping_path):
    rows = entries()
    mapping = mapping_path.read_text()
    for row in rows:
        row["mapping"] = Path(row["path"]).name in mapping
    return {
        "source_heads": {
            "power-loop": subprocess.check_output(
                ["git", "-C", str(WS / "power-loop"), "rev-parse", "HEAD"], text=True).strip(),
        },
        "tokens": TOKENS,
        "explicit_without_token": list(EXPLICIT),
        "count": len(rows),
        "unmapped": [row["path"] for row in rows if not row["mapping"]],
        "entries": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()
    current = manifest(args.mapping)
    assert not current["unmapped"], current["unmapped"]
    if args.output:
        args.output.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n")
    if args.check:
        saved = json.loads(args.check.read_text())
        assert saved["entries"] == current["entries"], "legacy test surface differs"
        assert saved["source_heads"] == current["source_heads"], "legacy test surface source head differs"
        assert saved["count"] == current["count"]
    print(f"legacy test surface: {current['count']} files, all mapped; "
          f"tokens={len([r for r in current['entries'] if r['tokens']])}, "
          f"explicit={len(EXPLICIT)}")


if __name__ == "__main__":
    main()
