"""Enumerate every absolute or relative import of power_loop.runtime.jobs."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
REPOS = {"power-loop": WS / "power-loop", "agent": WS / "agent"}
REPLACEMENTS = {
    "JobManager": "RuntimeInputStore/RuntimeClient + SessionStore session_state/messages; no manager replacement",
    "JobRef": "TargetRef/InputRef/send_index",
    "JobPage": "SessionPage",
    "JobActor": "TrustedActor/session actor",
    "JobError": "RuntimeRequestError or host error; no alias",
    "JobLease": "SessionOwnership + owner_epoch/token + active_send_index",
    "JobRegistration": "TargetRef + InputRef + Receipt",
    "JobRecord": "sessions/session_state/follow_up_queue/messages row models",
    "JobOperation": "ordinary messages + InputRef + tool_call_id",
    "KINDS": "SessionKind/ResultKind enums; no job-kind alias",
    "_record": "neutral session/message row models; no JobRecord",
    "_operation": "ordinary messages + InputRef + tool_call_id",
    "_json": "power_loop.runtime.canonical.canonical_json",
    "_hash": "power_loop.runtime.canonical.sha256_canonical",
    "TRUST_OWN_WRITES": "power_loop.runtime.store.verification.TRUST_OWN_WRITES",
    "verify_stored": "power_loop.runtime.store.verification.verify_stored",
}


def phase(repo, rel):
    if repo == "agent":
        return "P6"
    if rel.startswith(("tests/", "docs/")):
        return "see test-mapping.md"
    if rel.startswith("power_loop/workflow/"):
        return "P4"
    if rel.startswith("power_loop/tools/"):
        return "P3/P7"
    if rel.startswith(("power_loop/agent/runtime_types", "power_loop/agent/runtime_store")):
        return "P1"
    if rel.startswith(("power_loop/agent/", "power_loop/core/", "power_loop/runtime/model_calls")):
        return "P2"
    return "P7"


def resolved_jobs_module(path, node):
    if node.level == 0:
        return node.module
    parts = list(path.parts)
    if node.level > 1:
        parts = parts[:-(node.level - 1)]
    return ".".join([*parts, *(node.module or "").split(".")])


def scan_entries():
    entries = []
    for repo, root in REPOS.items():
        for path in sorted(root.rglob("*.py")):
            if any(part in {".git", ".venv", "legacy", "node_modules"} for part in path.parts):
                continue
            rel = path.relative_to(root).as_posix()
            try:
                tree = ast.parse(path.read_text(), filename=rel)
            except (SyntaxError, UnicodeDecodeError):
                continue
            package = path.relative_to(root).parent
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = resolved_jobs_module(package, node)
                    if module != "power_loop.runtime.jobs":
                        continue
                    symbols = [alias.name for alias in node.names]
                    import_form = f"from {'.' * node.level}{node.module or ''} import ..."
                elif isinstance(node, ast.Import) and any(
                        alias.name == "power_loop.runtime.jobs" for alias in node.names):
                    symbols = ["power_loop.runtime.jobs"]
                    import_form = "import power_loop.runtime.jobs"
                else:
                    continue
                scope = "production" if rel.startswith(("power_loop/", "app/")) else "test"
                entries.append({
                    "repo": repo,
                    "path": rel,
                    "line": node.lineno,
                    "scope": scope,
                    "phase": phase(repo, rel),
                    "import": import_form,
                    "symbols": symbols,
                    "replacement": {
                        symbol: REPLACEMENTS.get(
                            symbol, "delete with the old mechanism; normal behavior follows test-mapping.md")
                        for symbol in symbols
                    },
                })
    return sorted(entries, key=lambda entry: (
        entry["repo"], entry["path"], entry["line"], entry["import"]))


def manifest():
    entries = scan_entries()
    return {
        "source_heads": {
            repo: subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            for repo, root in REPOS.items()
        },
        "module": "power_loop.runtime.jobs",
        "count": len(entries),
        "counts": {
            repo: {
                "production": sum(entry["repo"] == repo and entry["scope"] == "production"
                                  for entry in entries),
                "test": sum(entry["repo"] == repo and entry["scope"] == "test" for entry in entries),
                "production_files": len({entry["path"] for entry in entries
                                         if entry["repo"] == repo and entry["scope"] == "production"}),
                "test_files": len({entry["path"] for entry in entries
                                   if entry["repo"] == repo and entry["scope"] == "test"}),
            }
            for repo in REPOS
        },
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    current = manifest()
    if args.output:
        args.output.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n")
    if args.check:
        saved = json.loads(args.check.read_text())
        assert saved["entries"] == current["entries"], "jobs import manifest differs from AST scan"
        assert saved["count"] == current["count"] == len(current["entries"])
        assert saved["counts"] == current["counts"]
        print(f"jobs import scan matches: {current['count']} entries; {current['counts']}")
    else:
        print(json.dumps(current["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
