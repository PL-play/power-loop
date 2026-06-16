from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from power_loop import create_default_tool_registry
from power_loop.runtime.env import RuntimeEnvError
from power_loop.runtime.human_input import HumanInputRequired
from power_loop.tools.default_tools import FILE_READ_STATE


def _sandbox(workspace: Path) -> Path:
    path = workspace / ".tmp-default-tools" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_workspace_tools_require_explicit_workspace() -> None:
    with pytest.raises(RuntimeEnvError, match="POWER_LOOP_WORKSPACE"):
        create_default_tool_registry(include=["read_file"])


def test_request_user_input_does_not_require_workspace() -> None:
    registry = create_default_tool_registry(include=["request_user_input"])
    with pytest.raises(HumanInputRequired) as exc:
        registry.invoke("request_user_input", {"prompt": "Approve?", "kind": "confirm"})
    assert exc.value.prompt == "Approve?"
    assert exc.value.kind == "confirm"


def test_write_read_and_overwrite_guard(tmp_path: Path) -> None:
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    root = _sandbox(tmp_path)
    rel = root.relative_to(tmp_path).as_posix()

    created = registry.invoke("write_file", {"path": f"{rel}/note.txt", "content": "alpha\nbeta\n"})
    assert "new file" in str(created)

    read = registry.invoke("read_file", {"path": f"{rel}/note.txt"})
    assert "1|alpha" in str(read)
    assert "2|beta" in str(read)

    # External modification invalidates the optimistic read stamp.
    (root / "note.txt").write_text("changed outside\n", encoding="utf-8")
    blocked = registry.invoke("write_file", {"path": f"{rel}/note.txt", "content": "overwrite\n"})
    assert "changed since last read" in str(blocked).lower()


def test_edit_requires_unique_exact_text_and_returns_diff(tmp_path: Path) -> None:
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    root = _sandbox(tmp_path)
    rel = root.relative_to(tmp_path).as_posix()
    target = f"{rel}/edit.txt"
    registry.invoke("write_file", {"path": target, "content": "one\ntwo\nthree\ntwo\n"})
    registry.invoke("read_file", {"path": target})

    ambiguous = registry.invoke("edit_file", {"path": target, "old_text": "two", "new_text": "TWO"})
    assert "matches 2 locations" in str(ambiguous)

    edited = registry.invoke(
        "edit_file",
        {"path": target, "old_text": "one\ntwo\nthree", "new_text": "one\nTWO\nthree"},
    )
    assert "Edited" in str(edited)
    assert "-two" in str(edited)
    assert "+TWO" in str(edited)
    assert (root / "edit.txt").read_text(encoding="utf-8") == "one\nTWO\nthree\ntwo\n"


def test_apply_patch_accepts_unified_hunks(tmp_path: Path) -> None:
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    root = _sandbox(tmp_path)
    rel = root.relative_to(tmp_path).as_posix()
    target = f"{rel}/patch.txt"
    registry.invoke("write_file", {"path": target, "content": "red\ngreen\nblue\n"})
    registry.invoke("read_file", {"path": target})

    result = registry.invoke(
        "apply_patch",
        {
            "path": target,
            "patch": """@@ -1,3 +1,4 @@
 red
-green
+emerald
 blue
+violet""",
        },
    )
    assert "Patched" in str(result)
    assert (root / "patch.txt").read_text(encoding="utf-8") == "red\nemerald\nblue\nviolet\n"


def test_glob_and_grep_are_scoped_and_capped(tmp_path: Path) -> None:
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    root = _sandbox(tmp_path)
    rel = root.relative_to(tmp_path).as_posix()
    (root / "a.py").write_text("needle = 1\n", encoding="utf-8")
    (root / "b.txt").write_text("needle = 2\n", encoding="utf-8")
    (root / ".hidden.py").write_text("needle = 3\n", encoding="utf-8")

    globbed = registry.invoke("glob", {"path": rel, "pattern": "*.py"})
    assert "a.py" in str(globbed)
    assert ".hidden.py" not in str(globbed)

    hidden = registry.invoke("glob", {"path": rel, "pattern": "*.py", "include_hidden": True})
    assert ".hidden.py" in str(hidden)

    grep = registry.invoke("grep", {"path": rel, "pattern": "needle", "include": "*.py", "max_results": 1})
    assert "a.py:1:needle" in str(grep)
    assert "b.txt" not in str(grep)


def _grep_matched_files(output: str) -> set[str]:
    """Filenames from grep output lines (``path:line:text``), env-independent."""
    files: set[str] = set()
    for line in output.splitlines():
        if line.startswith("...") or ":" not in line:
            continue
        files.add(line.split(":", 1)[0].lstrip("./"))
    return files


def test_grep_rg_command_excludes_all_common_skip_dirs_after_include(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rg command must carry a ``!**/<dir>/**`` exclusion for EVERY
    _COMMON_SKIP_DIRS entry (matches at any depth) placed AFTER the include
    glob, so rg's last-glob-wins precedence lets the exclusion win — matching
    the Python fallback, which prunes those dirs unconditionally. Hermetic: we
    capture the argv instead of depending on a real rg binary (rg may be a
    shell alias, not on PATH for subprocess)."""
    import subprocess as _sp

    import power_loop.tools.default_tools as dt
    from power_loop.tools.default_tools import _COMMON_SKIP_DIRS

    captured: dict[str, list[str]] = {}

    def _spy(cmd, **kw):  # noqa: ANN001, ANN202
        captured["argv"] = list(cmd)
        return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(dt.subprocess, "run", _spy)
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    registry.invoke("grep", {"path": ".", "pattern": "NEEDLE", "include": "*.py"})

    argv = captured["argv"]
    for d in _COMMON_SKIP_DIRS:
        assert f"!**/{d}/**" in argv, f"missing exclusion for {d}: {argv}"
    # Exclusions must come AFTER the include glob (last-glob-wins → exclusion wins).
    inc_idx = argv.index("*.py")
    first_skip_idx = min(argv.index(f"!**/{d}/**") for d in _COMMON_SKIP_DIRS)
    assert inc_idx < first_skip_idx, f"include must precede skip-dir globs: {argv}"


def test_grep_python_fallback_prunes_common_skip_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Python fallback (rg unavailable) must prune _COMMON_SKIP_DIRS at any
    depth — the contract the rg globs above are kept in parity with."""
    import power_loop.tools.default_tools as dt

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "keep.py").write_text("NEEDLE = 1\n", encoding="utf-8")
    for d in ("__pycache__", "node_modules", "target", "vendor", ".venv"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "skip.py").write_text("NEEDLE = 2\n", encoding="utf-8")
    (tmp_path / "pkg" / "__pycache__").mkdir()  # nested skip dir
    (tmp_path / "pkg" / "__pycache__" / "skip.py").write_text("NEEDLE = 3\n", encoding="utf-8")

    def _no_rg(*a, **k):  # noqa: ANN002, ANN003, ANN202
        raise FileNotFoundError("rg not installed")

    monkeypatch.setattr(dt.subprocess, "run", _no_rg)
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    out = str(registry.invoke("grep", {"path": ".", "pattern": "NEEDLE", "max_results": 50}))

    assert _grep_matched_files(out) == {"pkg/keep.py"}, out
    for leaked in ("__pycache__", "node_modules", "target", "vendor", ".venv"):
        assert leaked not in out, f"fallback leaked {leaked}: {out}"


def test_bash_runs_and_blocks_obvious_danger(tmp_path: Path) -> None:
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    ok = registry.invoke("bash", {"command": "printf 'hello-tools\\n'", "timeout": 5})
    assert "exit_code=0" in str(ok)
    assert "hello-tools" in str(ok)

    blocked = registry.invoke("bash", {"command": "sudo whoami"})
    assert "Dangerous command blocked" in str(blocked)


async def test_default_registry_contains_all_manifest_tools(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    registry = create_default_tool_registry(preset="full", workspace_dir=tmp_path, skills_dir=skills_root)

    todo = await registry.invoke_async(
        "todo",
        {
            "items": [
                {"id": "read", "text": "Read the file", "status": "completed"},
                {"id": "write", "text": "Write the patch", "status": "in_progress"},
            ]
        },
    )
    assert "write" in str(todo)

    skill = await registry.invoke_async("load_skill", {"name": "missing-skill-for-test"})
    assert "Unknown skill" in str(skill)

    started = await registry.invoke_async("background_run", {"command": "printf 'background-ok\\n'"})
    assert "Background task" in str(started)
    task_id = str(started).split()[2]
    checked = await registry.invoke_async("check_background", {"task_id": task_id})
    assert task_id or checked

    assert FILE_READ_STATE
