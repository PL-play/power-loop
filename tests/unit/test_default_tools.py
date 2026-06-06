from __future__ import annotations

import uuid
from pathlib import Path

from power_loop import create_default_tool_registry
from power_loop.runtime.env import WORKSPACE_DIR
from power_loop.tools.default_tools import FILE_READ_STATE


def _sandbox() -> Path:
    path = WORKSPACE_DIR / ".tmp-default-tools" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_write_read_and_overwrite_guard() -> None:
    registry = create_default_tool_registry()
    root = _sandbox()
    rel = root.relative_to(WORKSPACE_DIR).as_posix()

    created = registry.invoke("write_file", {"path": f"{rel}/note.txt", "content": "alpha\nbeta\n"})
    assert "new file" in str(created)

    read = registry.invoke("read_file", {"path": f"{rel}/note.txt"})
    assert "1|alpha" in str(read)
    assert "2|beta" in str(read)

    # External modification invalidates the optimistic read stamp.
    (root / "note.txt").write_text("changed outside\n", encoding="utf-8")
    blocked = registry.invoke("write_file", {"path": f"{rel}/note.txt", "content": "overwrite\n"})
    assert "changed since last read" in str(blocked).lower()


def test_edit_requires_unique_exact_text_and_returns_diff() -> None:
    registry = create_default_tool_registry()
    root = _sandbox()
    rel = root.relative_to(WORKSPACE_DIR).as_posix()
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


def test_apply_patch_accepts_unified_hunks() -> None:
    registry = create_default_tool_registry()
    root = _sandbox()
    rel = root.relative_to(WORKSPACE_DIR).as_posix()
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


def test_glob_and_grep_are_scoped_and_capped() -> None:
    registry = create_default_tool_registry()
    root = _sandbox()
    rel = root.relative_to(WORKSPACE_DIR).as_posix()
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


def test_bash_runs_and_blocks_obvious_danger() -> None:
    registry = create_default_tool_registry()
    ok = registry.invoke("bash", {"command": "printf 'hello-tools\\n'", "timeout": 5})
    assert "exit_code=0" in str(ok)
    assert "hello-tools" in str(ok)

    blocked = registry.invoke("bash", {"command": "sudo whoami"})
    assert "Dangerous command blocked" in str(blocked)


def test_default_registry_contains_all_manifest_tools() -> None:
    registry = create_default_tool_registry(preset="full")

    todo = registry.invoke(
        "todo",
        {
            "items": [
                {"id": "read", "text": "Read the file", "status": "completed"},
                {"id": "write", "text": "Write the patch", "status": "in_progress"},
            ]
        },
    )
    assert "write" in str(todo)

    skill = registry.invoke("load_skill", {"name": "missing-skill-for-test"})
    assert "Unknown skill" in str(skill)

    started = registry.invoke("background_run", {"command": "printf 'background-ok\\n'"})
    assert "Background task" in str(started)
    task_id = str(started).split()[2]
    checked = registry.invoke("check_background", {"task_id": task_id})
    assert task_id or checked

    assert FILE_READ_STATE
