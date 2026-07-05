"""apply_patch / edit_file emit a loud warning when a .json file ends up invalid."""

from __future__ import annotations

import pytest

from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools.default_tools import run_apply_patch, run_edit, run_read, run_write

pytestmark = pytest.mark.unit

GOOD = '{\n  "a": 1,\n  "b": 2\n}\n'


def _env(tmp_path):
    return runtime_env_context(RuntimeEnv(workspace_dir=tmp_path))


def test_edit_file_warns_on_broken_json(tmp_path) -> None:
    with _env(tmp_path):
        run_write("app.json", GOOD)
        run_read("app.json")
        out = run_edit("app.json", '"b": 2', '"b": 2,')  # trailing comma → invalid
        assert "Edited" in out
        assert "NOT valid JSON" in out


def test_edit_file_no_warning_when_valid(tmp_path) -> None:
    with _env(tmp_path):
        run_write("app.json", GOOD)
        run_read("app.json")
        out = run_edit("app.json", '"b": 2', '"b": 3')
        assert "Edited" in out
        assert "NOT valid JSON" not in out


def test_apply_patch_warns_on_broken_json(tmp_path) -> None:
    with _env(tmp_path):
        run_write("app.json", GOOD)
        run_read("app.json")
        patch = "@@ -3,1 +3,0 @@\n-  \"b\": 2\n"  # leaves '"a": 1,' dangling
        out = run_apply_patch("app.json", patch)
        assert "Patched" in out
        assert "NOT valid JSON" in out


def test_non_json_files_never_warn(tmp_path) -> None:
    with _env(tmp_path):
        run_write("notes.txt", "a: 1,\n")
        run_read("notes.txt")
        out = run_edit("notes.txt", "a: 1,", "a: {broken")
        assert "Edited" in out
        assert "NOT valid JSON" not in out
