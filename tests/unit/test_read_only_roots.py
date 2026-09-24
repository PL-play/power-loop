"""``RuntimeEnv.read_only_roots``: a shared knowledge base every agent may read but none may write.

Read tools (read_file / glob / grep) pass purpose "r" and get in; write tools (write_file /
edit_file / apply_patch) keep the default write purpose and are refused — even when the same
path would also sit inside the workspace or the home allowlist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from power_loop.runtime.env import RuntimeEnv, runtime_env_context, safe_path
from power_loop.tools import default_tools
from power_loop.tools.default_tools import run_edit, run_glob, run_grep, run_read, run_write

pytestmark = pytest.mark.unit


def _env(tmp_path: Path, *, skills_dir: Path | None = None) -> tuple[RuntimeEnv, Path]:
    home = tmp_path / "home"
    shared = home / "skills" / "_shared"
    (shared / "contract").mkdir(parents=True)
    (shared / "contract" / "SKILL.md").write_text("line one\nNEEDLE here\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    env = RuntimeEnv(workspace_dir=ws, home_dir=home, skills_dir=skills_dir, read_only_roots=(shared,))
    return env, shared


def test_read_purpose_resolves_under_read_only_root(tmp_path: Path) -> None:
    env, shared = _env(tmp_path)
    target = shared / "contract" / "SKILL.md"
    assert safe_path(str(target), "r", env=env) == target.resolve()
    assert safe_path("@home/skills/_shared/contract/SKILL.md", "r", env=env) == target.resolve()


@pytest.mark.parametrize("purpose", ["rw", "w"])
def test_write_purposes_are_refused(tmp_path: Path, purpose: str) -> None:
    env, shared = _env(tmp_path)
    with pytest.raises(ValueError, match="read-only"):
        safe_path(str(shared / "contract" / "SKILL.md"), purpose, env=env)


def test_read_only_wins_over_home_allowlist(tmp_path: Path) -> None:
    """Even if the shared dir is the agent's own skills_dir (rw allowlisted), it stays read-only."""
    home = tmp_path / "home"
    shared = home / "skills" / "_shared"
    shared.mkdir(parents=True)
    env = RuntimeEnv(workspace_dir=tmp_path, home_dir=home, skills_dir=shared, read_only_roots=(shared,))
    with pytest.raises(ValueError, match="read-only"):
        safe_path(str(shared / "x.md"), "rw", env=env)


def test_read_only_wins_over_workspace(tmp_path: Path) -> None:
    ref = tmp_path / "refs"
    ref.mkdir()
    env = RuntimeEnv(workspace_dir=tmp_path, read_only_roots=(ref,))
    assert safe_path("refs/a.md", "r", env=env) == (ref / "a.md").resolve()
    with pytest.raises(ValueError, match="read-only"):
        safe_path("refs/a.md", env=env)
    assert safe_path("other.md", env=env) == (tmp_path / "other.md").resolve()


def test_default_read_tools_reach_the_root_and_write_tools_do_not(tmp_path: Path) -> None:
    env, shared = _env(tmp_path)
    path = str(shared / "contract" / "SKILL.md")
    with runtime_env_context(env):
        assert "NEEDLE here" in run_read(path)
        assert "SKILL.md" in run_glob("*.md", str(shared))
        assert "NEEDLE" in run_grep("NEEDLE", str(shared))
        assert run_write(path, "overwritten").startswith("Error") and "read-only" in run_write(path, "x")
        assert run_edit(path, "line one", "changed").startswith("Error")
        assert default_tools.run_apply_patch(path, "@@ -1 +1 @@\n-line one\n+changed\n").startswith("Error")
    assert (shared / "contract" / "SKILL.md").read_text() == "line one\nNEEDLE here\n"


def test_no_read_only_roots_changes_nothing(tmp_path: Path) -> None:
    env = RuntimeEnv(workspace_dir=tmp_path)
    assert safe_path("a.txt", env=env) == (tmp_path / "a.txt").resolve()
