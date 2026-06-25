"""Path-boundary enforcement for the default file tools (``safe_path``).

``safe_path`` is the gate that keeps file/glob/grep/write/edit tools inside the
workspace (and an allowlisted slice of home). These cover its branches — and lock a
regression where a non-canonical/symlinked ``workspace_dir`` rejected every legitimate
in-workspace path because the resolved candidate was compared to an unresolved root.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from power_loop.runtime.env import RuntimeEnv, RuntimeEnvError, safe_path

pytestmark = pytest.mark.unit


def _ws(tmp_path: Path) -> RuntimeEnv:
    return RuntimeEnv(workspace_dir=tmp_path)


def test_relative_path_resolves_into_workspace(tmp_path: Path) -> None:
    assert safe_path("a/b.txt", env=_ws(tmp_path)) == (tmp_path / "a/b.txt").resolve()


def test_workspace_prefix(tmp_path: Path) -> None:
    assert safe_path("@workspace/x.txt", env=_ws(tmp_path)) == (tmp_path / "x.txt").resolve()


def test_absolute_inside_workspace_allowed(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "f.txt"
    assert safe_path(str(target), env=_ws(tmp_path)) == target.resolve()


@pytest.mark.parametrize("bad", ["../../etc/passwd", "/etc/passwd", "a/../../b"])
def test_escape_attempts_blocked(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="escapes allowed directories"):
        safe_path(bad, env=_ws(tmp_path))


def test_sibling_prefix_not_confused(tmp_path: Path) -> None:
    # /ws-evil must NOT be considered inside /ws (component-wise, not string-prefix).
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError, match="escapes allowed directories"):
        safe_path(str(tmp_path / "ws-evil" / "f"), env=RuntimeEnv(workspace_dir=ws))


def test_empty_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Path is required"):
        safe_path("   ", env=_ws(tmp_path))


def test_missing_workspace_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeEnvError, match="POWER_LOOP_WORKSPACE is required"):
        safe_path("f.txt", env=RuntimeEnv())


def test_symlinked_workspace_accepts_inside_and_blocks_out(tmp_path: Path) -> None:
    """Regression: a non-canonical (symlinked) workspace_dir must still accept files
    inside it (the resolved candidate vs unresolved-root mismatch bug) AND still
    block a symlink that points OUT of the workspace."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    (real / "f.txt").write_text("hi")
    env = RuntimeEnv(workspace_dir=link)  # unresolved symlink, as a host may pass

    # in-workspace file is accepted (was wrongly rejected before the fix)
    assert safe_path("f.txt", env=env) == (real / "f.txt").resolve()

    # a symlink inside the workspace pointing OUT is still blocked (resolve defeats it)
    secret = tmp_path / "secret.txt"
    secret.write_text("s")
    os.symlink(secret, real / "escape")
    with pytest.raises(ValueError, match="escapes allowed directories"):
        safe_path("escape", env=env)


def test_home_allowlist_and_restriction(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    home = tmp_path / "home"
    for d in (ws, home / ".cache", home / "logs", home / "secret"):
        d.mkdir(parents=True)
    env = RuntimeEnv(workspace_dir=ws, home_dir=home)

    # allowlisted home subdirs are reachable via @home/
    assert safe_path("@home/.cache/x", env=env) == (home / ".cache/x").resolve()
    assert safe_path("@home/logs/y", env=env) == (home / "logs/y").resolve()

    # other home paths are restricted
    with pytest.raises(ValueError, match="POWER_LOOP_HOME is restricted"):
        safe_path("@home/secret/z", env=env)


def test_read_file_with_limit_still_enforces_size_cap(tmp_path: Path) -> None:
    # M-default-tools-1: passing a `limit` no longer disables the size cap (which used to load the
    # WHOLE multi-GB file into memory). offset/limit page OUTPUT within a readable (<=cap) file.
    from power_loop.runtime.env import runtime_env_context
    from power_loop.tools.default_tools import TEXT_FILE_MAX_BYTES, run_read

    big = tmp_path / "big.txt"
    big.write_bytes(b"x\n" * (TEXT_FILE_MAX_BYTES // 2 + 16))  # > cap
    small = tmp_path / "small.txt"
    small.write_text("\n".join(f"line{i}" for i in range(20)), encoding="utf-8")
    with runtime_env_context(_ws(tmp_path)):
        out = run_read("big.txt", offset=1, limit=10)
        assert "too large" in out.lower()             # capped even with a limit set
        ok = run_read("small.txt", offset=1, limit=5)  # paging a <=cap file still works
        assert "line0" in ok and "line4" in ok and "line5" not in ok
