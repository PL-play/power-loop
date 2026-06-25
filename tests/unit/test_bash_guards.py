"""H2.4 / C14: tests for the bash safety guards (previously untested).

`_dangerous_command_reason` blocks privileged/device-level/root-deletion commands;
`_validate_bash_command_scope` blocks reads/writes under POWER_LOOP_HOME outside the
allowlist. Both gate `run_bash`/`run_background`, so a regression here is a security
regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools.default_tools import (
    _dangerous_command_reason,
    _validate_bash_command_scope,
)

pytestmark = pytest.mark.unit


# ── _dangerous_command_reason ──────────────────────────────────────────────

@pytest.mark.parametrize("command", [
    # recursive deletion of a root/home/system path
    "rm -rf /",
    "rm -fr ~",
    "rm -rf ~/Documents",
    "rm -rf $HOME",
    "rm -rf $HOME/.ssh",
    "rm -RF /etc",
    "rm -rf /usr/local",       # system-dir SUBPATH — was a false negative before
    "rm -rf /var/lib/data",
    "rm -r -f /boot",          # split flags
    "rm -rf /root",
    # redirection to a raw device
    "cat foo > /dev/sda",
    "echo x > /dev/nvme0n1",
    "dd if=/dev/zero of=/dev/sdb",   # also caught as the 'dd' binary
    # privileged / device-level binaries (basename-matched, even with a full path)
    "sudo rm file",
    "/usr/bin/sudo reboot",
    "su - root",
    "shutdown now",
    "reboot",
    "mkfs /dev/sdb",
    "diskutil eraseDisk x",
    # conservative over-match: 'dd' as ANY bareword is blocked. This is intentional
    # — it also catches 'x && dd ...' and '$(dd ...)'; a false positive on a harmless
    # 'echo dd' is the accepted price of never letting a real 'dd' slip through.
    "echo dd",
])
def test_dangerous_commands_are_blocked(command: str) -> None:
    assert _dangerous_command_reason(command) is not None, f"should block: {command!r}"


@pytest.mark.parametrize("command", [
    "rm -rf ./build",       # relative path → not a root/home/system deletion
    "rm -rf build/cache",
    "rm -rf /tmp/scratch",  # /tmp is scratch, not a system dir → allowed
    "ls -la /tmp",
    "git status",
    "echo hello world",
    "cat README.md",
    "python script.py --flag",
    "grep -rn pattern src/",
    "cat /dev/null",        # not a '> /dev/...' redirect
    "make test",
])
def test_safe_commands_are_allowed(command: str) -> None:
    assert _dangerous_command_reason(command) is None, f"should allow: {command!r}"


# ── _validate_bash_command_scope ───────────────────────────────────────────

_HOME = Path("/tmp/pl_agent_home_test")


def _home_env() -> RuntimeEnv:
    return RuntimeEnv(workspace_dir=Path("/tmp/pl_ws_test"), home_dir=_HOME)


def test_scope_no_home_configured_allows_everything() -> None:
    # default env has no home_dir → the scope guard is inert
    with runtime_env_context(RuntimeEnv(workspace_dir=Path("/tmp/pl_ws_test"))):
        assert _validate_bash_command_scope("cat /tmp/pl_agent_home_test/secret") is None


def test_scope_blocks_read_of_home_internals() -> None:
    with runtime_env_context(_home_env()):
        err = _validate_bash_command_scope(f"cat {_HOME}/secret.txt")
        assert err is not None and "Reading agent-home internals is blocked" in err


def test_scope_does_not_false_positive_on_superstring_path() -> None:
    # default-tools-3: a sibling dir whose path is a SUPERSTRING of POWER_LOOP_HOME (…_test ⊂
    # …_test_extra) must NOT trip the home-scope guard — match on path boundaries, not substrings.
    with runtime_env_context(_home_env()):
        assert _validate_bash_command_scope(f"cat {_HOME}_extra/file.txt") is None
        # the real home (boundary = '/') is still guarded
        assert _validate_bash_command_scope(f"cat {_HOME}/secret.txt") is not None


def test_scope_blocks_write_under_home() -> None:
    with runtime_env_context(_home_env()):
        err = _validate_bash_command_scope(f"echo x > {_HOME}/notes.txt")
        assert err is not None and "Writing under POWER_LOOP_HOME is blocked" in err


def test_scope_allows_allowlisted_home_paths() -> None:
    # .cache and logs under home are allowlisted (read/write OK)
    with runtime_env_context(_home_env()):
        assert _validate_bash_command_scope(f"cat {_HOME}/.cache/model.json") is None
        assert _validate_bash_command_scope(f"echo x > {_HOME}/logs/run.log") is None


def test_scope_allows_commands_not_touching_home() -> None:
    with runtime_env_context(_home_env()):
        assert _validate_bash_command_scope("cat /tmp/pl_ws_test/file.txt") is None
        assert _validate_bash_command_scope("ls -la") is None


@pytest.mark.parametrize(
    "command",
    [
        f"awk '{{print}}' {_HOME}/secret.txt",
        f"base64 {_HOME}/secret.txt",
        f"od -c {_HOME}/secret.txt",
        f"xxd {_HOME}/secret.txt",
        f"python -c \"print(open('{_HOME}/secret.txt').read())\"",
        f"dd if=/dev/zero of={_HOME}/payload bs=1 count=1",
        f"truncate -s 0 {_HOME}/notes.txt",
        f"ln -s {_HOME}/secret.txt /tmp/leak",
    ],
)
def test_scope_default_deny_blocks_unlisted_verbs_touching_home(command: str) -> None:
    # B6 (deep-review): the read/write verb hint lists are NOT exhaustive — these commands all reach
    # un-allowlisted agent-home but matched no hint, so the pre-fix guard fell through to "allow".
    # They must now be denied by default (the home path is provably un-allowlisted).
    with runtime_env_context(_home_env()):
        err = _validate_bash_command_scope(command)
        assert err is not None and "POWER_LOOP_HOME" in err


def test_background_manager_prunes_terminal_tasks() -> None:
    # default-tools-2: the in-memory task cache evicts oldest TERMINAL records beyond the cap so it
    # (and the store/event_loop refs each pins) can't grow without bound; running tasks are kept.
    from power_loop.tools.default_tools import _MAX_BG_TASKS, BackgroundManager

    mgr = BackgroundManager()
    with mgr._lock:
        mgr.tasks["run-1"] = {"status": "running"}  # oldest, but running → never evicted
        for i in range(_MAX_BG_TASKS + 50):
            mgr.tasks[f"done-{i}"] = {"status": "completed"}
        mgr._prune_locked()
    assert len(mgr.tasks) <= _MAX_BG_TASKS
    assert "run-1" in mgr.tasks                       # running task preserved
    assert f"done-{_MAX_BG_TASKS + 49}" in mgr.tasks  # newest terminal preserved
    assert "done-0" not in mgr.tasks                  # oldest terminal evicted
