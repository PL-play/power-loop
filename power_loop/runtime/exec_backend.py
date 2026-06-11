"""Pluggable launcher for the persistent shell used by default tools.

By default the shell runs in-process (`/bin/bash` in the workspace dir),
inheriting the host environment. A host that runs *model-authored* shell
commands (e.g. an agent runtime over untrusted chat input) can swap in a
backend that launches the shell inside an isolated sandbox instead — without
changing the tool implementations.

The contract is intentionally tiny: a backend only decides *how the shell
process is launched*. The pty / sentinel / drain machinery in
``BashSession`` is unchanged, so an in-process bash and a
``docker exec -i <sandbox> bash`` look identical to the rest of the runtime.

Security note: with the default ``LocalShellBackend`` the shell sees the full
host environment (including any secrets). Hosts that accept untrusted input
MUST install a sandboxing backend; see DeepTalk
``platform/docs/design/12-agent-execution-sandboxing.md``.
"""

from __future__ import annotations

import os
from collections.abc import Hashable
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ShellBackend(Protocol):
    """Decides how the persistent shell for a workspace is launched."""

    def launch_argv(self, workspace_dir: Path) -> list[str]:
        """argv used to start the persistent shell process."""
        ...

    def launch_cwd(self, workspace_dir: Path) -> str | None:
        """Working directory for the launched process, or None to leave default."""
        ...

    def launch_env(self, workspace_dir: Path) -> dict[str, str]:
        """Environment for the launched process.

        For a sandbox backend this is the env of the *client* (e.g. the
        ``docker`` CLI), not the sandbox itself — the sandbox's own env is set
        when the sandbox is created, so secrets never have to be passed here.
        """
        ...

    def session_key(self, workspace_dir: Path) -> Hashable:
        """Identity of *where* this shell runs, used to cache the persistent
        ``BashSession``. Two backends that target the same shell (e.g. the same
        sandbox container) should return equal keys so the session is reused;
        different targets must return different keys.
        """
        ...


class LocalShellBackend:
    """Run the shell in-process, in the workspace dir (original behavior)."""

    def launch_argv(self, workspace_dir: Path) -> list[str]:
        return ["/bin/bash", "--norc", "--noprofile"]

    def launch_cwd(self, workspace_dir: Path) -> str | None:
        return str(workspace_dir)

    def launch_env(self, workspace_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["TERM"] = "dumb"
        env["NO_COLOR"] = "1"
        return env

    def session_key(self, workspace_dir: Path) -> Hashable:
        return ("local", str(workspace_dir))


DEFAULT_SHELL_BACKEND: ShellBackend = LocalShellBackend()


__all__ = ["ShellBackend", "LocalShellBackend", "DEFAULT_SHELL_BACKEND"]
