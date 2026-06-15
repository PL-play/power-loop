"""28 · Docker sandbox shell / Docker 沙箱 shell: model-authored bash inside a container

What you learn / 你将学到
--------------------------
- The ``ShellBackend`` seam: swap power-loop's in-process ``/bin/bash`` for
  ``docker exec`` **without touching any tool code**.
  / 通过 ``ShellBackend`` 缝隙把内置的 bash 换成 ``docker exec``，工具代码零改动。
- Model-authored shell then runs **inside an isolated container**, not on the host.
  / 模型写的 shell 命令在隔离容器里执行，而不是宿主机。
- Inject the backend per-run via ``runtime_env_context(RuntimeEnv(shell_backend=...))``.
  / 用 ``runtime_env_context`` 按次注入后端。

Why it matters / 为什么重要
---------------------------
The ``bash`` / ``background_run`` tools are host-facing by default: the
``LocalShellBackend`` runs ``/bin/bash`` with your full environment (including
any secrets). **Any host that executes untrusted, model-authored commands MUST
sandbox them.** The ``ShellBackend`` protocol is that seam — the pty / sentinel
/ drain machinery in ``BashSession`` is unchanged, so an in-process bash and a
``docker exec -i <container> bash`` look identical to the rest of the runtime.
This is exactly how DeepTalk isolates per-conversation agent workspaces.

This example proves isolation two ways: the shell reports the **container's**
OS (Debian, not your host) and reads a file from a **bind-mounted** workspace.

Requires a running Docker daemon and the ``python:3.12-slim`` image (pulled on
first run). If Docker is unavailable the example prints a notice and returns
``None`` so it stays runnable everywhere.

Run / 运行
----------
    python examples/28_docker_shell_backend.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Hashable
from pathlib import Path

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    RuntimeEnv,
    StatefulAgentLoop,
    create_default_tool_registry,
    runtime_env_context,
)

IMAGE = "python:3.12-slim"  # Debian-based, ships bash; ~179MB
CONTAINER_WORKDIR = "/workspace"


class DockerShellBackend:
    """A ``ShellBackend`` that launches the persistent shell *inside* a running
    container via ``docker exec``.

    Implements the four-method ``ShellBackend`` protocol. Note ``launch_env``
    returns the *docker client's* environment (so ``docker`` is found on PATH),
    NOT the sandbox's — the container's own environment was fixed at
    ``docker run`` time, so no host secret has to cross this boundary.
    """

    def __init__(self, container: str, *, workdir: str = CONTAINER_WORKDIR) -> None:
        self._container = container
        self._workdir = workdir

    def launch_argv(self, workspace_dir: Path) -> list[str]:
        # -i keeps stdin open for the persistent session; -w sets the cwd INSIDE
        # the container. The host workspace_dir is bind-mounted at self._workdir
        # when the container starts, so we don't need it in the argv here.
        return [
            "docker", "exec", "-i", "-w", self._workdir,
            self._container, "bash", "--norc", "--noprofile",
        ]

    def launch_cwd(self, workspace_dir: Path) -> str | None:
        return None  # host-side cwd of the `docker` CLI is irrelevant

    def launch_env(self, workspace_dir: Path) -> dict[str, str]:
        env = os.environ.copy()  # the docker CLIENT's env (to find `docker`)
        env["TERM"] = "dumb"
        env["NO_COLOR"] = "1"
        return env

    def session_key(self, workspace_dir: Path) -> Hashable:
        # Same container+workdir → reuse one persistent shell; different → new one.
        return ("docker", self._container, self._workdir)


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(
            ["docker", "version"], capture_output=True, timeout=15
        ).returncode == 0
    except Exception:
        return False


def _ensure_image() -> bool:
    """Make sure IMAGE is present locally (pull once if needed)."""
    have = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    if have.returncode == 0:
        return True
    print(f"[setup] pulling {IMAGE} (first run only)…")
    return subprocess.run(["docker", "pull", IMAGE], capture_output=True).returncode == 0


async def main() -> dict | None:
    if not _docker_available():
        print("[skip] Docker not available — this example needs a running Docker daemon.")
        return None
    if not _ensure_image():
        print(f"[skip] could not obtain image {IMAGE}.")
        return None

    host_ws = Path(tempfile.mkdtemp(prefix="pl-docker-ws-"))
    # A file written on the HOST — visible inside via the bind mount, proving the
    # workspace boundary is shared while execution is isolated.
    (host_ws / "host_note.txt").write_text("hello from the host machine\n", encoding="utf-8")
    container = f"power-loop-example-{uuid.uuid4().hex[:8]}"

    # A long-lived sandbox with the workspace bind-mounted. In production you'd
    # also add --network none, a non-root --user, --read-only, memory/pids caps,
    # etc.; kept minimal here so the one concept (the backend) stays in focus.
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", container,
         "-v", f"{host_ws}:{CONTAINER_WORKDIR}", "-w", CONTAINER_WORKDIR,
         IMAGE, "sleep", "3600"],
        check=True, capture_output=True,
    )
    try:
        backend = DockerShellBackend(container)
        # Unbound registry: handlers resolve the RuntimeEnv at *call* time, so the
        # backend we set via runtime_env_context below is the one bash uses.
        registry = create_default_tool_registry(include=["bash"], bind=False)
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=500),
            db_path=":memory:",
            tool_registry=registry,
            config=AgentLoopConfig(
                system_prompt=(
                    "You answer by running shell commands with the bash tool. "
                    "The shell runs inside a Linux container. Run one command at a "
                    "time and base your final answer on what the commands print."
                ),
                max_rounds=6,
            ),
        )
        env = RuntimeEnv(workspace_dir=host_ws, shell_backend=backend)
        sid = loop.new_session()
        with runtime_env_context(env):
            res = await loop.send(
                "Two things, using bash: (1) print the PRETTY_NAME line from "
                "/etc/os-release so we can see the OS; (2) cat host_note.txt in the "
                "current directory. Then tell me the OS name and the file contents.",
                session_id=sid,
            )

        # Deterministic proof for the test: the raw bash tool outputs (not the
        # model's prose) must show the CONTAINER's OS and the bind-mounted file.
        bash_outputs = "\n".join(
            str(m.get("content", ""))
            for m in loop.get_messages(sid)
            if m.get("role") == "tool"
        )
        print("Final answer:\n", res.final_text)
        return {"final_text": res.final_text, "bash_outputs": bash_outputs}
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        shutil.rmtree(host_ws, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
