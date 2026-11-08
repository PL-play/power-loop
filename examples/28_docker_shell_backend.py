"""28 · An explicit factory environment runs bash inside a dedicated Docker container.

Requires a working Docker CLI/daemon and python:3.12-slim (pulled if missing).
Missing Docker is an error. The fixed model makes no remote calls. This example
uses a process-local shell; shared deployments also need shared process control
and stable workspace/resource profiles, as described in docs/runtime-11.md.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Hashable
from pathlib import Path

from _runtime import demo_runtime

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    RuntimeEnv,
    create_default_tool_registry,
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


class ShellModel(LLMService):
    async def complete(self, request, **kwargs):
        outputs = [m['content'] for m in request.messages if m.get('role') == 'tool']
        if outputs:
            return LLMResponse(raw_text=outputs[-1])
        return LLMResponse(raw_text='', tool_calls=[{'id':'shell','type':'function','function':{
            'name':'bash','arguments':json.dumps({'command':'cat /etc/os-release; cat host_note.txt'})}}])

    async def close(self):
        pass


async def main() -> dict:
    if not _docker_available():
        raise RuntimeError('Docker CLI/daemon access is required')
    if not _ensure_image():
        raise RuntimeError(f'Docker image unavailable: {IMAGE}')
    with tempfile.TemporaryDirectory(prefix='power_loop_docker_test_') as directory:
        host_ws = Path(directory)
        (host_ws / 'host_note.txt').write_text('hello from the host machine\n')
        container = f'power-loop-example-test-{uuid.uuid4().hex[:8]}'
        subprocess.run(['docker','run','-d','--rm','--name',container,'--network','none',
            '-v',f'{host_ws}:{CONTAINER_WORKDIR}','-w',CONTAINER_WORKDIR,IMAGE,'sleep','60'],
            check=True, capture_output=True)
        try:
            backend = DockerShellBackend(container)
            registry = create_default_tool_registry(include=['bash'], bind=False)
            env = RuntimeEnv(workspace_dir=host_ws, shell_backend=backend)
            async with demo_runtime(ShellModel(), registry, runtime_env=env,
                    config=AgentLoopConfig(max_rounds=3, compactor=None)) as (runtime, loop, target, actor):
                result = await loop.send(target, 'Inspect container and mounted file', message_id='docker', actor=actor,
                                         deadline_ms=int(time.time()*1000)+15000)
                assert result.status == 'completed', result
                outputs = '\n'.join(row.content for row in await runtime.store.load_all_messages(target.session_id) if row.role == 'tool')
                assert 'Debian' in outputs and 'hello from the host machine' in outputs
                print(outputs)
                return {'final_text':result.final_text, 'bash_outputs':outputs}
        finally:
            subprocess.run(['docker','rm','-f',container], check=True, capture_output=True)


if __name__ == '__main__':
    asyncio.run(main())
