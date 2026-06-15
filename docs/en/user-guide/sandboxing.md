# Sandboxing & Isolation

[中文](../../zh/user-guide/sandboxing.md) | [User Guide](../index.md)

By default power-loop runs model-authored shell **in-process, on the host**, with your full environment (including secrets). **Any host that executes untrusted, model-authored commands must sandbox them.** power-loop stays sandbox-agnostic and gives you two injection seams instead of baking in one runtime:

| Seam | Wraps | Scope | Example |
|---|---|---|---|
| `ShellBackend` | the persistent **shell** behind the `bash` tool | one shell per workspace | [28](../../../examples/28_docker_shell_backend.py) |
| `WorkerLauncher` | a whole **workflow leaf process** | one process per sub-agent | [30](../../../examples/30_subprocess_isolation.py) |

Both follow the same pattern as every other power-loop seam: a tiny `Protocol`, a safe default, and per-run injection. Neither changes any tool or engine code.

## ShellBackend — sandbox the `bash` tool

The `bash` / `background_run` tools delegate *how the shell process is launched* to a `ShellBackend`. The pty / sentinel / output-drain machinery in `BashSession` is unchanged, so an in-process bash and a `docker exec -i <container> bash` look identical to the rest of the runtime.

```python
from power_loop.runtime.exec_backend import ShellBackend  # Protocol
```

The default `LocalShellBackend` runs `/bin/bash` in the workspace dir, inheriting the host environment. To sandbox, implement the four-method protocol:

```python
from collections.abc import Hashable
from pathlib import Path

class DockerShellBackend:
    def __init__(self, container: str, workdir: str = "/workspace") -> None:
        self._container, self._workdir = container, workdir

    def launch_argv(self, workspace_dir: Path) -> list[str]:
        return ["docker", "exec", "-i", "-w", self._workdir,
                self._container, "bash", "--norc", "--noprofile"]

    def launch_cwd(self, workspace_dir: Path) -> str | None:
        return None  # host-side cwd of the `docker` CLI is irrelevant

    def launch_env(self, workspace_dir: Path) -> dict[str, str]:
        import os
        return {**os.environ, "TERM": "dumb", "NO_COLOR": "1"}  # the docker CLIENT's env

    def session_key(self, workspace_dir: Path) -> Hashable:
        return ("docker", self._container, self._workdir)  # same target → reuse one shell
```

`launch_env` returns the **docker client's** environment (so `docker` is found on PATH), *not* the sandbox's — the container's own env is fixed at `docker run` time, so no host secret crosses the boundary. `session_key` identifies *where* the shell runs: equal keys reuse one persistent `BashSession`; different keys get distinct shells.

### Injecting it

The backend lives on `RuntimeEnv.shell_backend` and is injected per-run, exactly like a workspace. Build the tool registry with `bind=False` so handlers resolve the current `RuntimeEnv` at call time:

```python
from power_loop import (
    RuntimeEnv, StatefulAgentLoop, create_default_tool_registry, runtime_env_context,
)

registry = create_default_tool_registry(include=["bash"], bind=False)
loop = StatefulAgentLoop(llm=..., db_path=":memory:", tool_registry=registry)

env = RuntimeEnv(workspace_dir=host_ws, shell_backend=DockerShellBackend("my-sandbox"))
with runtime_env_context(env):
    await loop.send("Run the tests with bash.", session_id=sid)
```

See [example 28](../../../examples/28_docker_shell_backend.py) for a complete, runnable version (it proves isolation: the shell reports the container's OS and reads a bind-mounted host file). DeepTalk uses this exact seam with a per-conversation gVisor (`runsc`) container.

## WorkerLauncher — sandbox a workflow leaf process

When workflow leaves run out-of-process (see [Workflows → SubprocessExecutor](workflows.md#out-of-process-subprocessexecutor)), `WorkerLauncher` decides *how each worker process is launched* — the process-level analog of `ShellBackend`. The default `DirectWorkerLauncher` runs the worker bare; inject your own to wrap it in `runsc` / `docker run` / `firejail` / `nsjail`:

```python
from power_loop.runtime.spec import AgentSpec
from power_loop.workflow import DirectWorkerLauncher, SubprocessExecutor, WorkerBootstrap

class DockerWorkerLauncher:
    def build(self, *, base_argv, base_env, spec: AgentSpec, db_path, workspace_dir):
        # Decide isolation PER LEAF from spec.tools, then wrap the command.
        argv = ["docker", "run", "--rm", "-i", "--network", "none",
                "-v", f"{db_path}:{db_path}", "python:3.12-slim", *base_argv]
        return argv, base_env

executor = SubprocessExecutor(
    bootstrap=WorkerBootstrap(llm_from_env=True),
    launcher=DockerWorkerLauncher(),
)
```

Because `build` receives the leaf's `AgentSpec`, you can choose isolation **per leaf** (e.g. only sandbox leaves that have the `bash` tool). This is the only way to give a sub-agent *stronger* isolation than its parent: the child runs confined while the orchestrator stays unconfined.

Key responsibilities and guarantees:

- **Paths are the launcher's job.** The worker uses `db_path` (and the bootstrap's `workspace_dir`) verbatim, so a sandbox must make those paths resolve to the same location inside it (e.g. an identity bind-mount).
- **Fail-closed.** A launch/sandbox error becomes a `failed` leaf, never a hang — the resume tier can re-run it.
- The reserved *shared scope → orchestrator* seam (for a future cross-process blackboard) is not implemented in this tier; isolated leaves don't share live state.

See [example 30](../../../examples/30_subprocess_isolation.py) for a runnable version that records the per-leaf launches and inspects each leaf's private DB afterward.

## Which seam do I need?

- Running a coding agent that executes `bash`? → **`ShellBackend`** (sandbox the shell).
- Fanning out a workflow whose leaves run untrusted code in parallel? → **`SubprocessExecutor` + `WorkerLauncher`** (sandbox each process).
- Both? Compose them — a sandboxed worker can itself install a `ShellBackend` for its own bash.

## See also

- [Tools](tools.md) — the `bash` tool and tool presets
- [Workflows](workflows.md) — the in-process vs out-of-process executors
- [Configuration](configuration.md) — `RuntimeEnv` and per-send injection
