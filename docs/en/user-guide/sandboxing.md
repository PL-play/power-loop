# Sandboxing & Isolation

[中文](../../zh/user-guide/sandboxing.md) | [User Guide](../index.md)

By default power-loop runs model-authored shell **in-process, on the host**, with your full environment (including secrets). **Any host that executes untrusted, model-authored commands must sandbox them.** power-loop stays sandbox-agnostic and gives you two injection seams instead of baking in one runtime:

| Seam | Wraps | Scope | Example |
|---|---|---|---|
| `ShellBackend` | the persistent **shell** behind the `bash` tool | one shell per workspace | [28](../../../examples/28_docker_shell_backend.py) |

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

> **Since 0.14.0** you don't *have* to use `bind=False`: a bound registry built with a workspace
> still honors a `ShellBackend` injected by an outer `runtime_env_context(...)` (the bind no longer
> resets it). You can also pass one directly:
> `create_default_tool_registry(include=["bash"], workspace_dir=ws, shell_backend=DockerShellBackend("my-sandbox"))`.
> `bind=False` remains the right choice when one registry is reused across many workspaces/sandboxes.

See [example 28](../../../examples/28_docker_shell_backend.py) for a complete, runnable version (it proves isolation: the shell reports the container's OS and reads a bind-mounted host file). DeepTalk uses this exact seam with a per-conversation gVisor (`runsc`) container.

## Runtime worker isolation

Workflow drivers and leaves run through the common Runtime worker. Deploy workers
in host-managed containers or sandboxes and resolve tools/resources using the frozen
RuntimeDescriptor. Shared execution requires a shared coordinator and transport;
the default memory adapters are for one process. The SDK no longer launches leaf
processes or creates private per-leaf databases. See [Workflows](workflows.md).

## See also

- [Tools](tools.md) — the `bash` tool and tool presets
- [Workflows](workflows.md) — the in-process vs out-of-process executors
- [Configuration](configuration.md) — `RuntimeEnv` and per-send injection

### Original-target process control

At shell creation the SDK binds the original argv, cwd and environment. Later
cancellation uses that bound target; it does not call `launch_argv` again. A host
may implement `bind_process_control(workspace_dir, *, argv, cwd, env)` and return
an instance of `power_loop.runtime.process_control.ShellProcessControl`.
Its `stop_tagged(tag, *, variable, grace_s)` returns `ShellControlResult` with a
strict boolean `confirmed` and an optional error string. Return unconfirmed when
original resource identity, transport or physical exit is uncertain. Never
resolve a replacement resource by reusable name inside this method.

The default bound controller runs a bundled Python 3 helper in the original
shell target. Linux pidfds prevent a PID discovered during a tag scan from
becoming authority to kill a later replacement. This requires Python 3 and pidfd
support/permission in the target. A missing capability remains unconfirmed;
there is no fallback to numeric PID kills. Command tags track cooperative
processes and are not a security boundary against hostile commands that remove
their tags or create remote work. Such resources need a stronger host adapter.

Background commands also have a local supervisor and an original cancellation
pipe. Creation-time cancellation is remembered; draining output after stop is
bounded. Failed stop confirmation remains visible in jobs and blocks completion,
including after the original controller is lost at restart. This extension does
not on its own admit shell work into native managed-task execution.

Before submitting a command, the default controller probes pidfd support and records the original target boot ID and probe process start time. Processes older than that probe cannot be descendants of the command; a changed boot or an unreadable newer owned-user process remains unconfirmed. Deferred status writes are serialized with later stop confirmation so old unknown receipts cannot overwrite a confirmed stop.
