# 沙箱与隔离

[English](../../en/user-guide/sandboxing.md) | [用户手册](../index.md)

默认情况下，power-loop 在**进程内、宿主机上**运行模型编写的 shell，并带有你的完整环境（包括各类密钥）。**任何执行不可信、由模型编写的命令的宿主，都必须对其进行沙箱隔离。** power-loop 保持与具体沙箱无关，不绑死某一种运行时，而是给你两个注入接缝：

| 接缝 | 包裹对象 | 作用范围 | 示例 |
|---|---|---|---|
| `ShellBackend` | `bash` 工具背后那个持久化的 **shell** | 每个 workspace 一个 shell | [28](../../../examples/28_docker_shell_backend.py) |
| `WorkerLauncher` | 整个 **workflow 叶子进程** | 每个 sub-agent 一个进程 | [30](../../../examples/30_subprocess_isolation.py) |

两者都遵循与 power-loop 其他接缝相同的模式：一个极小的 `Protocol`、一个安全的默认实现，以及按运行注入。它们都不改动任何工具或引擎代码。

## ShellBackend — sandbox the `bash` tool

`bash` / `background_run` 工具把 *shell 进程如何启动* 这件事委托给 `ShellBackend`。`BashSession` 中的 pty / sentinel / 输出抽取（output-drain）机制保持不变，因此对运行时的其余部分来说，进程内 bash 和 `docker exec -i <container> bash` 看起来完全一致。

```python
from power_loop.runtime.exec_backend import ShellBackend  # Protocol
```

默认的 `LocalShellBackend` 在 workspace 目录下运行 `/bin/bash`，继承宿主环境。要做沙箱，实现这个四方法协议：

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

`launch_env` 返回的是 **docker 客户端**的环境（这样才能在 PATH 上找到 `docker`），*而不是*沙箱内部的环境——容器自己的环境在 `docker run` 时就已固定，因此没有任何宿主密钥会越过边界。`session_key` 标识 shell *在哪里* 运行：键相等则复用同一个持久化 `BashSession`；键不同则得到各自独立的 shell。

### 注入它

backend 挂在 `RuntimeEnv.shell_backend` 上，按运行注入，和 workspace 完全一样。用 `bind=False` 构建工具注册表，让 handler 在调用时解析当前的 `RuntimeEnv`：

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

> **0.14.0 起**不一定非用 `bind=False`：带 workspace 的 bound 注册表仍会尊重外层
> `runtime_env_context(...)` 注入的 `ShellBackend`（bind 不再把它重置掉）。也可以直接传：
> `create_default_tool_registry(include=["bash"], workspace_dir=ws, shell_backend=DockerShellBackend("my-sandbox"))`。
> 当一个注册表要跨多个 workspace/沙箱复用时，`bind=False` 仍是正确选择。

完整可运行的版本见 [示例 28](../../../examples/28_docker_shell_backend.py)（它证明了隔离的有效性：shell 会报告容器的操作系统，并读取一个 bind-mount 进来的宿主文件）。DeepTalk 正是用这个接缝，配合一个按会话隔离的 gVisor（`runsc`）容器。

## WorkerLauncher — sandbox a workflow leaf process

当 workflow 叶子在进程外运行时（见 [Workflows → SubprocessExecutor](workflows.md#out-of-process-subprocessexecutor)），`WorkerLauncher` 决定 *每个 worker 进程如何启动*——它是 `ShellBackend` 在进程层面的对应物。默认的 `DirectWorkerLauncher` 裸跑 worker；注入你自己的实现，即可用 `runsc` / `docker run` / `firejail` / `nsjail` 把它包起来：

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

由于 `build` 拿到了该叶子的 `AgentSpec`，你可以**按叶子**选择隔离方式（例如只对带 `bash` 工具的叶子做沙箱）。这是唯一一种能让 sub-agent 获得比其父更*强*隔离的途径：子进程在受限环境中运行，而编排器本身保持不受限。

关键职责与保证：

- **路径是 launcher 的职责。** worker 会原样使用 `db_path`（以及 bootstrap 的 `workspace_dir`），所以沙箱必须让这些路径在其内部解析到相同的位置（例如用 identity bind-mount）。
- **失败即关闭（Fail-closed）。** 启动 / 沙箱错误会变成一个 `failed` 叶子，绝不会挂起——恢复层（resume tier）可以重新运行它。
- 预留的 *shared scope → orchestrator* 接缝（为将来的跨进程黑板预留）在本层尚未实现；被隔离的叶子之间不共享实时状态。

可运行的版本见 [示例 30](../../../examples/30_subprocess_isolation.py)，它会记录每个叶子的启动过程，并在之后检视每个叶子各自私有的 DB。

## 我该用哪个接缝？

- 运行一个会执行 `bash` 的编码 agent？→ **`ShellBackend`**（对 shell 做沙箱）。
- 把一个 workflow 扇出，其叶子并行运行不可信代码？→ **`SubprocessExecutor` + `WorkerLauncher`**（对每个进程做沙箱）。
- 两者都要？组合它们——一个被沙箱化的 worker 本身也可以为自己的 bash 安装一个 `ShellBackend`。

## 另见

- [Tools](tools.md) — `bash` 工具与工具预设
- [Workflows](workflows.md) — 进程内与进程外执行器
- [Configuration](configuration.md) — `RuntimeEnv` 与按 send 注入

### 原执行目标的进程控制

SDK在创建shell时绑定原argv、cwd和环境。停止时使用该绑定，不再调用launch_argv寻找或
创建替换目标。宿主可实现 `bind_process_control(workspace_dir, *, argv, cwd, env)`，
返回继承 `power_loop.runtime.process_control.ShellProcessControl` 的控制器。
`stop_tagged(tag, *, variable, grace_s)` 返回 `ShellControlResult`，包含严格布尔值
confirmed及可选错误文本。原身份、连接或实际退出不确定时必须返回未确认；不得在停止
方法里按可复用名称寻找替换资源。

默认控制器在原目标中执行随SDK提供的Python 3脚本，用Linux pidfd控制匹配原命令标签的
进程，避免扫描PID后数字复用造成误杀。目标缺少Python、pidfd或所需权限时保留未知，
不退回数字PID强杀。标签用于追踪配合的命令，不是对抗清除标签、远端工作等恶意行为的
安全隔离边界；这些资源需要宿主更强的生命周期适配器。

后台命令还由本地监督进程持有原组，通过专用管道取消。创建期间的取消不会丢失，停止后
读取输出有时间上限。未确认停止继续显示在jobs中并阻止完成；重启后找不到原控制器不再
伪报已停止。此扩展本身不放宽native受管任务的准入条件。

默认控制器在提交命令前探测pidfd支持，记录原目标boot ID和探测进程的启动时间。更早的进程不可能是该命令的子进程；boot变化或无法读取的新同用户进程仍保持未确认。延迟状态回写与后续停止确认串行，旧未知结果不会覆盖新的已确认停止。
