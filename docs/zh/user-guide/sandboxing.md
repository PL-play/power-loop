# 沙箱与进程资源

SDK 11.0.0 · 任务结构 15 · 协议 42 · 描述符结构 1。

[Runtime 契约](../../runtime-11.md) · [API](../api/index.md)

执行工厂返回 ExecutionEnvironment(runtime_env=..., process_control=..., workflow_file_io=...)。RuntimeEnv 在实际执行节点提供 workspace、skills 和 ShellBackend。模型生成的命令拥有该后端的权限；产品需要隔离时由宿主管理隔离容器。

ShellBackend 定义 launch_argv、launch_cwd、launch_env 和 session_key。Docker 适配器把稳定资源引用映射到指定容器和 workspace。创建 shell 时绑定原目标，之后取消操作仍指向该目标，不能按可复用的容器名重新找一个替代资源。

ShellProcessControl.stop_tagged 返回严格的 confirmed 布尔值。能力缺失或无法确认原进程时保留 unconfirmed，不能改成成功或暗中换资源。输出排空和取消有时限。后台命令使用 Runtime 任务，父任务退出后仍可查询控制，并保留原资源保护。

Runtime 与工作流文件 IO 显式装配。共享节点核验共享引用及内容，不能只检查目录存在。Docker 示例需要可用 daemon 和指定镜像；Python 进程控制器需要目标环境提供相应支持。依赖缺失应明确失败。

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design203_single_process.py。

| 示例 | 行为 |
|---|---|
| [28_docker_shell_backend.py](../../../examples/28_docker_shell_backend.py) | Docker shell |
| [20_default_tools.py](../../../examples/20_default_tools.py) | 工作区工具 |
| [27_dynamic_workflow.py](../../../examples/27_dynamic_workflow.py) | 工作流 IO |

```bash
.venv/bin/python examples/28_docker_shell_backend.py
```
