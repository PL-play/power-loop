# power-loop 11.0.0

可嵌入的 Agent 执行内核。主任务、子任务、后台工具与 workflow 使用同一套按 ID 投递和执行的接口。
默认单进程使用内存协调与通知，不需要 Redis；共享部署显式接入共享协调和消息传输。

当前结构版本为 15，协议版本为 42，描述版本为 1。使用新 SDK 数据库或表前缀，不导入历史任务。
本分支为源码交付，尚未发布或部署；完整回归状态以 P7 交付报告为准。

- [当前 API 与启动方式](docs/runtime-11.md)
- [可运行的单进程示例](examples/design203_single_process.py)
- [可运行的默认定时器](examples/26_timers.py)
- [版本变更](CHANGELOG.md)
- [公共稳定导出](power_loop/__init__.py)

在唯一源码目录安装开发包：`python -m pip install -e '.[dev]'`。
随后从独立测试目录运行 `python /absolute/path/to/examples/design203_single_process.py`。
示例使用固定模型，不访问外部服务。实际使用时由 `RuntimeFactory` 解析模型、工具、hooks 和资源。

`StatefulAgentLoop` 仅接受 `runtime=`；旧的直接 LLM/DSN 构造方式不再可用。
受理回执不代表执行完成，调用方必须按原 `InputRef` 等待结果。
正常权限、费用、上下文与清理继续保证；基础设施故障允许 interrupted/unknown，不自动重放未知外部动作。

子任务定时等待通过 `WaitPolicy` 与 `register_wait_handoff_tool` 装配。等待释放执行容量；原主 Agent 可以继续、取消或延后，始终受原截止时间与预算约束。详见 [运行时合同](docs/runtime-11.md#finite-child-review)。
