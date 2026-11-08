# Hooks

SDK 11.0.0 · 任务结构 15 · 协议 42 · 描述符结构 1。

[Runtime 契约](../../runtime-11.md) · [API](../api/index.md)

在 ExecutionEnvironment.hooks 提供 AgentHooks。按 HookPoint 注册处理器，并使用该点约定的 HookDirective；任意替换对象不等于有效指令。Hooks 在执行节点运行，工厂需通过声明的 hook profile 重建。

TOOL_BEFORE 处理权限和参数决策；模型 hooks 处理短路或缓存；memory hooks 处理注入与压缩策略。普通工具参数改写后按实际参数计价。Hook 不能把取消、所有权丢失或未知外部动作改成虚假的成功。

TIMER_FIRE 保留继续、跳过、取消和延后决策。TimerFireCtx 携带可信定时器和会话来源，模型不能伪造。Runtime 定时驱动只等待任务受理，不等待模型完成。

命名 hooks 支持 register、replace、remove、has，内置记忆 hook 可显式替换。EventBus 用于观察，hooks 参与执行并遵守时限。需要覆盖非模型调用方的宿主政策，仍在权威受理或写入接口执行。

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design203_single_process.py。

| 示例 | 行为 |
|---|---|
| [07_human_approval.py](../../../examples/07_human_approval.py) | 授权策略 |
| [26_timers.py](../../../examples/26_timers.py) | 定时 hook 与驱动 |
| [43_memory_hook_override.py](../../../examples/43_memory_hook_override.py) | 记忆替换 |
| [44_hook_events_audit.py](../../../examples/44_hook_events_audit.py) | Hook 指令 |

```bash
.venv/bin/python examples/07_human_approval.py
```
