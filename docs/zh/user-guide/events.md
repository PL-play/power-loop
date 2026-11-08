# 事件与输出

SDK 11.0.0 · 任务结构 15 · 协议 42 · 描述符结构 1。

[Runtime 契约](../../runtime-11.md) · [API](../api/index.md)

通过 ExecutionEnvironment.event_bus 提供 EventBus。执行前订阅，使用会话与执行身份关联模型、工具、子任务、用量和生命周期事件。共享 bus 按目标会话过滤。执行结束时 Runtime 删除自己的流式订阅，并限时排空异步订阅回调；宿主拥有自己的订阅和 bus。

事件用于观察，hooks 可以改变决策。它们都不是第二套计费账本；费用以持久任务结果和计费接口为准。流式增量是实时通知，重连后查询原输入结果，不能假设丢失增量能重放。

输出订阅的缓冲有界。慢消费者处理缺口并读取持久状态。不要在每个 token 回调中执行阻塞网络或数据库操作。事件可能包含用户和工具内容，访问权限与保留期由宿主决定。

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design203_single_process.py。

| 示例 | 行为 |
|---|---|
| [08_streaming.py](../../../examples/08_streaming.py) | 实时输出 |
| [09_audit_log.py](../../../examples/09_audit_log.py) | 事件观察 |
| [25_token_usage.py](../../../examples/25_token_usage.py) | 用量 |
| [36_observability.py](../../../examples/36_observability.py) | 指标 |
| [44_hook_events_audit.py](../../../examples/44_hook_events_audit.py) | Hooks 与事件 |

```bash
.venv/bin/python examples/08_streaming.py
```
