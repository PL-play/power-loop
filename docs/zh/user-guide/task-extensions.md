# 异步任务扩展：复用 Hook 与 Event

任务、workflow 节点和 subagent 共用 power-loop 的扩展机制。扩展应基于类型化接口和已注册的工具契约，不按工具名称、平台模块名或业务类型推断权限。

## 选择扩展接口

| 需求 | 接口 | 边界 |
|---|---|---|
| 模型上下文、轮次策略 | `AgentHooks` 的 `LLM_BEFORE`、`ROUND_DECIDE` | 复用 typed context；不得绕过实际调用准入与预算 |
| 自然结束前补充收尾指导 | `COMPLETE_DECIDE` | 可以请求继续，但不代表所有子任务已完成；仍须经过运行时状态与预算检查 |
| UI、日志、耗时与 token 指标 | `AgentEventBus` | 订阅既有 LLM、工具、subagent、timer 事件；慢处理用异步订阅 |
| 监管收尾的观察通知 | `TASK_SUPERVISION_CLOSED` | 只有原始证据验证及事务成功返回后才发布 |
| 真实取消、外部结果回执、费用结算 | 现有运行时/资源/预算接口 | 保留持久化、幂等和恢复协议，不能用普通事件订阅回调代替 |

新策略优先接入已有 hook；新增观测点优先扩展已有 event 类型与 payload。不新建并列的 hook/event 总线，也不把数据库事务移入可被用户移除的 hook。

## 订阅监管收尾

在创建运行时前，显式传入宿主自己的总线。可以和 loop 共用同一个 bus：

```python
from power_loop import AgentEventBus, AgentEventType, TaskSupervisionClosedPayload
from power_loop.runtime.jobs import JobManager

bus = AgentEventBus(suppress_subscriber_errors=True)

async def refresh_task(event):
    payload: TaskSupervisionClosedPayload = event.data
    # 使用 (payload.store_id, payload.event_id) 去重；按原有权限重新读取任务。
    await my_ui.invalidate_task(payload.store_id, payload.job_id)

bus.subscribe(AgentEventType.TASK_SUPERVISION_CLOSED, refresh_task)
manager = JobManager(store, event_bus=bus)
# TaskRuntime(manager, ...); StatefulAgentLoop(..., event_bus=bus)
```

`JobManager` 默认不发布到全局 bus，避免不同宿主混用任务通知。宿主应在自己控制的租户/权限范围内配置总线；event bus 本身不是权限隔离层。

payload 只有 `store_id`、`job_id`、`generation`、`event_id`、`revision`；不含任务正文、租约、执行令牌、预算或内部证据。envelope 的 `session_id` 是本次执行验证的调用会话，`source` 为 `task_registry`。订阅者修改 payload 不会修改落库原件或方法返回值。

`finalize_supervision` 验证成功后，为本次请求的 job 发出通知。幂等重验可以重复发出同一 `event_id`；内部递归收尾的每一个后代不保证独立发出通知。它不是“每次状态变化恰好一次”的事件流。权限拒绝、证据不完整、generation 冲突、回滚以及提交回执未知时不发送；落库后进程崩溃也可能丢通知。需要可靠消费时，使用持久化任务记录/回执与原有恢复机制，不能凭“没收到事件”判断任务还在运行或重新执行。

此事件仅表示某一代监管证据已闭合，不承诺业务成功、所有将来代次结束或费用结清。事件不得直接驱动 workflow 下游放行、主 Agent 结果 ACK 或业务扣费。

## 回调失败与退出

- 发布发生在事务退出后，不持有任务控制锁等待订阅者。
- 同步 handler 必须短且不阻塞；I/O 用 `async def`。异步 handler 的生命周期沿用现有总线机制，由宿主负责退出时的处理。
- 同步订阅异常（包括直接抛出的 `CancelledError`）不会使已提交的 `finalize_supervision` 变为失败；真实事务/调用方取消仍由原运行时处理。
- 建议 `suppress_subscriber_errors=True`，这样普通异常不阻止同一通知的后续订阅者。严格 bus 保留原有“遇错停止分发”语义，但 JobManager 仍隔离该错误。不要把不能丢失的业务动作放进订阅者。
- 本接口不新增查询、表、DDL 或后台轮询，也不改变任务协议版本。未传 bus 的宿主行为保持原样。
