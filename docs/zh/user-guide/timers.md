# Runtime 11 定时器

一个部署使用一个 `schedule_wakeup` 工具和一个权威 `TimerBackend`。
默认 `Runtime` 装配 `StoreTimerBackend`，随 worker 启停 `TimerRunner`，不需要 Redis。
纯 `RuntimeClient` 不启动扫描器；不需要定时器时传 `timers_enabled=False`。
显式宿主 backend 必须同时配置具有 start/stop/status 的 driver；未就绪时拒绝启动，不回退本地表。

模型调用 `schedule_wakeup(action="schedule", delay_seconds=60, note="检查进展")`，
或 `action="list"`、`action="cancel", timer_id=...`。`every_seconds` 指定周期。
只有 main 角色能创建；查询和取消也校验身份。SDK 向后端传入可信 `TimerContext`：
store/session、actor、原 job/tool-call、稳定 operation_id 和宿主上下文；身份不来自模型参数。

宿主也可使用相同 backend：

```python
from power_loop import TimerContext
source = TimerContext(store.store_id, target.session_id, authenticated_actor)
timer = await runtime.timer_backend.create(source, delay_s=60, note="检查进展")
items = await runtime.timer_backend.list(source, statuses=("armed", "firing"))
await runtime.timer_backend.cancel(source, timer.timer_id)
```

到期驱动按索引取有限批次，带条件领取，再用稳定 message_id 调 `RuntimeClient.append`。
只等待受理，不等待模型执行。忙时在轮边界追加，空闲时启动调用。
周期任务从此次投递时间加 interval 计算，不补跑全部错过的周期。
取消阻止未来触发，也阻止已领取周期任务重新 armed；已经受理的输入须用正常任务控制取消。

把 `AgentHooks` 显式传给 Runtime/driver，保留 `TIMER_FIRE`：CONTINUE 投递、SKIP 跳过本次、
BREAK 取消、正数 postpone_s 延后，也可修改 ctx.message。
自定义 driver 可通过 `TimerRunner(event_bus=...)` 发送结果事件。
status() 提供就绪、最近扫描/错误和最近一批到期待办，查询状态不扫描数据库。

基础设施异常允许丢失、重复或 unknown。陈旧领取会结束该次未知触发，不自动重放；后续周期可继续。
终态记录沿用七天保留期。
DeepTalk 使用 API backend 和 dispatcher；API 负责业务授权、限额以及每次新 run/预算，不写 SDK 定时表。
参见[可运行的编号例子 26](../../../examples/26_timers.py)和[当前 Runtime 合同](../../runtime-11.md)。

有限子任务可显式注册 `register_wait_handoff_tool(registry, policy=WaitPolicy(deadline_at_ms=...))`。
期限由宿主工厂冻结；子任务请求等待，原主 Agent 可继续、取消或延后。延后不能越过原期限。
等待复用原任务交互及当前 TimerBackend，不另建调度账本；未知创建只查询原结果，不自动重放。
