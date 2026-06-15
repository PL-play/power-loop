# 持久化定时器

[English](../../en/user-guide/timers.md) | [用户手册](../index.md)

定时器让 Agent（或你的宿主程序）可以表达*"在时刻 T 用这条备注唤醒该会话"*。它们是**持久化**的——一行 `timers` 表记录，而非内存中的任务——因此能在进程重启后存活。触发本身就是一次普通的轮次：备注通过 `follow_up` 投递进会话（空闲时为一次普通 `send`，运行中则为一次排队注入）。

## 创建定时器

两种方式，写入的是同一批记录。

**Agent 自行安排唤醒**，通过默认工具（`schedule_wakeup`、`cancel_wakeup`、`list_wakeups`）：

```python
from power_loop import create_default_tool_registry, get_tool_definitions
from power_loop.tools import DEFAULT_TOOL_HANDLERS

registry = create_default_tool_registry(preset="core", workspace_dir=ws)
for d in get_tool_definitions(include=["schedule_wakeup", "cancel_wakeup", "list_wakeups"]):
    registry.register(d, DEFAULT_TOOL_HANDLERS[d.name], overwrite=True)
# Now the model can call schedule_wakeup(delay_seconds=3600, note="check the build").
```

**宿主程序从外部安排**，通过 loop API：

```python
t = loop.schedule_timer(sid, delay_s=3600, note="check the report")   # or due_at_ms=…
loop.list_timers(sid)
loop.cancel_timer(sid, t.timer_id)
```

## 触发它们：TimerRunner

定时器只在 `TimerRunner` 运行时触发——它扫描 store 中到期的记录并投递。（如果你从自己的调度器轮询 `loop.store.due_timers()`，则不需要它。）

```python
from power_loop import TimerRunner

runner = TimerRunner(loop)
await runner.start()    # re-arms stale rows, then scans every scan_interval
# ...
await runner.stop()
```

参见 [示例 26](../../../examples/26_timers.py)。

## 一次性 vs 周期性

是否周期在创建时声明。无 `interval` → 一次性（`firing → fired`）。设置 `every_seconds`（工具）/ `interval_s`（API）→ 定时器在触发时刻 + interval 处重新装载（固定延迟），直到被取消。**取消是周期性定时器结束的唯一途径。**

```python
loop.schedule_timer(sid, delay_s=60, note="heartbeat", interval_s=300)   # every 5 min
```

## TIMER_FIRE 钩子——编排层的否决点

每次投递前，runner 会以一个 `TimerFireCtx` 运行 `HookPoint.TIMER_FIRE`。由该钩子决定接下来发生什么：

```python
from power_loop import HookDirective, HookPoint

def gate(ctx):                 # ctx: TimerFireCtx(session_id, timer_id, note, due_at, message)
    if system_is_busy():
        ctx.postpone_s = 300   # re-arm 5 min later
    # ctx.directive = HookDirective.SKIP    # drop THIS firing (recurring still re-arms)
    # ctx.directive = HookDirective.BREAK   # cancel the timer entirely

hooks.register(HookPoint.TIMER_FIRE, gate)
```

未注册钩子 → 投递。这里也是你在**重复触发后去重**的地方（见下文）。

## 投递语义

定时器是**至少一次**的。认领是一次比较并设置（`armed → firing`）；某行因进程在触发途中死亡而卡在 `firing`，会被恢复扫描重新装载，可能投递两次。一次*活跃*的慢投递**不会**被重新认领——runner 在 `follow_up` 运行期间对 `firing` 行发送心跳（节奏 = `stale_firing_s` 的四分之一，可通过 `TimerRunner(heartbeat_interval_s=…)` 覆盖）。若某个副作用要求恰好一次，请在 `TIMER_FIRE` 钩子中去重。

## 参见

- [钩子](hooks.md) —— 完整的 `TIMER_FIRE` 契约
- [会话](sessions.md) —— `follow_up` 与持久化 store
- [工作流](workflows.md) —— 分离运行（detached run）用一个定时器唤醒父会话
