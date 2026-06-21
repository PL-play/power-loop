# 异步编排与长任务

[English](../../en/user-guide/async-orchestration.md) | [用户指南](../index.md)

power-loop 有几种方式做"一轮 LLM 里做不完"的工作——[后台任务](advanced-runtime-tools.md)、
[子代理](subagents.md)、[工作流](workflows.md)、[持久定时器](timers.md)。它们共享**同一套模型**：
异步结果如何回到 agent、如何持久化、如何与[投影](send-context-projection.md)/
[压缩](compaction.md)配合。本页就是这套共享模型——读懂它，各个分页就都顺理成章了。最后还讲了如何用
**纯公开、可调用的 API**（没有黑魔法）写你自己的异步工具，以及问题排查。

## 唯一的铁律：power-loop 由宿主驱动（无守护进程）

`StatefulAgentLoop` 在服务层是**无状态的**，且**自己不跑任何后台线程**去推进一段对话。store
（`pl_*` 表）是唯一事实源；循环每次调用都从中读最新状态（read-latest）。**每一次"唤醒"都由你的
宿主进程驱动。** power-loop 内部不会自己发现"某个定时器到点了""某个后台任务完成了"然后开新一轮——
必须是*你*（或 power-loop 附带的一个小 runner）调回来。

驱动一个 session 的全部 API：

| 调用 | 何时用 | 做什么 |
|---|---|---|
| `loop.send(text, sid)` | 一条新的用户消息 | 追加一条用户消息，运行若干轮直到循环让出（完成/等待/pending/超预算）。分配一个**新的** `send_index`。 |
| `loop.follow_up(text, sid)` | 异步结果 / 带外插话 | 若该 session 上有 `send` **正在执行**，则把 `text` 入队，运行中的循环会在**下一个轮边界**把它作为 `<follow_up>` 用户消息注入。若 session **空闲**，等价于 `send`。这是通用的"唤醒/引导"路径。 |
| `loop.resume(sid)` | 崩溃后继续，或 `pending_tools` 让出后继续 | 重新执行持久化的 pending tool_calls，然后续跑。 |
| `loop.submit_input(sid, interaction_id, value)` | 回答一个挂起的 `request_user_input` | 解除人类介入的暂停并续跑。 |

> 循环**让出**（返回给你）时带一个 `status`：`completed`、`waiting_for_input`、`pending_tools`、
> `hit_round_limit`、`budget_exceeded`、`cancelled`、`degraded`。status 告诉你用哪个调用续跑
> （`waiting_for_input` → `submit_input`；`pending_tools` → `resume`；其余 → `send`/`follow_up`）。

下面所有内容都是同一句话的变体："异步工作把自己的状态存进 store，然后某个东西调上面四个方法之一把结果带回来。"

## 各类异步结果如何回到循环

| 机制 | 怎么跑 | 结果怎么回到 agent |
|---|---|---|
| **后台任务**（[`background_run`](advanced-runtime-tools.md)） | 守护线程；工具立刻返回 `task_id` | **自动、在循环内**：一个 `RuntimeProjector`（默认开的 `BackgroundRuntimeProjector`）在*每轮开头*运行，把**未读**的已完成/更新任务作为 `<background_updates>` 系统消息注入，然后标记已读。所以运行中的循环下一轮就看到；**空闲**的循环要等下一次 `send`/`follow_up`/`resume`。 |
| **子代理**（[`spawn_agent` / `AgentSpec`](subagents.md)） | 跑**自己**的循环到完成，就在父代理的那次工具调用里（受 `MAX_SPAWN_DEPTH` 限深） | **内联**：子代理的结果就是工具的返回值，落在父代理同一轮的 tool-result 行里——不需要单独唤醒。 |
| **工作流**（[`create_workflow`](workflows.md)） | 进程内（或子进程）执行器逐步执行 DSL | 同步步骤内联返回；**detached** 工作流靠完成时唤醒（一次 `follow_up`）回来——和后台任务一样。 |
| **定时器**（[`schedule_wakeup`](timers.md)） | 一行持久的 `pl_timers`；自己不会触发 | 一个 **`TimerRunner`**（或你自己轮询 `store.due_timers()` 的调度器）发现到期行，调 `loop.follow_up(note, sid)`——运行中则入队，空闲则新开一次 `send`。投递前 `TIMER_FIRE` hook 可否决/跳过/改期。 |
| **pending tool_calls**（崩溃，或 `pending_tools` 让出） | assistant 发了 tool_calls，结果没写全 | 持久化的 `session_state.pending` 由 `loop.resume(sid)` 重放。 |
| **人类输入**（`request_user_input`） | 循环以 `waiting_for_input` 暂停 | `loop.submit_input(sid, interaction_id, value)`。 |

两个要记牢的推论：

- **后台任务完成本身不会跑一轮。** 它只在循环下一次取轮时变可见——也就是下一次
  `send`/`follow_up`/`resume`。如果你的 agent 起了个长任务然后空闲了，由*你*决定何时唤醒它
  （比如一个定时器，或你自己的"任务完成"回调里调 `follow_up`）。
- **子代理会阻塞父代理那一轮，后台任务不会。** 想*现在这一轮*就要结果用 `spawn_agent`；想继续
  聊、稍后再取结果用 `background_run`。

## 持久化、session 状态与崩溃恢复

循环产出的每一行都经 **sink** 按单调递增的 `seq`（永不重置）追加进 `pl_messages`，并打上当前
`send_index`。每个 session 的持久状态在 `pl_session_state`：`next_seq`、`round_index`、
`last_compact_seq`，以及 **`pending`**（飞行中 tool_calls 的 JSON 快照）。后台任务在
`pl_background_tasks`，定时器在 `pl_timers`，笔记在 `pl_notes`。

因为 store 是唯一事实源，一个 session **可凭 `(dsn, session_id)` 跨进程恢复**——没有内存里的
session 状态会丢。你要处理的是：

- **工具调用中途崩溃会留下 `pending`。** 该 session 下一次 `loop.send()` 会抛
  **`SessionPendingError`** 而不是悄悄丢掉这半截。用 `loop.resume(sid)`（重放 pending 工具并续跑）、
  `loop.abort_pending(sid)`（丢弃）、或 `loop.send(..., heal_pending=True)`（先丢再发）恢复。
  请有意识地选——这是特性不是 bug：它阻止一对悬空的 tool 调用污染下一轮。
- **`resume()` 会重新执行工具处理器**，所以有副作用的处理器必须**幂等**（或自己防重）——resume 的
  工具调用会再跑一次。
- **每 session 的锁和 follow-up 队列都在内存里**（`_locks[sid]`、`_follow_up_queues[sid]`）。它们
  只在*一个进程内*串行化并发 send，重启后什么都不留。如果进程死时一条 follow-up 只入了队还没被
  drain，它就丢了——所以**需要持久的唤醒就用定时器**（持久化），不要用裸 `follow_up`。而且共享同一
  个 store 的两个进程**不会**被这些锁互相串行化（见"问题排查"）。

## 与投影、压缩的配合

`send_index` **每次 `send()` 分配一次**，并被该 send 产出的每一行继承——**包括** `resume()`
追加的行、drain 进来的 `follow_up`、以及定时器经 `follow_up` 投递的 note。所以一个中途到达的异步
结果属于**那一次 send**，不是新的。一个唤醒空闲 session 的定时器（经 `follow_up`）会成为一次新的
`send`（有自己的 `send_index`）。

- **[投影](send-context-projection.md)模式**下，一次 send 在 end-of-send 被投影。reason-gate 会
  **推迟**投影一个以 `waiting_for_input` / `pending_tools` 让出的 send（等 resume 完成后在同一个
  `send_index` 下重新定稿），所以被打断的异步轮不会被投两次或投一半。缺失或版本过期的投影行会回退到
  从 `pl_messages` **逐字**渲染那次 send（绝不丢）；投影器从不抛异常——它降级。被折叠的 send 仍可
  用 `recall_send` 取回。
- **[压缩](compaction.md)模式**（默认）下，异步行就是普通历史，正常折叠。投影与压缩**互斥**
  （`history_projector` ⇒ `compactor=None`）。
- `BackgroundRuntimeProjector` 注入的 `<background_updates>` 是**每轮临时**消息（不作为正常一轮
  持久化）——既能让结果浮现，又不撑大审计日志。

## 写你自己的异步唤醒工具

上面没有任何私有机制。一个"起异步工作、再把结果带回来"的自定义工具，用的是同一批公开、可调用的接缝
（基础见[扩展工具](extending-tools.md)）：

1. **定义 + 注册**一个工具（`ToolDefinition` + 一个 **`async def` 处理器**——见下方异步陷阱——经
   `ToolRegistry.register`）。
2. 在处理器里用 `get_tool_runtime_context()` **拿到 session** → `store`、`session_id`、`loop`、
   `config`。该存的存（你自己的表，或经后台 API 复用 `pl_background_tasks`）。
3. **把结果带回来**，挑合适的回流方式：
   - *让运行中的循环自己取* —— 写进 `pl_background_tasks`（或像 `BackgroundRuntimeProjector` 那样
     暴露你自己的 `RuntimeProjector`，每轮注入未读状态）；
   - *稍后唤醒空闲 session* —— 排一个持久定时器（`loop.schedule_timer(sid, delay_s=..., note=...)`），
     或在你的完成回调里调 `loop.follow_up(result, sid)`；
   - *本轮阻塞等结果* —— 在处理器里直接 `await` 你的工作并返回（子代理就是这么做的）。

```python
from power_loop import ToolDefinition, get_tool_runtime_context

async def _start_export(rows: int) -> str:
    ctx = get_tool_runtime_context(required=True)          # store / session_id / loop / config
    sid = ctx.session_id
    job_id = await my_jobs.start(rows)                      # 你的异步系统
    # 持久唤醒：任务完成时间到了，定时器经 follow_up 回流。
    # （或在别处的"任务完成"回调里：await ctx.loop.follow_up(summary, sid)。）
    await ctx.loop.schedule_timer(sid, delay_s=5, note=f"导出任务 {job_id} 应该好了——去查一下")
    return f"已启动导出任务 {job_id}；我稍后会被提醒去查结果。"

EXPORT_TOOL = ToolDefinition(
    name="start_export",
    description="开始一个长导出；之后你会被提醒去查结果。",
    input_schema={"type": "object", "properties": {"rows": {"type": "integer"}}, "required": ["rows"]},
)
# registry.register(EXPORT_TOOL, _start_export)
```

你会用到的唤醒 API——`loop.follow_up`、`loop.schedule_timer` / `cancel_timer`、`loop.resume`、
`loop.submit_input`——加上 `TimerRunner`、`runtime_env_context` / `RuntimeEnv`（给工具按 send 注入
后端）、`RuntimeProjector`（每轮注入临时状态）、`register_spawn_agent` / `run_agent_spec`，全都在包的
`STABLE_API` 里。如果工具里需要一个后端（DB 连接池、HTTP 客户端、沙箱）又不想用全局变量，就用
`runtime_env_context(...)` 按 send 注入，再用 `get_tool_runtime_context()` 读出来。

## 问题排查

**"后台任务完成了，agent 却没反应。"** 完成的任务只在*下一轮*被注入，而空闲的循环不取轮。唤醒它：
排个定时器，或在完成回调里 `follow_up`。也确认 `BackgroundRuntimeProjector` 在 `runtime_projectors`
里（默认就在）——没有它，`pl_background_tasks` 的更新永远不会浮现。

**"后台任务卡在 `running`，结果从没写回。"** 守护线程通过 `run_coroutine_threadsafe` 把终态写回循环
的 event loop。如果该 loop 在写落地前**关停时被关闭**，写就丢了。关闭循环前先 drain/await 飞行中的工作
（`await loop.aclose(...)`），并把长期 `running` 的任务在下次启动时当作可能孤儿处理。

**"`send()` 抛 `SessionPendingError`。"** 上一次运行在 assistant 发了 tool_calls 但结果没写全时崩了。
调 `resume(sid)`（写完）、`abort_pending(sid)`（丢弃）、或 `send(..., heal_pending=True)`。别忽略它
——它在保护下一轮不被一对悬空 tool 调用搞坏。

**"我的异步工具跑了但结果没出现 / 跑在了错误的线程上。"** 把处理器注册成 **`async def`**（或 `__call__`
是 async 的可调用对象）。一个*返回协程的同步 lambda* 会被当作同步、用 `asyncio.to_thread` 跑，它的协程
永远不会被 await。相应地，async 处理器必须经 `invoke_async` 调用（循环会这么做）——对 async 处理器调
`invoke()` 会抛 `AsyncToolInSyncContext`。还有，如果处理器里 `asyncio.create_task`，要拷贝 context
（`contextvars.copy_context()`），否则会丢 `session_id` / `store`。

**"定时器触发了两次。"** 投递是**至少一次**：在认领定时器（`firing`）和结束它之间崩溃，会让 stale-row
恢复把它重新 arm。让效果幂等，并/或用 **`TIMER_FIRE` hook**（唯一能在投递前按 `timer_id` 去重/否决/
改期的地方）。

**"重启后一条入队的 `follow_up`（或定时器 note）不见了。"** follow-up 队列在内存里。任何必须扛崩溃的东西
用**持久定时器**；`follow_up` 是给活着的、进程内的运行做引导用的。

**"同一个 session 上的两个进程互相踩了。"** 每 session 的锁只在进程内——它**不**跨进程协调。每个 session
跑一个写者，或自己加一个按 `session_id` 的分布式锁；store 本身保持一致（它串行化自己的写），但循环不会
自己排队。

**"我把一个 session 切到投影模式，旧轮看起来被折叠/压缩了。"** 那是一次性的切换迁移
（见[发送上下文投影](send-context-projection.md)）；它是 best-effort 且从不抛异常。投影前的行
（`send_index = NULL`）作为前缀逐字渲染；带 in-place `compact_note` 历史的 session 会降级为逐字渲染，
而不是错误分区。

## 另见

- [高级运行时工具](advanced-runtime-tools.md) —— `background_run`、运行时上下文、运行时投影器、hook
- [子代理](subagents.md) · [工作流](workflows.md) · [持久定时器](timers.md)
- [扩展工具](extending-tools.md) —— 工具定义/处理器/注册 配方
- [发送上下文投影](send-context-projection.md) · [压缩](compaction.md) · [会话](sessions.md)
