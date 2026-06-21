# 自己造：用原语重建内置工具

[English](../../en/user-guide/build-your-own-tools.md) | [用户指南](../index.md)

power-loop 里每个"特殊"内置工具——后台任务、子代理、定时器、人类输入、黑板、记忆——都建立在**你也能用
的公开原语**之上。本页把每一个都重写成一个普通的自定义工具，让你看到它们没有黑魔法，并能把这套模式套到
你自己的后端上。这里所有代码就是可运行的 [`examples/42_build_your_own_tools.py`](https://github.com/)，
并由 `tests/unit/test_byo_tools.py` 实测，因此不会与文档漂移。

> 前置：[异步编排](async-orchestration.md)模型（宿主驱动；`send`/`resume`/`submit_input`/`follow_up`
> 唤醒 API）与[工具配方](extending-tools.md)。

## 原语清单

下面全部只用这些公开接缝（都从 `power_loop` 导出）：

| 原语 | 给自定义工具什么 |
|---|---|
| `ToolDefinition` + `ToolRegistry.register(defn, handler)` | 声明工具 + 一个 **`async def`** 处理器 |
| `get_tool_runtime_context()` → `ToolRuntimeContext(session_id, store, loop, config)` | 在处理器里拿到当前 session |
| `SessionStore` 方法（`add_note`/`list_notes`、`get_runtime_state`/`set_runtime_state`、`upsert_background_task`/`list_unseen_background_updates`/`mark_background_seen` …） | 持久的每-session 状态 |
| `loop.schedule_timer` / `follow_up` / `resume` / `submit_input` | 唤醒 / 续跑循环 |
| `run_agent_spec(spec, input, *, parent_loop)` + `AgentSpec` | 运行子代理 |
| `RuntimeProjector`（+ `config.runtime_projectors`） | 每轮把临时状态注入提示 |
| `HumanInputRequired` | 暂停循环等待输入的控制信号 |
| `MemoryProvider`（+ `config.memory`） | 每次 send 把持久记忆召回进上下文 |

"从原语造"的铁律：**绝不 import 内置工具的私有模块**——只用上面的公开接缝。哪里内置用了私有小助手，就把
那一小段逻辑就地复刻（每个特性的 *Gaps* 会注明）。

---

## 1. 后台任务 — `background_run` → `bg_run`

**内置：** `background_run` 在守护线程跑一条 shell 命令，立刻返回 `task_id`，结果自动回流到循环。

**原语：** `get_tool_runtime_context()`（→ `store`、`session_id`）、`asyncio.get_running_loop()` 捕获
owning loop、`store.upsert_background_task(...)` 记录状态，以及一个每轮注入
`list_unseen_background_updates(...)` 再 `mark_background_seen(...)` 的 `RuntimeProjector`。

```python
class CustomBackgroundManager:
    async def run(self, command: str) -> str:
        ctx = get_tool_runtime_context(required=True)
        store, sid = ctx.store, ctx.session_id
        owning_loop = asyncio.get_running_loop()          # 守护线程写回到这个 loop
        task_id = uuid.uuid4().hex[:8]
        await store.upsert_background_task(sid, task_id=task_id, command=command, status="running")
        def _work():
            p = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
            status = "completed" if p.returncode == 0 else f"failed({p.returncode})"
            asyncio.run_coroutine_threadsafe(
                store.upsert_background_task(sid, task_id=task_id, command=command,
                                            status=status, output_tail=(p.stdout + p.stderr)[:4000]),
                owning_loop)          # 若 loop 在关停时已关闭，这次写会被丢弃（陷阱）
        threading.Thread(target=_work, daemon=True).start()
        return f"Background task {task_id} started."

class CustomBackgroundProjector(RuntimeProjector):       # 回流：经 config.runtime_projectors 安装
    def __init__(self, *, mark_seen: bool = True): self.mark_seen = mark_seen
    async def project(self, *, store, session_id, round_index, context):
        updates = await store.list_unseen_background_updates(session_id)
        if not updates: return []
        body = "<background_updates>" + "".join(
            f'<task id="{t.task_id}" status="{t.status}">{(t.output_tail or "").strip()}</task>'
            for t in updates) + "</background_updates>"
        if self.mark_seen: await store.mark_background_seen(session_id, [t.task_id for t in updates])
        return [{"role": "user", "name": "background_updates", "content": body}]
```

**Parity：** 起并发活、立刻返回、结果在之后某轮回流。**陷阱：** 自定义 projector 必须经
`AgentLoopConfig(runtime_projectors=(CustomBackgroundProjector(),))` **替换**默认的 `runtime_projectors`
（默认已含 `BackgroundRuntimeProjector`），否则两个都注入。**Gaps：** 内置还有危险命令校验、持久 bash
会话、LRU 会话缓存——功能 parity 都不需要。

---

## 2. 子代理 — `spawn_agent` → `delegate`

**内置：** `spawn_agent` 在父代理那次工具调用里把子代理跑到完成。

**原语：** `run_agent_spec(spec, task, *, parent_loop)` 正是干这个——在父下创建子 session、跑它的循环、
返回 `{"status", "final_text", ...}`。

```python
async def _delegate(task: str, max_rounds: int = 6) -> str:
    ctx = get_tool_runtime_context(required=True)
    spec = AgentSpec(name="delegate", system_prompt="You are a focused worker…",
                     tools=[], max_rounds=max_rounds)          # tools=[]：无工具；None：继承父的工具
    res = await run_agent_spec(spec, task, parent_loop=ctx.loop)
    out = res.get("final_text") or "(no output)"
    return out if res.get("status") == "completed" else f"[child {res.get('status')}] {out}"
```

**Parity：** 完整——内联跑子代理并返回其答案；深度受 `MAX_SPAWN_DEPTH` 限制。**Gaps：** 内置另有按子代理的
`model` 覆盖、完整的 `SUBAGENT_*` 事件生命周期、`spawn_tool_call_id` 审计关联（可观测性，非行为）。

---

## 3. 迷你工作流 — `WorkflowSpec` → `run_pipeline`

**内置：** `WorkflowSpec` DSL/引擎跑声明式的步骤图，带并行、journaled resume、结构化输出。

**原语：** 编排*内核*其实就是顺序调 `run_agent_spec`，把每步输出串到下一步：

```python
async def _run_pipeline(goal: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    carry = goal
    for step in ("research", "draft"):
        spec = AgentSpec(name=step, system_prompt=f"You are the '{step}' step.", tools=[], max_rounds=4)
        res = await run_agent_spec(spec, f"...\n\n{carry}", parent_loop=ctx.loop)
        carry = res.get("final_text") or carry
    return carry
```

**Parity：** 固定的顺序管线。**Gaps（真引擎多了什么）：** 并行 / foreach / 分支节点、journaled 跨重启
resume、按步的结构化输出 schema、detached 执行、整体 spec 校验。需要这些就用内置 `WorkflowSpec`；这里
展示的是它底下的原语。

---

## 4. 持久定时器 — `schedule_wakeup` → `remind_me`

**内置：** `schedule_wakeup` 排一个持久的自我唤醒；note 之后回到 session。

**原语：** `loop.schedule_timer(sid, delay_s=…, note=…)` 写同一行 `pl_timers`。**宿主**必须跑一个
`TimerRunner`（或轮询 `store.due_timers()`）才会真正触发——power-loop 没有守护进程
（见[异步编排](async-orchestration.md)和[定时器](timers.md)）。

```python
async def _remind_me(delay_seconds: int, note: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    timer = await ctx.loop.schedule_timer(ctx.session_id, delay_s=float(delay_seconds), note=note)
    return f"Reminder armed (#{timer.timer_id}) in {delay_seconds}s."
```

**Parity：** 持久唤醒本身。**Gaps：** 触发侧——`TIMER_FIRE` hook（否决/跳过/改期）、stale-row 心跳、
at-least-once 去重——在 `TimerRunner` / 你的调度器里，不在工具里。自己写调度器就得自己处理这些。

---

## 5. 人类输入 / 审批 — `request_user_input` → `ask_human`

**内置：** `request_user_input` 让循环以 `status="waiting_for_input"` + `pending_interactions` 暂停；
宿主用 `loop.submit_input(...)` 续跑。

**原语：** 这个暂停是一个**控制信号异常** `HumanInputRequired`（公开）。自定义工具靠 raise 它即可获得*完整
parity*：

```python
async def _ask_human(prompt: str, options: list[str] | None = None) -> str:
    raise HumanInputRequired(kind="choice" if options else "text", prompt=prompt,
                             options=[{"value": o, "label": o} for o in (options or [])])
```

宿主流程：`send()` → `result.status == "waiting_for_input"` → 读
`result.pending_interactions[0]["interaction_id"]` → `loop.submit_input(sid, interaction_id, answer)`
→ 循环把答案当作工具结果续跑。

**Parity：** 完整。**Gap：** 它依赖 pipeline 里显式的 `except HumanInputRequired` 捕获。
`HumanInputRequired` 在 `__all__`（公开）但不在 `STABLE_API`；那个捕获是框架里有意、稳定的一部分，但要知道
这是一处架构依赖。

---

## 6. 黑板 — `board_*` → `board_write` / `board_read`

**内置：** `board_*` 是多个 agent 读写以协作的共享便笺。

**原语：** store 的每-session 键值，`get_runtime_state` / `set_runtime_state`（值需 JSON 可序列化）：

```python
_BOARD_KEY = "byo_board"
async def _board_write(text: str) -> str:
    ctx = get_tool_runtime_context(required=True)
    board = list(await ctx.store.get_runtime_state(ctx.session_id, _BOARD_KEY, default=[]) or [])
    board.append(text)
    await ctx.store.set_runtime_state(ctx.session_id, _BOARD_KEY, board)
    return f"Posted (entry #{len(board)})."
```

**Parity：** session 内共享的便笺（同一 session 里的多个 agent 都看得到）。**Gap：** 内置
`SqliteBlackboard` 按 `board_id` 跨 session 共享并带版本化 JSON 乐观并发；这个 session-local 版更简单。
跨 session/进程共享时，改用 `runtime_env_context` 注入一个 board 后端。

---

## 7. 记忆 / 笔记 — `note_add` + 召回 → `remember` + `CustomNotesMemory`

**内置：** `note_add` 写持久笔记；`SQLiteNoteMemory`（一个 `MemoryProvider`）在每次 send 开头把它们召回。

**原语：** 工具侧用 `store.add_note` / `list_notes`；召回用 `MemoryProvider` 协议（`config.memory`）+
`render_notes(...)`：

```python
async def _remember(content: str, pinned: bool = False) -> str:
    ctx = get_tool_runtime_context(required=True)
    note = await ctx.store.add_note(ctx.session_id, content.strip(), pinned=bool(pinned))
    return f"Remembered as note #{note.note_id}."

class CustomNotesMemory:                                   # config.memory = CustomNotesMemory(store)
    def __init__(self, store, *, policy=DEFAULT_NOTES_POLICY): self._store, self._policy = store, policy
    async def recall(self, *, messages, session_id, budget_tokens=1500):
        if session_id is None: return []
        text = render_notes(await self._store.list_notes(session_id), policy=self._policy)
        return [{"role": "system", "name": "memory_notes", "content": text}] if text else []
    async def remember(self, *, snapshot, session_id=None): return None   # 笔记是实时写的
```

**Parity：** agent 写的持久笔记 + 每次 send 召回。**Gaps：** 内置带策略校验的 add/update 助手
（`NotesPolicy` 的上限/淘汰）是私有的——需要就就地复刻那一小段；且 `NoteRow` 未导出，按属性访问返回的笔记
（`note.note_id`、`note.content`、`note.pinned`）。

---

## 崩溃恢复（附带）

以上都不改变恢复故事：工具调用中途崩溃会留下 `pending`；下次 `send()` 抛 `SessionPendingError`；用
`resume` / `abort_pending` / `heal_pending` 恢复。见 [`examples/05_pending_recovery.py`](https://github.com/)
和[异步编排](async-orchestration.md)的恢复一节。

## 另见

- [`examples/42_build_your_own_tools.py`](https://github.com/) —— 以上全部，可运行，带真实-LLM 演示
- [异步编排](async-orchestration.md) · [扩展工具](extending-tools.md)
- 深入：[高级运行时工具](advanced-runtime-tools.md) · [子代理](subagents.md) · [定时器](timers.md) · [工作流](workflows.md) · [黑板](blackboard.md) · [记忆](memory.md)
