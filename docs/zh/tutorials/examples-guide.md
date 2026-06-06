# 示例指南 / Examples Guide

[返回文档](../index.md) | [English](../../en/tutorials/examples-guide.md)

本指南结合代码、真实运行输出和要点，逐一讲解 `examples/` 下的每个示例。每个示例只教**一个概念**——按顺序排列，建议从 00 开始。

所有示例共享 `_helpers.py`（加载 `.env`、构建 LLM 客户端）。如果要把示例复制到你的项目里，只需把 `make_llm()` 的两行代码内联，删掉 import 即可。

---

## 目录

| # | 文件 | 概念 |
|---|---|---|
| [00](#00-最简示例) | `hello_world.py` | 发一条消息、拿一条回复 |
| [01](#01-多轮对话) | `multi_turn_chat.py` | 用 session_id 续话 |
| [02](#02-工具调用) | `tool_calling.py` | 自定义工具 + JSON Schema |
| [03](#03-子代理委托) | `subagent_delegation.py` | 命令式子代理 spawn_agent |
| [04](#04-上下文压缩) | `compaction.py` | 自动上下文压缩 |
| [05](#05-悬挂态恢复) | `pending_recovery.py` | 工具调用中途崩溃恢复 |
| [06](#06-声明式子代理) | `declarative_subagent.py` | AgentSpec 声明式子代理 |
| [07](#07-用户确认) | `human_approval.py` | Hook 实现用户确认门 |
| [08](#08-流式渲染) | `streaming.py` | 实时 token 流 |
| [09](#09-审计日志) | `audit_log.py` | 全量事件审计 → JSONL |
| [10](#10-并发会话) | `concurrent_sessions.py` | 并行会话 + 异步审批队列 |
| [11](#11-跨进程恢复) | `cross_process_resume.py` | 进程重启后恢复会话 |
| [12](#12-重试与取消) | `retry_and_cancel.py` | 重试策略 + 取消 |
| [13](#13-sqlite-记忆) | `memory_sqlite.py` | 跨会话 SQLite 记忆 |
| [14](#14-结构化输出) | `structured_card.py` | 结构化 JSON 提取 |
| [15](#15-markdown-技能) | `skills_from_markdown.py` | SKILL.md → 系统提示词 |
| [16](#16-自定义压缩器) | `custom_compactor.py` | 自定义压缩策略 |
| [17](#17-自定义记忆提供者) | `custom_memory_provider.py` | HTTP 后端 MemoryProvider |
| [18](#18-多-provider) | `multi_provider.py` | 多个 LLM 供应商 |
| [19](#19-旗舰示例) | `full_chatbot.py` | **全部功能集合** |
| [20](#20-默认工具) | `default_tools.py` | 内置文件/搜索/bash 工具 |
| [21](#21-可恢复人类输入) | `request_user_input.py` | 可恢复的外部输入 |
| [高级运行时](../../../examples/advanced_runtime/) | `advanced_runtime/` | 运行时绑定工具模式 |

---

## 00 · 最简示例

**概念**：最小化——创建 loop、发一条消息、打印回复。

### 代码

```python
from power_loop import StatefulAgentLoop

loop = StatefulAgentLoop(llm=make_llm(), db_path=":memory:")
sid = loop.new_session()
result = await loop.send("In one sentence: what is HTTP?", session_id=sid)
print(result.final_text)
```

### 要点

- `StatefulAgentLoop` 是**唯一**的公开入口
- `new_session()` 显式创建会话——返回 `session_id`
- `send(user_input, session_id=sid)` 跑完整 agent 循环，返回 `StatefulResult`
- `db_path=":memory:"` → 不落盘的临时 store；生产环境请传文件路径

### 输出

```
HTTP (HyperText Transfer Protocol) is an application-layer protocol that enables
communication between web clients and servers by defining how requests and
responses are formatted and transmitted over the internet.
```

---

## 01 · 多轮对话

**概念**：用 `session_id` 在多轮交互中保持上下文。

### 代码

```python
loop = StatefulAgentLoop(
    llm=make_llm(), db_path=":memory:",
    config=AgentLoopConfig(
        system_prompt="You are a friendly assistant with perfect memory of this chat.",
        max_rounds=1, compactor=None,
    ),
)
sid = loop.new_session()

# 第 1 轮：建立事实
r1 = await loop.send("My favorite color is teal.", session_id=sid)

# 第 2 轮：传同一个 session_id，模型应该记得
r2 = await loop.send("What did I just tell you my favorite color was?", session_id=sid)

# 查看持久化的 history
msgs = loop.get_messages(sid)

# 用完即删
loop.close_session(sid)
```

### 要点

- `new_session()` 返回的 `session_id` 是你**唯一**需要保管的东西
- 每次 `send(...)` 自动加载历史 → 模型看到完整上下文
- `get_messages(sid)` 返回持久化的完整历史
- `close_session(sid)` 物理删除所有会话数据

### 输出

```
turn 1: Got it! I'll remember that your favorite color is teal.

turn 2: You told me your favorite color is teal!

history has 4 messages: roles = ['user', 'assistant', 'user', 'assistant']
deleted 1 session row(s)
```

---

## 02 · 工具调用

**概念**：把自定义 Python 函数注册为工具，让 LLM 自主调用。

### 代码

```python
# 1. 定义工具
DISHES = {"tokyo": "sushi", "bangkok": "pad thai", ...}

def lookup_dish(**kwargs) -> str:
    city = str(kwargs.get("city") or "").strip().lower()
    return DISHES.get(city, f"No data for {city!r}")

LOOKUP_TOOL = ToolDefinition(
    name="lookup_dish",
    description="Return the signature local dish for a given city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    required_params=("city",),
)

# 2. 注册并运行
registry = ToolRegistry()
registry.register(LOOKUP_TOOL, lookup_dish)

loop = StatefulAgentLoop(
    llm=make_llm(), db_path=":memory:", tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt="You answer questions about local cuisine.",
        max_rounds=4,   # 工具调用必须 ≥ 2
        compactor=None,
    ),
)
result = await loop.send("What is Bangkok's signature dish?", session_id=sid)
```

### 要点

- `ToolDefinition` 声明工具名 / 描述 / JSON Schema / required 参数
- `ToolRegistry.register(definition, handler)` 把定义和处理函数绑定
- handler 可以是 sync 或 `async def`——ToolRegistry 自动适配
- **工具调用需要两轮**：Round 1 LLM 决定调用 → Round 2 工具结果回灌
- `max_rounds=1` **跑不通**工具调用——必须 ≥ 2

### 输出

```
status: hit_round_limit, rounds: 4
reply : [hit_round_limit]
**Accomplished:**
I successfully used the available tool to look up the signature local dish
for Bangkok, which is **Pad Thai**.
```

> 模型成功调用了 `lookup_dish(city="Bangkok")`，拿到 "pad thai"，正确回答。`hit_round_limit` 说明模型用满了 4 轮（工具调用 + 结果 + 后续轮 + 最终总结）——增大 `max_rounds` 或加明确的停止指令可以避免。

---

## 03 · 子代理委托

**概念**：父 agent 通过 `spawn_agent` 把任务委托给子 agent。

### 代码

```python
registry = ToolRegistry()
register_spawn_agent(registry)   # 注入 spawn_agent + run_agent 两个 meta-tool

loop = StatefulAgentLoop(
    llm=make_llm(), store=store, tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt="You are a delegating orchestrator. For any factual "
                     "question, call the `spawn_agent` tool...",
        max_rounds=5, compactor=None,
    ),
)
result = await loop.send(
    "Delegate this: what is the capital of Japan?", session_id=sid,
)
print(f"surviving subs: {store.list_children(result.session_id)}")
```

### 要点

- `register_spawn_agent(registry)` 注入两个 meta-tool：`spawn_agent` 和 `run_agent`
- 父 LLM 自主决定调用 `spawn_agent` → 自动新建子 session，跑独立小循环
- 子结果作为 `tool` 消息回灌父 session
- **EPHEMERAL** 生命周期：成功后子 session 物理删除（失败者保留供调试）
- `store.list_children(parent_sid)` 查看还存活的子 session

### 输出

```
status        : completed, rounds: 2
reply         : The capital of Japan is Tokyo.
surviving subs: []
```

> 子 session 独立运行，查到 "Tokyo"，完成后被清理（EPHEMERAL → 空列表）。

---

## 04 · 上下文压缩

**概念**：`DefaultCompactor` 自动把长历史折叠成摘要。

### 代码

```python
# 灌入大量填充消息
for i in range(4):
    store.append_message(sid, role="user", content="filler " + "u" * 400, round_index=i)
    store.append_message(sid, role="assistant", content="filler ack " + "a" * 400, round_index=i)

# 强制低阈值保证触发
os.environ["CONTEXT_COMPACT_THRESHOLD"] = "500"

loop = StatefulAgentLoop(
    llm=make_llm(), store=store,
    config=AgentLoopConfig(
        system_prompt="Answer the user's latest question concisely.",
        max_rounds=1,
        compactor=DefaultCompactor(trigger_ratio=0.5, keep_last_n=1),
    ),
)
r = await loop.send("Name the largest planet in our solar system.", session_id=sid)

# 查看压缩痕迹
folded = sum(1 for m in all_rows if m.state is MessageState.COMPACTED_OUT)
notes = [m for m in all_rows if m.name == "compact_note"]
```

### 要点

- 触发条件：`estimate_tokens(history) ≥ max_tokens × trigger_ratio`
- `CONTEXT_COMPACT_THRESHOLD` 环境变量可设绝对阈值
- 被折叠的消息标 `state='compacted_out'`
- 插入 `role=system, name=compact_note` 摘要消息
- `compactions` 表新增审计行
- 模型基于 "system + compact_note + 最近尾巴" 继续答题

### 输出

```
status   : completed, rounds: 1
reply    : Jupiter is the largest planet in our solar system.
compactions recorded : 1
messages compacted   : 8
compact_note preview : 'The transcript contains only repeated filler/keep-alive messages...'
```

> 8 条消息（4 轮填充）被折叠成一条摘要。模型根据 compact_note + 最新用户消息正确回答了问题。

---

## 05 · 悬挂态恢复

**概念**：优雅处理工具调用中途的进程崩溃。

### 代码

```python
def _simulate_crash_pending(store, sid):
    """模拟：LLM 已返回但 tool 调用还没跑完进程就挂了。"""
    asst_seq = store.append_message(
        sid, role="assistant",
        tool_calls=[{"id": "tc-stuck", "function": {"name": "echo", "arguments": '{"text":"x"}'}}],
    )
    store.set_pending(sid, {"assistant_seq": asst_seq, ...})

# 直接 send 会抛——协议禁止把悬挂态丢给 LLM
try:
    await loop.send("anything", session_id=sid)
except SessionPendingError as exc:
    print(f"[blocked] pending: {[tc['id'] for tc in exc.pending_tool_calls]}")

# 选择 abort_pending（也可以 await loop.resume(sid)）
n = loop.abort_pending(sid, reason="user_cancelled")

# 现在 send 可以正常往下走
r = await loop.send("What does HTML stand for?", session_id=sid)
```

### 要点

- 协议：`assistant(tool_calls=[A,B])` 之后必须有对应的 `tool` 消息
- 进程在 assistant 落库后、tool 没全部落库时挂掉 → 下次 send 抛 `SessionPendingError`
- **两条恢复路径**：
  - `resume(sid)` — 跑完剩余 tool_calls，继续循环
  - `abort_pending(sid, reason=...)` — 写 `<aborted>` 消息，恢复协议合法性

### 输出

```
[blocked] pending tool_calls: ['tc-stuck']
[abort_pending] aborted 1 tool_call(s); pending now None
[send]  status=completed, reply=HTML stands for HyperText Markup Language.
```

---

## 06 · 声明式子代理

**概念**：父 LLM 提交完整 `AgentSpec` JSON，精确控制子代理。

### 代码

```python
# 直接调用——代码拼 AgentSpec，绕过 LLM 驱动
spec = AgentSpec(
    name="math-helper",
    system_prompt="Compute the expression. Reply with the number only.",
    tools=["calc"],                         # 只白名单 calc
    max_rounds=3, max_tokens=128,
    lifecycle=SubagentLifecycle.LINKED,     # 保留供审计
)
result = await run_agent_spec(spec, "What is 12 * 11?", parent_loop=parent_loop)

# 通过 meta-tool——让父 LLM 自己拼 AgentSpec 调 run_agent
```

### 要点

- **两种子代理入口**：
  - `spawn_agent` — 命令式（kwargs，库构造 AgentSpec），见 [03](#03-子代理委托)
  - `run_agent` — 声明式（父 LLM 提交完整 AgentSpec JSON），本例
- `AgentSpec` 是 **strict-schema**：未知字段 → `AgentSpecError`
- `tools` 是父 registry 的白名单——限制子的可见能力
- `run_agent_spec()` 可直接调用（绕过 LLM，适合测试/编排框架）

### 输出

```
[strict-schema]
  reject unknown field   : unknown AgentSpec field(s): ['evil']
  reject max_rounds=999  : AgentSpec.max_rounds must be in [1, 50]
  valid spec parsed      : name='ok', rounds=3

[direct_call]
  child sid : sess_cdcb10096b6189fe5d46c907  (LINKED → preserved)
  reply     : 132
  surviving children: ['sess_cdcb10096b6189fe5d46c907']

[via_meta_tool]
  reply  : The sub-agent calculated that **(17 + 25) × 3 = 126**.
```

> 直接调用算出 132（12×11）。Meta-tool 调用算出 126（(17+25)×3）。LINKED 生命周期保留了子 session。

---

## 07 · 用户确认

**概念**：通过 `TOOL_BEFORE` hook 在工具执行前拦截，要求用户确认。

### 代码

```python
async def ask_before_bash(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name != "bash":
        return
    cmd = str(ctx.tool_args.get("command") or "")

    if is_safe(cmd):       # ls/pwd/echo/cat → 自动放行
        return

    approved = await confirm_fn(cmd)   # 可以 async——模型真的在等
    if not approved:
        ctx.output = f"[denied by user — command was not executed: {cmd!r}]"
        ctx.directive = HookDirective.SKIP

hooks.register(HookPoint.TOOL_BEFORE, ask_before_bash)
```

### 要点

- `TOOL_BEFORE` hook 是 **async** 的：handler 可以 `await` 任意 UI/CLI 输入
- 模型**真的在等**——没有定时器，没有轮询
- 同意 → 默认 CONTINUE，工具正常跑
- 拒绝 → `ctx.output` 成为工具返回结果；`HookDirective.SKIP` 跳过执行
- 协议保持合法：pending 状态自动清零
- 白名单安全命令（ls/pwd/echo）避免无意义打扰

### 输出

```
[auto-approve] ls -la
[CONFIRM] 'rm README.md' → simulating user input N (deny)

[reply] status=completed, rounds=3
[reply] The deletion of `README.md` was denied, so the file remains in the project.
        I will not retry the command.
[stats] commands actually executed: ['ls -la']
```

> `ls -la` 自动放行并执行。`rm README.md` 被拒——LLM 看到 `[denied by user]` 作为工具结果，自然改变了方向。

---

## 08 · 流式渲染

**概念**：订阅 `STREAM_DELTA` 事件做打字机效果。

### 代码

```python
def on_delta(event: AgentEvent) -> None:
    if not isinstance(event.data, StreamDeltaPayload):
        return
    if event.data.is_think:
        return                     # 跳过 reasoning 流
    print(event.data.text, end="", flush=True)

bus.subscribe(AgentEventType.STREAM_STARTED, on_start)
bus.subscribe(AgentEventType.STREAM_DELTA, on_delta)
bus.subscribe(AgentEventType.STREAM_COMPLETED, on_done)
```

### 要点

- `AgentEventBus` 是只读旁路通道——订阅不影响主循环
- `STREAM_DELTA` 在 LLM 每吐一片 token 时触发
- `stream_id` 区分流（默认 `"main"`）
- `STREAM_THINK_DELTA` 是 reasoning/thinking 段（部分模型才有）
- 订阅可以 sync 或 async；bus 自动判断
- `bus.subscribe(None, fn)` 订阅**所有**事件（debug 用）

### 输出

```
[stream main starting...] HTTP sends data in plaintext, allowing anyone to intercept
and read it. HTTPS encrypts the connection to keep your sensitive information
hidden. It also verifies the website's identity to prevent phishing attacks.
[stream done — 213 chars rendered]

[result] status=completed, final_text len=213
```

---

## 09 · 审计日志

**概念**：订阅所有事件，写入 JSONL 文件。

### 代码

```python
def on_event(event: AgentEvent) -> None:
    record = {
        "ts": time.time(),
        "type": event.type.value,
        "session_id": event.session_id,
        "round_index": event.round_index,
        "payload": event.payload,
    }
    fh.write(json.dumps(record, default=str) + "\n")

bus.subscribe(None, on_event)   # None = 订阅所有事件类型
```

### 要点

- `bus.subscribe(None, fn)` 订阅**所有**事件类型
- `AgentEvent.payload` 是 dict（`data.to_dict()` 自动填充）
- 订阅者错误默认隔离（`suppress_subscriber_errors=True`）
- 适合 hook 到 ELK / Datadog / 自家 audit pipeline

### 输出

```
[reply] audit log demo

[audit] wrote 129 events to /tmp/power_loop_audit.jsonl
[audit] event type histogram:
         109  stream_think_delta
           2  round_started
           2  stream_started
           2  stream_delta
           1  session_started
           1  tool_call_started
           1  tool_call_completed
           1  session_ended
```

> 捕获了 129 个事件。`stream_think_delta` 占大头，因为模型大量使用推理 token。循环的每个阶段（session start → round → stream → tool → round end → session end）都完整可观测。

---

## 10 · 并发会话

**概念**：并发驱动多个 session，配合异步审批队列。

### 代码

```python
# 审批 hook 派发到 asyncio.Queue
async def gate(ctx: ToolBeforeCtx) -> None:
    cmd = str(ctx.tool_args.get("command") or "")
    sid = get_session_id() or "?"
    if cmd.strip().startswith(("ls", "pwd", "echo", "cat ")):
        return                                # 自动放行
    req = ApprovalRequest(session_id=sid, command=cmd)
    await queue.put(req)
    approved = await req.response             # 真的等
    if not approved:
        ctx.output = f"[denied: {cmd!r}]"
        ctx.directive = HookDirective.SKIP

# 三个 session 并发跑
worker = asyncio.create_task(approval_worker(queue, decide=decide))
results = await asyncio.gather(
    drive_session(loop, "S1", "List the project files."),
    drive_session(loop, "S2", "Delete the file named cache.tmp."),
    drive_session(loop, "S3", "Print the current working directory."),
)
await queue.put(_STOP)
```

### 要点

- 一个 `StatefulAgentLoop` 实例**并发驱动多个 session**（每 session 一把 `asyncio.Lock`）
- `asyncio.Queue` 把工具审批异步派发给独立 worker
- 审批 worker 决定速度——主循环**真的会等**，没有 timeout/轮询
- 拒绝路径仍走 `HookDirective.SKIP`（同 07）

### 输出

```
  [gate] auto-approve 333f90: 'pwd'
  [worker] ef8632 → 'rm cache.tmp': DENY
  [worker] 29b7b6 → 'find . -type f | head -100': APPROVE
[S2] done: status=completed, rounds=2
[S3] done: status=completed, rounds=4
[S1] done: status=hit_round_limit, rounds=4

[worker] handled 2 approval request(s)
[sessions] 3 sessions completed
```

> 三个 session 并行运行。`rm cache.tmp` 被拒，`find` 被批准。每个 session 有独立的上下文和工具执行。

---

## 11 · 跨进程恢复

**概念**：把会话持久化到真实 SQLite 文件，在完全不同的进程里恢复。

### 代码

```python
# Phase 1：父进程建 session、塞事实、退出
async def phase1(db_path: str) -> str:
    loop = StatefulAgentLoop(llm=make_llm(), db_path=db_path, ...)
    sid = loop.new_session()
    r = await loop.send("Remember: my name is Alan, favorite number is 37.", session_id=sid)
    loop.close()
    return sid

# Phase 2：子进程打开同一个 db 文件，直接续
async def phase2(db_path: str, sid: str) -> str:
    loop = StatefulAgentLoop(llm=make_llm(), db_path=db_path, ...)
    r = await loop.send("What is my name? What is my favorite number?", session_id=sid)
    loop.close()
    return r.final_text

# 父进程用 subprocess 拉起子进程
subprocess.run([sys.executable, __file__, "phase2", db_path, sid])
```

### 要点

- `db_path="./real_file.db"` 落到**真实文件**（不是 `:memory:`）
- 整个会话活在 SQLite 里：messages / pending / usage / compactions
- 进程退出后，新进程拿着 `session_id` + 同一个 db 路径 → 看到完整历史
- 不需要任何「恢复」API——`SessionStore.open(path)` 无缝接上
- WAL + busy_timeout 让单文件串行复用安全

### 输出

```
[phase1] sid=sess_ae76592b763efa2c61be58da
[phase1] reply=Got it, Alan. I've noted your name and favorite number, 37.

--- parent exits phase1, db left at /tmp/.../real_file.db ---
--- spawning child process for phase2 ---

[phase2] reply=Your name is Alan and your favorite number is 37.
```

> Phase 2 进程**完全不知道** Phase 1 的内存。只打开了同一个 db 文件，LLM 就从持久化的历史里正确召回了事实。

---

## 12 · 重试与取消

**概念**：指数退避重试 LLM 瞬时故障；干净地取消。

### 代码

```python
class FlakyWrap(LLMService):
    """包装真实 LLM；前 N 次抛异常模拟抖动，之后透传。"""
    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError(f"injected transient failure #{self.calls}")
        return await self.inner.complete(request, **kwargs)

# 场景 1：两次失败后第三次成功
config=AgentLoopConfig(
    retry_policy=LLMRetryPolicy(
        max_attempts=4, backoff_initial=0.1, backoff_max=0.3, total_timeout=15,
    ),
)

# 场景 3：retry backoff 期间外部取消
token = CancellationToken()
send_task = asyncio.create_task(loop.send("hi", session_id=sid, stop_event=token))
await asyncio.sleep(0.2)
token.cancel("user_pressed_stop")
```

### 要点

- `LLMRetryPolicy` 指数退避到 `backoff_max` 封顶
- `total_timeout` 跨所有 attempt 累计
- `CancellationToken` 统一所有取消形状（`threading.Event` / `asyncio.Event` / `Callable`）
- Cancel 在 retry backoff 期间**立刻生效**——不等满 backoff
- 三种结果：`completed` / `degraded` / `cancelled`

### 输出

```
── Scenario 1: transient failures, eventually completes ──
  status=completed llm_calls=3 text='OK'
  events: ['llm_retry_attempted', 'llm_retry_attempted']

── Scenario 2: all attempts fail → degraded ──
  status=degraded llm_calls=2
  final_text='[degraded: LLM retry_exhausted — RuntimeError: ...]'
  events: ['llm_retry_attempted', 'llm_degraded']

── Scenario 3: external cancel during retry backoff ──
  status=cancelled llm_calls=1
  events: ['llm_retry_attempted', 'loop_cancelled']
```

> 三条路径都用注入失败的 LLM 包装确定性演示——不依赖真实网络抖动。

---

## 13 · SQLite 记忆

**概念**：通过 `MemoryProvider` 协议实现跨会话事实记忆。

### 代码

```python
class SqliteFactMemory:
    _FACT_RE = re.compile(r"FACT:\s*([A-Za-z_][\w]*)\s*=\s*(.+?)\s*$", re.M)

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        # 从 SQLite 拉全部事实，返回一条 system 消息
        rows = c.execute("SELECT key, value FROM facts").fetchall()
        text = "Known facts:\n" + "\n".join(f"- {r[0]}: {r[1]}" for r in rows)
        return [{"content": text}]

    async def remember(self, *, snapshot: MemorySnapshot, session_id):
        # 从 final_text 抽 FACT: key=value 行入库
        captured = self._FACT_RE.findall(snapshot.final_text or "")
        c.executemany("INSERT INTO facts VALUES (?, ?)", captured)

# Session A 教事实 → Session B 召回
config=AgentLoopConfig(system_prompt=SYSTEM, memory=memory)
```

### 要点

- `MemoryProvider` 两个方法：`recall()`（session 开始）/ `remember()`（session 结束）
- 召回的事实注入为 `role=system, name=memory_*`——扛过压缩
- `recall`/`remember` 任一抛错都不打断主循环 → `MEMORY_FAILED` 事件
- **库内零实现**——让业务策略留在应用层

### 输出

```
── Session A: teach the agent a fact ──
[A] reply=Hello Alan, it is nice to meet you and I have noted your favorite number is 37.
FACT: name=Alan
FACT: favorite_number=37
[memory.db] facts after session A: [('favorite_number', '37'), ('name', 'Alan')]

── Session B: new session — does the agent remember? ──
[B] reply=Your name is Alan and your favorite number is 37.
[B] memory events: ['memory_recalled']
```

> Session B 是**全新的 session**，不同的 `session_id`。`MemoryProvider` 通过 `recall()` 从 SQLite 带回了事实。

---

## 14 · 结构化输出

**概念**：让 LLM 输出匹配 schema 的合法 JSON，自动修复常见缺陷。

### 代码

```python
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "favorite_number": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "favorite_number"],
    "additionalProperties": False,
}

SPEC = StructuredOutputSpec(name="UserCard", schema=SCHEMA, strict=True)

req = LLMRequest(
    messages=[{"role": "user", "content": user_text}],
    system_prompt="Extract the user's profile into JSON.",
    response_format=SPEC.to_openai_response_format(),
)
resp = await llm.complete(req)
card = parse_structured(resp, schema=SCHEMA)
```

### 要点

- `StructuredOutputSpec.to_openai_response_format()` 渲染 OpenAI 兼容的 `response_format`
- `parse_structured()` 自动剥 markdown 围栏、抓第一段 `{...}`、修补尾逗号
- 缺 required 字段 → `StructuredOutputError(reason="missing_required:<field>")`
- 错误带 `raw_text` 和 `reason`——可调试，不会 silent 吞掉

### 输出

```
[ok] card = {
  "name": "Alan",
  "location": "Shanghai",
  "favorite_number": 37,
  "hobbies": ["hiking", "coding", "cooking"]
}

[repair] parsed = {'name': 'Xiao Ming', 'favorite_number': 7}
  ← markdown 围栏剥掉，尾逗号修复

[caught] reason='missing_required:favorite_number' raw_text='{"name": "Xiao Ming"}'
  ← schema 校验捕获了缺失字段
```

---

## 15 · Markdown 技能

**概念**：从 `SKILL.md` 文件加载领域知识注入系统提示词。

### 代码

```python
SKILL_PYTHON = """\
---
name: python-expert
description: Answer Python questions with best practices
---
# Python Expert
1. Always provide a runnable example.
2. Use type hints in all code examples.
"""

def build_system_prompt(*skills: str) -> str:
    parts = [parse_skill_md(s) for s in skills]
    prompt = "You are a helpful assistant with these skills:\n\n"
    for p in parts:
        prompt += f"## {p['name']}\n{p['instructions']}\n\n"
    return prompt

# 组合多个 skill
prompt = build_system_prompt(SKILL_PYTHON, SKILL_SECURITY)
config = AgentLoopConfig(system_prompt=prompt)
```

### 要点

- `SKILL.md` 是最轻量的领域知识注入方式：YAML frontmatter + markdown 正文
- 多个 skill 自由组合成一个 system prompt
- 本例不依赖 `runtime/skills.py`（内部实现）——演示外部加载方式

### 输出

**场景 A**（只有 Python Expert）：模型提供了完整的可运行示例，带 type hints，遵循 skill 指令。

**场景 B**（Python Expert + Security Reviewer）：模型同时遵循两个 skill——先审查 SQL 注入漏洞，再提供修复代码。

---

## 16 · 自定义压缩器

**概念**：实现 `Compactor` 协议，定义自己的压缩策略。

### 代码

```python
class TailOnlyCompactor:
    """只保留最后 N 条消息——最简单的压缩器。"""

    async def maybe_compact(self, messages, *, llm, max_tokens, round_index):
        total = estimate_tokens(messages)
        if total <= max_tokens * self.trigger_ratio:
            return None          # 不需要压缩
        fold_end = n - self.keep - 1
        summary = f"[Compacted {removed} earlier messages.]"
        return CompactionPlan(
            fold_start_idx=fold_start, fold_end_idx=fold_end,
            summary_text=summary, before_tokens=total, after_tokens=...,
        )

compactor = TailOnlyCompactor(keep=4, trigger_ratio=0.2)
config = AgentLoopConfig(system_prompt="...", compactor=compactor)
```

### 要点

- `Compactor` 协议：`async def maybe_compact(...) → CompactionPlan | None`
- `None` = 跳过本轮压缩
- 压缩器在每轮开始前调用；pipeline 自动处理消息折叠和持久化
- 返回 `CompactionPlan(fold_start_idx, fold_end_idx, summary_text, before_tokens, after_tokens)`

### 输出

```
Reply: You drink coffee every morning and your pet's name is Luna.
Rounds: 1
```

> 6 轮事实教学后，TailOnlyCompactor 丢掉了最早的消息。最后 4 个交互保留——足够模型召回 "coffee" 和 "Luna"。

---

## 17 · 自定义记忆提供者

**概念**：HTTP API 后端的 `MemoryProvider`，软失败语义。

### 代码

```python
class MockMemoryAPI:
    """Mock 后端——用内存 dict 代替真实 DB。"""
    async def get(self, user_id, endpoint, payload):
        if endpoint == "/api/memory/recall":
            return {"facts": [{"key": k, "value": v} for k, v in facts.items()]}

class HttpMemoryProvider:
    async def recall(self, *, messages, session_id, budget_tokens=1500):
        try:
            resp = await self.api.get(self.user_id, "/api/memory/recall", {})
            # ... 格式化事实为 system 消息
        except Exception:
            return []     # 软失败：空召回，不崩

    async def remember(self, *, snapshot, session_id):
        try:
            # ... 提取并推送事实
        except Exception:
            pass          # 软失败：静默跳过
```

### 要点

- `recall()` 在 session 开始时调用，`remember()` 在结束时调用
- 失败**绝不**阻塞用户获取回复
- 框架发 `MEMORY_FAILED` 事件，照常返回
- 生产环境把 `MockMemoryAPI` 换成 `httpx` 或 `aiohttp`

### 输出

```
[Session A] reply: It is nice to meet you, Alan, and I have noted your employment at Acme Corp.
FACT: name=alan
FACT: company=acme_corp
[Session A] events: ['memory_recalled']

[Session B] reply: Your name is Alan and you work at Acme Corp.
[Session B] events: ['memory_recalled']
```

> Session A 通过 mock HTTP API 存了事实。Session B 召回了它们——跟真实 HTTP 服务一样的行为。

---

## 18 · 多 Provider

**概念**：按需切换 LLM 供应商（OpenAI / DashScope / DeepSeek）。

### 代码

```python
def _cfg_from_env(prefix: str) -> LLMProviderConfig | None:
    return LLMProviderConfig.from_env(prefix=prefix)

# 用主 provider 跑
primary = _cfg_from_env("POWER_LOOP")
await run_with_provider("Primary", primary, "What color is the sky?")

# 用备用 provider（不同 env 前缀）
alt = _cfg_from_env("ALT_LLM")
await run_with_provider("Alternate", alt, "What is the opposite of hot?")

# 代码构建（无需 env）
manual_cfg = LLMProviderConfig(
    provider="openai", base_url="https://api.openai.com/v1",
    api_key="sk-...", model="gpt-4o-mini",
)
```

### 要点

- `LLMProviderConfig.provider` 是标签，不是路由器——今天都走 OpenAI 兼容传输
- `create_llm_service_from_env(prefix=...)` 支持自定义前缀
- 切换模型只需换 `LLMProviderConfig.model`——不改业务代码

### 输出

```
[Primary] model=qwen3.7-plus
[Primary] reply: Blue

[ALT_LLM] skipped: LLMProviderConfig missing required field(s): base_url, api_key, model

[Manual] cfg.provider=openai, cfg.model=gpt-4o-mini, is_ready=True
```

> Primary provider 回答了 "Blue"。ALT_LLM 被跳过（没配凭证）。Manual config 纯代码构建——有凭证时即可使用。

---

## 19 · 旗舰示例

**概念**：全部功能集合——session 持久化、工具、hook、事件、记忆、压缩。

### 代码

```python
# 工具
REGISTRY.register(ToolDefinition(name="get_weather", ...), get_weather)
REGISTRY.register(ToolDefinition(name="calculator", ...), calculator)

# Hook——安全门
def safety_gate(ctx: ToolBeforeCtx) -> None:
    if any(d in args_str.lower() for d in ("rm -rf", "sudo", "delete all")):
        ctx.output = "[blocked by safety gate]"
        ctx.directive = HookDirective.SKIP

# 事件——流式 + 工具追踪
bus.subscribe(AgentEventType.STREAM_DELTA, on_stream)
bus.subscribe(AgentEventType.TOOL_CALL_STARTED, on_tool)

# 记忆——SQLite 事实库
memory = SqliteFactMemory(mem_path)

# 全功能组合
loop = StatefulAgentLoop(
    llm=llm, db_path=db_path, tool_registry=REGISTRY,
    hooks=HOOKS, event_bus=bus,
    config=AgentLoopConfig(
        system_prompt=SYSTEM, max_rounds=4, memory=memory,
    ),
)

# Session A：工具调用 + 记忆
r1 = await loop.send("What's the weather in Tokyo and what is 15 * 7?", session_id=sid_a)

# Session B：教事实
r2 = await loop2.send("My name is Alan. I live in Shanghai.", session_id=sid_b)

# Session C：从记忆召回
r3 = await loop3.send("What is my name and where do I live?", session_id=sid_c)
```

### 要点

- **会话持久化**：多个 session 共用同一个 `db_path`，通过 `session_id` 延续
- **工具**：`get_weather` + `calculator`——两个自定义工具
- **Hook**：`TOOL_BEFORE` 安全门拦截危险操作
- **事件**：`STREAM_DELTA` 打字机效果，`TOOL_CALL_STARTED` 工具追踪
- **记忆**：SQLite 事实库跨 session 召回
- **压缩**：默认 `DefaultCompactor` 自动压缩长会话

### 输出

```
=== Session A: tool calling + memory ===
User: What's the weather in Tokyo and what is 15 * 7?
- **Weather in Tokyo:** Sunny, 22°C
- **15 × 7 = 105**
[Session A] status=completed, rounds=2
[Session A] tools used: ["get_weather({'city': 'Tokyo'})", "calculator({'expression': '15 * 7'})"]

=== Session B: cross-session memory recall ===
User: My name is Alan. I live in Shanghai.
I can confirm that your name is Alan and you live in Shanghai.
FACT: name=Alan, location=Shanghai
[Session B] status=completed, rounds=1
[Memory] facts stored: {'name': 'Alan, location=Shanghai'}

=== Session C: recall from memory ===
User: What is my name and where do I live?
Your name is Alan and you live in Shanghai.
[Session C] status=completed, rounds=1

[Done] 3 sessions completed.
```

> Session A 在一轮里同时用了两个工具（天气 + 计算器）。Session B 把事实存进了记忆。Session C——全新的 session、新的 `session_id`——从共享的 `SqliteFactMemory` 召回了 Alan 的名字和城市。

---

## 20 · 默认工具

**概念**：不依赖真实 LLM，逐个演示内置默认工具。

### 代码

```python
registry = create_default_tool_registry(preset="full", workspace_dir="/path/to/project")

registry.invoke("write_file", {"path": target, "content": "alpha\nbeta\n"})
registry.invoke("read_file", {"path": target})
registry.invoke("edit_file", {"path": target, "old_text": "beta", "new_text": "BETA"})
registry.invoke("apply_patch", {"path": target, "patch": "@@ -1,2 +1,3 @@\n alpha\n BETA\n+gamma"})
registry.invoke("glob", {"path": rel_root, "pattern": "*.py"})
registry.invoke("grep", {"path": rel_root, "pattern": "VALUE", "include": "*.py"})
registry.invoke("bash", {"command": "python -m py_compile path/to/code.py"})
```

### 要点

- `create_default_tool_registry(preset="full", workspace_dir=...)` 会注册文件、搜索、shell、todo、skill 和后台任务工具。
- 修改已有文件前，必须先用 `read_file` 读取，之后 `write_file`、`edit_file`、`apply_patch` 才会执行。
- 定位文件优先用 `glob`，搜索内容优先用 `grep`；`bash` 更适合测试和构建。
- 这个示例是确定性的，不需要 API 凭证。

---

## 21 · 可恢复人类输入

**概念**：让 loop 暂停等待外部输入，但不阻塞 Python 进程。

### 代码

```python
waiting = await loop.send("Draft and send a summary.", session_id=sid)
interaction = waiting.pending_interactions[0]

result = await loop.submit_input(
    sid,
    interaction["interaction_id"],
    {"choice": "send"},
)
```

### 要点

- `request_user_input` 返回 `status="waiting_for_input"` 和可序列化的 `pending_interactions`。
- 业务方负责把 prompt/options 展示到 UI 或 API；即使跨进程重启，也可以之后调用 `submit_input`。
- `submit_input` 会补上对应 tool message，并用合法的 LLM tool-call 历史继续执行。

---

## 速查表

| 我想… | 看哪个 |
|---|---|
| 发一条消息拿一条回复 | [00](#00-最简示例) |
| 多轮聊天 | [01](#01-多轮对话) |
| 加自定义工具 | [02](#02-工具调用) |
| 委托给子代理 | [03](#03-子代理委托)、[06](#06-声明式子代理) |
| 处理长对话 | [04](#04-上下文压缩)、[16](#16-自定义压缩器) |
| 应对崩溃 | [05](#05-悬挂态恢复)、[11](#11-跨进程恢复) |
| 加用户确认门 | [07](#07-用户确认)、[10](#10-并发会话) |
| 流式渲染 | [08](#08-流式渲染) |
| 审计追踪 | [09](#09-审计日志) |
| 应对 LLM 抖动 | [12](#12-重试与取消) |
| 跨会话记忆 | [13](#13-sqlite-记忆)、[17](#17-自定义记忆提供者) |
| 结构化 JSON 提取 | [14](#14-结构化输出) |
| 注入领域知识 | [15](#15-markdown-技能) |
| 切换多个 LLM 供应商 | [18](#18-多-provider) |
| 试用内置工具 | [20](#20-默认工具) |
| 构建运行时绑定工具 | [高级运行时](../../../examples/advanced_runtime/) |
| 看全部功能 | [19](#19-旗舰示例) |
