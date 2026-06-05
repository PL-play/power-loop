# Hooks 完整参考

[English](en/user-guide/hooks.md) | [回到文档站](README.md)

Hook 是 power-loop 的**控制流**通道。和 [Events](events.md) 不同：
event 只读、用于观测；hook 可以改 messages、改 LLM 请求、改工具输入输出，甚至
让循环跳到下一轮或直接结束。

> ⚠️ Hook 在循环热路径上同步触发。**handler 越短越好**；耗时操作放进后台任务或写
> event 让旁路 worker 处理。

## 目录

- [1. 概念速览](#1-概念速览)
- [2. HookDirective](#2-hookdirective)
- [3. 完整 HookPoint 列表](#3-完整-hookpoint-列表)
  - [3.1 Session](#31-session)
  - [3.2 Round](#32-round)
  - [3.3 LLM](#33-llm)
  - [3.4 Round Decide](#34-round-decide)
  - [3.5 Tools batch](#35-tools-batch)
  - [3.6 Tool（单个）](#36-tool单个)
  - [3.7 Compact](#37-compact)
  - [3.8 Message](#38-message)
  - [3.9 Memory](#39-memory)
- [4. 注册 hook](#4-注册-hook)
- [5. 常见模式](#5-常见模式)

---

## 1. 概念速览

每个 `HookPoint` 对应一个 **typed Ctx 类**（`power_loop.contracts.hook_contexts`），
带 `round_index` 和 `directive` 两个公共字段。handler 收到 ctx → 原地改字段 → 设
`ctx.directive` → 返回（无返回值）。

```python
def on_llm_before(ctx: LlmBeforeCtx) -> None:
    ctx.request.temperature = 0.2          # 改请求
    if "danger" in ctx.system_prompt:
        ctx.directive = HookDirective.BREAK  # 直接结束循环
```

Hook 链是有序的；前一个 handler 设置的 ctx 字段会被下一个 handler 看到。同一
HookPoint 内若任一 handler 返回非 `CONTINUE`，pipeline 立即按 directive 处理，
后续 handler 不再执行。

## 2. HookDirective

| Directive | 语义 |
|---|---|
| `CONTINUE` | 默认。pipeline 按业务逻辑往下走。 |
| `SKIP` | 跳过当前阶段。具体含义因 HookPoint 而异（见下表）。 |
| `BREAK` | 终止整个 loop，按 status="completed" 返回。 |
| `SHORT_CIRCUIT` | 跳过真实操作，用 ctx 里给的"假结果"。 |

并非每个 HookPoint 都支持所有 directive；不支持的会被忽略并按 `CONTINUE` 处理。
下面的表里只列**有效组合**。

## 3. 完整 HookPoint 列表

### 3.1 Session

#### `session.start`

每个 send/resume 进入主循环前触发一次。

| Ctx 字段 | 类型 | 说明 |
|---|---|---|
| `scope` | `str` | 主循环固定 `"main"`，子代理 `"subagent"` |
| `messages` | `list[LoopMessage]` | 当前加载的 active history（可改） |
| `stop_event` | `threading.Event \| None` | 用于外部取消 |

**支持的 directive**：（无）

**典型用途**：
- 注入预热消息或上下文片段（改 `ctx.messages`）
- 给 metrics 系统打 session 开始时间戳

#### `session.end`

main / subagent 结束前触发，read-only。

| 字段 | 类型 | 说明 |
|---|---|---|
| `scope` | `str` | `"main"` / `"subagent"` |
| `reason` | `str` | `"completed"` / `"hit_round_limit"` / `"cancelled"` / `"hook_break"` |
| `messages` | `list[LoopMessage]` | 终态 history |
| `final_text` | `str \| None` | 最终回复 |

**典型用途**：审计、关闭外部连接、写日志。

### 3.2 Round

#### `round.start`

每轮开始时触发，**在** prepare_round（todo 提示 + 压缩）之前。

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `list[LoopMessage]` | 可改 |
| `stop_event` | `threading.Event \| None` | |
| `reason` | `str` | `BREAK` 时填，会写进 session.end.reason |

**支持的 directive**：
- `BREAK` — 直接结束 loop（status="completed"）
- `SKIP` — 跳过本轮，进入下一轮

**典型用途**：上限式预算检查（"这个 session 已花 $0.10，停"）。

#### `round.end`

每轮结束触发，read-only。

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `list[LoopMessage]` | |
| `has_tools` | `bool` | 这一轮 LLM 是否调了工具 |
| `response_text` | `str` | 这一轮 assistant text |
| `used_todo` | `bool` | 是否用了 todo 工具 |

**典型用途**：每轮上报 metrics、push 进度条更新。

### 3.3 LLM

#### `llm.before`

LLM 请求构造完毕、调用前触发。**最常用的 hook**——改请求、注入缓存命中、做安全门都在这里。

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `list[LoopMessage]` | 可改 |
| `system_prompt` | `str` | 可改 |
| `tools` | `list[dict] \| None` | OpenAI tool schemas，可改 |
| `max_tokens` | `int` | 可改 |
| `temperature` | `float` | 可改 |
| `output` | `LLMResponse \| None` | SHORT_CIRCUIT 时必须填 |

**支持的 directive**：
- `SHORT_CIRCUIT` — 跳过真实 LLM 调用，用 `ctx.output` 作为返回；用于缓存命中、mock。
- `BREAK` — 终止 loop。

**典型用途**：
```python
def cache_hit(ctx: LlmBeforeCtx) -> None:
    key = hash_of(ctx.messages, ctx.system_prompt)
    cached = redis.get(key)
    if cached:
        ctx.output = LLMResponse(raw_text=cached)
        ctx.directive = HookDirective.SHORT_CIRCUIT
```

#### `llm.after`

LLM 返回、未解析 tool_calls 前触发。

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `list[LoopMessage]` | |
| `output` | `LLMResponse \| None` | 可替换为新的 LLMResponse |

**支持的 directive**：`BREAK`（终止 loop，把 assistant text 作为 final_text）。

**典型用途**：基于 LLM 输出内容做内容审核 / 立即终止。

### 3.4 Round Decide

#### `round.decide`

LLM 已返回、要不要执行工具的决策点。

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `list[LoopMessage]` | |
| `tool_calls` | `list[dict]` | 即将执行的工具调用 |
| `assistant_text` | `str` | |
| `output` | `str` | SKIP 时作为所有工具的占位结果 |

**支持的 directive**：
- `SKIP` — 跳过本轮所有工具执行，把 `ctx.output` 作为每个 tool 的结果。
- `BREAK` — 终止 loop。

### 3.5 Tools batch

#### `tools.batch.before`

一批工具执行前触发。

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `list[LoopMessage]` | |
| `tool_calls` | `list[dict]` | |
| `output` | `str` | SKIP 时作为所有工具的占位结果 |

**支持的 directive**：`SKIP`（跳过整批）。

**典型用途**：全局工具关闭开关、batch-level 限流。

#### `tools.batch.after`

read-only。

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `list[LoopMessage]` | |
| `used_todo` | `bool` | |

### 3.6 Tool（单个）

#### `tool.before`

每个工具调用前触发。

| 字段 | 类型 | 说明 |
|---|---|---|
| `tool_call` | `dict` | 原始 OpenAI tool_call |
| `tool_name` | `str` | 可改（路由 / 重命名） |
| `tool_args` | `dict` | 可改 |
| `output` | `str` | SKIP 时作为该工具的结果 |

**支持的 directive**：`SKIP`（跳过这个工具，用 `ctx.output`）。

**典型用途**：
```python
def block_dangerous(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name == "bash" and "rm -rf" in ctx.tool_args.get("cmd", ""):
        ctx.output = "[blocked: destructive command]"
        ctx.directive = HookDirective.SKIP
```

#### `tool.after`

工具调用成功返回后触发。

| 字段 | 类型 | 说明 |
|---|---|---|
| `tool_call` | `dict` | |
| `tool_name` | `str` | |
| `tool_args` | `dict` | |
| `output` | `str` | 可改（替换/裁剪结果） |
| `failed` | `bool` | 可改 |

**支持的 directive**：`BREAK`（不再执行剩余 tool_calls，直接进入 round.end）。

#### `tool.error`

工具抛异常时触发。

| 字段 | 类型 | 说明 |
|---|---|---|
| `tool_call` | `dict` | |
| `tool_name` | `str` | |
| `tool_args` | `dict` | |
| `error` | `Exception \| None` | |
| `error_message` | `str` | |
| `output` | `str` | SKIP 时作为该工具的结果 |

**支持的 directive**：
- `SKIP` — 吞掉异常，用 `ctx.output` 作为结果。
- `SHORT_CIRCUIT` — 重试整个工具调用。

### 3.7 Compact

#### `compact.before`

`Compactor.maybe_compact` 即将运行前触发。

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `list[LoopMessage]` | 当前 history |

**支持的 directive**：`SKIP`（跳过本轮压缩）。

**典型用途**：业务方对某些 session 强制关闭压缩，或注入自定义 compactor。

#### `compact.after`

压缩完成、history 已被替换后触发，read-only。

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `list[LoopMessage]` | 压缩后 history |
| `messages_before_count` | `int` | |
| `messages_after_count` | `int` | |

### 3.8 Message

#### `message.append`

每条消息（user / assistant / tool / system）被加入 history 之前触发。

| 字段 | 类型 | 说明 |
|---|---|---|
| `message` | `dict` | 可改（脱敏、加 metadata） |
| `session_id` | `str \| None` | |

**典型用途**：
- 写 audit log
- 对 tool 输出做 PII 脱敏
- 给消息加 `_meta` 字段供后续 hook 使用

### 3.9 Memory

#### `memory.recalled`

`MemoryProvider.recall()` 返回之后、注入 history 之前触发。**M1.9 起可用。**

| 字段 | 类型 | 说明 |
|---|---|---|
| `recalled` | `list[dict]` | 待注入的 memory 消息（可改：过滤 / 重排 / 去敏） |
| `session_id` | `str \| None` | |
| `budget_tokens` | `int` | recall 时传入的 token 预算 |

**支持的 directive**：`SKIP`（丢弃整批 memory，不注入任何消息）。

**典型用途**：双方授权 gate（"这个 session 应当看到记忆吗？"）；PII 去敏后再注入。

```python
def redact_memory(ctx: MemoryRecalledCtx) -> None:
    if not user_has_consented(ctx.session_id):
        ctx.directive = HookDirective.SKIP
        return
    for m in ctx.recalled:
        m["content"] = redact_pii(m.get("content", ""))
```

---

## 4. 注册 hook

```python
from power_loop import StatefulAgentLoop, AgentHooks, HookPoint, HookDirective
from power_loop.contracts.hook_contexts import ToolBeforeCtx

hooks = AgentHooks()

def safety_filter(ctx: ToolBeforeCtx) -> None:
    if "secret" in str(ctx.tool_args):
        ctx.output = "[blocked]"
        ctx.directive = HookDirective.SKIP

hooks.register(HookPoint.TOOL_BEFORE, safety_filter)

loop = StatefulAgentLoop(llm=..., db_path="...", hooks=hooks, ...)
```

支持 `def` 和 `async def`；pipeline 自动判断并 await。

## 5. 常见模式

### 安全门

```python
def block_destructive(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name == "bash":
        cmd = ctx.tool_args.get("command", "")
        if any(bad in cmd for bad in ("rm -rf", "sudo", ":(){:|:&};:")):
            ctx.output = "[blocked: destructive]"
            ctx.directive = HookDirective.SKIP

hooks.register(HookPoint.TOOL_BEFORE, block_destructive)
```

### 用户交互中断（async confirm）

`TOOL_BEFORE` 是 async 的，handler 里可以 `await` 任意 UI / WebSocket / CLI 输入，
模型会**真的在等**——没有定时器，没有 polling。同意 → CONTINUE 放行；拒绝 →
`SKIP + ctx.output`，pipeline 把 output 当成工具结果回灌给 LLM，pending 状态自
动清零。

```python
async def ask_before_bash(ctx: ToolBeforeCtx) -> None:
    if ctx.tool_name != "bash":
        return
    cmd = ctx.tool_args.get("command", "")
    if not await your_confirm_ui(cmd):     # await 用户决定
        ctx.output = f"[denied by user: {cmd!r}]"
        ctx.directive = HookDirective.SKIP

hooks.register(HookPoint.TOOL_BEFORE, ask_before_bash)
```

完整可运行版（含白名单 / always-deny 注入式回调 / CLI input 默认实现）：
[`examples/07_user_confirmation.py`](../examples/07_user_confirmation.py)。

### 缓存短路

```python
def llm_cache(ctx: LlmBeforeCtx) -> None:
    key = sha256(json.dumps(ctx.messages).encode()).hexdigest()
    if (cached := redis.get(key)) is not None:
        ctx.output = LLMResponse(raw_text=cached.decode())
        ctx.directive = HookDirective.SHORT_CIRCUIT
```

### 预算上限

```python
total_tokens = 0

def cap_budget(ctx: RoundStartCtx) -> None:
    if total_tokens > 50_000:
        ctx.reason = "budget_exhausted"
        ctx.directive = HookDirective.BREAK

hooks.register(HookPoint.ROUND_START, cap_budget)
```

### 工具结果裁剪

```python
def trim_huge_output(ctx: ToolAfterCtx) -> None:
    if len(ctx.output) > 8000:
        ctx.output = ctx.output[:8000] + "\n...[truncated]"
```

### 提前结束（content moderation）

```python
def block_pii(ctx: LlmAfterCtx) -> None:
    text = getattr(ctx.output, "raw_text", "") or ""
    if PII_RE.search(text):
        ctx.output.raw_text = "[response withheld: contained PII]"
        ctx.directive = HookDirective.BREAK
```

---

需要观测而非控制？看 [`docs/events.md`](events.md)。
