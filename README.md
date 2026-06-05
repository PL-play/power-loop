# power-loop

> **可嵌入的、有状态的 Agent 执行内核。** 调用方只管「给一段最新输入 + session_id」，
> power-loop 自治管理：LLM 多轮循环、工具调用、上下文压缩、子代理、消息持久化（SQLite）、
> 悬挂态恢复。
>
> Python ≥ 3.10 · MIT · OpenAI 兼容 + Anthropic 双家底 · 零强依赖（SQLite 用 stdlib）

```python
from power_loop import StatefulAgentLoop, AgentLoopConfig

loop = StatefulAgentLoop(
    llm=my_llm,
    db_path="./sessions.db",
    config=AgentLoopConfig(system_prompt="You are helpful.", max_rounds=8),
)
r1 = await loop.send("hello")                            # 自动新建 session
r2 = await loop.send("more please", session_id=r1.session_id)  # 续话
loop.close_session(r1.session_id)                        # 物理删除
```

---

## 目录

- [1. 它是什么 / 不是什么](#1-它是什么--不是什么)
- [2. 安装](#2-安装)
- [3. Quickstart](#3-quickstart)
- [4. 核心概念](#4-核心概念)
- [5. API 参考](#5-api-参考)
- [6. Examples](#6-examples)
- [7. 配置（环境变量）](#7-配置环境变量)
- [8. 内部机制](#8-内部机制)
- [9. 测试](#9-测试)
- [10. Roadmap & Changelog](#10-roadmap--changelog)

---

## 1. 它是什么 / 不是什么

**是**：
- 一个 Python 库（非服务），可被任意后端 / CLI / 测试 `import` 使用。
- **唯一公开入口** `StatefulAgentLoop`：有状态、`send/resume/abort_pending/close_session` 四个动词。
- 持久化层 `SessionStore`：SQLite，5 张表，承诺消息顺序、悬挂态、压缩审计的所有不变量。
- 工具循环 + 生命周期 Hook + 事件总线 + 子代理（命令式 + 声明式 `AgentSpec`）。
- 上下文压缩：可插拔 `Compactor` 协议，自带 `DefaultCompactor` 默认开启。

**不是**：
- 不是 IM / 业务服务。不感知会话、用户、Kafka、HTTP；这些放在调用方。
- 不是大而全的 Agent Framework。没有内置 RAG / 向量库 / Planner / DAG。
- 不绑死任何模型厂商：`base_url` / `api_key` / `model` 都通过配置传入。

---

## 2. 安装

```bash
pip install -e .                  # 开发安装
pip install -e ".[dev]"           # 含 pytest / ruff / mypy
```

依赖（见 `pyproject.toml`）：`openai`、`anthropic`、`socksio`、`python-dotenv`、`pyyaml`、`rich`、`pypdf`。**SQLite 用 Python 标准库**，零新增。

---

## 3. Quickstart

本节按由浅入深的顺序排：每一步只引入一个新概念。每个代码片段都对应 `examples/`
下一个可独立运行的文件。先按顺序读完，再回头看 §4 的核心概念会很顺。

### 3.1 第一次发送（最小用法）

最少的样板：构造 LLM → 构造 loop → `send`。

```python
import asyncio
from power_loop import StatefulAgentLoop

async def main():
    loop = StatefulAgentLoop(llm=my_llm, db_path=":memory:")
    result = await loop.send("In one sentence: what is HTTP?")
    print(result.final_text)

asyncio.run(main())
```

要点：
- `db_path=":memory:"` 是临时 store；生产换成文件路径就能跨进程保留。
- 不传 `session_id` → 自动创建新 session。返回的 `result.session_id` 是后续续话的钥匙。
- 没传 `AgentLoopConfig` 也行：全部用默认值（`max_rounds=24`、`DefaultCompactor()` 等）。

→ 完整版：[`examples/00_minimal.py`](examples/00_minimal.py)

### 3.2 多轮对话

唯一新东西：把上一轮返回的 `session_id` 传回去。

```python
r1 = await loop.send("My favorite color is teal.")
r2 = await loop.send("What did I just say?", session_id=r1.session_id)
# r2.final_text 会引用 "teal"
```

power-loop 会自动从 store 加载历史，模型每轮看到的都是完整上下文。你不用维护
"messages list"，只管最新的一句输入。

→ 完整版：[`examples/01_multi_turn.py`](examples/01_multi_turn.py)

### 3.3 工具调用

要让模型调用你的 Python 函数：写 `ToolDefinition` + handler，注册到 `ToolRegistry`，
传给 loop。

```python
from power_loop import ToolDefinition, ToolRegistry, AgentLoopConfig

def lookup_dish(**kwargs) -> str:
    return {"lima": "ceviche", "tokyo": "sushi"}.get(kwargs["city"].lower(), "?")

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="lookup_dish",
        description="Return the local dish for a city.",
        input_schema={"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
        required_params=("city",),
    ),
    lookup_dish,
)

loop = StatefulAgentLoop(
    llm=my_llm, db_path=":memory:", tool_registry=registry,
    config=AgentLoopConfig(max_rounds=4),
)
result = await loop.send("What's Lima's signature dish?")
```

为什么 `max_rounds ≥ 2`：工具调用本质是两步——第 1 轮 LLM 决定调工具，第 2 轮看
到工具结果给最终答案。`max_rounds=1` 跑不通工具。

→ 完整版：[`examples/02_tool_use.py`](examples/02_tool_use.py)

### 3.4 子代理

`register_spawn_agent(registry)` 一行注入两个 meta-tool：`spawn_agent`（命令式）
和 `run_agent`（声明式 `AgentSpec`）。父 LLM 自主决定调用，子会话在同一个
`SessionStore` 里独立跑完，把 `final_text` 当 tool 结果回灌给父。

```python
from power_loop import register_spawn_agent

registry = ToolRegistry()
register_spawn_agent(registry)
loop = StatefulAgentLoop(
    llm=my_llm, db_path=":memory:", tool_registry=registry,
    config=AgentLoopConfig(
        system_prompt="Delegate factual Qs via spawn_agent.",
        max_rounds=5,
    ),
)
result = await loop.send("Delegate this: capital of Japan?")
```

→ 完整版：[`examples/03_subagent.py`](examples/03_subagent.py)

### 3.5 上下文压缩

历史变长后默认压缩自动触发：被折叠的消息在 store 里标 `compacted_out`，
插入一条 `compact_note` 摘要替代。开关、阈值、保留条数都在 `DefaultCompactor`
构造参数里。完整版演示了如何检查 `store.list_compactions(sid)` 看审计行。

→ [`examples/04_compaction.py`](examples/04_compaction.py)

### 3.6 悬挂态恢复

如果进程在 `assistant(tool_calls)` 已落库、`tool` 消息还没全部落库时挂掉，session
处于悬挂态。下次 `send` 会抛 `SessionPendingError`，由调用方选 `resume` 还是
`abort_pending`。

→ [`examples/05_pending_resume.py`](examples/05_pending_resume.py)

---

## 4. 核心概念

### Session

一次 send/resume 循环跑在一个 **session** 上。session 是 `SessionStore` 里 sessions 表的一行，承载 system prompt、AgentLoopConfig 快照、metadata 与所有 messages。`send(user_input)` 不传 `session_id` 自动新建；传则续话。

### SessionStore

power-loop 的**唯一**持久化入口（SQLite）。5 张表：

| 表 | 作用 |
|---|---|
| `sessions` | 元数据 + 父子链接 + 生命周期 |
| `messages` | 每条消息 + `state ∈ {active, compacted_out}` + seq |
| `compactions` | 每次压缩的审计行（覆盖了哪些 seq → 哪个 note） |
| `usage_rounds` | 每轮 token 用量 |
| `session_state` | next_seq / round_index / pending |

并发：单连接 + `threading.RLock`；SQLite WAL 模式，允许多读者跨进程共享文件。

### Sink

`MessageSink` 是 pipeline 与持久化之间的协议。`StatefulAgentLoop` 默认装 `SQLiteSink`，把每条消息、压缩、usage 持久化到 store。NullSink 是测试用的 no-op。

### Compactor

可插拔的上下文压缩策略。`AgentLoopConfig.compactor` 默认为 `DefaultCompactor()`；传 `None` 关闭。

**触发**：`estimate_tokens(history) ≥ max_tokens × trigger_ratio`（默认 0.75）；
环境变量 `CONTEXT_COMPACT_THRESHOLD` 可设绝对值覆盖。

**不变量**（DefaultCompactor 实现，自定义 Compactor 应遵守）：
1. 保留所有 `role=system` 消息（含先前 `compact_note`）；
2. 保留尾部 `keep_last_n` 个 user 段（默认 4）；
3. **绝不切开 `assistant(tool_calls) ↔ tool(tool_call_id=…)` 原子对**；
4. 摘要 LLM 抛错 → 返回 `None` → 主循环用未压缩 history 继续（软降级）。

### Pending 状态机

一轮工具调用的协议是：`assistant(tool_calls=[A,B])` → `tool(tool_call_id=A)` + `tool(tool_call_id=B)`。

如果进程在 assistant 已落库、tool 还没全部落库时挂掉，session 处于**悬挂态**。下次 `send` 会抛 `SessionPendingError`。调用方两个选项：
- `await loop.resume(sid)` — 把剩余 tool_calls 跑完，继续循环；
- `loop.abort_pending(sid, reason="…")` — 给每个未完成 tool_call 写一条 `<aborted: reason>` tool 消息，恢复协议合法性，再 `send` 即可继续。

### 子代理

- `spawn_agent(task, ...)` — 命令式 meta-tool，LLM 用 kwargs 调用，自动包成 `AgentSpec`。
- `run_agent(spec, input)` — 声明式 meta-tool，LLM 提交完整 spec（严格 schema，未知字段拒绝）。

两者都走同一份内部实现 `run_agent_spec`，差异只在入口形态。子会话与父共享同一个 `SessionStore`，建立 `parent_session_id` / `spawn_tool_call_id` 链接，`spawn_depth ≤ 3` 强校验。

**生命周期** `SubagentLifecycle`：
- `EPHEMERAL`（默认）：子 session 完成时物理删除；非完成态（hit_round_limit / cancelled）保留供 debug；
- `LINKED`：保留；父 `close_session(cascade=True)` 时随之级联删；
- `DETACHED`：保留；父 close 时不影响（解链）。

### Hooks & Events

两条互不污染的通道：

- **Hooks**（控制流）：15 个 `HookPoint`，每个 hook 返回 `HookDirective ∈ {CONTINUE/SKIP/BREAK/SHORT_CIRCUIT}`。改 LLM 请求、注入 mock 结果、安全门、提前终止都靠它。完整 hook 表与示例：[`docs/hooks.md`](docs/hooks.md)。
- **Events**（旁路只读）：`AgentEventBus` 发布 token usage / stream delta / tool call started / completed / 等。指标、审计、UI 推送都订阅它。完整 event 表与示例：[`docs/events.md`](docs/events.md)。

---

## 5. API 参考

### `StatefulAgentLoop`

唯一公开入口。一个实例可并发驱动多个 session（每 session 一把 `asyncio.Lock`）。

```python
StatefulAgentLoop(
    *,
    llm: LLMService,
    store: SessionStore | None = None,          # 传 None → 用 db_path 自建
    db_path: str = "./power_loop_sessions.db",
    config: AgentLoopConfig | None = None,
    tool_registry: ToolRegistry | None = None,
    hooks: AgentHooks | None = None,
    event_bus: AgentEventBus | None = None,
)
```

| 方法 | 说明 |
|---|---|
| `await send(user_input, session_id=None, *, metadata=None, stop_event=None) -> StatefulResult` | 主入口。无 `session_id` 自动新建；悬挂态 → `SessionPendingError`。 |
| `send_sync(...)` | 同上的同步壳。 |
| `await resume(session_id)` | 把悬挂的 tool_calls 跑完，继续循环。 |
| `abort_pending(session_id, *, reason="aborted") -> int` | 给悬挂 tool_calls 写 `<aborted>` tool 消息，返回 abort 数量。 |
| `close_session(session_id, *, cascade=True) -> int` | 物理删除 session（含 LINKED 子树）。 |
| `close()` | 关 store（若 owned）。不删数据。 |
| `get_messages(session_id, *, include_compacted=False) -> list[dict]` | 取当前 active history。 |
| `get_pending(session_id) -> dict \| None` | 查悬挂态。 |

### `StatefulResult`

```python
@dataclass
class StatefulResult:
    session_id: str
    status: str                       # "completed" / "hit_round_limit" / "cancelled" / "pending_tools"
    final_text: str = ""
    rounds: int = 0
    pending_tool_calls: list[dict] = []
```

### `AgentLoopConfig`

```python
@dataclass
class AgentLoopConfig:
    system_prompt: str | None = None
    max_rounds: int = 24
    temperature: float | None = 0.0
    max_tokens: int | None = 8000
    compactor: Compactor | None = DefaultCompactor()   # 传 None 关闭压缩
```

### `SessionStore`

```python
SessionStore.open(path="./power_loop_sessions.db") -> SessionStore
store.close()
store.create_session(*, system_prompt=None, model=None, config=None,
                     parent_session_id=None, spawn_tool_call_id=None,
                     kind=SessionKind.ROOT,
                     lifecycle=SubagentLifecycle.EPHEMERAL,
                     metadata=None, session_id=None) -> str
store.get_session(sid) -> SessionRow | None
store.list_children(parent_sid) -> list[SessionRow]
store.close_session(sid, *, cascade=True) -> int   # 物理删除，返回行数
store.archive_session(sid)                          # 改 status，不删
store.append_message(sid, *, role, content=None, tool_calls=None,
                      tool_call_id=None, name=None, round_index=None,
                      meta=None) -> int             # 返回新 seq
store.load_active_messages(sid) -> list[MessageRow]
store.load_all_messages(sid) -> list[MessageRow]    # 含 compacted_out
store.record_compaction(sid, *, from_seq, to_seq, note_content,
                         before_tokens, after_tokens, round_index) -> tuple[int, int]
store.list_compactions(sid) -> list[CompactionRow]
store.record_usage(sid, *, round_index, prompt_tokens,
                    completion_tokens, total_tokens, model=None)
store.get_state(sid) -> SessionStateRow | None
store.set_pending(sid, pending: dict | None)
```

### 子代理：`AgentSpec` + 工具

```python
@dataclass(frozen=True)
class AgentSpec:
    name: str
    system_prompt: str
    tools: list[str] | None = None        # parent tool whitelist, None=inherit all
    max_rounds: int = 8                    # 1..50
    max_tokens: int = 4000
    temperature: float = 0.0
    model: str | None = None
    lifecycle: str = "ephemeral"           # "ephemeral" / "linked" / "detached"
    metadata: dict[str, Any] = {}
# 工厂：AgentSpec.from_dict(d) / AgentSpec.from_json(s)
# 严格 schema：未知字段 / 非法 lifecycle / max_rounds 越界 → AgentSpecError

from power_loop import register_spawn_agent
register_spawn_agent(registry, *, include_run_agent=True, overwrite=False)
```

直接 API（绕过 meta-tool）：

```python
from power_loop import run_agent_spec
result = await run_agent_spec(spec, "user input", parent_loop=loop)
# → {"session_id", "status", "final_text", "rounds", "depth"}
```

### Errors

```python
class PowerLoopError(Exception): ...
class SessionNotFoundError(PowerLoopError):
    session_id: str
class SessionPendingError(PowerLoopError):
    session_id: str
    assistant_seq: int
    pending_tool_calls: list[dict]
```

### Compactor

```python
@dataclass(frozen=True)
class CompactionPlan:
    fold_start_idx: int    # inclusive
    fold_end_idx: int      # inclusive
    summary_text: str
    before_tokens: int
    after_tokens: int

class Compactor(Protocol):
    async def maybe_compact(self, messages, *, llm, max_tokens, round_index) -> CompactionPlan | None: ...

class DefaultCompactor:
    def __init__(self, *, trigger_ratio=0.75, keep_last_n=4,
                  summary_max_tokens=512, summary_llm=None, absolute_threshold=None): ...
```

自定义 compactor 实现上面的 Protocol 即可注入 `AgentLoopConfig.compactor=YourCompactor()`。

### Public API 稳定性约定

power-loop 采用 **三层分级**，与 `power_loop/__init__.py` 的 `STABLE_API` 元组同步：

#### STABLE（跨 minor 保证向后兼容）

破坏性变更必须升 minor 版本号（0.x → 0.x+1）+ CHANGELOG 独立条目。**业务方只应依赖这些符号。**

| 符号 | 一句话 |
|---|---|
| `StatefulAgentLoop` | 主入口：`send()` / `resume()` / `abort_pending()` |
| `StatefulResult` | `send()` 返回值：`session_id` / `status` / `final_text` / `rounds` |
| `AgentLoopConfig` | 配置单：`system_prompt` / `max_rounds` / `compactor` / `retry_policy` / `memory` / … |
| `AgentLoopResult` | Pipeline 内部返回值（`status` / `final_text` / `rounds` / `messages`） |
| `SessionStore` | SQLite 持久化：`open(path)` / `create_session()` / `append_message()` / … |
| `SubagentLifecycle` | Enum：`EPHEMERAL` / `LINKED` / `DETACHED` |
| `PowerLoopError` | 所有异常的基类，`except PowerLoopError` 一把抓 |
| `SessionNotFoundError` | `session_id` 不在 store 里 |
| `SessionPendingError` | 上次崩溃留下未完成的 `tool_calls` |
| `LLMTimeout` | LLM 调用（或一系列 retry）超 `total_timeout` |
| `LLMRetryExhausted` | `max_attempts` 次重试仍未成功 |
| `CancellationRequested` | `CancellationToken` 已 flip |
| `ToolNotFound` | 调用了一个未注册的 tool 名字 |
| `ToolValidationError` | tool args 未通过 schema / required 校验 |
| `SpecValidationError` | `AgentSpec` 严格 schema 拒绝（`AgentSpecError` 的父类） |
| `LLMRetryPolicy` | 重试策略：`max_attempts` / `backoff_*` / `total_timeout` / `retry_on` |
| `CancellationToken` | 统一 cancel 形状：`from_any(ev)` / `cancel(reason)` |
| `AgentHooks` | Hook 管理器：`register(pt, fn)` / `register_async(pt, async_fn)` |
| `AgentEventBus` | 事件总线：`subscribe(type, fn)` / `publish(event)` |
| `HookPoint` | Enum：`SESSION_START` … `MEMORY_RECALLED`（18 个） |
| `HookDirective` | Enum：`CONTINUE` / `SKIP` / `BREAK` / `SHORT_CIRCUIT` |
| `ToolRegistry` | 工具注册表：`register(def, handler)` / `invoke_async(name, args)` |
| `ToolDefinition` | 工具声明：`name` / `description` / `input_schema` / `required_params` |

#### PROVISIONAL（0.x 阶段可能调整）

从 `power_loop` 顶层导入，但**不在 STABLE 列表中**。生产代码引用前确认版本号。

例：`MessageSink` / `SQLiteSink` / `AgentSpec` / `run_agent_spec` / `MemoryProvider` / `MemorySnapshot` / `StructuredOutputSpec` / `parse_structured` / `trim_history` / `LLMProviderConfig` / 全部 `*Payload` / 全部 `*Ctx` 等。

#### INTERNAL（无版本承诺）

从 `power_loop.core.*` / `power_loop.runtime.*` 等子模块导入的符号视为 internal，可随时变更或删除。Pipeline / Runner / ContextManager 等都在这一层。

---

## 6. Examples

`examples/` 下每个文件可独立 `python examples/NN_*.py` 运行，并由 `tests/real/test_examples.py` 持续验证。

推荐按编号顺序读：每个文件只引入一个新概念。

| 文件 | 你会学到 |
|---|---|
| [`00_minimal.py`](examples/00_minimal.py) | 最小用法：`StatefulAgentLoop(llm=…).send(text)` |
| [`01_multi_turn.py`](examples/01_multi_turn.py) | 用 `session_id` 续话 + `get_messages` / `close_session` |
| [`02_tool_use.py`](examples/02_tool_use.py) | 自定义 `ToolDefinition` + 多轮工具调用 |
| [`03_subagent.py`](examples/03_subagent.py) | `spawn_agent` meta-tool + EPHEMERAL 自动清理 |
| [`04_compaction.py`](examples/04_compaction.py) | `DefaultCompactor` 自动折叠 + 查看 store 审计行 |
| [`05_pending_resume.py`](examples/05_pending_resume.py) | `SessionPendingError` + `resume` / `abort_pending` |
| [`06_declarative_subagent.py`](examples/06_declarative_subagent.py) | `AgentSpec` 严格 schema + `run_agent` meta-tool + 直接调 `run_agent_spec` |
| [`07_user_confirmation.py`](examples/07_user_confirmation.py) | 用 async `TOOL_BEFORE` hook 实现「执行前问用户」中断 |
| [`08_streaming.py`](examples/08_streaming.py) | 订阅 `STREAM_DELTA` event 做打字机渲染 |
| [`09_audit_log.py`](examples/09_audit_log.py) | `bus.subscribe(None, …)` 全量审计写 JSONL |
| [`10_async_approval_queue.py`](examples/10_async_approval_queue.py) | 多并发 session + asyncio.Queue 审批 worker |
| [`11_persistence.py`](examples/11_persistence.py) | `db_path` 跨进程恢复：子进程拿同一个 SQLite 文件续上 |
| [`12_retry_and_cancel.py`](examples/12_retry_and_cancel.py) | `LLMRetryPolicy` + 注入失败 → 重试 / degraded / cancel 三条路径 |
| [`13_memory_sqlite.py`](examples/13_memory_sqlite.py) | `MemoryProvider` 跨 session SQLite 事实记忆 |
| [`14_structured_card.py`](examples/14_structured_card.py) | `StructuredOutputSpec` + `parse_structured` 抽取 JSON 卡片 |

`examples/_helpers.py` 是共享的 `.env` 读取 + LLM 构造辅助，每个示例 `from _helpers import make_llm`，省掉 boilerplate。复制到自己项目时把那两行内联即可。

---

## 7. 配置（环境变量）

LLM 凭证与端点 **不入代码**，统一走环境变量（建议 `.env` + `python-dotenv`）：

**推荐**（`POWER_LOOP_*`，M1.4 起）：

| 变量 | 说明 |
|---|---|
| `POWER_LOOP_BASE_URL` | OpenAI 兼容端点 |
| `POWER_LOOP_API_KEY` | API key |
| `POWER_LOOP_MODEL` | 默认模型名 |
| `POWER_LOOP_PROVIDER` | 标签（openai / dashscope / deepseek / …） |

向后兼容 `OPENAI_COMPAT_*`（旧 `.env` 不改名继续工作）。详见 [`docs/providers.md`](docs/providers.md)。

构造 LLM 一行：

```python
from power_loop import create_llm_service_from_env
llm = create_llm_service_from_env()  # 读 POWER_LOOP_*（回退到 OPENAI_COMPAT_*）
```

旧方式（仍可用）：

```python
from llm_client.interface import OpenAICompatibleChatConfig
from llm_client.llm_factory import OpenAICompatibleChatLLMService
import os

cfg = OpenAICompatibleChatConfig(
    base_url=os.environ["OPENAI_COMPAT_BASE_URL"],
    api_key=os.environ["OPENAI_COMPAT_API_KEY"],
    model=os.environ["OPENAI_COMPAT_MODEL"],
    max_tokens=512, temperature=0.2,
)
llm = OpenAICompatibleChatLLMService(cfg)
```

---

## 8. 内部机制

完整版（含 Mermaid 架构图、序列图、状态机）：[`docs/architecture.md`](docs/architecture.md)。
本节是浓缩。

### Pipeline 一回合

```
session.start
  ↓
for round in 0..max_rounds:
  ├ round.start  →  sink.on_round_started
  ├ prepare_round
  │   ├ todo reminder（每 5 轮）
  │   ├ microcompact（大 tool 输出溢盘到 .cache/）
  │   └ compactor.maybe_compact → 命中则 sink.on_compaction → 持久化
  ├ llm.before  →  LLM.complete  →  llm.after
  ├ assistant 消息落 sink（带 tool_calls 时立即 set_pending）
  ├ 若无 tool_calls → round.end → 返回 "completed"
  ├ round.decide
  ├ tools.batch.before
  │   ├ tool.before  →  tool.invoke  →  tool.after / tool.error
  │   └ tool 消息落 sink（同 tool_call_id 解 pending）
  ├ tools.batch.after
  └ round.end  →  sink.on_round_ended(usage=…)
session.end
```

15 个 HookPoint：见 `power_loop/contracts/hooks.py`。

### 消息持久化与 seq

每条消息在 store 里有唯一 `(session_id, seq)`。`SQLiteSink` 在内存里维护一份 `_history_seqs` 与 pipeline.history 一一对应：
- 加载老 session：`init_history_seqs([row.seq for row in active_rows])`
- 新追加：`store.append_message` 返回 seq，追加到尾
- 压缩：`on_compaction(fold_start_idx, fold_end_idx, …)` → 用索引转 seq → `store.record_compaction` → 重写 `_history_seqs`（折叠区间替换为 note 的 seq）

### Pending 状态机

```
LLM 返回 tool_calls
  ↓
assistant 消息落库 → set_pending({assistant_seq, tool_call_ids, tool_calls})
  ↓
tool A 落库 → 自动从 _unresolved 移除 A → set_pending(剩余)
  ↓ (process killed here)
进程重启 → send() 检测 pending → SessionPendingError
  ├ resume()         → 跑剩余 tool_calls，pending 清零，继续
  └ abort_pending()  → 写 <aborted> tool 消息，pending 清零，下次 send 即可继续
```

---

## 9. 测试

```bash
# 全跑（含真实 LLM，要 .env 配 OPENAI_COMPAT_*）
pytest

# 只跑单元测试（不连真实 LLM）
pytest -m "not real_llm"

# 跳过真实 LLM
pytest --no-real
```

| 目录 | 内容 |
|---|---|
| `tests/unit/` | 纯控制流 / 契约测试，fake LLM |
| `tests/integration/` | 多组件场景，fake LLM |
| `tests/real/` | 跑真实 DashScope；缺 env 自动 skip |

`tests/real/judge.py` 提供 **LLM-as-judge**：业务方在测试里调 `assert_passes(question, answer, rubric)`，
内部 spawn 一个 power-loop 作 evaluator，按 rubric 返回 `{passed, reason}` JSON。
专门解决 LLM 输出非确定性下的断言难题。

---

## 10. Roadmap & Changelog

- 详细路线：[`ROADMAP.md`](ROADMAP.md)
- 版本记录：[`CHANGELOG.md`](CHANGELOG.md)

Issues / PRs welcome.
