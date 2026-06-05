# power-loop

> 一个可嵌入的 **Agent 执行内核**：LLM 抽象 + 多轮消息循环 + 工具调用 + 生命周期 Hook + 事件总线 + 子代理。
> 目标：让上层服务（例如 DeepTalk 的 `agent` 服务）只写「编排 / 业务壳」，不重复写 LLM 适配、tool 调用解析、循环控制和观测。
>
> 状态：**开发中（v0.1）**。本 README 与代码同步迭代；新增能力先落代码与测试，再回流到这里。
> 最后更新：2026-06-04。

---

## 1. 它是什么 / 不是什么

**是**：
- 一个 Python 库（非服务），可被任意后端 / CLI / 测试用例 `import` 调用。
- OpenAI 兼容（含国内同协议厂商）与 Anthropic 双家底的 LLM 客户端集合。
- 一套带 hook 与事件的「消息 → LLM → 工具 → 回合」主循环，可单回合或多回合。
- 一个可动态注册工具的 `ToolRegistry`，自带一组开箱即用的本地工具（read / write / edit / bash / spawn_agent…）。

**不是**：
- 不是 IM / 业务服务。**不感知**会话、用户、Kafka、HTTP；这些应在调用方（DeepTalk `agent` 服务等）里实现。
- 不是大而全的 Agent Framework；不内置 RAG、向量库、Planner 等高层抽象。
- 不绑死任何模型厂商：`base_url` / `api_key` / `model` 全部从配置传入。

### 实现亮点 / 卖点

> 状态标记：✅ 已实现 · 🚧 计划中（ROADMAP 标号） · 📐 设计已锁

| # | 能力 | 状态 | 一句话价值 |
|---|---|---|---|
| 1 | **可编程控制流**：15 个 HookPoint × 4 个 Directive（`CONTINUE/SKIP/BREAK/SHORT_CIRCUIT`） | ✅ | 不改主循环就能实现安全策略、缓存短路、提前终止、注入 mock 结果 |
| 2 | **观测/控制双通道**：`AgentEventBus`（只读旁路）+ `AgentHooks`（可改控制流） | ✅ | 指标、审计、typing 推送走 event；安全门、降级走 hook，二者不互相污染 |
| 3 | **contextvars 会话隔离**：`hooks / event_bus / session_id` 通过 contextvars 注入 | ✅ | 一个进程并发跑多会话不串台；子代理自动继承独立子 session |
| 4 | **声明式 + 命令式两种子代理**：`spawn_agent` 工具（命令式）+ `AgentSpec` JSON（声明式，主 Agent 可自描述生成） | ✅ + 🚧M1.8 | dynamic workflow 不需要 DAG 框架：主 Agent 自己决定何时分裂、用什么工具子集、什么人格 |
| 5 | **OpenAI 兼容 + Anthropic 双家底**：单一 `LLMRequest` 抽象，`base_url / api_key / model` 全走 env | ✅ | 切厂商不改业务代码；DashScope、MiniMax、Kimi、DeepSeek 直连 |
| 6 | **工具调用 = 一等公民**：动态 `ToolRegistry` + JSON Schema 必填校验 + sync/async handler 自动适配 + OpenAI tool schema 一键导出 | ✅ | 业务工具随注册随用；无需为 async handler 手写 event loop hack |
| 7 | **运行时工具注册**（meta-tool `register_tool`）：主 Agent 临时声明脚本/命令为新工具，默认沙箱 + 默认关闭 | 🚧M2.6 | 把"主 Agent 自己长出新能力"做成受控、可审计的一等机制，不是 `eval` 后门 |
| 8 | **上下文压缩**（M1.7a） | 📐已锁设计 | 详见下方"压缩策略"；解决长会话 `context_length_exceeded` 这个**运行时正确性问题** |
| 9 | **统一重试 / 超时 / 取消** + 软降级 | 🚧M1.1 | LLM 抖动不让 Agent 卡死；用户撤回 @ 时能真正取消正在跑的循环 |
| 10 | **结构化输出**：`response_format=json_schema` + JSON 修复 + Schema 校验 | 🚧M1.3 | 卡片 / 表单 / 报告类输出由 LLM 直出可解析对象，不靠 prompt 哄 |
| 11 | **可序列化的会话快照**：`SessionSnapshot.save/load`，支持断点续跑 / 调试回放 | 🚧M1.7b | 长任务可中断恢复；事故现场可一比一重放 |
| 12 | **记忆分层**：4 层清晰拆分（回合上下文 / 会话工作记忆 / 跨会话连续 / 长期事实），库提供 `MemoryProvider` 协议 + 生命周期接线，**业务方 30 行实现长期记忆** | 🚧M1.9 | 库不绑存储栈（不押注 vector DB / RAG 框架），业务方任选 SQLite / Redis / PG / Chroma 接入；examples 给 3 个开箱即用参考实现 |

#### 压缩策略（M1.7a 设计已锁）

参考 Anthropic 官方 [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)、Claude Code auto-compact、LangChain `ConversationSummaryBufferMemory`，落到 vendor-neutral 的客户端实现：

- **抽象**：`Compactor` 协议 + `DefaultCompactor`，**默认开启**（`AgentLoopConfig.compactor=DefaultCompactor()`，传 `None` 才关）。
- **触发**：每回合 `round.start` 之前；`estimate_tokens(messages) ≥ max_tokens × 0.75`（或 env `CONTEXT_COMPACT_THRESHOLD` 绝对值）。**幂等**：同一回合最多压缩 1 次。
- **保留区**（核心不变量）：
  1. 所有 `role=system` 消息（含先前 `compact_note`）；
  2. 工具定义（不在 messages 里，天然不动）；
  3. 末尾 `keep_last_n` 条**完整 exchange**——`assistant(tool_calls)` 必须与对应 `tool(tool_call_id=…)` **作为原子对**保留，绝不被切开；
  4. 任何悬挂未完结的 tool_call。
- **摘要**：可压缩中段 → 一次额外 LLM 调用（默认主 LLM，可注入更便宜的 `summary_llm`）→ 输出包在 `<summary>…</summary>` 中，prompt 显式禁止调用工具（避开 Anthropic 文档警告的失败模式）。
- **注入**：替换为一条 `{role: "system", name: "compact_note", content, _meta: {compacted_at_round, original_count, original_tokens, summary_tokens}}`，位置在 system 之后、保留尾巴之前——**保持顺序稳定，prompt cache 友好**。
- **多轮累积**：默认 Option A（旧 `compact_note` 视为 system 不再压缩）；超过 `max_compact_notes=5` 时切换 Option B（合并最旧 2 条 notes 为 1 条）。
- **失败软降级**：压缩 LLM 失败（含 retry 用尽）→ `compact.failed` 事件 → **用未压缩 history 跑该回合**；若主 LLM 因此 `context_length_exceeded` → 升级 `loop.degraded` 终止；连续 3 次失败 → 本 session 永久禁用 compactor + warning。
- **可换**：业务方可注入自定义 `Compactor` 实现（选择性压缩、外部摘要服务、本地小模型摘要等），或通过 `compact.before` hook 一票否决/替换。

> 与"裁剪"（M1.2 `trim_history`）的关系：裁剪是无成本纯删，压缩是有成本但保留语义；二者正交，业务可任选。

---

## 2. 目录结构

```
power-loop/
├── llm_client/                # LLM 适配层（独立顶级包）
│   ├── interface.py           # LLMRequest / LLMResponse / LLMService 抽象
│   ├── llm_factory.py         # 多 provider 工厂（OpenAI 兼容 + Anthropic）
│   ├── llm_tooling.py         # OpenAI tool-call 协议解析 / 规整
│   ├── multimodal.py          # 多模态内容拼装（图片等）
│   ├── capabilities.py        # 模型能力探测（多模态 / 工具 / 流式…）
│   ├── web_search.py          # 内置 web search 工具的 LLM 侧接线
│   └── qwen_image.py          # 通义千问图像专用通道
│
├── power_loop/                # Agent 执行内核
│   ├── agent/
│   │   ├── loop.py            # AgentLoop 外观类（run / run_sync）
│   │   ├── types.py           # AgentLoopConfig / AgentLoopResult / LoopMessage
│   │   └── system_prompt.py   # 默认 system prompt 模板
│   ├── core/
│   │   ├── pipeline.py        # 主循环（按 phase 拆分 + hook/event 编排）
│   │   ├── agent.py           # 旧入口 agent_loop_async（薄包装）
│   │   ├── runner.py          # 会话生命周期与上下文注入
│   │   ├── agent_context.py   # contextvars：hooks / event_bus / session_id
│   │   ├── state.py           # ContextManager（消息累积、token 估算等）
│   │   ├── phase.py           # 单回合的 phase 步骤拆分
│   │   ├── hooks.py           # AgentHooks：有序 sync/async hook 管理
│   │   └── events.py          # AgentEventBus：事件订阅 / 发布
│   ├── contracts/             # 类型契约（dataclass / pydantic）
│   │   ├── hooks.py           # HookPoint / HookDirective / HookContext
│   │   ├── hook_contexts.py   # 每个 hook 点的 typed Ctx
│   │   ├── events.py          # AgentEvent / AgentEventType
│   │   ├── event_payloads.py  # 每类事件 payload
│   │   ├── tools.py           # ToolDefinition / 校验
│   │   ├── messages.py        # LoopMessage 协议
│   │   ├── handlers.py        # handler 协议
│   │   └── protocols.py       # 公共 Protocol
│   ├── tools/
│   │   ├── registry.py        # ToolRegistry（动态 register / invoke / OpenAI schema 导出）
│   │   ├── default_manifest.py# 默认工具的 ToolDefinition 集合
│   │   ├── default_tools.py   # 默认工具实现：bash/read/write/edit/grep/glob/patch…
│   │   └── spawn_agent.py     # 子代理 spawn 工具
│   └── runtime/
│       ├── env.py             # WORKSPACE / AGENT_DIR / 路径白名单
│       └── skills.py          # SKILL.md 加载（frontmatter + body）
└── tests/                     # smoke + 真实/伪 LLM 测试用例
```

> `llm_client` 当前是 **顶级包**，独立于 `power_loop`，可单独 import 用作"只调 LLM"的薄客户端。

---

## 3. 安装

```bash
cd power-loop
pip install -e ".[dev]"      # 含 pytest / pytest-asyncio
```

依赖（见 `pyproject.toml`）：`anthropic`、`openai`、`socksio`、`python-dotenv`、`pyyaml`、`rich`、`pypdf`。
Python ≥ 3.10。

---

## 4. 配置（环境变量）

LLM 凭证与端点 **不入代码 / 不入配置文件**，统一走环境变量（推荐配合 `.env` + `python-dotenv`）：

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | OpenAI 兼容厂商凭证与默认模型 |
| `ANTHROPIC_API_KEY` | Anthropic 凭证（如需） |
| `POWER_LOOP_WORKSPACE` | 默认工作目录（默认工具的相对路径以此为根） |
| `POWER_LOOP_SKILLS_DIR` | SKILL.md 目录（默认 `<agent_dir>/.skills`） |

> 多 provider 的具体 key 名见 `llm_client/llm_factory.py`；新增厂商时在工厂里扩，不要在业务代码里写死。

---

## 5. 最小用法

### 5.1 仅用 LLM 客户端（不进 agent loop）

```python
from llm_client.llm_factory import create_llm_service
from llm_client.interface import LLMRequest

llm = create_llm_service(provider="openai")  # 或 "anthropic"
resp = llm.complete_sync(LLMRequest(
    messages=[{"role": "user", "content": "你好"}],
    system_prompt="你是一个简洁的助手。",
    temperature=0.0,
))
print(resp.text)
```

> 完整可运行版本见 [`examples/00_minimal.py`](examples/00_minimal.py)（真实 DashScope）。
> `tests/real/test_examples.py::test_example_00_minimal_runs` 把它作为活文档锁定——
> 若 example 跑挂，要么 example 该更新，要么有 Public API 回归。

### 5.2 运行一个完整 Agent Loop（含工具）

```python
from llm_client.llm_factory import create_llm_service
from power_loop.agent.loop import AgentLoop
from power_loop.agent.types import AgentLoopConfig
from power_loop.tools.registry import build_registry
from power_loop.tools.default_manifest import DEFAULT_TOOL_DEFINITIONS
from power_loop.tools.default_tools import DEFAULT_TOOL_HANDLERS

llm = create_llm_service(provider="openai")
tools = build_registry(DEFAULT_TOOL_DEFINITIONS, DEFAULT_TOOL_HANDLERS)

loop = AgentLoop(
    llm=llm,
    config=AgentLoopConfig(
        system_prompt="你是一个能调用工具的工程师助手。",
        max_rounds=8,
        temperature=0.0,
        max_tokens=4000,
    ),
    tool_registry=tools,
)

result = loop.run_sync(messages=[
    {"role": "user", "content": "读取 README.md 的前 20 行并总结。"},
])
print(result.status, result.rounds)
print(result.final_text)
```

### 5.3 不要工具的单回合「LLM 应答」

```python
loop = AgentLoop(llm=llm, config=AgentLoopConfig(system_prompt=SYS, max_rounds=1))
result = loop.run_sync(messages=conversation)   # 直接拿 final_text
```

> 这就是 DeepTalk `agent` 服务 MVP 的典型用法：业务侧把"系统人格 + 历史 + 当前任务"拼成 messages 传进来，
> power-loop 走一次 LLM 调用就返回。

---

## 6. Hook & Event 模型

### 6.1 Hook 点（控制流）

`power_loop.contracts.hooks.HookPoint` 当前覆盖：

```
session.start / session.end
round.start   / round.end   / round.decide
llm.before    / llm.after
tools.batch.before / tools.batch.after
tool.before   / tool.after  / tool.error
compact.before / compact.after
message.append
```

每个 hook 点接收 **typed Ctx**（`power_loop/contracts/hook_contexts.py` 定义对应 dataclass），可：

- 原地修改 ctx 字段（如改写 messages、注入 tool_output、改 LLM 请求）；
- 返回 `HookDirective`：`CONTINUE / SKIP / BREAK / SHORT_CIRCUIT`，组合语义见 `HookDirective` 的 docstring。

注册示例：

```python
from power_loop.core.hooks import AgentHooks
from power_loop.contracts.hooks import HookPoint, HookDirective

hooks = AgentHooks()

def on_llm_before(ctx):
    ctx.request.temperature = 0.2   # 改写请求

def on_tool_error(ctx):
    ctx.tool_output = "（工具失败，已忽略）"
    return HookDirective.SKIP

hooks.register(HookPoint.LLM_BEFORE, on_llm_before)
hooks.register(HookPoint.TOOL_ERROR, on_tool_error)

loop = AgentLoop(llm=llm, config=cfg, tool_registry=tools, hooks=hooks)
```

### 6.2 Event Bus（观测 / 旁路）

`AgentEventBus` 发布只读事件（不改控制流），用于上报指标、推送"正在输入"、日志/审计等：

- `session.started / session.ended`
- `round.started / round.completed`
- `stream.started / stream.delta / stream.completed`
- `tool_call.started / tool_call.completed / tool_call.failed`
- `usage.updated`、`user.notification`、`auto_compact.status`、`hit_round_limit.status`…

Payload 类型见 `power_loop/contracts/event_payloads.py`。

---

## 7. 工具系统

- `ToolDefinition`（`contracts/tools.py`）：名字 / 描述 / JSON Schema / 必填参数；可直接 `to_openai_tool()` 喂给 LLM。
- `ToolRegistry`：`register / unregister / invoke / invoke_async`，自带必填参数校验，兼容 dict / kwargs 两种 handler 签名。
- 默认工具集（`default_tools.py` / `default_manifest.py`）：`run_bash` `run_read` `run_write` `run_edit` `run_grep` `run_glob` `apply_patch`…（含路径白名单与读后写检查）。
- 子代理：`tools/spawn_agent.py` 提供 `spawn_agent` 工具，允许在工具调用里启一个子 AgentLoop 处理子任务。

> DeepTalk agent 服务用到的"业务能力"（如 `/summary`、`/deep`、生成卡片 JSON）应在 **调用方** 注册成工具或直接靠
> `parse_json` 结构化输出实现；不要把业务工具塞进 `power_loop/tools/default_*`。

---

## 8. 测试

```bash
cd power-loop
pytest                                  # 全跑
pytest tests/test_agent_loop_events_hooks.py
pytest tests/smoke_agent_loop_v1.py
```

- `smoke_*` 用例：跑通主循环骨架。
- `test_real_*` 用例：连真实 LLM 验证流式/工具调用/sub-agent；需要环境变量里有 key，否则跳过。

---

## 9. DeepTalk 集成定位

`agent` 服务对 power-loop 的预期分工（来自 `platform/docs/design/06-agent-orchestration.md`）：

| 关注点 | 归属 |
|---|---|
| Kafka 消费 `im.messages` / 触发判定（@、斜杠、卡片） | **agent 服务** |
| `message_id` 幂等去重、防自激（`sender_type=agent` 过滤） | **agent 服务** |
| 调 api 拉历史 / 调 api 受信内部接口写回 | **agent 服务** |
| 双方授权门 / 能力开关 / 降级文案 | **agent 服务** |
| 历史窗口拼装（含 A/B/Agent 身份标注） | **agent 服务**（power-loop 后续可提供 trim helper） |
| 系统人格 + LLM 调用 + 工具循环 + 结构化输出 | **power-loop** |
| `typing` 瞬时指示（通过事件总线 + 业务侧推送） | 协作：power-loop 出事件，agent 服务推 gateway |

---

## 10. Roadmap（边用边补）

按当前对接需要优先级：

1. **结构化卡片输出约定**：定义 DeepTalk 侧 card schema → 在 `llm_client` 提供 `response_format=json_schema` 便捷封装。
2. **token 预算窗口工具**：`trim_history(messages, max_tokens, model)` 公共助手，避免每个调用方重复实现。
3. **统一 LLM 重试 / 超时策略**：`LLMRetryPolicy`（指数退避 + 最大次数 + 整体超时），让业务侧降级更可靠。
4. **事件命名稳定化**：固定一组对外暴露的事件名，方便 DeepTalk admin 后台采集。
5. **README 中加入"事件 / Hook 完整一览表"**：随着 hook / event 增减保持同步。
6. **provider 配置文档化**：把 `llm_factory` 支持的厂商与所需 env 列成对照表。
7. **README 示例代码自动化校验**（doctest 或 smoke）。

---

## 11. 变更与对接备注

- 主入口稳定：`AgentLoop(llm, config, tool_registry, *, event_bus, hooks).run / run_sync`。
- 旧入口 `power_loop.core.agent.agent_loop_async` 仍保留，作为 `AgentPipeline` 的薄包装。
- 业务方不应直接 import `power_loop.core.pipeline` 内部；通过 `AgentLoop` 外观使用。
- 新增 hook 点或事件类型，需同时更新 `contracts/hooks.py` 或 `contracts/events.py` + payload，并在本 README §6 补充。
