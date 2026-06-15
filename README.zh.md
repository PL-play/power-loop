# power-loop

[![PyPI](https://img.shields.io/pypi/v/power-loop.svg)](https://pypi.org/project/power-loop/)
[![Python](https://img.shields.io/pypi/pyversions/power-loop.svg)](https://pypi.org/project/power-loop/)
[![CI](https://github.com/PL-play/power-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/PL-play/power-loop/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-see%20LICENSE-blue.svg)](LICENSE)

[English](README.md) · **中文** · [文档](docs/zh/index.md) · [示例](examples/README.md) · [更新日志](CHANGELOG.md)

> **一个会"消失"进你应用里的 Agent 运行时。** 一个类、一个 SQLite 文件、零基础设施——你就得到:持久的多轮会话、工具调用、子代理、**能跨崩溃恢复的确定性多 Agent 工作流**、持久定时器、进程级沙箱。没有要跑的服务,没有要学的框架,不用立 Redis/Postgres/队列。

```python
from power_loop import StatefulAgentLoop, create_llm_service_from_env

loop = StatefulAgentLoop(llm=create_llm_service_from_env(), db_path="app.db")
sid = loop.new_session()
await loop.send("记住我最喜欢的颜色是青色。", session_id=sid)
print((await loop.send("我最喜欢的颜色是什么?", session_id=sid)).final_text)
# → "你最喜欢的颜色是青色。"   (已持久化到 app.db;重启也在)
```

这就是全部配置。对话已经是持久、可恢复、可用工具的。

```bash
pip install 'power-loop[openai]'      # 或 [anthropic],或 [all]
```

> **1.0 —— 稳定版。** 公共 API 已在 SemVer 下冻结(破坏性变更需要大版本升级),由 CI 里的基线守卫机器化强制。**核心零运行时依赖**(纯标准库;由一个"不装任何 extra 也能 import"的 CI job 验证)。见 [稳定性](#稳定性与-semver) 与 [诚实声明](#诚实声明)——一个年轻、单维护者的项目,会直说。

---

## 从这里开始

| 你是… | 去 |
|---|---|
| 🚀 **新手** —— 给我 5 分钟上手版 | [快速开始](docs/zh/getting-started.md) |
| 🛠️ **边做边学** | [教程](docs/zh/tutorials/index.md) —— 聊天机器人 · 工具 · 人在回路 · 多 Agent |
| 🧩 **想看可运行代码** | [39 个示例](examples/README.md) —— `00_hello_world.py` → 完整聊天机器人 |
| 📚 **查阅参考** | [用户指南](docs/zh/user-guide/index.md) · [API 参考](docs/zh/api/index.md) |
| 🤔 **判断是否适合** | [横向对比](#横向对比) · [诚实声明](#诚实声明) |

**按目标找路:** 跨重启持久化 → [会话](docs/zh/user-guide/sessions.md) · 给它工具 → [工具](docs/zh/user-guide/tools.md) / [扩展](docs/zh/user-guide/extending-tools.md) · 多 Agent → [工作流](docs/zh/user-guide/workflows.md) · 沙箱化不可信代码 → [沙箱](docs/zh/user-guide/sandboxing.md) · 长期运行(备份/保留)→ [会话](docs/zh/user-guide/sessions.md) · 监控(日志/指标/追踪)→ [可观测性](docs/zh/user-guide/observability.md) · 测量与扩展 → [扩展性](docs/zh/user-guide/scaling.md) · 接入 MCP server → [扩展](docs/zh/user-guide/extending-tools.md) · 挺过崩溃 → [Pending 恢复](docs/zh/user-guide/sessions.md)。

---

## 为什么是 power-loop

大多数"Agent 框架"要你把应用建在它们**里面**。power-loop 相反:它是一个你**嵌入**的库。你保留自己的 HTTP 层、鉴权、队列、RAG、UI、部署。它只负责把 Agent 循环跑好——而且跑得持久。

- 🪶 **羽量级 & 零依赖。** 没有 `pydantic`、没有 LangChain、没有要学的图 DSL。一个紧凑的纯标准库内核(约 1.7 万行),公共面基本就是一个类——而且**零运行时依赖**。OpenAI/Anthropic transport 只在你装对应 extra 时才被拉入。
- 💾 **零基础设施。** 会话、定时器、子代理树、工作流日志、共享黑板——全在**一个 SQLite 文件**里。拷贝文件就是拷贝状态。靠在多进程间分片文件来扩展。
- ⏱️ **默认持久。** 跑到一半崩了就 `resume()`。Agent 给自己排**持久定时器**,重启不丢。工作流在进程死亡后**重放已完成步骤、只重跑未完成的尾巴**。存储**能跨版本升级**(真正的 schema 迁移阶梯),还能**裁剪、VACUUM、导出**。
- 🧩 **从一个循环到一支舰队都可组合。** 从 `send()` 开始。加工具。派生子代理。展开确定性**工作流**(`sequence`/`parallel`/`foreach`/`branch`)。让每个叶子跑在**自己的进程和 DB**里、套上沙箱。一路上都是同一套原语。
- 🛡️ **隔离接缝用在刀刃上。** 工具级沙箱用 `ShellBackend`(给 `bash` 套 gVisor/Docker);进程级用 `WorkerLauncher`(每个叶子包一整个子代理 worker)。power-loop 对沙箱无关;策略你定。
- 🔬 **为可观测而生。** 每个流式分片、工具调用、轮次、**单次 LLM 调用**都有带类型的事件——每个都带进程级 `seq` 与单调时钟。出厂 sink 在 extras 之后可插:持久 **JSONL**(带 `replay`)、**Prometheus/StatsD** 指标、**OpenTelemetry** span 树。每运行 + 每会话的 token 计量,以及硬性每运行预算。
- 🔌 **开放生态。** Provider 无关(任意 OpenAI 兼容端点或原生 Anthropic,按环境变量切)。用 `ToolRegistry` 接任意工具,或一个适配器接入 **Model Context Protocol** server。
- ✅ **真机 LLM 测试。** 专门的 `tests/real/` 套件用真实模型跑库本身——工作流、resume、沙箱子进程代理、结构化输出、压缩、一个真实 MCP server——不只是 mock。

---

## 你能得到什么

| 能力 | 一句话 | 文档 |
|---|---|---|
| **有状态会话** | 持久多轮记忆 + 跨进程恢复,全在一个 SQLite 文件 | [会话](docs/zh/user-guide/sessions.md) |
| **工具调用** | JSON-Schema 校验的工具;内置 `bash`/文件/搜索/skills 预设 | [工具](docs/zh/user-guide/tools.md) · [扩展](docs/zh/user-guide/extending-tools.md) |
| **子代理** | 通过 `AgentSpec` 委托给子循环(自带 prompt/工具/模型) | [子代理](docs/zh/user-guide/subagents.md) |
| **动态工作流** | LLM 可编写的 JSON DSL(`sequence`/`parallel`/`foreach`/`branch`);确定性引擎 | [工作流](docs/zh/user-guide/workflows.md) |
| **工作流恢复** | 每步入账;崩溃后重放已完成步骤、只重跑尾巴 | [工作流](docs/zh/user-guide/workflows.md) |
| **进程沙箱** | 每个工作流叶子在自己的进程 + DB;每叶子套 gVisor/Docker | [沙箱](docs/zh/user-guide/sandboxing.md) |
| **持久定时器** | Agent 自排唤醒;重启存活;一次性或循环 | [定时器](docs/zh/user-guide/timers.md) |
| **上下文压缩** | 自动摘要旧轮次(绝不拆开工具调用对);`recall_compacted` 取回原文 | [压缩](docs/zh/user-guide/compaction.md) |
| **持久化运维** | schema 迁移阶梯、保留/裁剪、VACUUM、`export_session`/`import_session`、优雅 `aclose()` | [会话](docs/zh/user-guide/sessions.md) |
| **可观测性** | 带类型、`seq` 有序的事件 → 持久 JSONL + `replay`、Prometheus/StatsD 指标、OTel span;可选背压 | [可观测性](docs/zh/user-guide/observability.md) |
| **可测量的扩展** | 带真实数据的 `bench/` 压测台;可选只读 WAL 连接池;诚实的单进程上限 | [扩展性](docs/zh/user-guide/scaling.md) |
| **MCP 工具** | 把 Model Context Protocol server 的工具接成 power-loop 工具 | [扩展](docs/zh/user-guide/extending-tools.md) |
| **Hooks & 事件** | 在每个生命周期点否决/观测;强类型事件 payload | [Hooks](docs/zh/user-guide/hooks.md) · [Events](docs/zh/user-guide/events.md) |
| **结构化输出** | `output_schema` → provider `response_format` → 解析并校验 | [结构化](docs/zh/user-guide/structured-output.md) |
| **可插拔记忆** | 通过 `MemoryProvider` Protocol 做跨会话召回 | [记忆](docs/zh/user-guide/memory.md) |
| **重试/取消/预算** | Provider 感知的重试、统一取消令牌、硬性每运行 token 上限 | [重试与取消](docs/zh/user-guide/retry-cancel.md) |
| **稳定错误码** | 每个 `PowerLoopError` 带冻结的机器可读 `code`——按 `exc.code` 分支 | [API:错误码](docs/zh/api/index.md#错误码) |
| **崩溃恢复** | `heal_pending` / `resume` / `abort_pending` 处理工具调用中途被杀的运行 | [Pending 恢复](docs/zh/user-guide/sessions.md) |

---

## 亮点

### 确定性多 Agent 工作流——模型能编写,且能跨崩溃存活

子代理委托是*模型驱动*的("去做这个")。当你想要**代码驱动、确定性**的编排——在列表上展开、按结果分支、跑流水线——就把它写成 `WorkflowSpec` 让引擎解释。唯一的 LLM 调用是叶子;`sequence`/`parallel`/`foreach`/`branch` 都是纯代码。

```python
from power_loop.workflow import create_workflow

spec = {
    "name": "research", "input": "日本茶道",
    "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "plan",
         "spec": {"name": "planner", "system_prompt": "把主题拆成 3 个子主题。"},
         "output_schema": {"name": "Plan", "schema": {"type": "object", "required": ["subtopics"],
            "properties": {"subtopics": {"type": "array", "items": {"type": "string"}}}}}},
        {"type": "foreach", "id": "research", "items_from": "plan.subtopics", "as": "t",
         "parallel": True, "max_concurrency": 3,
         "body": {"type": "agent", "id": "r",
                  "spec": {"name": "researcher", "system_prompt": "就 {{t}} 写两句话。"},
                  "input": "子主题:{{t}}"}},
        {"type": "agent", "id": "write",
         "spec": {"name": "writer", "system_prompt": "把这些笔记综合起来。"},
         "inputs_from": ["research"]},
    ]},
}
result = await create_workflow(spec, parent_loop=loop).run()
```

创建时即校验(一次报出所有问题——很适合让 LLM 修复)。**detached** 运行,父 Agent 会在完成时经持久定时器被唤醒。展开到一半崩了?`resume_run(loop, parent_sid, run_id)` 从日志重放 planner + 已完成的 researcher,只重跑剩下的。把它注册成工具,Agent 自己就能构造并提交工作流。

### 在真沙箱里跑不可信子代理——而不必沙箱化父进程

默认执行器在进程内跑叶子。**子进程执行器**让每个叶子跑在自己的 OS 进程、对着自己的 SQLite 文件(一文件一写者天然成立),`WorkerLauncher` 按叶子(并审视其被授予的工具)把那个进程包进 gVisor / Docker / firejail。

```python
from power_loop.workflow import SubprocessExecutor, WorkerBootstrap, create_workflow

ex = SubprocessExecutor(
    bootstrap=WorkerBootstrap(llm_from_env=True, tool_preset="core"),
    launcher=my_gvisor_launcher,   # 按叶子包裹 worker 命令;fail-closed
    timeout_s=120,
)
await create_workflow(spec, parent_loop=loop, executor=ex).run()
```

### 持久、可运维的存储——多数"Agent 库"跳过的那部分

磁盘存储就是产品本身,所以它为长期运行而造:

```python
store.export_session(sid)                 # 整个会话 → JSON 归档(含已压缩轮次)
store.prune_compacted_messages(sid)       # 对折叠出的原文做按需保留/裁剪
store.vacuum(); store.checkpoint()        # 回收磁盘
async with StatefulAgentLoop(...) as loop:  # 优雅 aclose():排空在飞 send 再关库
    ...
```

它能跨版本升级(`PRAGMA user_version` 迁移阶梯会**拒绝**打开比当前代码更新的库,而不是把它搞坏),读还能跑在可选的只读 WAL 池上,不必排在写者后面。

### 全程可观测,任意导出

```python
from power_loop.contrib.jsonl_sink import attach_jsonl_sink, replay
from power_loop.contrib.metrics_sink import attach_metrics_sink, PrometheusBackend

attach_jsonl_sink(bus, "events.jsonl")        # 持久;之后 replay("events.jsonl")
attach_metrics_sink(bus, PrometheusBackend()) # power-loop[prometheus] · 或 StatsD,或 OTel span
```

每个事件都带进程级 `seq` 与单调时钟,所以多个流可全序、可重建。同步订阅者默认内联运行;当某个 sink 可能阻塞时,可选启用有界队列的后台分发。

### 接入 Model Context Protocol server

```python
from power_loop.contrib.mcp import StdioMCPClient, register_mcp_tools   # power-loop[mcp]

client = await StdioMCPClient("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"]).connect()
await register_mcp_tools(registry, client, prefix="fs.")   # MCP 工具 → power-loop ToolDefinition
```

接入点是一个极小的 `MCPToolSource` Protocol,所以 `mcp` SDK 是可选的,任意客户端都能用。

> 还有:硬性 token 预算、结构化输出、崩溃恢复、记忆、黑板——见 [`examples/`](examples/README.md)(39 个可运行程序)与 [文档](docs/zh/index.md)。

---

## 横向对比

power-loop 是**内核**,不是平台——这就是全部取舍。

- **对比 LangChain / LangGraph / LlamaIndex / CrewAI / AutoGen** —— 那些是开箱即用的框架,生态庞大(连接器、向量库、集成)、依赖树很重。power-loop 刻意**一概不带**:一个紧凑(约 1.7 万行)的纯标准库内核、核心零依赖,工具你自带(或接一个 MCP server)。你开箱得到持久会话、可跨崩溃恢复的工作流、真正的沙箱接缝;你**不会**得到捆绑的 RAG 栈或上百个连接器。
- **选 power-loop**:当你想把 Agent *嵌入*现有应用、把依赖面压到最小,并在意持久化 + 隔离 + 稳定契约。
- **选框架**:当你想要开箱即用、庞大的集成目录,且不介意重量。

诚实地说:power-loop 在**生态广度上落后**(集成、社区、项目年龄),在**可嵌入性、持久化、机器化守卫的稳定 API 上领先**。据此取舍。

---

## 安装与配置

```bash
pip install 'power-loop[openai]'      # 任意 OpenAI 兼容端点
pip install 'power-loop[anthropic]'   # 原生 Anthropic Messages API
pip install 'power-loop[all]'         # 两个 transport + skills/pdf/可观测/mcp 等 extras
```

指向任意 OpenAI 兼容端点(或 `POWER_LOOP_PROVIDER=anthropic`):

```bash
POWER_LOOP_BASE_URL=https://api.openai.com/v1
POWER_LOOP_API_KEY=sk-...
POWER_LOOP_MODEL=gpt-4o-mini
```

Python 3.10+。见 [快速开始](docs/zh/getting-started.md)。可选 extras:`skills`、`pdf`、`prometheus`、`statsd`、`otel`、`mcp`。

---

## 稳定性与 SemVer

自 **1.0** 起,**STABLE** API(列在 `power_loop.STABLE_API`)处于 SemVer 之下:破坏性变更需要大版本升级(`2.0.0`),由 CI 里的冻结基线测试强制——包括旗舰 `StatefulAgentLoop` *以及构造它所需的 LLM 契约*(所以你能仅用 STABLE 符号构造、使用、实现自定义 provider)。错误 `.code` 字符串同样被冻结。

| 层级 | 含义 |
|---|---|
| **Stable** | 同一大版本内向后兼容;在 `power_loop.STABLE_API` 中。 |
| **Provisional** | 从顶层 re-export;未来 minor 可能调整。 |
| **Internal** | `power_loop.core.*` 等;不作兼容承诺。 |

见 [API 参考](docs/zh/api/index.md)。

---

## 诚实声明

power-loop **做编排;它本身不做隔离。** 内置的 `bash`/文件工具在进程内运行、继承宿主环境——方便用于可信的本地场景,**不是安全边界**。对不可信/模型编写的命令,用 `ShellBackend` 接缝(工具级)注入沙箱,或让叶子走 `SubprocessExecutor` + `WorkerLauncher`(进程级)。密钥留在你的编排层。见 [SECURITY.md](SECURITY.md)。

**一个存储文件 = 一个写者进程。** 每会话的顺序由进程内 `asyncio.Lock` 保证;两个进程对同一会话 `send()` 会绕过它。一进程一存储文件(把会话分片到多个文件)——[扩展性指南](docs/zh/user-guide/scaling.md) 给了实测数据和多进程模式。多写者横向扩展不在范围内。

**成熟度。** 这里的 1.0 标签是对 **API/持久化契约**的信心声明——不是多年实战检验的宣称。power-loop 还年轻、主要由单一维护者维护、公开生产记录有限。契约是机器化守卫的,项目是 MIT、可分叉;请按你的场景权衡 bus factor。

---

## 项目与链接

- **被谁使用:** DeepTalk —— 一款一对一关系型 IM 产品里会话内 Agent 的运行时。*(在生产用了它?欢迎 PR 加一行。)*
- **开发:** `pip install -e ".[dev]"` · `ruff check .` · `pytest -q --no-real`(去掉 `--no-real` 跑真机 LLM 套件)。
- [文档](docs/zh/index.md) · [架构](docs/architecture.md) · [1.0 路线图](ROADMAP_1.0.md) · [更新日志](CHANGELOG.md) · [贡献](CONTRIBUTING.md) · [安全](SECURITY.md) · [许可](LICENSE)
