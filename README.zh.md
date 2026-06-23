# power-loop

[![PyPI](https://img.shields.io/pypi/v/power-loop.svg)](https://pypi.org/project/power-loop/)
[![Python](https://img.shields.io/pypi/pyversions/power-loop.svg)](https://pypi.org/project/power-loop/)
[![CI](https://github.com/PL-play/power-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/PL-play/power-loop/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-see%20LICENSE-blue.svg)](LICENSE)

[English](README.md) · **中文** · [文档](docs/zh/index.md) · [示例](examples/README.md) · [更新日志](CHANGELOG.md)

> **循环工程,而非框架投靠。** power-loop 是一个可嵌入的 **Agent 执行内核**:你**工程化** Agent 的*循环*——在每个生命周期点埋 hook、可插拔存储、沙箱接缝、上下文压缩、确定性工作流——而不是把应用建在一个框架*里面*。循环本身是一个**轻量、无状态的句柄**,架在一个**可插拔存储**之上(默认 SQLite——零基础设施——或按 DSN 切到 PostgreSQL/MySQL)。由此你得到:持久的多轮会话、工具调用、子代理、能跨崩溃恢复的多 Agent 工作流、持久定时器、进程级沙箱。没有要跑的服务,没有要学的图 DSL。

```python
from power_loop import StatefulAgentLoop, create_llm_service_from_env

# 循环是架在存储之上的一个轻薄、无状态句柄。默认 = 一个 SQLite 文件(零基础设施);
# 把 dsn= 换成 "postgresql://…/app" 或 "mysql://…/app",其它代码一行都不用改。
loop = StatefulAgentLoop(llm=create_llm_service_from_env(), dsn="app.db")
sid = await loop.new_session()
await loop.send("记住我最喜欢的颜色是青色。", session_id=sid)
print((await loop.send("我最喜欢的颜色是什么?", session_id=sid)).final_text)
# → "你最喜欢的颜色是青色。"   (已持久化;重启也在)
```

对话已经是持久、可恢复、可用工具的。又因为循环**不持有任何权威状态**,一个全新进程仅凭一个 DSN + 会话 id 就能从零恢复它:

```python
# 冷启动,另一个进程——重建循环并继续。没有状态要序列化或携带。
loop = StatefulAgentLoop(llm=create_llm_service_from_env(), dsn="app.db")
print((await loop.send("那第二喜欢的呢?", session_id=sid)).final_text)
```

```bash
pip install 'power-loop[openai]'      # 或 [anthropic] · 要那两个后端就加 [postgres] / [mysql]
```

> **自 1.0 起稳定;现已 3.x。** 公共 API 已在 SemVer 下冻结,由 CI 里的基线守卫机器化强制——而自那以来的两次大版本升级恰恰印证了这份纪律,而非削弱它:**2.0** 把存储换成可插拔的异步后端,**3.0** 把上下文处理变成两条正交的轴。两者都是真正的破坏性变更,所以都吃了一次大版本升级。**核心零运行时依赖**(纯标准库;由一个"不装任何 extra 也能 import"的 CI job 验证)——LLM transport *以及数据库驱动*都是可选 extra。背后有 **900+ 单元测试**、一套**真机 LLM** 套件,以及一套**三后端一致性套件**(SQLite/PostgreSQL/MySQL)。见 [稳定性](#稳定性与-semver) 与 [诚实声明](#诚实声明)——一个年轻、单维护者的项目,会直说。

---

## 从这里开始

| 你是… | 去 |
|---|---|
| 🚀 **新手** —— 给我 5 分钟上手版 | [快速开始](docs/zh/getting-started.md) |
| 🛠️ **边做边学** | [教程](docs/zh/tutorials/index.md) —— 聊天机器人 · 工具 · 人在回路 · 多 Agent |
| 🧩 **想看可运行代码** | [43 个示例](examples/README.md) —— `00_hello_world.py` → 完整聊天机器人 |
| 📚 **查阅参考** | [用户指南](docs/zh/user-guide/index.md) · [API 参考](docs/zh/api/index.md) |
| 🤔 **判断是否适合** | [横向对比](#横向对比) · [诚实声明](#诚实声明) |

**按目标找路:** 跨重启持久化与恢复 → [会话](docs/zh/user-guide/sessions.md) · 选后端(SQLite/PG/MySQL)→ [存储后端](docs/zh/user-guide/storage-backends.md) · 给它工具 → [工具](docs/zh/user-guide/tools.md) / [扩展](docs/zh/user-guide/extending-tools.md) · 多 Agent → [工作流](docs/zh/user-guide/workflows.md) · 沙箱化不可信代码 → [沙箱](docs/zh/user-guide/sandboxing.md) · 监控 → [可观测性](docs/zh/user-guide/observability.md) · 扩展 → [扩展性](docs/zh/user-guide/scaling.md) · 挺过崩溃 → [Pending 恢复](docs/zh/user-guide/sessions.md)。

---

## 为什么是 power-loop —— "循环工程"

大多数"Agent 框架"要你把应用建在它们**里面**。power-loop 相反:它是一个你**嵌入**的库。你保留自己的 HTTP 层、鉴权、队列、RAG、UI、部署。它负责把 Agent 循环跑好——并让你*工程化*它。

- 🪶 **羽量级 & 零依赖。** 没有 `pydantic`、没有 LangChain、没有要学的图 DSL。一个紧凑的纯标准库内核(约 2.4 万行),公共面基本就是一个类——而且**零运行时依赖**。LLM transport *以及* Postgres/MySQL 驱动,只在你装对应 extra 时才被拉入。
- 🗄️ **可插拔存储,零基础设施为默认。** 会话、定时器、子代理树、工作流日志、共享黑板——全部由一份后端无关的存储,基于一个极小的 `Database`/`Dialect` 端口写就。默认是**一个 SQLite 文件**(拷贝文件就是拷贝状态);想要真正的多写者服务时,把 DSN 指向 **PostgreSQL 或 MySQL**——同一份代码、同一套一致性测试。表自动创建,或**带外预置**并打印出 DDL 脚本(见 [存储后端](docs/zh/user-guide/storage-backends.md))。
- ♻️ **无状态、可恢复的循环。** `StatefulAgentLoop` 不携带任何权威状态——全在存储里。所以循环创建廉价,仅凭一个 **DSN + 会话 id** 就能轻易**恢复**(适合 web handler、worker、冷启动)。它会自缓存每个会话的活动窗口(一个可重建的加速器,绝不改变模型看到的内容),在热路径上免去重复读取。
- ⏱️ **默认持久。** 跑到一半崩了就 `resume()`。Agent 给自己排**持久定时器**,重启不丢。工作流在进程死亡后**重放已完成步骤、只重跑未完成的尾巴**。存储**能跨版本升级**(一个可移植、后端无关的迁移版本表),还能**裁剪、VACUUM、导出**。
- 🧠 **上下文工程,而非一种固定策略。** 每个已结束的 send 如何被*记录/渲染*(**representation**:完整**逐字**,或每个 send 一份简短**投影**),以及历史超预算后如何被*压缩*(**fold strategy**:一次 **LLM 摘要**,或一次还会写下持久笔记的 **agentic** 处理)——是两条**正交、由配置驱动**的轴:任意 representation 都能与任意 fold strategy 组合,且两者都接受你自己的 `Representation` / `FoldStrategy` 实现。折叠永远保留整个 send(绝不拆开工具调用对);`recall_send` / `recall_compacted` 从不可变审计日志里取回原始明细。
- 🧩 **从一个循环到一支舰队都可组合。** 从 `send()` 开始。加工具。派生子代理。展开确定性**工作流**(`sequence`/`parallel`/`foreach`/`branch`)。让每个叶子跑在**自己的进程和 DB**里、套上沙箱。一路上都是同一套原语。
- 🛡️ **隔离接缝用在刀刃上。** 工具级沙箱用 `ShellBackend`(给 `bash` 套 gVisor/Docker);进程级用 `WorkerLauncher`(每个叶子包一整个子代理 worker)。power-loop 对沙箱无关;策略你定。
- 🔬 **为可观测而生。** 每个流式分片、工具调用、轮次、**单次 LLM 调用**都有带类型的事件——每个都带 `seq` 与单调时钟。出厂 sink 在 extras 之后可插:持久 **JSONL**(带 `replay`)、**Prometheus/StatsD** 指标、**OpenTelemetry** span 树。每运行 + 每会话的 token 计量,以及硬性每运行预算。
- 🔌 **开放生态。** Provider 无关(任意 OpenAI 兼容端点或原生 Anthropic,按环境变量切)。用 `ToolRegistry` 接任意工具,或一个适配器接入 **Model Context Protocol** server。
- ✅ **真机测试。** 专门的 `tests/real/` 套件用真实模型跑库本身——工作流、resume、沙箱子进程代理、结构化输出、压缩、一个真实 MCP server;存储层则有一套**后端无关的一致性测试**,针对 SQLite、PostgreSQL、MySQL 跑。

---

## 你能得到什么

| 能力 | 一句话 | 文档 |
|---|---|---|
| **有状态会话** | 持久多轮记忆 + 按 id 恢复,运行于 SQLite/PG/MySQL | [会话](docs/zh/user-guide/sessions.md) |
| **可插拔后端** | 一份存储,`dsn=` 选 SQLite(默认)/ PostgreSQL / MySQL;可配置 schema 预置 | [存储后端](docs/zh/user-guide/storage-backends.md) |
| **无状态 / 可恢复循环** | 循环不持状态;从 `dsn` + `session_id` 重建;创建廉价 | [会话](docs/zh/user-guide/sessions.md) |
| **工具调用** | JSON-Schema 校验的工具;内置 `bash`/文件/搜索/skills 预设 | [工具](docs/zh/user-guide/tools.md) · [扩展](docs/zh/user-guide/extending-tools.md) |
| **子代理** | 通过 `AgentSpec` 委托给子循环(自带 prompt/工具/模型) | [子代理](docs/zh/user-guide/subagents.md) |
| **动态工作流** | LLM 可编写的 JSON DSL(`sequence`/`parallel`/`foreach`/`branch`);确定性引擎 | [工作流](docs/zh/user-guide/workflows.md) |
| **工作流恢复** | 每步入账;崩溃后重放已完成步骤、只重跑尾巴 | [工作流](docs/zh/user-guide/workflows.md) |
| **进程沙箱** | 每个工作流叶子在自己的 OS 进程 + 自己的 DB;每叶子套 gVisor/Docker | [沙箱](docs/zh/user-guide/sandboxing.md) |
| **持久定时器** | Agent 自排唤醒;重启存活;一次性或循环 | [定时器](docs/zh/user-guide/timers.md) |
| **上下文 — representation** | 把每个已结束的 send 以**逐字**记录/渲染,或记成每个 send 一份简短**投影**(派生表 `pl_project_messages`);`pl_messages` 保持不可变;`recall_send` 重新展开 | [投影](docs/zh/user-guide/send-context-projection.md) |
| **上下文 — fold strategy** | 历史超预算后压缩较旧部分:**LLM 摘要**或 **agentic**(还会写下笔记);可插拔 `FoldStrategy`;绝不拆开工具调用对;`recall_compacted` 重新展开 | [压缩](docs/zh/user-guide/compaction.md) |
| **持久化运维** | 可移植迁移版本表、保留/裁剪、VACUUM、`export_session`/`import_session`、优雅 `aclose()` | [会话](docs/zh/user-guide/sessions.md) |
| **可观测性** | 带类型、`seq` 有序的事件 → 持久 JSONL + `replay`、Prometheus/StatsD 指标、OpenTelemetry span | [可观测性](docs/zh/user-guide/observability.md) |
| **MCP 工具** | 把 Model Context Protocol server 的工具接成 power-loop 工具 | [扩展](docs/zh/user-guide/extending-tools.md) |
| **Hooks & 事件** | 在每个生命周期点否决/观测;强类型事件 payload | [Hooks](docs/zh/user-guide/hooks.md) · [Events](docs/zh/user-guide/events.md) |
| **结构化输出** | `output_schema` → provider `response_format` → 解析并校验 | [结构化](docs/zh/user-guide/structured-output.md) |
| **可插拔记忆** | 通过 `MemoryProvider` Protocol 做跨会话召回；默认开启的内置 hook 在请求尾部临时注入(可做前缀缓存) | [记忆](docs/zh/user-guide/memory.md) |
| **重试/取消/预算** | Provider 感知的重试、统一取消令牌、硬性每运行 token 上限 | [重试与取消](docs/zh/user-guide/retry-cancel.md) |
| **稳定错误码** | 每个 `PowerLoopError` 带冻结的机器可读 `code`——按 `exc.code` 分支 | [API:错误码](docs/zh/api/index.md#错误码) |
| **崩溃恢复** | `heal_pending` / `resume` / `abort_pending` 处理工具调用中途被杀的运行 | [Pending 恢复](docs/zh/user-guide/sessions.md) |

---

## 亮点

### 可插拔存储——默认 SQLite,按 DSN 切 PostgreSQL/MySQL

整套存储(会话、消息、定时器、压缩日志、子代理树、黑板)**只写一次**,基于一个极小的异步 `Database` + `Dialect` 端口。用 DSN 选后端;它之上的代码从不改变。

```python
from power_loop import StatefulAgentLoop, SchemaPolicy

StatefulAgentLoop(llm=llm, dsn="app.db")                                  # SQLite(零基础设施,默认)
StatefulAgentLoop(llm=llm, dsn="postgresql://u:p@host/app")               # PostgreSQL  → pip install 'power-loop[postgres]'
StatefulAgentLoop(llm=llm, dsn="mysql://u:p@host/app", table_prefix="pl_")  # MySQL    → pip install 'power-loop[mysql]'

# schema 预置是一个策略。AUTO_CREATE(默认)在表缺失时创建;VERIFY 只检查,
# 若 schema 不存在则抛错,并带上要以特权用户身份运行的精确 DDL。
StatefulAgentLoop(llm=llm, dsn="postgresql://readonly@host/app", schema=SchemaPolicy.VERIFY)
```

SQLite 是单写者文件(零基础设施,在多进程间分片)。PostgreSQL/MySQL 是真正的**多写者**服务——每会话的序号分配经一个 `SELECT … FOR UPDATE` 行锁在多进程间保持正确。同一套后端无关的**一致性测试**针对这三者都跑。每个后端的 DDL 与预置选项见 [存储后端](docs/zh/user-guide/storage-backends.md)。

### 无状态、可恢复的循环

`StatefulAgentLoop` 是一个*句柄*,不是会话。它不持有任何对话状态——全在存储里——所以创建廉价,你能从一个冷进程按 id 恢复任意会话:

```python
# Web handler / worker:每个请求建一个循环,恢复用户的会话,完事。
loop = StatefulAgentLoop(llm=create_llm_service_from_env(), dsn=DSN)
await loop.prewarm(session_id)                       # 可选:预加载活动窗口
result = await loop.send(user_text, session_id=session_id)
```

底层循环为每个会话保留一份**活动窗口缓存**——但它只缓存*持久*投影,由一个单调的 `next_seq` 令牌校验,所以它是一个纯加速器:一个缓存为空的冷循环会产出逐字节相同的 prompt(由 warm-vs-cold 一致性测试证明,含 recall/压缩/prompt 编辑等边界场景)。

### 上下文工程——你来选的两条正交轴(也能自己实现)

长对话迟早撑爆窗口。多数库只给你*一种*固定的压缩行为;power-loop(3.0)把它拆成两条独立、由配置驱动的轴:

- **Representation** —— 每个*已结束的 send* 如何被记录与渲染:`VerbatimRepresentation`(完整、逐字节一致的历史)或 `ProjectedRepresentation`(每个 send 一份简短的纯文本投影)。原始明细始终留在不可变的 `pl_messages` 审计日志里。
- **Fold strategy** —— 当渲染出的前缀越过预算后,*较旧*的历史如何被压缩:`LLMSummaryFold`(一次摘要调用)或 `AgenticFold`(一段有界的工具循环,还会把持久事实落成笔记)。

```python
from power_loop import (
    StatefulAgentLoop, AgentLoopConfig,
    ProjectedRepresentation, AgenticFold,   # 两条轴随意搭配——或传入你自己的实现
)

cfg = AgentLoopConfig(
    representation=ProjectedRepresentation(max_chars=300),  # 简短投影(或 VerbatimRepresentation)
    fold_strategy=AgenticFold(keep_last_sends=4),           # 摘要较旧的 send + 写笔记
)
loop = StatefulAgentLoop(llm=llm, dsn="app.db", config=cfg)
```

任意 representation 都能与任意 fold strategy 组合,而每条轴都是一个你可以自己实现的小 `Protocol`。折叠永远保留**整个 send**(绝不拆开原子的工具调用/结果对),模型可以调用 `recall_send(send_index=N)` / `recall_compacted()` 从审计日志取回完整原始明细。(上面两个类是公开但**暂定的**——3.0 新增,尚未冻结进 `STABLE_API`;`AgentLoopConfig` 本身是 Stable。)

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

存储就是产品本身,所以它为长期运行而造:

```python
await store.export_session(sid)                 # 整个会话 → JSON 归档(含已压缩轮次)
await store.prune_compacted_messages(sid)       # 对折叠出的原文做按需保留/裁剪
await store.vacuum(); await store.checkpoint()  # 回收磁盘(SQLite;在不适用的后端为 no-op)
async with StatefulAgentLoop(...) as loop:      # 优雅 aclose():排空在飞 send 再关库
    ...
```

它能跨版本升级——一个可移植的 `pl_schema_migrations` 版本表(不是 SQLite 专属的 `PRAGMA`)会**拒绝**打开比当前代码更新的库,而不是把它搞坏,并且在每个后端上行为一致。

### 全程可观测,任意导出

```python
from power_loop.contrib.jsonl_sink import attach_jsonl_sink, replay
from power_loop.contrib.metrics_sink import attach_metrics_sink, PrometheusBackend

attach_jsonl_sink(bus, "events.jsonl")        # 持久;之后 replay("events.jsonl")
attach_metrics_sink(bus, PrometheusBackend()) # power-loop[prometheus] · 或 StatsD,或 OpenTelemetry span
```

每个事件都带进程级 `seq` 与单调时钟,所以多个流可全序、可重建。同步订阅者默认内联运行;当某个 sink 可能阻塞时,可选启用有界队列的后台分发。

### 接入 Model Context Protocol server

```python
from power_loop.contrib.mcp import StdioMCPClient, register_mcp_tools   # power-loop[mcp]

client = await StdioMCPClient("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"]).connect()
await register_mcp_tools(registry, client, prefix="fs.")   # MCP 工具 → power-loop ToolDefinition
```

接入点是一个极小的 `MCPToolSource` Protocol,所以 `mcp` SDK 是可选的,任意客户端都能用。

> 还有:硬性 token 预算、结构化输出、崩溃恢复、记忆、黑板——见 [`examples/`](examples/README.md)(43 个可运行程序)与 [文档](docs/zh/index.md)。

---

## 横向对比

power-loop 是**内核**,不是平台——这就是全部取舍。

- **对比 LangChain / LangGraph / LlamaIndex / CrewAI / AutoGen** —— 那些是开箱即用的框架,生态庞大(连接器、向量库、集成)、依赖树很重。power-loop 刻意**一概不带**:一个紧凑(约 2.4 万行)的纯标准库内核、核心零运行时依赖,工具你自带(或接一个 MCP server)。你开箱得到跨 SQLite/PG/MySQL 的持久会话、可跨崩溃恢复的工作流、真正的沙箱接缝;你**不会**得到捆绑的 RAG 栈或上百个连接器。
- **选 power-loop**:当你想把 Agent *嵌入*现有应用、把依赖面压到最小、自选数据库,并在意持久化 + 隔离 + 稳定契约。
- **选框架**:当你想要开箱即用、庞大的集成目录,且不介意重量。

诚实地说:power-loop 在**生态广度上落后**(集成、社区、项目年龄),在**可嵌入性、持久化、存储灵活性、以及机器化守卫的稳定 API 上领先**。据此取舍。

---

## 安装与配置

```bash
pip install 'power-loop[openai]'      # 任意 OpenAI 兼容端点
pip install 'power-loop[anthropic]'   # 原生 Anthropic Messages API
pip install 'power-loop[postgres]'    # PostgreSQL 后端(asyncpg)
pip install 'power-loop[mysql]'       # MySQL 后端(aiomysql)
pip install 'power-loop[all]'         # transport + postgres + mysql + skills/pdf/可观测/mcp
```

指向任意 OpenAI 兼容端点(或 `POWER_LOOP_PROVIDER=anthropic`):

```bash
POWER_LOOP_BASE_URL=https://api.openai.com/v1
POWER_LOOP_API_KEY=sk-...
POWER_LOOP_MODEL=gpt-4o-mini
```

Python 3.10+。见 [快速开始](docs/zh/getting-started.md)。可选 extras:`postgres`、`mysql`、`skills`、`pdf`、`prometheus`、`statsd`、`otel`、`mcp`。

---

## 稳定性与 SemVer

自 **1.0** 起,**STABLE** API(列在 `power_loop.STABLE_API`)处于 SemVer 之下:破坏性变更需要大版本升级,由 CI 里的冻结基线测试强制——包括旗舰 `StatefulAgentLoop` *以及构造它所需的 LLM 契约*。错误 `.code` 字符串同样被冻结。自那以来的两次大版本升级(2.0 可插拔异步存储、3.0 正交的上下文轴)正是这条策略在起作用——破坏性变更换来一次大版本升级,各自记录在 [更新日志](CHANGELOG.md) 里。

| 层级 | 含义 |
|---|---|
| **Stable** | 同一大版本内向后兼容;在 `power_loop.STABLE_API` 中。 |
| **Provisional** | 从顶层 re-export(如 `open_store`、`SchemaPolicy`);未来 minor 可能调整。 |
| **Internal** | `power_loop.core.*`、`power_loop.runtime.store.*` 等内部;不作兼容承诺。 |

见 [API 参考](docs/zh/api/index.md)。

---

## 诚实声明

power-loop **做编排;它本身不做隔离。** 内置的 `bash`/文件工具在进程内运行、继承宿主环境——方便用于可信的本地场景,**不是安全边界**。对不可信/模型编写的命令,用 `ShellBackend` 接缝(工具级)注入沙箱,或让叶子走 `SubprocessExecutor` + `WorkerLauncher`(进程级)。密钥留在你的编排层。见 [SECURITY.md](SECURITY.md)。

**每会话单写者。** 每会话的顺序由进程内 `asyncio.Lock` 保证;它不提供跨进程互斥。用 **SQLite** 时,一文件一写者进程(把会话分片到多个文件)。用 **PostgreSQL/MySQL** 时,序号分配是多写者安全的(`SELECT … FOR UPDATE`),但 *pending 状态机*仍假设同一会话同一时刻只有一个写者驱动(上层的 dispatcher/队列由你负责)。在全新服务 schema 上的并发首次启动应带外预置(`SchemaPolicy.VERIFY`)。见 [扩展性指南](docs/zh/user-guide/scaling.md)。

**成熟度。** 这里的 1.0 标签是对 **API/持久化契约**的信心声明——不是多年实战检验的宣称。power-loop 还年轻、主要由单一维护者维护、公开生产记录有限。契约是机器化守卫的,项目是 MIT、可分叉;请按你的场景权衡 bus factor。

---

## 项目与链接

- **被谁使用:** DeepTalk —— 一款一对一关系型 IM 产品里会话内 Agent 的运行时。*(在生产用了它?欢迎 PR 加一行。)*
- **开发:** `pip install -e ".[dev]"` · `ruff check .` · `pytest -q --no-real`(去掉 `--no-real` 跑真机 LLM 套件;设置 `POWER_LOOP_TEST_PG_DSN` / `POWER_LOOP_TEST_MYSQL_DSN` 跑服务端后端的一致性套件)。
- [文档](docs/zh/index.md) · [架构](docs/architecture.md) · [存储后端](docs/zh/user-guide/storage-backends.md) · [更新日志](CHANGELOG.md) · [贡献](CONTRIBUTING.md) · [安全](SECURITY.md) · [许可](LICENSE)
