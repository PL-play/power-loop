# power-loop 文档与示例升级计划

> 目标：将 power-loop 从内部库文档升级为**开源级专业文档**，中英双语、用户手册 + 教程 + API 参考 + 示例全覆盖。
> 预计总工期：**5–7 天**（按两天完成一个 Phase 估算）。

---

## 现状分析

| 维度 | 当前 | 差距 |
|---|---|---|
| 语言 | README + docs 全中文 | 无英文，海外用户无法上手 |
| 文档类型 | 4 篇 reference（架构/hooks/events/providers） | 缺少 tutorial、how-to、FAQ、migration |
| 示例 | 15 个（00–14），每个 ~100 行 | 无 skills 示例、无多 provider 示例、无真实场景 mock |
| 图表 | 9 个 Mermaid（仅在 architecture.md） | 教程区无图、Quickstart 无图 |
| 项目元信息 | 无 CONTRIBUTING.md、无 CODE_OF_CONDUCT、无 LICENSE | 开源标配缺失 |
| API 文档 | README 内嵌表格 | 无独立 API reference、无符号索引 |
| 版本 | v0.2.0 tag | 无 release notes page |

---

## Phase 1：中英双语基础设施 + 文档站骨架（1.5 天）

### 1.1 文档站框架

```
docs/
├── README.md                  # 文档站入口（中英双语）
├── _config.yml                # 可选：GitHub Pages / mdBook 配置
├── en/                        # 英文文档
│   ├── index.md               # 英文入口
│   ├── getting-started.md     # 5 分钟上手
│   ├── user-guide/            # 用户手册
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   ├── configuration.md
│   │   ├── sessions.md
│   │   ├── tools.md
│   │   ├── subagents.md
│   │   ├── hooks.md
│   │   ├── events.md
│   │   ├── memory.md
│   │   ├── retry-cancel.md
│   │   ├── structured-output.md
│   │   └── compaction.md
│   ├── tutorials/             # 教程
│   │   ├── build-a-chatbot.md
│   │   ├── tool-calling.md
│   │   ├── human-in-the-loop.md
│   │   └── multi-agent.md
│   ├── api/                   # API 参考
│   │   ├── stateful-loop.md
│   │   ├── config.md
│   │   ├── hooks.md
│   │   ├── events.md
│   │   ├── tools.md
│   │   └── errors.md
│   ├── architecture.md        # 架构（翻译 + 补充）
│   ├── providers.md           # Provider 配置（翻译）
│   ├── examples.md            # 示例索引
│   ├── migration.md           # 迁移指南
│   ├── faq.md
│   └── contributing.md
├── zh/                        # 中文文档（镜像结构）
│   └── ...（同 en 结构）
├── architecture.md            # 保留（中文，向后兼容）
├── hooks.md                   # 保留
├── events.md                  # 保留
└── providers.md               # 保留
```

### 1.2 中英双语切换机制

- 每个 Markdown 文件顶部加语言切换链接 `[English](../en/xxx.md) | [中文](../zh/xxx.md)`
- README.md 根目录保持中文为主，顶部显眼位置放 `[English Documentation](docs/en/index.md)`
- 不引入 i18n 框架（gettext/po），纯 Markdown 文件结构实现双语

### 1.3 图表策略

- 所有 Mermaid 图表内嵌在 `.md` 文件中，GitHub 原生渲染
- 每个核心概念至少一张图：
  - 架构总览（已有）
  - send 全链路（已有）
  - Pipeline 一回合（已有）
  - Session 生命周期（已有）
  - Pending 状态机（已有）
  - Compaction 流程（已有）
  - **新增**：Quickstart 流程图（最小用法 → 多轮 → 工具 → 子代理 → 持久化）
  - **新增**：Hook 决策树（"我该用哪个 HookPoint？"）
  - **新增**：Event 时序图（一份完整 send 的所有事件发射顺序）
  - **新增**：Memory 生命周期图（recall → inject → remember）
  - **新增**：Retry 状态机（attempt → sleep → retry → degraded / cancel）

---

## Phase 2：英文文档全量翻译 + 教程撰写（2 天）

### 2.1 核心文档翻译（en/）

| 文件 | 来源 | 内容 |
|---|---|---|
| `en/getting-started.md` | 新写 | 5 分钟从零到第一条回复 |
| `en/user-guide/installation.md` | README §2 | pip install + env 配置 |
| `en/user-guide/quickstart.md` | README §3 | 扩写 3.1–3.6 为独立教程 |
| `en/user-guide/configuration.md` | README §7 + docs/providers.md | 合并改写 |
| `en/user-guide/sessions.md` | 新写 | SessionStore 详解 + 跨进程恢复 |
| `en/user-guide/tools.md` | README §3.3 + 新写 | 工具注册、校验、async handler |
| `en/user-guide/subagents.md` | README §3.4 + 新写 | spawn_agent + AgentSpec |
| `en/user-guide/hooks.md` | docs/hooks.md | 翻译 + 补 memory.recalled |
| `en/user-guide/events.md` | docs/events.md | 翻译 + 补 M1.1/M1.9 事件 |
| `en/user-guide/memory.md` | 新写 | MemoryProvider 协议 + 示例 |
| `en/user-guide/retry-cancel.md` | 新写 | LLMRetryPolicy + CancellationToken |
| `en/user-guide/structured-output.md` | 新写 | StructuredOutputSpec + parse_structured |
| `en/user-guide/compaction.md` | 新写 | DefaultCompactor 详解 |
| `en/architecture.md` | docs/architecture.md | 翻译 + 内容更新 |
| `en/providers.md` | docs/providers.md | 翻译 |
| `en/migration.md` | 新写 | 0.1.x → 0.2.0 迁移指南 |
| `en/faq.md` | 新写 | 常见问题 |

### 2.2 教程（en/tutorials/）

每个教程 = 目标说明 + 代码 + 运行结果 + 关键概念解释 + 下一步。

| 文件 | 概念覆盖 | 难度 |
|---|---|---|
| `build-a-chatbot.md` | install → send → multi-turn → sessions | 入门 |
| `tool-calling.md` | ToolDefinition → register → invoke → 多轮工具 | 入门 |
| `human-in-the-loop.md` | TOOL_BEFORE hook → async approve → cancel | 进阶 |
| `multi-agent.md` | spawn_agent → AgentSpec → 子代理生命周期 | 进阶 |

### 2.3 API 参考（en/api/）

每个 API 文件 = 签名 + 参数表 + 返回值 + 示例 + 相关链接。

| 文件 | 覆盖符号 |
|---|---|
| `stateful-loop.md` | `StatefulAgentLoop` + `StatefulResult` |
| `config.md` | `AgentLoopConfig` + `LLMRetryPolicy` + `LLMProviderConfig` |
| `hooks.md` | `AgentHooks` + `HookPoint` + `HookDirective` + 所有 `*Ctx` |
| `events.md` | `AgentEventBus` + `AgentEventType` + 所有 `*Payload` |
| `tools.md` | `ToolRegistry` + `ToolDefinition` + `AsyncToolInSyncContext` |
| `errors.md` | 所有 `PowerLoopError` 子类 |

---

## Phase 3：示例体系升级（2 天）

### 3.1 示例重命名 + 编号对齐

当前 00–14 按概念递增，但编号与 ROADMAP 的 M2.3 目标名不完全一致。重命名为**场景驱动**的名称，同时保留编号：

```
examples/
├── 00_hello_world.py              # 最小用法（不变）
├── 01_multi_turn_chat.py          # 多轮对话（原 01）
├── 02_tool_calling.py             # 工具调用（原 02）
├── 03_subagent_delegation.py      # 子代理（原 03）
├── 04_compaction.py               # 压缩（原 04）
├── 05_pending_recovery.py         # 悬挂态恢复（原 05）
├── 06_declarative_subagent.py     # 声明式子代理（原 06）
├── 07_human_approval.py           # 用户确认（原 07）
├── 08_streaming.py                # 流式输出（原 08）
├── 09_audit_log.py                # 审计日志（原 09）
├── 10_concurrent_sessions.py      # 并发会话（原 10）
├── 11_cross_process_resume.py     # 跨进程恢复（原 11）
├── 12_retry_and_cancel.py         # 重试取消（原 12）
├── 13_memory_sqlite.py            # 跨会话记忆（原 13）
├── 14_structured_card.py          # 结构化输出（原 14）
├── _helpers.py                    # 共享（不变）
└── README.md                      # 新增：示例索引
```

### 3.2 新增示例

| 文件 | 学到什么 | 优先级 |
|---|---|---|
| `15_skills_from_markdown.py` | 加载 SKILL.md → 转 system prompt → Agent 按 skill 行动 | 高 |
| `16_custom_compactor.py` | 实现 Compactor 协议 → 注入自定义压缩策略 | 中 |
| `17_custom_memory_provider.py` | 实现 MemoryProvider → HTTP API 后端 | 中 |
| `18_multi_provider.py` | 用 `LLMProviderConfig` 切换三家 provider | 中 |
| `19_full_chatbot.py` | 综合示例：session + tools + hooks + events + memory + compaction | 高（旗舰示例） |

### 3.3 每个示例的文档化

每个示例文件头部 docstring 升级为标准格式：

```python
"""标题 · 一句话描述

## What you'll learn
- 要点 1
- 要点 2

## Prerequisites
- 需要 .env 配置 OPENAI_COMPAT_*

## Run
    python examples/NN_name.py

## Expected output
    $ python examples/NN_name.py
    [output ...]

## Key concepts
- 概念 A：简短解释
- 概念 B：简短解释

## Next
下一步看 `examples/MM_name.py`
"""
```

---

## Phase 4：项目元信息 + 发布准备（1 天）

### 4.1 开源标配文件

| 文件 | 内容 |
|---|---|
| `CONTRIBUTING.md`（中英） | 如何贡献：环境搭建、代码规范、PR 流程、CLA 声明 |
| `CODE_OF_CONDUCT.md`（英） | Contributor Covenant 2.1 |
| `LICENSE` | MIT 或 Apache 2.0（需用户确认） |
| `SECURITY.md`（英） | 安全漏洞报告流程 |
| `.github/ISSUE_TEMPLATE/bug_report.md` | Bug 模板 |
| `.github/ISSUE_TEMPLATE/feature_request.md` | Feature 模板 |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR 模板 |

### 4.2 发布检查清单

- [ ] `pyproject.toml` 补 `description` / `readme` / `homepage` / `repository`
- [ ] `CHANGELOG.md` [Unreleased] 合并到 v0.2.0 条目
- [ ] `__version__` 更新为 `"0.2.0"`
- [ ] GitHub Release 页面写 release notes
- [ ] PyPI 发布准备（`python -m build` + `twine check`）

---

## 执行优先级

| 优先级 | Phase | 理由 |
|---|---|---|
| **P0** | Phase 1（文档站骨架 + 图表） | 基础架构，后续所有文档的容器 |
| **P0** | Phase 2 核心（getting-started + quickstart + 核心 user-guide） | 新用户第一天就看这些 |
| **P1** | Phase 3 示例升级 + 旗舰示例 | 开源项目的"活文档"，最直接的信任来源 |
| **P1** | Phase 2 剩余（tutorials + API ref + FAQ） | 深度用户查阅 |
| **P2** | Phase 4（元信息 + 发布准备） | PyPI 发布前补齐即可 |

---

## 建议推进顺序

1. **Phase 1 全部**（骨架 + 图表）→ 1.5 天
2. **Phase 2 核心**（英文 getting-started + quickstart + installation + configuration + tools + sessions）→ 1 天
3. **Phase 3 示例**（重命名 + 新示例 + docstring 升级）→ 1.5 天
4. **Phase 2 剩余**（tutorials + API ref + FAQ + migration）→ 1 天
5. **Phase 4**（元信息 + 发布）→ 0.5 天

**总计 5.5 天。** 是否需要调整优先级或增减内容？