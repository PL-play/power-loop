# power-loop ROADMAP

> 目标：在开始 DeepTalk `agent` 服务集成 **之前**，把 power-loop 做成一个独立可发布、定位清晰、短板补齐的库。集成期间发现的真实需求按需回流。
>
> 总节奏：**M0 → M1 → M2 完成即可开始 DeepTalk**；**M3 与 DeepTalk 并行、按需触发**。
> 预计时长：M0 ≈ 2–3 天，M1 ≈ 7–10 天，M2 ≈ 4–5 天，共 **约 3 周**。
>
> 最后更新：2026-06-04。

---

## 定位（约束所有规划的总纲）

power-loop 是 **可嵌入的 Agent 执行内核**，提供：

1. LLM 抽象 + 多轮消息循环 + 工具调用 + 生命周期 Hook + 事件总线；
2. **声明式动态子代理**（`AgentSpec` JSON → 一次性 sub-loop）；
3. **运行时工具注册**（受控 meta-tool）；
4. **上下文压缩**（运行时必备，默认开启） + 会话持久化（可选）。

不做：业务语义（IM/会话/账号）、长期记忆 / 向量库 / RAG、Planner / 编排 DAG、UI。

API 稳定性：`power_loop/__init__.py` 暴露的为 **Public API**，破坏性变更必须 minor 版本号 + CHANGELOG。其余视为 internal，可随时变更。

---

## 四条护栏

1. **API 表面在 M0 就锁定**，之后所有改动都问"会破坏 Public API 吗"。
2. **M1/M2 期间禁止为 DeepTalk 写专有逻辑**；可写进 `examples/`，不进 `power_loop/`。
3. **每个 milestone 收尾必须更新 README + CHANGELOG**，否则该 milestone 不算完。
4. **测试策略**：
   - `pytest` 本地默认 **跑真实 API**（marker `real_llm` 默认 run，`--no-real` 反向跳过）。
   - CI 无 key → 自动 skip `real_llm`，保持 CI 绿。
   - 纯控制流单测（hook directive、spec schema、tool registry 校验等）继续可用 fake LLM —— 不依赖模型行为，避免抖动。
   - **凡涉及 LLM 行为的 milestone 项（M1.1 重试 / M1.3 结构化输出 / M1.7a 压缩 / M1.8 spec 子代理 / M2.6 动态工具 等）必须至少配 1 个真实 LLM 集成测试**，不可只跑 fake。

---

## M0 · 工程化基线（2–3 天）

让仓库看起来像个库，而不是脚本集合。

- [ ] **Public API 固化**：`power_loop/__init__.py` 显式 export `AgentLoop / AgentLoopConfig / AgentLoopResult / AgentHooks / AgentEventBus / HookPoint / HookDirective / ToolRegistry / ToolDefinition`。
- [ ] **pyproject 元信息**：`license` / `classifiers` / `project.urls` / `keywords`；`__version__` 单一来源。
- [ ] **CHANGELOG.md**（Keep a Changelog 风格），从 v0.1.0 起记。
- [ ] **工具链统一**：
  - `ruff` line-length=120，启用 `E,F,I,UP,B`；
  - `mypy` strict-on-public-api（先只覆盖 `power_loop/agent`、`power_loop/contracts`、`power_loop/tools/registry.py`）；
  - `pytest.ini`：注册 marker `unit / integration / real_llm`，real_llm 默认 skip，`--run-real` 才跑。
- [ ] **CI（GitHub Actions）**：`ruff` + `mypy(部分)` + `pytest -m "not real_llm"`；real_llm 走 nightly + 手动触发。
- [ ] **测试目录重排**：`tests/{unit,integration,real}/`；smoke_* 归 unit，showcase 归 integration，test_real_* 归 real。
- [ ] **examples/ 占位**：建目录，放 `00_minimal.py`（与 README §5.1 同步）。

**完成判定**：CI 绿；`pytest -m "not real_llm"` 全过；`ruff check .` 0 错；example 00 可运行。

---

## M1 · 补关键短板（7–10 天）

每项 = **能力 + 单元测试 + 文档段落**。按列表顺序推进。

### M1.1 LLM 重试 / 超时 / 取消（最关键）

- `power_loop/runtime/retry.py`：`LLMRetryPolicy(max_attempts, backoff_initial, backoff_max, total_timeout, retry_on=(RateLimitError, TimeoutError, …))`。
- 接入 `AgentPipeline.call_llm`：失败按策略重试；整体超时 BREAK，结果 `status="degraded"`。
- `stop_event` 升级为 `CancellationToken`，兼容 `asyncio.Event` / `threading.Event` / `Callable[[], bool]`；每个 await 边界检查。
- 新事件：`llm.retry.attempted` / `llm.degraded` / `loop.cancelled`。
- 测试：连续失败 / 超时 / 用户取消三条路径。

### M1.2 历史窗口工具 `trim_history`

- `power_loop/runtime/budget.py`：
  - `estimate_tokens(messages, model)`（OpenAI 用 tiktoken，Anthropic 近似）；
  - `trim_history(messages, max_tokens, model, *, keep_system=True, keep_last_n=2)` —— 仅裁剪，不摘要（摘要走 M1.7）。
- 不进 pipeline 默认行为，作为业务侧 helper。
- 测试：不同预算下的裁剪结果固定快照。

### M1.3 结构化输出（卡片 JSON）

- `LLMRequest.response_format: dict | None`（OpenAI `json_schema` / Anthropic tool-use 两种适配）。
- `power_loop/runtime/structured.py`：
  - `StructuredOutputSpec(name, schema, examples=None)`；
  - `parse_structured(response, schema) -> dict`，含 JSON 修复（提炼 pipeline 现有 `_tool_call_args` 的逻辑）。
- 测试：fake LLM 返回带噪声 JSON；schema mismatch 时报错清晰。

### M1.4 Provider 配置统一

- `power_loop/runtime/provider.py`：`LLMProviderConfig(provider, base_url, api_key, model, extra)`；
  - `create_llm_service_from_env(prefix="POWER_LOOP")`；
  - `create_llm_service_from_config(cfg)`。
- 现 `llm_factory.py` 内部各厂商分支保留，外部只暴露这一层。
- 文档：provider × env 变量对照表（写进 `docs/providers.md`）。
- 测试：mock env，三家以上 provider 实例化通过。

### M1.5 取消语义统一（与 M1.1 合并实现，单独验收）

- 在 `tool.after` 触发取消，验证不会再启动下一回合；
- 取消可由 hook 主动发起（`HookDirective.CANCEL` 或外部 token），二者等价。

### M1.6 async tool handler 工效学

- `ToolRegistry.invoke` 自动检测 async handler 并走 `invoke_async`，去掉 `test_real_streaming_subagent.py` Part 3 的 `get_event_loop().is_running()` hack。
- 测试：同名工具 sync / async 双版本。

### M1.7a **上下文压缩**（must-have，默认 ON，影响运行时正确性）

> 不做这条，长会话会 hard-fail 在 `context_length_exceeded`。这是运行时正确性问题，不是可选优化。

- `power_loop/runtime/compact.py`：
  - `Compactor` 协议 + `DefaultCompactor(trigger_ratio=0.75, keep_last_n=4, summary_max_tokens=512)`；
  - `AgentLoopConfig.compactor: Compactor | None = DefaultCompactor()`（**默认开启**，传 `None` 才关）。
- **不变量**（测试必须盯死）：
  1. **触发**：每回合 `round.start` 之前，若 `estimate_tokens(messages, model) ≥ max_tokens × trigger_ratio`；env `CONTEXT_COMPACT_THRESHOLD` 可覆盖为绝对 token 数。
  2. **幂等**：同一回合最多压缩 1 次，标记 `round_compacted=True`。
  3. **保留段**：所有 `role=system` 消息 + 最后 `keep_last_n` 条对话；**未完成的 `assistant(tool_calls)` ↔ `tool(tool_call_id=…)` 对必须作为原子单元**，不可被切开。
  4. **摘要消息形状**：`{role: "system", name: "compact_note", content: <text>, _meta: {compacted_at_round, original_count, original_tokens}}`，注入位置在原 system 之后、保留段之前。
  5. **摘要 LLM 调用**：走 M1.1 `LLMRetryPolicy`，独立超时（默认 30s）。
  6. **软降级**：压缩失败 → 发 `compact.failed` 事件 → 继续用未压缩 history 跑该回合；若该回合 LLM 因 context overflow 失败 → 升级为 `loop.degraded` 终止。
- 钩子：`compact.before` / `compact.after` 仍可被业务方拦截、改写或注入自定义 compactor。
- 事件：`compact.triggered / compact.completed / compact.failed`，payload 含 `before_tokens / after_tokens / messages_removed`。
- 测试：
  - 人造长 history → 自动触发 → 消息数下降，摘要注入正确位置；
  - 中段含 `tool_calls / tool` 配对 → 验证不被切开；
  - mock 压缩 LLM 抛错 → 软降级路径走通；
  - 触发后同回合再判定 → 不重复压缩；
  - `compactor=None` → 不压缩，长 history 直接报 context overflow（验证关闭语义）。

### M1.7b 会话持久化（nice-to-have，独立交付）

> 与压缩解耦；DeepTalk MVP 不依赖（事实源在 PG / api）。若 M1 时间紧可推迟到 M2 或 M3。

- `power_loop/runtime/session_store.py`：`SessionSnapshot` dataclass（`messages / usage / round_index / session_id / compacted_notes / metadata`）；`save(path) / load(path)`。
- 明确**不**持久化：hooks / event_bus / llm / tool_registry / stop_event；由调用方重建。
- `AgentLoop.run(..., resume_from: SessionSnapshot | None = None)`。
- 测试：跑 N 轮 → snapshot → 新实例 resume → 继续跑 → 结果一致。

### M1.9 **MemoryProvider 协议 + 生命周期接线**（新增）

> 来源：长期记忆 / 跨会话连续性的分层落地。**库内零实现**，只提供协议、接线、参考示例。

- **协议**（`power_loop/runtime/memory.py`）：
  ```python
  class MemoryProvider(Protocol):
      async def recall(self, *, messages, session_id, budget_tokens=1500) -> list[dict]: ...
      async def remember(self, *, snapshot, session_id) -> None: ...
  ```
- **接线**：
  - `session.start` 之前，若 `AgentLoopConfig.memory` 非空 → 调 `memory.recall(...)` → 结果作为 `role=system, name=memory_*` 消息插入在原 system 之后、`compact_note` 之后（与压缩注入位置统一，cache 友好）。
  - `session.end` 时 → 调 `memory.remember(snapshot)`；失败软降级（warn 不阻断主流程）。
  - 新增 hook 点 `memory.recalled`：业务可拦截 / 改写召回结果（去敏、限流、A/B）。
- **不变量**：
  - 召回的消息**不会被压缩**（它们是 system 区一员）；
  - 召回失败 → 返回 `[]` 视为无记忆，loop 继续，不报错；
  - `remember` 失败 → 不影响 `result` 返回，但发 `memory.failed` 事件。
- **参考实现**（仅在 `examples/`，不在库内）：
  - `09a_memory_facts_sqlite.py` —— 键值事实库，SQLite 后端；
  - `09b_memory_diary_via_api.py` —— 摘要日记，DeepTalk-style HTTP api 后端（mock）；
  - `09c_memory_semantic_chroma.py` —— 向量检索 RAG，Chroma 后端。
- 测试：
  - fake MemoryProvider 验证注入顺序、注入位置；
  - `recall` raise → 返回 `[]` 软降级；
  - `remember` raise → 不影响 `result.status` + 发 `memory.failed`；
  - `memory.recalled` hook SKIP 时不注入。

### M1.8 **AgentSpec 与 dynamic sub-agent**（新增）

> 来源：定位调整中"动态运行时调用"in-process 部分。Subprocess 留 M3。

- `power_loop/runtime/spec.py`：
  - `AgentSpec`（pydantic）：`{name, system_prompt, model, tools: list[str] | None, max_rounds, max_tokens, temperature, response_format, hooks_preset: str | None}`；
  - **严格 schema**：未知字段拒绝，便于上游 prompt 输出后校验。
- `run_agent_sync(spec: AgentSpec | dict | str, messages, *, parent_registry: ToolRegistry | None) -> AgentLoopResult`：
  - 在同进程内造一个 `AgentLoop` 跑一次；
  - `tools` 字段是工具白名单：从 `parent_registry` 里筛子集，避免主 agent 把全部能力都暴露给子 agent；
  - contextvars 自动隔离子 session 的 hook / event_bus。
- 同时暴露成 meta-tool `run_agent`（接 JSON 字符串），方便主 agent 在 LLM 阶段自己唤起。
- 测试：fake LLM 验证 spec 解析 / 工具白名单 / 嵌套深度限制（沿用 spawn_agent 的 depth guard）。

**M1 完成判定**：8 项各自有 unit + integration 测试；CHANGELOG 升 v0.2.0；README §6/§7 + Roadmap §10 同步更新；`docs/providers.md`、`docs/persistence.md` 写完。

---

## M2 · 通用化 / API 稳定化（4–5 天）

让外部敢用、敢长期依赖。

### M2.1 Public API 表

- README 增节："Public API（稳定）vs Internal（可变）"。
- 列出 v0.2.0 公开面，约定后续破坏性变更走 minor + CHANGELOG。

### M2.2 Hook 点 / Event 全表

- `docs/hooks.md`：每个 HookPoint 的 ctx 字段、可返回 directive、典型用途、示例片段。
- `docs/events.md`：每类 AgentEvent 的 payload 字段、触发时机、订阅示例。

### M2.3 examples/ 充实

每个独立可运行，README 链接：

- `01_single_shot_reply.py` —— DeepTalk Agent MVP 同款用法
- `02_tool_calling.py`
- `03_hooks_security_policy.py` —— real_showcase Level 4 抽出
- `04_subagent_delegation.py` —— `spawn_agent` + `run_agent(spec)` 对比
- `05_structured_card_output.py`
- `06_retry_and_cancel.py`
- `07_persistence_resume.py` —— M1.7 演示
- `08_dynamic_agent_spec.py` —— M1.8 演示

### M2.4 性能 smoke

- `bench/loop_no_tools.py`：10 回合空工具循环 + fake LLM，记录基线数字进 CHANGELOG。
- 非门禁；后续退化能看见即可。

### M2.5 错误体系

- 统一基类 `PowerLoopError`，子类：`ToolNotFound / ToolValidationError / LLMTimeout / LLMRetryExhausted / CancellationRequested / SpecValidationError / CompactionFailed`。
- 业务侧 `except PowerLoopError` 即可一把抓。

### M2.6 **动态工具注册（meta-tool）**（新增）

> 来源：定位调整中"动态工具注册（运行时塞脚本/命令）"。

- meta-tool `register_tool(name, description, input_schema, kind, interpreter, source, scope="session", timeout=30)`：
  - `kind ∈ {python_inline, shell, script_file, http}`；
  - 注册一个 wrapper handler，按 kind 走对应 executor；
  - **默认沙箱**：cwd 限定（复用 `runtime/env.py` 白名单）；超时强制 kill；stdout/stderr 截断；env 变量白名单；
  - **作用域**：默认 `session`（当前 loop 结束即丢）；可选 `persist_to=path` 写盘；
  - 默认在 `ToolRegistry` 中**关闭**，调用方显式 `registry.enable_dynamic_registration(policy=…)` 才开。
- 业务侧通过 `tools.batch.before` hook 可一票否决（DeepTalk 一定要关）。
- 测试：注册 python_inline / shell 两种；超时 kill；作用域回收；未启用时 LLM 调 `register_tool` 报清晰错误。

**M2 完成判定**：8 个 examples 全部可运行；`docs/hooks.md` / `docs/events.md` / `docs/providers.md` / `docs/persistence.md` / `docs/dynamic-tools.md` 齐全；README Roadmap 区段勾掉；**v0.2.0 正式 tag**。**到这里可以开始 DeepTalk 集成。**

---

## M3 · 高阶能力（按需触发，不前置）

候选清单。每项要等 DeepTalk 真撞上需求才启动。

- **Subprocess agent executor**：把 `run_agent(spec)` 的 executor 抽象成接口，新增 `subprocess` executor —— `python -m power_loop.runner --spec spec.json --input msgs.json`。spec 协议保持不变；IPC 走 stdin/stdout JSON，事件回传走管道。
- **断点 / 单步**：在指定 HookPoint 暂停，外部 controller 读取 / 修改 state 后 resume。基于 `asyncio.Event` + 一个轻量 `DebugController`。主要用于开发调试。
- **OpenTelemetry tracing**：把 AgentEvent 桥接到 OTel span，便于 DeepTalk admin 后台 trace。
- **流式 token 上屏的分片协议**：与 gateway / 客户端协议联调时再做（DeepTalk PRD §8 后置）。
- **多模型路由**：成本/能力路由器；DeepTalk 风险§4 后置。
- **Skill 系统打磨**：`runtime/skills.py` 当前粗糙，等真要给 Agent 加可热插能力时再回头。
- **更多 provider**：按 DeepTalk 实际选型推动；MVP 锁单家。

**显式不做**（避免手痒）：

- ❌ **长期记忆 / 向量库 / RAG**。DeepTalk 的"记忆"是 PG 里的会话历史 + 关系信息，由 api 提供，agent 服务拼进 messages 即可。库里强行做反而冲突。
- ❌ **Planner / DAG workflow**。`run_agent(spec)` + 主 agent 决策已经覆盖 90% 动态 workflow 场景；DAG 抽象会过度。
- ❌ **业务语义（会话 / 用户 / IM 协议）**。属于 DeepTalk `agent` 服务壳层。

---

## 版本与里程碑

| 里程碑 | 版本 tag | 含义 |
|---|---|---|
| M0 完成 | `v0.1.0` | 工程化基线打住 |
| M1 完成 | `v0.2.0-m1` | 关键短板补齐（pre-release） |
| M2 完成 | `v0.2.0` | 通用化 / API 稳定 / 可发布 |
| M3 起 | `v0.3.x+` | 按需迭代，DeepTalk 并行驱动 |

每个 milestone 提交独立 tag，失败可回滚。

---

## 与 DeepTalk 的衔接

DeepTalk `agent` 服务的开工条件 = **M2 v0.2.0 tag 完成**。届时：

- DeepTalk 直接 `pip install -e ../power-loop`（或锁版本）；
- 在 DeepTalk 侧只写业务壳（Kafka 消费 / 触发判定 / 调 api / 双方授权 / 降级）；
- 集成中暴露的通用需求按 M3 候选清单回流。
