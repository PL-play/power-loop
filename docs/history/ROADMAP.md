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

### M1.1 LLM 重试 / 超时 / 取消（最关键）✅ 2026-06-05

- `power_loop/runtime/retry.py`：`LLMRetryPolicy(max_attempts, backoff_initial, backoff_max, total_timeout, retry_on)` + `with_retry(call, *, policy, token, on_retry=None)`。
- 接入 `AgentPipeline.call_llm`：失败按策略重试；整体超时 / 重试耗尽 → pipeline 翻译成 `status="degraded"` 并 append 合成 assistant 消息。
- `stop_event` 升级为 `CancellationToken`（`runtime/cancellation.py`），兼容 `asyncio.Event` / `threading.Event` / `Callable[[], bool]` / `None`；retry backoff sleep cancel-aware。
- 新事件：`LLM_RETRY_ATTEMPTED` / `LLM_DEGRADED` / `LOOP_CANCELLED`（+ 对应 Payload）。新错误：`LLMTimeout` / `LLMRetryExhausted` / `CancellationRequested` / `CompactionFailed`。
- 测试：`tests/unit/test_retry_cancel.py`（12 个）+ `tests/real/test_real_retry.py`（2 个真实 LLM）。example `12_retry_and_cancel.py` 演示三条路径。

### M1.2 历史窗口工具 `trim_history` ✅ 2026-06-05

- `power_loop/runtime/budget.py`：
  - `trim_history(messages, max_tokens, *, keep_system=True, keep_last_n=2)` —— 纯裁剪（不摘要），保留 leading system + 最后 N exchanges + `assistant(tool_calls) ↔ tool` 原子对。
  - `estimate_tokens` / `estimate_text_tokens` 从顶层导出。
- 不进 pipeline 默认行为，作为业务侧 helper。
- 测试：`tests/unit/test_budget.py`（9 个，覆盖不同预算下的裁剪结果固定快照）。

### M1.3 结构化输出（卡片 JSON）✅ 2026-06-05

- `LLMRequest.response_format: dict | None` 已加；`_request_kwargs` 透传到 OpenAI 兼容 API。Anthropic tool-use 适配后置（DeepTalk MVP 单 provider）。
- `power_loop/runtime/structured.py`：
  - `StructuredOutputSpec(name, schema, strict=True, description, examples)` + `.to_openai_response_format()`。
  - `parse_structured(output, *, schema=None)`：直接 → 围栏剥离 → 抓平衡 `{...}` → 修尾逗号；失败抛 `StructuredOutputError(reason, raw_text, detail)`，原因 `no_json` / `invalid_json` / `not_object` / `missing_required:<field>`。
  - 本地仅校验 `type=="object"` + 顶层 `required`；深层校验留 provider strict mode。
- 测试：`tests/unit/test_structured.py`（14 个）+ `tests/real/test_real_structured.py`（1 个真实 LLM）。example `14_structured_card.py` 三段（真实抽取 / 修复 / schema 缺字段）。

### M1.4 Provider 配置统一 ✅ 2026-06-05

- `power_loop/runtime/provider.py`：`LLMProviderConfig` + `from_env(prefix, fallback_prefix, env=None)` + `to_openai_compatible()` + `create_llm_service_from_config()` / `create_llm_service_from_env()`。
- `OPENAI_COMPAT_*` 作为 fallback 前缀，老 `.env` 无须改名即可工作；缺必填字段在构造时抛 `ValueError`（非首个 `complete()` 时）。
- 文档：`docs/providers.md`（变量表 + OpenAI / DashScope / DeepSeek / 本地 OpenAI-compatible 4 个 snippet + 迁移指引）。
- 测试：`tests/unit/test_provider.py`（11 个，覆盖必填守卫 / 前缀优先 / 回退 / 三家参数化 / 适配回环）。`provider` 字段当前仅是 informational tag，M3 引入第二条 transport 时升级为路由 key。

### M1.5 取消语义统一（与 M1.1 合并实现，单独验收）

- 在 `tool.after` 触发取消，验证不会再启动下一回合；
- 取消可由 hook 主动发起（`HookDirective.CANCEL` 或外部 token），二者等价。

### M1.6 async tool handler 工效学 ✅ 2026-06-05

- `register()` 用 `inspect.iscoroutinefunction` 在登记时缓存 `is_async`，覆盖 `async def` 与 `async __call__` 两类 callable。
- `invoke()`（sync）对 async handler 抛 `AsyncToolInSyncContext`，错误指向 `invoke_async`；`invoke_async()` 同时处理 sync + async，保留「sync handler 返回 awaitable」的回退路径。
- 旧 `test_real_streaming_subagent.py` hack 在 stateful 重构时已随旧测试删除，本次 polish 让上游 API 工效学也补齐。
- 测试：`tests/unit/test_tool_registry_async.py`（7 个）。

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

### M1.9 **MemoryProvider 协议 + 生命周期接线**（新增）✅ 2026-06-05

> 来源：长期记忆 / 跨会话连续性的分层落地。**库内零实现**，只提供协议、接线、参考示例。

**实现状态**：协议 / Snapshot / pipeline 注入 / 软失败 / `MEMORY_RECALLED` hook / 两类事件全部落地；`tests/unit/test_memory.py`（6 个）+ `examples/13_memory_sqlite.py` 跨 session 演示通过。三个候选 example（09a-c）按 ROADMAP 标注「仅在 examples」原则只保留 SQLite 实现，HTTP API / vector 留给真正用上的人写。

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

### M2.1 Public API 表 ✅ 2026-06-05

- README §5 末尾增节："Public API 稳定性约定"，STABLE 24 符号全表（一句话功能描述）+ PROVISIONAL / INTERNAL 边界。
- Examples 表补齐 11–14；§7 环境变量更新为 `POWER_LOOP_*` + `create_llm_service_from_env()`。

### M2.2 Hook 点 / Event 全表 ✅ 2026-06-05

- `docs/hooks.md` 补 §3.9 `memory.recalled` hook（Ctx 字段 / SKIP / 双方授权示例）。
- `docs/events.md` 补 §2.7 LLM retry/cancel lifecycle（3 事件）+ §2.8 Memory（2 事件），含完整 payload 表、触发时机、典型订阅者。

### M2.5 错误体系 ✅ 2026-06-05

- 新增 `ToolNotFound(tool_name)` / `ToolValidationError(tool_name, message)` / `SpecValidationError(message, *, field=None)`，全部 `PowerLoopError` 子类。
- `AgentSpecError` 改继承 `SpecValidationError`（不再直接从 `ValueError`）；旧 `except AgentSpecError` 继续有效。
- `ToolRegistry.invoke / invoke_async` 对 unknown tool / invalid args 现在 raise；pipeline `execute_tool` 内部 catch 并返回 `(str(exc), True)` 使 LLM 可见。
- `ToolRegistry.validate` 保留为 internal legacy 接口（仍返回 `str | None`）。

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
