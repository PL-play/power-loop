# Changelog

本项目采用 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格，
并遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

破坏性变更只能出现在 minor（0.x 阶段）或 major 版本，
并且必须在此文件中显式列出受影响的 Public API。

## [Unreleased]

## [0.11.0] — 2026-06-12

### Added

- **持久定时唤醒（durable timers）**：store 新表 `timers`（timer = 数据非任务，
  跨重启存活，随 session 级联删除）。
  - Agent 侧默认工具：`schedule_wakeup(delay_seconds, note)` /
    `list_wakeups` / `cancel_wakeup(timer_id)` / `current_time`（不在任何
    preset 里，按需 `get_tool_definitions(include=[...])` 注册）。
  - 宿主侧 API：`loop.schedule_timer(sid, delay_s=|due_at_ms=, note=)` /
    `loop.cancel_timer` / `loop.list_timers`（与工具写同一批行）。
  - **`TimerRunner(loop)`**：进程内扫描器——`start()` 时回收 stale `firing`
    行（at-least-once，可能二次投递），到期 CAS 认领后经 **`follow_up`**
    投递（空闲 = send，运行中 = 轮边界注入；进会话只有一条路）。
    不启动 runner（或外部调度器轮询 `store.due_timers()`）则永不触发。
  - **`HookPoint.TIMER_FIRE`** + `TimerFireCtx`：投递前编排否决点——
    CONTINUE 投递 / SKIP 跳过 / BREAK 取消 / `postpone_s` 改期，可改写
    投递文本；无 hook 默认投递。
  - 事件 `timer_fired`（`TimerFiredPayload`，outcome: delivered / queued /
    skipped / cancelled / postponed / error）。
  - `loop.hooks` / `loop.event_bus` 公开只读属性。
- 新示例 `examples/26_timers.py`；新增单测 `tests/unit/test_timers.py`（9 个）。

## [0.10.0] — 2026-06-12

### Added

- **`AgentLoopConfig.max_tokens_per_run`**：per-run 真实 token 预算护栏。轮边界
  检查（越界的那一轮完整结束，不留未决 tool_calls），命中后 status =
  `budget_exceeded`（新 LoopStatus 值），发 `status_changed`
  （`BudgetExceededStatusPayload`，kind="budget_exceeded"）。默认关闭。
- **Session 统计**：store 新表 `session_stats`（每次 send 结束累加一次：sends /
  rounds / llm_calls / tool_calls / prompt / completion / total tokens /
  first_send_at / last_send_at），随
  `close_session` 级联删除；新 API `loop.get_session_stats(sid)` /
  `loop.list_session_stats()`（`SessionStatsRow`）。注意 `usage_rounds` 表按
  (session, round_index) 覆盖、跨 send 不可累计——记账请用 session_stats。
- **`power_loop.contrib.logging_sink.attach_logging_sink(bus)`**：标准结构化
  日志 sink，每个事件一行 JSON（stdlib-only，长字段截断），消灭每个接入方
  重写"事件→日志"胶水的重复劳动。

- `AgentLoopResult.tool_calls` / `StatefulResult.tool_calls`：本次 run 执行的
  工具调用次数（`ContextManager.tool_calls` 计数）。

### Changed

- **同步工具 handler 现在跑在工作线程**（`asyncio.to_thread`，contextvars 正常
  传播）：慢的同步工具不再阻塞事件循环和同进程的其它 session。需要留在事件循环
  线程的 handler 请改 `async def`。
- `@phase` 装饰器发布的 start/end 事件改为携带 `PhaseEventPayload`（typed
  `data`）——至此**所有**内部事件发射路径都保证 `event.data` 非 None（新增契约
  测试）。
- README：补「一个 store 文件 = 一个进程」的多进程边界声明（跨进程并发安全
  暂不实现，先文档约束）。

## [0.9.0] — 2026-06-12

### Added

- **`StatefulResult.usage` / `AgentLoopResult.usage`**：每次 `send`/run 返回累计
  token 用量（对该 run 的全部 LLM 调用求和：`prompt_tokens` / `completion_tokens` /
  `cache_read_tokens` / `reasoning_tokens` / `total_tokens` / `calls`）。此前只能
  订阅 `usage_updated` 事件自行累加（事件是单次调用量、覆盖式），编排方做成本
  记账需要自建 tracker——现在直接读返回值。
- **`ContextManager.usage_totals`**：`update_usage` 在保持 `token_usage`（末次
  调用）语义不变的同时累计总量。
- **`send(..., heal_pending=True)`**（含 `send_sync`）：session 因上一个 run
  被杀死在 tool-call 中途而带有未决 `tool_calls` 时，自动 `abort_pending` 后
  继续本次 send，不再抛 `SessionPendingError`。默认仍为 raise（自愈会丢弃
  未完成的工具结果，应由调用方显式选择）。
- `SessionPendingError` 报错信息补充三条恢复路径指引（`resume` /
  `abort_pending` / `heal_pending=True`）。
- 文档：README 增加 token 记账与 heal_pending 小节、明确「无内置定时器」的
  范围边界；events 文档明确 `usage_updated` 为单次调用量、整 run 总量应读
  `result.usage`、handler 中 `event.payload` 按 dict 取值。
- 新示例 `examples/25_token_usage.py`；新增单测 `tests/unit/test_usage_and_heal.py`。

### Notes

- 本版本由 DeepTalk 多 agent 编排落地反推：token 成本面板需要 per-run 用量、
  人类中断 run 后 session 被未决 tool_calls 卡死，是两处真实暴露的不足。

## [0.8.1] — 2026-06-11

### Fixed

- `SQLiteNoteMemory.remember` 的签名对齐 `MemoryProvider` 协议（keyword-only
  `snapshot=` / `session_id=`）；0.8.0 的位置参数签名会在 session 结束时触发
  `MEMORY_FAILED`（软失败，不影响回复，但有噪音日志）。

## [0.8.0] — 2026-06-11

### Added

- **Agent-authored notes（自主记忆）**：模型用工具自己维护的持久笔记，存在 session store 新增的
  `notes` 表里（按 session 隔离，`close_session` 级联删除）。
  - 新默认工具 `note_add` / `note_update` / `note_delete`（进入 `full` preset；`core`/`explore` 不含）。
  - `SQLiteNoteMemory`：实现 `MemoryProvider` 协议，`recall()` 把该 session 的笔记渲染成一条
    `memory_notes` system 消息每次 send 注入；`remember()` no-op（写入由工具实时完成）。
  - `NotesPolicy(max_notes=50, max_note_chars=1000, inject_max_chars=8000, eviction="reject")`：
    默认**拒绝式**容量——满了报错并提示模型先删/合并（静默遗忘是 agent 记忆最糟的失败模式）；
    `eviction="fifo"` 切换为队列式淘汰（pinned 永不被自动淘汰）。注入超预算时优先隐藏最老的
    未 pin 条目并在文本中声明隐藏数量。
  - `AgentLoopConfig.notes_policy` 字段；顶层导出 `NotesPolicy` / `NotesFullError` /
    `SQLiteNoteMemory` / `DEFAULT_NOTES_POLICY` / `render_notes`。
  - `SessionStore` 新方法：`add_note` / `update_note` / `delete_note` / `list_notes` / `count_notes`，
    `NoteRow` dataclass。旧数据库自动建表，完全向后兼容。
  - 新示例 `examples/24_agent_notes.py`；单测 `tests/unit/test_notes.py`（18 例）。

## [0.7.2] — 2026-06-11

### Fixed

- Export `runtime_env_context` from the top-level package so the documented `bind=False` flow works without an internal import.
- Forward `tools=` and `system_prompt=` through `send_sync()` and idle `follow_up_sync()` as well as the async APIs.

### Docs and tests

- Document per-call overrides, unbound registries, and `ShellBackend` in the English and Chinese guides/API reference.
- Extend example 23 with a real unbound-registry invocation across two runtime workspaces.
- Add regression coverage for sync overrides and runtime resolution of unbound handlers.

## [0.7.1] — 2026-06-11

### Docs

- New **example 23** (`examples/23_per_send_overrides.py`) demonstrating per-call `tools=` allowlisting and `system_prompt=` override, plus a README "Per-call overrides" section. No code changes vs 0.7.0.

## [0.7.0] — 2026-06-11

### Added — Per-call overrides & cleaner public surface

- **`StatefulAgentLoop.send(..., tools=, system_prompt=)`** and **`follow_up(..., tools=, system_prompt=)`** — per-call overrides that do not mutate loop/session state. `tools` accepts a sequence of tool names (allowlisted from the loop registry) or a `ToolRegistry`; the model only *sees* the permitted subset. `system_prompt` overrides for that run only (precedence: per-call > session > config). Enables multi-tenant reuse of one cached loop without runtime hook gating.
- **`ToolRegistry.subset(names)`** and **`ToolRegistry.names()`** — derive a restricted registry.
- **`create_default_tool_registry(..., bind=False)`** — build an **unbound** registry whose handlers read the current `RuntimeEnv` at call time (caller supplies it per call via `runtime_env_context`); no eager workspace requirement. `DEFAULT_TOOL_HANDLERS` is now part of the public API.
- **`ShellBackend.session_key(workspace_dir)`** — the persistent `BashSession` is now cached by the backend's execution-target key, so swapping backends (e.g. local ↔ sandbox, or distinct sandbox containers) no longer needs ad-hoc rebuilds.

### Fixed

- Follow-up dropped on a terminal round (0.6.0) — see below; plus `__init__` export/lint hygiene (`FollowUpQueued`, `DEFAULT_TOOL_HANDLERS` now exported).

### Docs

- README: explicit **"orchestration, not isolation"** scope note — built-in `bash`/file tools run in-process and are not a security boundary; sandbox via the `ShellBackend` seam.

## [0.6.0] — 2026-06-11

### Fixed — Follow-up on a terminal round

- A follow-up enqueued during an otherwise-terminal round (model returned a final answer with no tool calls) was never drained — the queue only drained at the *next* round start, which never came. The loop now drains pending follow-ups before completing and runs another round to address them, so absorbed steering input is always processed.

## [0.5.0] — 2026-06-11

### Added — Pluggable shell backend

- **`runtime.exec_backend`** (`ShellBackend` protocol, `LocalShellBackend`, `DEFAULT_SHELL_BACKEND`) and **`RuntimeEnv.shell_backend`** — host code can route the persistent shell into an isolated sandbox (e.g. `docker exec`) instead of an in-process `/bin/bash`, without changing tool implementations. Default behavior unchanged.

## [0.4.1] — 2026-06-08

### Added — In-flight steering (`follow_up`)

- **`StatefulAgentLoop.follow_up()` / `follow_up_sync()`** — enqueue steering input while a session run is in flight; idle sessions degrade to `send()`.
- **`FollowUpQueued`** — immediate return shape when input is queued for the next pipeline round.
- **Round-boundary drain** — merged follow-ups append as a wrapped `<follow_up>` user message before `prepare_round`.
- **Example 22**, bilingual docs, and unit/real tests for the steering path.

### Added — M2.8 Anthropic Messages API 传输（2026-06-06）

- **`AnthropicMessagesLLMService`**（`llm_client.anthropic_factory`）—— 新增原生 Anthropic Messages API transport，复用统一 `LLMRequest` / `LLMResponse`。
- **`LLMProviderConfig.provider` 成为路由键**：`provider="anthropic"` / `"claude"` / `"dashscope-anthropic"` 使用 Anthropic transport；其它 provider 仍使用 OpenAI-compatible transport。
- **消息转换**：OpenAI-style `tool_calls` → Anthropic `tool_use` blocks；`role="tool"` → `tool_result` blocks；返回的 `tool_use` 统一转回 `LLMResponse.tool_calls`，pipeline 无需分支。
- **测试配置**：real LLM helper 改为 `create_llm_service_from_env()`，支持 `POWER_LOOP_*` 与 legacy `OPENAI_COMPAT_*` 两组环境变量。
- **版本**：`power_loop.__version__ = "0.4.0"`。

### Public API（M2.8 新增）

`AnthropicChatConfig` / `AnthropicMessagesLLMService` 可从子模块导入；顶层 `LLMProviderConfig` 的 `provider` 字段现在影响 transport 路由。

### Changed — M2.7 显式 Session 创建（2026-06-06）

- **`StatefulAgentLoop.new_session(metadata=None, system_prompt=None) -> str`** —— 新增显式会话创建入口。调用方先拿到 `session_id`，再传给每次 `send()` / `send_sync()`。
- **Breaking**：`StatefulAgentLoop.send(user_input, session_id, *, stop_event=None)` 与 `send_sync(...)` 现在必须传入 `session_id`；不再在首次 `send()` 时隐式创建 session。
- **Breaking**：`metadata` 从 `send(metadata=...)` 移到 `new_session(metadata=...)`。这样会话级信息在会话创建时固定，避免首条消息和会话生命周期耦合。
- **文档 / 示例 / 测试**：README、双语 docs、examples、unit/real 测试全部改为 `sid = loop.new_session(); await loop.send(..., session_id=sid)`。
- **版本**：`power_loop.__version__ = "0.3.0"`。

### Public API（M2.7 变更）

`StatefulAgentLoop.new_session` 顶层入口新增；`StatefulAgentLoop.send / send_sync` 签名破坏性变更，`session_id` 必填。

### Added — M1.1 LLM 重试 / 超时 / 取消（2026-06-05）

- **`LLMRetryPolicy`**（`power_loop.runtime.retry`）—— 配置 `max_attempts` / `backoff_initial` / `backoff_max` / `total_timeout` / `retry_on`。指数退避（capped），跨所有 attempt 共享总超时；退避 sleep 是 cancel-aware 的（cancel 触发时不会傻等到底）。
- **`with_retry(call, *, policy, token, on_retry=None)`** —— 库内通用 helper，pipeline 用它包 `await self.llm.complete(...)`。``CancellationRequested`` / ``asyncio.CancelledError`` 直接透传，不会被吞。
- **`CancellationToken`**（`power_loop.runtime.cancellation`）—— 统一 cancel 形状：`from_any(...)` 接受 `asyncio.Event` / `threading.Event` / `Callable[[], bool]` / 已存在的 token / `None`。自带 owned 模式（`token.cancel(reason)`），供 hook `HookDirective.CANCEL`（M1.5）和外部 controller 使用。`is_cancelled()` 对用户 callable 抛出做容错（**绝不让 cancel 检查本身污染主循环控制流**）。
- **`AgentLoopConfig.retry_policy: LLMRetryPolicy | None = None`** —— 默认 None（保持现有 fail-fast 行为）；显式赋值即开启。
- **新事件**：
  - `AgentEventType.LLM_RETRY_ATTEMPTED` + `LlmRetryAttemptedPayload(attempt, max_attempts, error_type, error_message, next_sleep_seconds)`
  - `AgentEventType.LLM_DEGRADED` + `LlmDegradedPayload(reason, attempts, error_type, error_message)` —— `reason ∈ {"retry_exhausted", "timeout"}`
  - `AgentEventType.LOOP_CANCELLED` + `LoopCancelledPayload(reason, round_index)`
- **新错误**（`power_loop.contracts.errors`，全部 `PowerLoopError` 子类）：
  - `LLMTimeout(elapsed, attempts, total_timeout)`
  - `LLMRetryExhausted(attempts, last_error)`（`__cause__` 保留 last error）
  - `CancellationRequested(reason)`
  - `CompactionFailed`（M2.5 占位）
- **`LoopStatus`** 新增 `"degraded"`。Pipeline 在 `call_llm` 抛 `LLMRetryExhausted` / `LLMTimeout` 时：append 一条合成的 `assistant` 消息（`[degraded: …]`），emit `LLM_DEGRADED`，`status="degraded"` 返回。`CancellationRequested` 翻译为 `status="cancelled"` + `LOOP_CANCELLED`。
- **Pipeline 内部统一**：`stop_event` 仍接受任意 cancel-like 对象（API 向后兼容），但内部统一存为 `CancellationToken`；`StatefulAgentLoop.send / send_sync / resume` 的 `stop_event` 类型放宽为 `CancellationLike`。
- **测试**：`tests/unit/test_retry_cancel.py`（12 个，覆盖 `with_retry` 直测 + token 各形态 + pipeline 端到端三条路径）；`tests/real/test_real_retry.py`（2 个真实 LLM 集成 —— 注入 transient 失败后真实 complete 通；全失败走 degraded 不打真实网络）。

### Public API（M1.1 新增）

`LLMRetryPolicy` / `with_retry` / `CancellationToken` / `CancellationLike` / `LLMTimeout` / `LLMRetryExhausted` / `CancellationRequested` / `CompactionFailed` / `LlmRetryAttemptedPayload` / `LlmDegradedPayload` / `LoopCancelledPayload` 全部从 `power_loop` 顶层导出；`AgentEventType.LLM_RETRY_ATTEMPTED` / `LLM_DEGRADED` / `LOOP_CANCELLED` 已加入枚举。

### Added — M1.9 MemoryProvider 协议（2026-06-05）

> 库内**零实现**：定义协议 + 接线 + 注入位置不变量。具体后端（SQLite / HTTP API / 向量库）一律留在调用方或 `examples/`。

- **`MemoryProvider` Protocol**（`power_loop.runtime.memory`）—— 两个方法：
  - `async recall(*, messages, session_id, budget_tokens=1500) -> list[dict]`
  - `async remember(*, snapshot: MemorySnapshot, session_id) -> None`
- **`MemorySnapshot`** dataclass —— `session_id / messages / final_text / rounds / status / metadata`，在 SESSION_END 时传给 `remember`。
- **`tag_as_memory(messages)`** —— 工具函数，把任意 dict 列表规范化成 `role=system, name=memory_*`。Pipeline 在注入前自动调用，业务方不必关心。
- **`AgentLoopConfig.memory: MemoryProvider | None = None`** + **`memory_budget_tokens: int = 1500`**。默认 None（保持原有行为）。
- **注入位置不变量**：召回结果插在 ``self.history`` 的「最长 leading `role=system` 段」之后、对话历史之前。这与 `compact_note` 同区，受压缩器系统区保留保护。
- **失败模型**（库强制不破坏主流程）：
  - `recall` 抛 → 视为返回 `[]`，emit `MEMORY_FAILED(phase="recall")`，loop 照常跑。
  - `remember` 抛 → emit `MEMORY_FAILED(phase="remember")`，`StatefulResult` 原样返回。
- **新 hook**：`HookPoint.MEMORY_RECALLED` + `MemoryRecalledCtx(recalled, session_id, budget_tokens)`。业务可在注入前 redact / 去敏 / `directive=SKIP` 跳过整批注入（典型场景：双方授权 gate）。
- **新事件**：`AgentEventType.MEMORY_RECALLED` + `MemoryRecalledPayload(returned, injected, budget_tokens)`；`AgentEventType.MEMORY_FAILED` + `MemoryFailedPayload(phase, error_type, error_message)`。
- **Pipeline 内部**：`_finalize` 多了一个 `rounds` 形参，使 `MemorySnapshot.rounds` 正确反映已完成回合数；老调用点（cancelled 早出）保留默认行为。
- **测试**：`tests/unit/test_memory.py`（6 个，覆盖注入位置 + tag 规范化 + recall 软失败 + remember 软失败 + 快照内容 + MEMORY_RECALLED SKIP）。
- **example**：`examples/13_memory_sqlite.py` —— SQLite 事实 KV，跨 session 把「我叫阿岚 / 喜欢 37」记忆带回。

### Public API（M1.9 新增）

`MemoryProvider` / `MemorySnapshot` / `tag_as_memory` / `MemoryRecalledCtx` / `MemoryRecalledPayload` / `MemoryFailedPayload` 顶层导出；`HookPoint.MEMORY_RECALLED`、`AgentEventType.MEMORY_RECALLED` / `MEMORY_FAILED` 入枚举。

### Added — M1.3 结构化输出（2026-06-05）

- **`LLMRequest.response_format: dict[str, Any] | None = None`** —— OpenAI 兼容 `response_format` 字段；`llm_factory._request_kwargs` 与 `_build_resume_request` 透传。
- **`StructuredOutputSpec(name, schema, strict=True, description=None, examples=...)`**（`power_loop.runtime.structured`）—— 声明式包装；`.to_openai_response_format()` 渲染成 `{"type":"json_schema","json_schema":{name, schema, strict, description}}`。
- **`parse_structured(output, *, schema=None) -> dict`** —— 四级修复链：
  1. 直接 `json.loads`
  2. markdown ```json``` 围栏剥离
  3. 抓出第一个**括号平衡**的 `{...}` 子串（跳过字符串里的引号）
  4. 修补**尾逗号** `,]` / `,}`
- **`StructuredOutputError(reason, raw_text, detail)`** —— 失败原因机器可读：`no_json` / `invalid_json` / `not_object` / `missing_required:<field>`，`raw_text` 截断到 1000 字符方便调试。
- **本地 schema 校验有限**：仅强制 `type=="object"` 与顶层 `required` 字段存在。更深的 type / enum / pattern 留给 provider 在 strict mode 服务端校验，**避免本地实现与 provider 静默分歧**。
- **测试**：`tests/unit/test_structured.py`（14 个）+ `tests/real/test_real_structured.py`（1 个真实 LLM —— card 抽取来回跑通）。
- **example**：`examples/14_structured_card.py` —— 真实 LLM 抽取 → 修复带噪 JSON → schema 缺字段失败三段。

### Public API（M1.3 新增）

`StructuredOutputSpec` / `parse_structured` / `StructuredOutputError` 顶层导出。`LLMRequest.response_format` 已在 `llm_client` 层落地。

### Added — M1.6 ToolRegistry async-handler 工效学（2026-06-05）

- **`async def` 自动识别**：`ToolRegistry.register()` 用 `inspect.iscoroutinefunction` 在登记时缓存 `RegisteredTool.is_async`，覆盖普通 `async def` 与 `async __call__` callable 对象两种形态。
- **`invoke()`（sync）对 async 处理器抛 `AsyncToolInSyncContext`**：取代之前「silently 返回未 await 的 coroutine」的隐式坑，错误信息明确指向 `invoke_async`。
- **`invoke_async()` 是通用入口**：async handler 直接 `await tool.handler(...)`；sync handler 跑完后若返回 awaitable 仍会被自动 await（保留向后兼容）。
- pipeline 早已用 `invoke_async`，业务侧无需改动；只有显式调 `invoke()` 把 async 当 sync 用的旧代码会立刻看到清晰报错。
- ROADMAP 里提到的 `tests/real/test_real_streaming_subagent.py` 的 `get_event_loop().is_running()` hack 已在 stateful 重构时随旧测试一并删除，本次 polish 把上游 API 工效学也补齐。
- **测试**：`tests/unit/test_tool_registry_async.py`（7 个）—— async 检测、callable 对象、sync-on-async 报错、双形态 invoke_async、sync-returning-awaitable 兼容。

### Public API（M1.6 新增）

`AsyncToolInSyncContext` 顶层导出。`ToolRegistry.invoke / invoke_async` 行为变更（前者更严格，对 async handler 抛清晰错；后者更优雅，省一次 `inspect.isawaitable`）—— 既有调 `invoke_async` 的代码完全不受影响。

### Added — M1.4 LLMProviderConfig 统一（2026-06-05）

- **`LLMProviderConfig`**（`power_loop.runtime.provider`）—— provider-agnostic 配置：`base_url` / `api_key` / `model` 必填，`provider` 标签（informational，今天只走 openai-compatible 一条 transport，预留 M3 多 transport 路由 key），加 `timeout_s` / `max_tokens` / `temperature` / `max_retries` 等默认值。
- **`LLMProviderConfig.from_env(prefix="POWER_LOOP", fallback_prefix="OPENAI_COMPAT", env=None)`** —— 读 `POWER_LOOP_*` 环境变量，缺则回退 `OPENAI_COMPAT_*`，**老 `.env` 无须改字段**。`env` 形参用于测试（注入 dict）。
- **`create_llm_service_from_config(cfg)` / `create_llm_service_from_env(*, prefix=…)`** —— 一行造服务；内部通过 `to_openai_compatible()` 适配现有 `OpenAICompatibleChatLLMService`。
- **失败模式**：必填字段缺失 → 构造时 `ValueError`（不是首个 `complete()` 时），让配置错误在 pytest 阶段就暴露。
- **docs/providers.md** —— 环境变量表 + 4 个 provider snippet（OpenAI / DashScope / DeepSeek / 本地 OpenAI-compatible）+ 老调用方式迁移指引。
- **测试**：`tests/unit/test_provider.py`（11 个）—— 必填守卫 / 主前缀 / 回退前缀 / 主前缀优先 / 三家 provider 参数化建造 / `from_env` 一行入口 / `to_openai_compatible` 适配回环。

### Public API（M1.4 新增）

`LLMProviderConfig` / `create_llm_service_from_config` / `create_llm_service_from_env` 顶层导出。`OPENAI_COMPAT_*` 环境变量名继续可用，仅为回退；新代码请用 `POWER_LOOP_*`（或自定义 prefix）。

### Added — M1.2 trim_history（2026-06-05）

- **`trim_history(messages, max_tokens, *, keep_system=True, keep_last_n=2)`**（`power_loop.runtime.budget`）—— 纯裁剪 helper：保留 leading system + 最后 N 个 user-bounded 交换，从中间删消息直到落在 token 预算内。不调 LLM（不摘要），仅是业务侧调用前裁剪。
- **不变量**：
  1. 预算已够 → 返回原 list（不复制）。
  2. `keep_system=True` → 所有 leading `role=system` 消息保留；`keep_last_n` 个 user-bounded 交换在尾部保留。
  3. `assistant(tool_calls) ↔ tool(tool_call_id=...)` 对永不拆分 — 裁剪边界通过 tool_call_id 配对检测自动调整。
  4. 当 system + tail 都放不下时，降级为 tail-only（丢 system）再按需从尾部裁剪。
  5. 不修改输入（返回新 list）。
- **测试**：`tests/unit/test_budget.py`（9 个）—— 已合预算 / 空 / 零预算 / 系统保留 / 去系统 / 工具对原子性 / 工具对在边界 / 仅 tail / 非突变。
- `estimate_tokens` / `estimate_text_tokens` / `trim_history` 从 `power_loop` 顶层导出。

### Public API（M1.2 新增）

`trim_history` / `estimate_tokens` / `estimate_text_tokens` 顶层导出。

### Added — M2.5 错误体系收口（2026-06-05）

- **`ToolNotFound(tool_name)`** —— `ToolRegistry.invoke / invoke_async` 找不到工具时 raise。
- **`ToolValidationError(tool_name, message)`** —— 参数校验失败时 raise。
- **`SpecValidationError(message, *, field=None)`** —— 新的规范验证错误；`AgentSpecError` 现继承于它（而它继承 `PowerLoopError`），旧代码 `except AgentSpecError` 继续有效，新代码 `except SpecValidationError` 或 `except PowerLoopError` 一把抓。
- `ToolRegistry.invoke / invoke_async` 对 unknown tool 和 invalid args 现在 **raise 异常而非 return 字符串**；`invoke` 也 raise 而非 return 字符串作为错误。Pipeline 的 `execute_tool` 内部 catch 这两个异常并返回 `(str(exc), True)` 给 LLM 看到（保持向后兼容）。
- **`ToolRegistry.validate` 保留为 internal legacy**（仍返回 `str | None`），管线仍用它做第一层检测，但新代码应直接 invoke + catch。
- 所有 `__init__.py` 和 `__all__` 同步更新。

### Added — M2.1 Public API 稳定性约定（2026-06-05）

- **README §5** 新增 "Public API 稳定性约定" 节：**STABLE**（24 符号，跨 minor 保证兼容 + CHANGELOG 独立条目）、**PROVISIONAL**（顶层导入但 0.x 可调）、**INTERNAL**（无版本承诺）。与 `power_loop/STABLE_API` 元组同步。
- Examples 表补齐 11–14（persistence / retry-cancel / memory-sqlite / structured-card）。
- §7 环境变量节更新为 `POWER_LOOP_*` 优先 + `create_llm_service_from_env()` 一行法。

### Added — M2.2 Hook/Event 全表文档（2026-06-05）

- **`docs/hooks.md`** §3.9 新增 `memory.recalled` hook 点文档（Ctx 字段 / SKIP directive / 双方授权示例）。
- **`docs/events.md`** 新增 §2.7 LLM retry/cancel lifecycle（`llm_retry_attempted` / `llm_degraded` / `loop_cancelled`）和 §2.8 Memory（`memory_recalled` / `memory_failed`），含完整 payload 字段表、触发时机、典型订阅者。

## [0.2.0] — 2026-06-05

Stateful refactor. The library now revolves around `StatefulAgentLoop` and a SQLite-backed `SessionStore`; the stateless `AgentLoop` is removed. **Hard break — no compatibility shim.**

### Added

- **`StatefulAgentLoop`** — the only public entry point. `new_session()` / `send(user_input, session_id)` / `send_sync` / `resume(sid)` / `abort_pending(sid)` / `close_session(sid, cascade=True)` / `close()` / `get_messages(sid)` / `get_pending(sid)`. Per-session `asyncio.Lock` so one instance can drive any number of sessions concurrently.
- **`SessionStore`** (`power_loop.runtime.session_store`) — SQLite-backed, the **only** thing that writes to disk. Five tables: `sessions` / `messages` / `compactions` / `usage_rounds` / `session_state`. Single connection + `threading.RLock`; WAL + busy_timeout. Public API surface for sessions, messages, compactions, usage, lifecycle.
- **`MessageSink`** Protocol + `NullSink` + `SQLiteSink` — pipeline persistence hook. SQLiteSink owns the in-memory `_history_seqs` list that mirrors `pipeline.history` so the compactor can translate fold indices back to store rows.
- **Pending state machine** — `assistant(tool_calls)` falling-into-store immediately marks `session_state.pending`; each matching `tool` message clears it. Mid-tool crash leaves a recoverable state. Next `send` raises `SessionPendingError`; caller picks `resume()` (replay remaining tools) or `abort_pending(sid, reason=…)` (synthesize `<aborted>` tool messages).
- **Subagent on top of `SessionStore`** — `spawn_agent` rewritten as a thin shell over the shared store. Children get their own row with `parent_session_id` / `spawn_tool_call_id` / `spawn_depth` (`MAX_SPAWN_DEPTH=3` enforced at insert time).
- **`AgentSpec`** (`power_loop.runtime.spec`) — strict-schema declarative subagent: `name / system_prompt / tools / max_rounds / max_tokens / temperature / model / lifecycle / metadata`. Unknown fields → `AgentSpecError`. `from_dict` / `from_json` factories.
- **`run_agent` meta-tool** — declarative companion to `spawn_agent`. The parent LLM submits a full `AgentSpec` JSON; the library validates and dispatches via `run_agent_spec`. Both meta-tools registered by a single `register_spawn_agent(registry)`.
- **`SubagentLifecycle`** enum — `EPHEMERAL` (default, deleted on success, preserved on failure for debug) / `LINKED` (cascade-deleted with parent) / `DETACHED` (independent of parent's lifecycle).
- **`Compactor`** Protocol + **`DefaultCompactor`** (`power_loop.runtime.compact`) — pluggable LLM-summary compaction. Trigger at `max_tokens × trigger_ratio` (default 0.75) or absolute `CONTEXT_COMPACT_THRESHOLD`. Preserves all `role=system`, last `keep_last_n` user-bounded exchanges, and the `assistant(tool_calls) ↔ tool` atomic pair. Soft-fails to `None` on summary errors so the loop degrades gracefully.
- **`runtime/budget.py`** — `estimate_tokens(messages)` heuristic (≈4 chars/token, stdlib-only) used by the compactor's trigger logic.
- **Error hierarchy** — `PowerLoopError` base + `SessionNotFoundError` + `SessionPendingError(session_id, assistant_seq, pending_tool_calls)`. Caller catches the base class to handle every library-raised exception.
- **`_current_loop` contextvar** (`power_loop.core.agent_context`) — threads the active `StatefulAgentLoop` through tool invocations so meta-tools like `spawn_agent` find their parent without ambient state.
- **Examples 00–05** — progressive tutorial: minimal send → multi-turn → tool calling → subagent → compaction → pending recovery. Each file introduces exactly one new concept. `examples/_helpers.py` shares `.env` loading + LLM construction.
- **Real-LLM test suite** — `tests/real/test_real_stateful_loop.py` / `test_real_tool_use.py` / `test_real_subagent.py` / `test_real_pending_resume.py` / `test_real_compaction.py` / `test_examples.py` (6 examples). `tests/real/judge.py` provides an **LLM-as-judge** helper: tests assert `await assert_passes(question, answer, rubric)` and a separate power-loop evaluator returns `{passed, reason}` JSON, solving the LLM-non-determinism assertion problem.

### Changed — Breaking

- **Removed `AgentLoop` and `agent_loop_async`**. Replace `AgentLoop(llm, config).run(messages=…)` with `StatefulAgentLoop(llm=…, db_path=…, config=…).send(user_input, session_id=…)`. The stateless model is gone — callers no longer ship the full messages list per turn; power-loop loads history from the store.
- **`AgentLoopConfig.compactor: Compactor | None = DefaultCompactor()`** — default-on. Pass `None` to disable.
- **`CompactBeforeCtx`** loses `input_tokens` and `compact_threshold` — neither carried useful data after the runtime/compact.py rewrite. **`AutoCompactStatusPayload`** trades them for `before_tokens` / `after_tokens` (sourced from the `CompactionPlan`).
- **`ContextManager`** loses `compact_async` / `should_compact` / `compact_threshold` / `last_input_tokens` / `_compact_count` / `reset_usage`. It now owns only `update_usage` (telemetry parsing), `microcompact` (large tool-output spill-to-disk), `recent_files`, and `TodoManager`. LLM-summary compaction has moved to `runtime/compact.py`.
- **`power_loop.contracts.errors`** is now a real module (was unused).

### Removed

- `power_loop/agent/loop.py` — `AgentLoop` shell.
- `power_loop/core/agent.py` — `agent_loop_async` entry point.
- Six stale integration / real tests that depended on the removed API; replaced by the new `tests/real/test_real_*.py` suite + the 6 example-driven tests.

### Migration

| Before (0.1.x) | After (0.2.0) |
|---|---|
| `AgentLoop(llm, config).run(messages=[…])` | `sid = loop.new_session(); await loop.send(user_input, session_id=sid)` |
| Caller manages `messages` list | Library loads from `SessionStore` by `session_id` |
| No persistence | `db_path` (default `./power_loop_sessions.db`); `":memory:"` for tests |
| No pending detection | Crash mid-tool → next `send` raises `SessionPendingError`; pick `resume()` or `abort_pending()` |
| Compaction via `ContextManager.compact_async` | `AgentLoopConfig.compactor = DefaultCompactor()` (default-on); pluggable via `Compactor` protocol |
| `spawn_agent` with private `AgentLoop` | `register_spawn_agent(registry)` + `run_agent` meta-tool; shared `SessionStore` with parent linking |
| No declarative subagent | `AgentSpec` + `run_agent_spec(spec, input, parent_loop=…)` |

### Documentation

- README rewritten around `StatefulAgentLoop`. Sections: 1. what it is/isn't · 2. install · 3. quickstart (mirrors examples 00→05) · 4. core concepts (Session / SessionStore / Sink / Compactor / Pending / Subagent / Hooks vs Events) · 5. flat API reference · 6. examples table · 7. env-var config · 8. pipeline ASCII trace + persistence/seq notes + pending state machine · 9. tests (including the LLM-as-judge pattern) · 10. roadmap pointer.
- `docs/hooks.md` — every `HookPoint` with its typed Ctx fields + accepted directives + typical use cases.
- `docs/events.md` — every `AgentEventType` with its payload fields + when fired + typical subscriber.

### Added — M0 工程化基线（2026-06-05）
- `power_loop.__version__ = "0.1.0"` 单一来源 + `STABLE_API` 元组声明稳定面（`AgentLoop / AgentLoopConfig / AgentLoopResult / AgentHooks / AgentEventBus / HookPoint / HookDirective / ToolRegistry / ToolDefinition`）。
- `pyproject.toml` 补 license / classifiers / urls / dynamic version；新增 dev extras (`ruff` / `mypy`)；统一 `ruff` (line=120, E/F/I/UP/B) 与 `pytest` marker (`unit` / `integration` / `real_llm`)。
- `CHANGELOG.md` 与 `ROADMAP.md` 落地（M0–M3 + 关键短板拆解）。
- `tests/` 重排为 `unit/ integration/ real/`；`tests/conftest.py` 实现 `real_llm` 默认 ON、`--no-real` 反向跳、缺 env 自动 skip；`tests/real/conftest.py` 自动给 `tests/real/*` 打 marker。
- GitHub Actions：`ci.yml`（ruff + mypy 非阻塞 + pytest --no-real，3.10/3.12 矩阵）+ `real-llm-nightly.yml`（凭 repo secrets 跑真实 LLM）。
- `examples/00_minimal.py` —— DeepTalk Agent MVP 同款单回合用法；`tests/real/test_examples.py` 把 example 作为活文档锁定。
- README 新增 "实现亮点 / 卖点"（含 M1.7a 压缩策略、M1.9 记忆分层）与 STABLE / PROVISIONAL / INTERNAL 三层稳定性约定。

### Changed
- `ruff --fix` 自动现代化（PEP 585/604、import 排序、unused import 清理）共 458 处；剩余 28 处为真实代码债，按规则码登记到 `pyproject.toml` ignore，统一 M1 接触模块时清理。

### Fixed — M0.8 / M0.9（同日）
- ruff 残留 28 → 0：清掉 19 处 `__init__.py` E402（`STABLE_API` 重排到 imports 之后）；修真实 bug `llm_client/llm_factory.py:855` 未定义的 `e`；修闭包变量延迟绑定（B023）；`Union[…]` → `X | Y`；`zip(...)` 加 `strict=`；其余移除未用变量与 E402。
- mypy 残留 32 → 0：`LLMRequest.messages/tools` 类型放宽为 `list[dict[str, Any]]`（list 不变性问题）；`default_tools.py` 4 处隐式 Optional 改 `int | None` / `str | None`；Popen None 守卫；3 处 var-annotated；`pipeline.execute_tool` 加 ToolRegistry None 守卫；`phase.py` event_meta 显式 `dict[str, Any]`；`system_prompt.py` lambda 改 def 以保留类型推导。
- CI 中 mypy 改为阻塞（不再 `|| true`）。

### Known limitations
- M1 起的所有短板（重试 / 取消 / 压缩 / 持久化 / 记忆 / spec sub-agent / 动态工具）仍未实现。

## [0.1.0] — 2026-06-05

首个基线版本（M0 起点的现状快照）。

### Added
- `AgentLoop` 主外观类 + `AgentLoopConfig` / `AgentLoopResult`。
- `AgentPipeline`：按 phase 拆分的主循环，含 LLM 调用 / 工具调用 / 压缩 hook 点。
- `AgentHooks`：有序 sync/async hook 管理，支持 typed Ctx 与 legacy dict 两种回调。
- `AgentEventBus`：事件订阅 / 发布，订阅者错误隔离。
- `HookPoint` 15 个：`session.* / round.* / llm.* / tools.* / tool.* / compact.* / message.append`。
- `HookDirective`：`CONTINUE / SKIP / BREAK / SHORT_CIRCUIT` 控制流。
- `ToolRegistry`：动态工具注册 + JSON Schema 必填校验 + OpenAI tool schema 导出。
- 默认工具集：`run_bash / run_read / run_write / run_edit / run_grep / run_glob / apply_patch`，含路径白名单。
- `spawn_agent` 工具：命令式子代理，含深度守卫与父总线事件冒泡。
- `llm_client/`：OpenAI 兼容（含 DashScope）+ Anthropic 双家底；多模态、web search、tool-call 解析。
- `SystemPromptBuilder` + `SystemPromptContext`：可拼装的系统提示。
- `runtime/skills.py`：SKILL.md frontmatter 加载（粗糙版）。
- `runtime/env.py`：WORKSPACE / AGENT_DIR 路径白名单。
- 测试 3.7k 行：契约单测、fake LLM 集成、真实 DashScope 端到端 showcase。

### Documentation
- `README.md`：定位、目录结构、安装、env 配置、最小用法、Hook & Event 模型、工具系统、与 DeepTalk 分工表、Roadmap。
- `ROADMAP.md`：M0–M3 四阶段规划，含上下文压缩、AgentSpec、动态工具注册等关键短板。

### Known limitations
- Public API 边界尚未文档化（M0 中解决）。
- 缺乏 LLM 重试 / 超时 / 取消的统一策略（M1.1）。
- 历史窗口 / token 预算修剪缺工具（M1.2）。
- `LLMRequest` 无 `response_format`（结构化输出，M1.3）。
- Provider 实例化散落在 `llm_factory.py`，无 unified `LLMProviderConfig`（M1.4）。
- `compact.*` hook 点存在但**无默认 compactor 实现**，长会话会 hard-fail 在 context overflow（**M1.7a 必须**）。
- 无声明式 sub-agent（`AgentSpec`，M1.8）。
- 无运行时工具注册 meta-tool（M2.6）。
