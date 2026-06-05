# Changelog

本项目采用 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格，
并遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

破坏性变更只能出现在 minor（0.x 阶段）或 major 版本，
并且必须在此文件中显式列出受影响的 Public API。

## [Unreleased]

## [0.2.0] — 2026-06-05

Stateful refactor. The library now revolves around `StatefulAgentLoop` and a SQLite-backed `SessionStore`; the stateless `AgentLoop` is removed. **Hard break — no compatibility shim.**

### Added

- **`StatefulAgentLoop`** — the only public entry point. `send(user_input, session_id=None)` / `send_sync` / `resume(sid)` / `abort_pending(sid)` / `close_session(sid, cascade=True)` / `close()` / `get_messages(sid)` / `get_pending(sid)`. Per-session `asyncio.Lock` so one instance can drive any number of sessions concurrently.
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
| `AgentLoop(llm, config).run(messages=[…])` | `StatefulAgentLoop(llm=…, db_path=…, config=…).send(user_input)` |
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
