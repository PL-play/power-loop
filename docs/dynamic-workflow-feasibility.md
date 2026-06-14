# power-loop 动态工作流（Dynamic Workflow）能力可行性报告

> 目标读者：power-loop 库作者 / 维护者
> 版本基线：power-loop **v0.12.0**
> 定义采用：本报告所称「dynamic workflow」= 运行时决定的多代理编排，其**控制流是确定性的**（loop / branch / fan-out / fan-in / budget 驱动 / loop-until-condition），由它去 spawn 并协调 sub-agent；区别于 (a) 静态预定义 DAG，(b) 纯粹主 LLM 用 tool call 即时决策。按维护者指令 D1，最终形态取「LLM 运行时产出声明式 JSON DSL（WorkflowSpec）+ 引擎确定性解释执行」。

---

## 1. 结论先行（Verdict）

**可行（推荐做成可选 `power_loop.workflow` 子模块，而非进 core）。** 理由：

1. **底座已经具备**。`run_agent_spec` 已被证实可由宿主代码直接调用（`examples/06` 的 `direct_call()`、docstring spec.py:14-15 明言"good for tests / orchestration"），它就是 `agent()` 原语；一个 `StatefulAgentLoop` + 共享 `SessionStore` + per-session `asyncio.Lock` 已经支撑 `asyncio.gather` 并发驱动多会话（`examples/10`），这就是 `parallel()` 的底座；`TimerRunner` + `follow_up` 的唯一投递路径就是"完成回调唤醒主 agent"的现成骨架；`SQLiteNoteMemory` / `session_runtime_state` 可作黑板；`AgentEventBus` / `RuntimeProjector` / `get_session_stats` 可作 introspection。
2. **缺的是确定性"编排组合子层"和跨子代理协调层**，不是基础能力——`parallel/pipeline/foreach/while/branch`、跨子代理**共享 token 预算池**、**编排级 resume/journal**、**fan-out 批次的 fan-in rollup**、**活进度树**这些全部为净新增。
3. **两条硬约束必须绕开设计**：`SessionStore` 的 **one-process-per-store-file（写）** 约束（阻断 M3 subprocess executor 的多进程同写），以及 `run_agent_spec` 依赖 **contextvar 传递 parent session id**（嵌套并行 fan-out 下有 clobber 风险，未测试）。
4. 维护者已**撤销** ROADMAP M3"不做 DAG/Planner"的结论，本能力正式在范围内；因此焦点不是"能不能做"，而是"放哪一层、core 改多少"。建议 core 只做**少量、有正当性的扩展**（surface child usage、forward stop_event/budget、给 `AgentSpec` 加 `output_schema`、depth 可配），编排策略全部进 `power_loop.workflow`。

---

## 2. 什么是 dynamic workflow（澄清范围）

| 模型 | 谁决定控制流 | power-loop 现状 | 本报告立场 |
|---|---|---|---|
| **A. Model-driven 委派** | 主 LLM 用 `spawn_agent` / `run_agent` meta-tool **即时**决定 spawn 谁 | **已支持**（`register_spawn_agent`，`SPAWN_AGENT_DEFINITION` / `RUN_AGENT_DEFINITION`） | 保留，但**不是**本能力；它是 workflow agent 内部 ad-hoc 委派的便利退路 |
| **B. 静态 DAG / Planner** | 宿主预先写死的图 | 无 | 维护者明确**不要**这个 |
| **C. 确定性编排（本能力）** | **确定性代码**解释执行一份**声明式 WorkflowSpec**；其中 `foreach/while/branch` 在运行时动态展开 | 无（缺组合子层） | **目标形态** |

**按 D1 的关键定位**：dynamism **不**来自宿主手写命令式脚本，也**不**是静态图。它来自两处：(1) **LLM 现场创建** WorkflowSpec（把 schema 告诉它，它像生成 `AgentSpec` JSON 一样生成 workflow JSON）；(2) spec **内部的动态构造**（`foreach` 对运行时数据 map、`while/until` 按条件循环、`branch` 按运行时结果分支）。引擎只负责**确定性地解释执行**这份 spec——校验在前、报错精确、行为可复现。这与现有 `AgentSpec`/`from_json` 拒绝未知键、`__post_init__` loud-fail 的声明式 JSON 思路一脉相承，可直接复用 pydantic / `StructuredOutputSpec` 校验。

---

## 3. 可直接复用的机制清单（映射表）

| Workflow 原语/需求 | power-loop 现有机制（真实 API） | 复用价值 | 注意事项（来自 source） |
|---|---|---|---|
| **`agent(prompt,{schema})`** | `run_agent_spec(spec, user_input, *, parent_loop)` + `AgentSpec` + `filtered_registry` | **高** | 可宿主直接调用，但 (a) 返回**仅 text**（`{session_id,status,final_text,rounds,depth}`），**丢弃 result.usage**；(b) 依赖 `get_session_id()/get_current_loop()` contextvar——standalone 调用须手动 `set_current_loop/set_session_id`（`examples/06`，且这些 setter 是 **INTERNAL** 未导出）；(c) `parent_loop: Any`，伸手进 `parent_loop._runner.hooks/.event_bus`（脆耦合私有属性）；(d) `spec.model` 被丢、child config 不继承父的 retry/memory/projector/`max_tokens_per_run` |
| **`schema` 化返回** | `StructuredOutputSpec(name,schema,strict)` / `.to_openai_response_format()` / `parse_structured` / `StructuredOutputError` | **高** | **未接线**：`AgentSpec` 无 `output_schema` 字段，child_config 不设 `response_format`，返回不过 `parse_structured`。本地校验**浅**（只查 top-level object + required keys；不查嵌套/类型/enum，靠 provider strict）。glue 须 workflow 层自己拼或在 core 给 `AgentSpec.output_schema` 接线 |
| **`parallel([...])` (fan-out)** | `StatefulAgentLoop.new_session`/`send` + per-session `asyncio.Lock`（`_lock_for`） + `asyncio.gather`（`examples/10`）；底层 `run_agent_spec` 子 sid 各自独立锁，store 单 `RLock` 串行 | **高** | **无** `parallel()`/聚合/部分失败策略/并发上限 helper——全要自建。**嵌套并行 sharp edge**：单个 parent turn 内并发多个 `run_agent_spec`，子的 `session_async` 会覆写同一 contextvar（task 间可能共享 context），**未测试不显然安全**；workflow 层须**显式传 session id**而非靠 contextvar |
| **`pipeline(items, ...stages)`** | 同上 + `SessionKind`/`SubagentLifecycle`/`list_children` 的父子树 | **中** | 无 stage 概念、无 barrier 语义、无跨 stage 结果传递的 workflow 可见通道（结果只走 `final_text` 或 session-private notes）。pipeline-of-pipeline 易撞 `MAX_SPAWN_DEPTH=3` |
| **`phase(title)` 阶段边界** | ⚠️ **没有合适机制**。`@phase` 装饰器（`core/phase.py`）是**内部 pipeline 方法**的 hook/event 包装，`self` 必须是 `AgentPipeline`——**不可**当编排级 stage 原语，是命名冲突陷阱 | 低 | 净新增；不要 conflate |
| **`log(msg)` / 进度** | `AgentEventBus`（`subscribe/publish`，`DEFAULT_EVENT_BUS`）+ `AgentEventType`（~25 类）+ `SystemLogPayload`/`UserNotificationPayload` + `contrib.logging_sink.attach_logging_sink`（已读 `event.source`） | **高** | bus 仅观察、无 replay/buffer（迟到订阅者丢失早期事件）、async 无背压。**无编排级事件**（无 PHASE/FANOUT/JOIN/STEP）；事件**无 parent_session_id**——进度树须从 `SessionStore` 重建 |
| **共享 token 预算** | per-run `AgentLoopConfig.max_tokens_per_run`（round 边界检查，`status='budget_exceeded'`）+ `usage` dict + `get_session_stats`/`list_session_stats`（持久累计）+ `USAGE_UPDATED` 事件 | **中** | **只 per-run**，无跨子代理池；`run_agent_spec` **丢 usage**，parent 看不到子花费；`get_session_stats` 不能按 subtree 聚合（须先 `list_children` 再 app 层 join）。这是 budget-driven scaling 的核心缺口 |
| **`abort`** | `CancellationToken` / `CancellationLike` / `from_any`；`send(..., stop_event=)`；round 边界协作式检查 | **高** | `cancel()` 只翻**自有** token 的 event（包 asyncio.Event/callable 时 `cancel()` 是 no-op）；**`run_agent_spec` 不接收/不转发 stop_event**——取消 parent 不会 abort spec 子代理。无 fan-out token helper（一父 N 子 / 失败取消兄弟） |
| **`timers` / 延时·调度·loop-until** | `TimerRunner`（`scan_once`/`due_timers`/CAS claim）+ `schedule_timer/cancel_timer/list_timers`（loop 方法，宿主可写）+ `TIMER_FIRE` hook（`TimerFireCtx`：CONTINUE/SKIP/BREAK/`postpone_s`） | **高** | **at-least-once**（须按 `timer_id` 幂等去重）；分辨率=scan_interval；**timer 只"用 note 唤醒一个 session"**，无"调度执行一个 AgentSpec/stage"的 job 原语——须把意图编码进 note 或在 `TIMER_FIRE` hook 里跑确定性代码。recurring + 在 hook 里 BREAK = loop-until-condition |
| **detached + 完成回调** | `follow_up`/`FollowUpQueued` 是**唯一投递路径**，timer firing 与 human steering 都走它；`loop.schedule_timer` + `TIMER_FIRE` hook = "唤醒某 session 注入结果" | **高** | follow-up 队列**在内存**（crash 丢失，不如 timer/human-input 持久）；只在 round 边界落地 |
| **memory/notes（黑板）** | `MemoryProvider` 协议 + `SQLiteNoteMemory`/`NotesPolicy` + `note_add/update/delete` 工具 + store `add_note/list_notes(session_id,...)` | **高（作私有记事本）/ 中（作共享黑板）** | **严格 session-scoped**：每个查询 `WHERE session_id=?`，子代理拿全新 child_sid → **父子 notes 命名空间互不相交**，agent-facing 工具永不暴露别的 session id。store API 接受**任意 session_id**，故可在编排层选一个"board" session 预置/读取——但**无 API/工具/锁/读隔离/typed entry**，且绕过 `NotesPolicy`（enforcement 在 `add_note_checked` 不在 store）。memory 是**边界 seam**（recall 仅 SESSION_START、remember 仅 SESSION_END），非 mid-run 读写通道 |
| **hooks（策略/否决/回调）** | `AgentHooks`（`register/run_typed`）+ `HookPoint`（SESSION/ROUND/LLM/TOOL/COMPACT/`TIMER_FIRE`…）+ `HookDirective`（CONTINUE/SKIP/BREAK/SHORT_CIRCUIT）+ typed `*Ctx` | **高** | 跑在热路径（handler 要小）；同一 `AgentHooks` 传给子 loop，**无 main/sub 角色判别字段**。**无编排级 hook**（无 BEFORE_SPAWN/AFTER_JOIN/PHASE_BOUNDARY）——否决 spawn 只能 hook `TOOL_BEFORE` 在 `spawn_agent` 上，间接 |
| **events（观察）** | 同 `log` 行；另 `Subagent*Payload`/`SUBAGENT_*` 事件类型 | **中** | ⚠️ **SUBAGENT_\* 已定义/导出/文档化但从不 publish**（grep 零 publish site）；子 loop 还**硬编码 `scope="main"`**；文档的 `source="subagent:<sid>"` 约定**从未被任何 producer 赋值**（但 `logging_sink` 消费侧已就绪）。fan-out 进度树**今天无法从 events 建**，须从 `SessionStore.parent_session_id` 重建，或 workflow 层自己 wire 这些发射（廉价，consumer 已在） |
| **background projector（后台投影）** | `RuntimeProjector` 协议 + `TodoRuntimeProjector` + `BackgroundRuntimeProjector`（每 round 从 `session_runtime_state` 注入）+ `ToolRuntimeContext`/`get_tool_runtime_context` | **高** | projector 是把状态注入**给 LLM**，非投影**给 UI**；真正的进度 source-of-truth 是 `SessionStore.get_runtime_state`/`list_unseen_background_updates`（UI 须订阅 `TODO_UPDATED` 或轮询）。**无 subagent-tree projector**——进度树须写自定义 poller over `parent_session_id` |
| **动态工具注册** | `ToolRegistry.register/unregister/subset` + `build_registry` + `create_default_tool_registry`（preset core/explore/full）+ `invoke_async`（sync 工具走 `to_thread`，不阻塞并发会话） | **高** | **仅程序化**——**无 LLM-facing `register_tool` meta-tool**（grep 确认；唯一 meta-tool 是 spawn/run_agent）。LLM 不能自定义工具。`subset` 共享同一 handler 实例（绑定的 workspace/runtime-env 不随委派 re-scope） |
| **skills** | `SkillLoader` / `register_skill_tools` / `load_skill` 工具 / `build_system_prompt_section` | **中** | 注入**知识/指令**（Markdown），**不注册可执行工具**；filesystem-only（需磁盘 `skills_dir`，无 in-memory 源）；无 per-skill tool-allowlist/sandbox |
| **human-input（审批门）** | `request_user_input`（raise `HumanInputRequired`）+ `submit_input` + `StatefulResult.status=='waiting_for_input'`/`pending_interactions` | **高** | per-tool-call 暂停、可跨重启/进程；但**无 workflow 级门**（无 "wait N of M approvals"、无超时/过期、无 approve/deny→确定性分支——分支由 LLM 看到答案后决定，非代码决定）。无内建 expiry（须配 timer） |
| **resume / 跨进程** | pending 状态机（`SQLiteSink` 即时写 `pending_json`，`SessionPendingError`/`resume`/`abort_pending`/`heal_pending`）+ 共享 db 文件跨进程续跑（`examples/11`） | **中** | 粒度=**单 session 的在途 tool_calls**，**不是**"pipeline 第 3 阶段、fan-out 2/5 完成"。`resume` **重跑** leftover tool_calls，**无幂等键**——非幂等工具（spawn/send/写文件）会**双发**。EPHEMERAL 子仅 `status=='completed'` 删除→成功 fan-out **不留持久痕迹**（要审计/恢复须用 LINKED） |

---

## 4. 缺口（Gaps，按重要性排序）

1. **确定性编排层/组合子 API（最重要，净新增）**。无 `agent()/parallel()/pipeline()/foreach()/while/branch/phase()/log()`，无结果聚合、无 partial-failure 策略（continue-on-error vs cancel-all）、无 max-concurrency 限流、无 loop-until-condition。全部要在 workflow 层从 `run_agent_spec` 之上自建。
2. **跨子代理"共享 token 预算池"**。现有 budget 严格 per-run（`max_tokens_per_run`），`run_agent_spec` **丢弃 `result.usage`**，`get_session_stats` 不能按 subtree 聚合。budget-driven scaling（"剩 X% 就停止 spawn"）所需的 **debit/reserve/commit/refund + 全局上限**对象**完全不存在**。需要：(a) core 让 `run_agent_spec`/spawn meta-tools **surface usage**；(b) workflow 层一个 `SharedBudget` 池，原子扣减。
3. **编排级 resume / journal**。store 只建模 agent session 树，**无 `workflow_run`、无 step/stage 行、无 fan-out 批次 id、无 checkpoint journal**。要让 `parallel()/pipeline()` 可恢复，须自建持久 journal（新表或在专用 driver session 的 `session_runtime_state` 上有纪律地写）。pending 状态机解决不了"哪些步骤已完成"。
4. **fan-in barrier 与 pipeline 语义**。`parallel()` 的 N 个子只是 `list_children` 里按 `created_at` 排的兄弟——**无 batch id、无 done/running/failed rollup**。fan-in 全靠内存里的 `StatefulResult` 重建，crash 后**无"该批次哪些已完成"的持久查询**。需要：有 barrier 的 `parallel`（全到齐再继续）vs 无 barrier 的流式 `pipeline`（一个 stage 出结果就喂下一 stage）两套语义。
5. **活进度树（live progress tree）**。`SUBAGENT_*` 事件**inert**、`scope` 硬编码 `"main"`、`source` 槽**未赋值**、事件**无 parent_session_id**、bus **无 replay/late-join**。进度树今天只能从 `SessionStore.parent_session_id` + `get_session_stats` 轮询重建。需要：wire 现有 dead 的 `SUBAGENT_*` + `source` 槽（廉价，consumer 已在），新增 PHASE/JOIN 事件类型 + 事件 buffer。
6. **one-process-per-store-file 对 fan-out 的约束**。WAL 只保证跨进程**读**安全，**写必须单进程**。直接阻断 M3 subprocess executor（每子代理一个 `python -m power_loop.runner` 进程写同一 store = 并发写）。in-process fan-out 没问题（单 writer + per-session 锁 + store RLock）；多进程 fan-out 需要单一 owning writer（RPC 漏斗，未提供）或 per-worker 分文件（则 fan-in 无共享 store）。**无 lease/leader/fencing 强制单写**。
7. **次要但要处理**：~~`MAX_SPAWN_DEPTH=3` 模块常量**不可配**~~（已修：现为 `store.max_spawn_depth`，经 `SessionStore.open(max_spawn_depth=)` / `StatefulAgentLoop(max_spawn_depth=)` 可配，默认仍 3）；非 `completed` 终态的 EPHEMERAL 子**泄漏 session 行**（长 fan-out 无 GC）；contextvar session 身份在嵌套并行下脆弱；notes/memory **无 typed/keyed/scoped** 黑板；human-input 无 workflow 级 gating；follow-up 队列**非持久**。

---

## 5. 与既有 ROADMAP 的关系

- **"不做 DAG/Planner"的结论已被维护者撤销**，本能力正式在范围内。所以焦点不是"能不能做"，而是**怎么放**。
- **本能力 ≠ 静态 DAG/Planner**：它是**命令式确定性控制流**（loop/branch/fan-out/fan-in 由代码引擎解释一份声明式 spec 而成），**复用现有 sub-agent**（`run_agent_spec`）做执行单元，而非预编译一张静态图、也不是 router LLM 决策。原 M3 论点"`run_agent(spec)` + 主 agent 决策覆盖 ~90%"仍部分成立——但它覆盖的是**模型驱动委派（模型 A）**，覆盖不了**确定性控制流（模型 C）**：预算驱动扩缩、loop-until-condition、有 barrier 的 fan-in、可恢复 journal 都不能交给主 LLM 即兴。
- **放哪一层（明确建议）**：

| 层 | 放什么 | 理由 |
|---|---|---|
| **core**（小改、有正当性） | `run_agent_spec`/spawn meta-tools **surface `result.usage`**；`run_agent_spec` **接收并转发 `stop_event`（和可选 budget）**；`AgentSpec` 加 **`output_schema`**（接线 `response_format` + `parse_structured`）；**显式 `parent_loop`/`parent_sid` 参数**取代 contextvar 依赖；`MAX_SPAWN_DEPTH` **可配**；修 `spec.model` 透传；wire dead 的 `SUBAGENT_*` 事件 + `source` 槽 | 这些是**原语的硬化**，不是业务；它们让上层能干净地建。其余 per-run budget 检查、`CancellationToken`、retry、bus、hook 已在 core 且应留在 core |
| **`power_loop.workflow`（推荐，可选子模块）** | WorkflowSpec DSL + pydantic 校验、确定性引擎（sequence/parallel/foreach/while/branch/wait_timer/human_gate）、`SharedBudget` 池、编排级 journal（新表 `workflow_runs`/`workflow_steps`）、fan-in rollup、进度树 assembler、PHASE/JOIN 事件、共享黑板（按 run id keyed、scoped 读写工具）、detached + 完成回调 wiring | **保住"小而稳的核心、不掺业务"**：编排策略 = policy，应消费 core 原语而非污染 core。可选导入，不增加 core 依赖面 |
| **contrib / examples** | 参考 WorkflowSpec、logging/progress 渲染器、subprocess executor 实验 | 实验性、易变 |

> 不建议把编排逻辑塞进 `TIMER_FIRE`/human-input/follow-up hook——那会把业务编排耦合进 kernel hook（维护者明确顾虑）。

---

## 6. 建议的落地形态（Recommended shape，落实 D1–D4）

### (a) WorkflowSpec JSON DSL（D1）

声明式、严格 pydantic schema、**创建即校验、精确报错**，复用 `AgentSpec`/`StructuredOutputSpec` 的 loud-fail 思路（拒绝未知键）。

**节点类型**：
- **`agent` 节点** = 复用/扩展 `AgentSpec`（加 `output_schema`、`max_tokens_per_run`、`budget_ref`）。
- **控制构造**：`sequence`、`parallel`（带 `barrier`/`max_concurrency`/`on_error`）、`foreach`（map，over 运行时数据）、`while`/`until`（带 `max_iters`/`budget_ref`）、`branch`（按上游 typed 输出走 case）、`wait_timer`（`delay_s`/`due_at_ms`，落 timer 行）、`human_gate`（`request_user_input` 暂停）。

**示例 WorkflowSpec JSON**：

```json
{
  "name": "research_and_summarize",
  "budget": { "max_tokens": 200000, "stop_at_remaining_pct": 5 },
  "root": {
    "type": "sequence",
    "steps": [
      { "type": "agent", "id": "plan",
        "spec": { "name": "planner", "system_prompt": "Break the topic into 3-6 subtopics.",
                  "tools": ["grep", "read_file"], "max_rounds": 6,
                  "output_schema": { "name": "Plan", "schema":
                    { "type": "object", "required": ["subtopics"],
                      "properties": { "subtopics": { "type": "array", "items": { "type": "string" } } } } } } },
      { "type": "foreach", "id": "research",
        "items_from": "plan.subtopics",
        "as": "subtopic",
        "body": { "type": "agent", "spec": { "name": "researcher",
                   "system_prompt": "Research: {{subtopic}}", "tools": ["bash", "read_file"],
                   "max_rounds": 10, "lifecycle": "linked" } },
        "parallel": { "barrier": true, "max_concurrency": 4, "on_error": "continue" } },
      { "type": "agent", "id": "synthesize",
        "spec": { "name": "writer", "system_prompt": "Synthesize all findings.", "max_rounds": 8 },
        "inputs_from": ["research.*"] }
    ]
  }
}
```

**校验报错示例**（创建时即报，引用真实风格）：

```text
WorkflowSpecError: invalid WorkflowSpec — node 'research' (foreach):
  - 'parallel.max_concurrency' must be >= 1 (got 0)
  - 'body.spec.lifecycle' unknown value 'temporary'; expected one of: ephemeral, linked, detached
  - 'items_from' references 'plan.subtopics' but node 'plan' declares no output_schema with key 'subtopics'
  - unknown key 'retires' in node 'synthesize' (did you mean 'retries'?)
```

（实现：每个节点一个 pydantic model，`model_config = ConfigDict(extra='forbid')`；`from_json` 聚合所有错误一次性报，复用 `AgentSpecError` 模式。）

### (b) 持久化执行引擎（D2）

- **运行态落 SQLite**：复用 `SessionStore`，新增两表 **`workflow_runs`**（run_id, name, spec_json, status, budget_json, created/updated_at, driver_session_id）和 **`workflow_steps`**（run_id, step_id, node_path, kind, child_session_id, status, usage_json, result_json, started/finished_at）。或退一步：以一个**父 driver session** 串起子 session 树 + 在其 `session_runtime_state` 上写 journal。**推荐新表**（编排级 resume/rollup 的一等查询需要它）。
- **每个 agent 步骤 = 一次 `run_agent_spec` / 子 session**，`lifecycle='linked'`（**不要 EPHEMERAL**——成功 fan-out 才留持久痕迹供 resume/审计）。
- **`TimerRunner` 驱动推进**：`wait_timer` 节点写 timer 行；引擎在 `TIMER_FIRE` hook 里推进 journal（按 `timer_id` 幂等）。延时/调度步骤天然走这条。
- **notes/memory 当共享黑板**：按 **run_id keyed 的 board session** 存（store `add_note/list_notes` 接任意 session_id），workflow 层提供 scoped、typed（`results[step_id]`）读写——**不直接复用 session-private 的 `note_*` 工具**。
- **fan-out 注意 one-process-per-store-file**：最小/中档**保持 in-process**（单 writer + `asyncio.gather` + per-session 锁），fan-out 用 `run_agent_spec` 并发并**显式传 session id**（绕开 contextvar clobber）。多进程 executor 留到完整档作可选。

### (c) detached + 完成回调（D3）

workflow 异步运行，主 agent 触发后即可 `pass_turn` 退出；workflow 完成/失败/退出经 hook（`TIMER_FIRE` 或新增 `WORKFLOW_*` HookPoint）+ `follow_up`/`schedule_timer` **唤醒主 agent 的 session 并注入结果**——完全复用"timer firing 走唯一 `follow_up` 投递路径"的现成模式。

### (d) introspection（D4）

`list_workflows()` / `get_workflow(run_id, detail=True)`：从 `workflow_runs`/`workflow_steps` 持久表 + `AgentEventBus` 订阅（运行中实时）+ `get_session_stats(child_sid)` 投影**运行中/已完成、步骤进度树、token 用量**。

### 三个档位 + Python 伪代码 API

```python
from power_loop import StatefulAgentLoop
from power_loop.workflow import create_workflow, list_workflows, get_workflow
from power_loop.workflow import WORKFLOW_COMPLETED  # 新 HookPoint

loop = StatefulAgentLoop(db_path="./app.db")

# ---- 创建（D1：JSON 创建即校验，错则抛 WorkflowSpecError）----
wf = create_workflow(spec_json, parent_loop=loop)   # 复用 pydantic 校验

# ---- 最小档：in-process 同步 sequence+parallel+foreach + JSON 校验 ----
#   引擎内部全部 await run_agent_spec(node.spec, item_input, parent_loop=loop)
#   parallel -> asyncio.gather(显式传 child sid); foreach -> map over items_from
result = await wf.run()        # 阻塞直到完成，返回 typed 聚合结果

# ---- 中档：+SQLite 持久化 + timer 驱动 + 完成回调唤醒主 agent ----
@loop.hooks.register(WORKFLOW_COMPLETED)            # 复用 hook + follow_up 模式
async def on_done(ctx):
    await loop.schedule_timer(ctx.parent_session_id, delay_s=0,
                              note=f"[workflow {ctx.run_id} done] {ctx.summary}")
    # 或直接 await loop.follow_up(f"workflow done: {ctx.result}", ctx.parent_session_id)

run_id = await wf.start(detached=True)              # D3：主 agent 触发后即可 pass_turn
# ...主 agent 退出，TimerRunner/引擎推进，完成时上面的 hook 唤醒主 session...

# ---- 完整档：+编排级 resume/journal + 共享预算池 + 活进度树 + 可选 subprocess ----
wf2 = create_workflow(spec_json, parent_loop=loop, budget=SharedBudget(max_tokens=200_000))
run_id = await wf2.start(detached=True)
# 进程崩溃后另起进程（复用跨进程 db 文件 resume）：
wf2 = get_workflow(run_id, parent_loop=loop)
await wf2.resume()             # 从 workflow_steps journal 续跑：已 completed 的步骤跳过

# ---- introspection（D4）----
for row in list_workflows():                         # workflow_runs 表
    print(row.run_id, row.status, row.tokens_used)
detail = get_workflow(run_id, detail=True)           # 表 + events + get_session_stats 投影
#   detail.tree -> 步骤进度树; detail.usage -> 跨子代理 SharedBudget 已扣 / 剩余
```

| 档位 | 内容 | 需要的 core 改动 |
|---|---|---|
| **最小** | sequence+parallel+foreach 的 in-process **同步**执行 + JSON 校验 | 仅 contextvar sharp edge：显式传 sid（可纯在 workflow 层 set/reset INTERNAL setter，或推动 core 加显式 `parent_sid` 参数） |
| **中** | + SQLite 持久化（新表）+ TimerRunner 驱动 + `WORKFLOW_*` hook 完成回调唤醒主 agent + LINKED 子 session | + `run_agent_spec` surface usage、forward stop_event；`AgentSpec.output_schema` |
| **完整** | + 编排级 resume/journal + `SharedBudget` 跨子代理池 + 活进度树（wire `SUBAGENT_*`/`source` + parent_session_id 关联 + 事件 buffer）+ 可选 subprocess executor | + `MAX_SPAWN_DEPTH` 可配；subprocess 需单 writer 漏斗（最难，单独立项） |

---

## 7. 风险与工作量（Risks & effort）

**主要风险 / 陷阱**：
1. **contextvar session 身份在嵌套并行 fan-out 下 clobber**（未测试）。**缓解**：workflow 引擎**永不**依赖 contextvar 传 parent sid，全部显式入参；并推动 core 给 `run_agent_spec` 加显式 `parent_loop`/`parent_sid`。
2. **one-process-per-store-file**：in-process 档安全；任何多进程 fan-out 都会破坏单写不变量、无 fencing 强制。**缓解**：完整档前不碰 subprocess；真要做就单 owning writer 漏斗（与项目"api 唯一写入口"风格一致）。
3. **resume 双发非幂等工具**（`resume()` 重跑 leftover tool_calls；timer at-least-once）。**缓解**：journal 里给每个步骤幂等键 / "已执行"记录，spawn/send 走 dedup。
4. **EPHEMERAL 泄漏 / 非 completed 子保留**：长 fan-out 留垃圾行。**缓解**：用 LINKED + 完成后由引擎 cascade `close_session`。
5. **budget 粗粒度**（round 边界检查，单大 round 会 overshoot）+ `SharedBudget` 并发扣减 race。**缓解**：`SharedBudget` 用 store 单 RLock 下的原子 upsert + 乐观版本；接受 round 级粒度。
6. **bus 无 replay/late-join**：UI/resume 后进度树丢早期事件。**缓解**：进度树以**持久表 + `get_session_stats` 为真相源**，events 仅作实时增量；可加事件 buffer。
7. **notes 黑板无锁/无版本**：fan-in 并发写同一 key 互相覆盖。**缓解**：typed/keyed entry（`results[step_id]` 天然分键），不共享单 key。

**粗略工作量分档**（单人）：
- **最小档**：~1–1.5 周（DSL + pydantic 校验 + in-process 引擎；不动 store schema）。
- **中档**：+~2–3 周（新表 + journal 写入 + TimerRunner wiring + 完成回调 + 几处 core 硬化：usage surface / stop_event / output_schema）。
- **完整档**：+~3–4 周（编排级 resume + `SharedBudget` + 活进度树 + wire dead 的 `SUBAGENT_*`/`source`）；**subprocess executor 单独再 +2–4 周**且风险最高（破单写）。

---

## 8. 开放问题（需 maintainer 拍板）

1. **执行器形态**：先只做 **in-process**（单 writer，安全、覆盖绝大多数 demo/MVP），还是要把 **subprocess executor** 列为目标？后者必须先解决 one-process-per-store-file（单 owning writer 漏斗 or per-worker 分文件 + 聚合），是整个方案最大的架构岔路。
2. **resume 粒度**：编排级 resume 做到"步骤级"（已 completed 步骤跳过、运行中重跑）即可，还是要"步骤内"也精确（接 pending 状态机）？前者简单且够用，后者复杂且与非幂等工具冲突。
3. **是否进核心库**：是否接受把 `power_loop.workflow` 作为**可选子模块**随包发布（vs 仅 `examples`/`contrib`）？以及是否同意我列的那批**小 core 改动**（surface usage / forward stop_event / `AgentSpec.output_schema` / depth 可配 / wire `SUBAGENT_*`）进 core——它们是原语硬化、非业务，但确实动 PROVISIONAL 公共面。
4. **是否暴露给 LLM 自身调用**：除了宿主/引擎确定性解释，是否提供一个 **LLM-facing `create_workflow` meta-tool**（让主 agent 现场生成 WorkflowSpec JSON 并提交执行）？这正是 D1 的"agent 自身生成 workflow"——但需要决定权限/沙箱/递归深度边界（会再吃一层 `MAX_SPAWN_DEPTH`）。
5. **共享预算 enforcement 语义**：`SharedBudget` 超限时是 **soft（停止 spawn 新步骤、已在跑的让其跑完）** 还是 **hard（取消在途子代理，复用 `CancellationToken` fan-out）**？以及预算是否随 retry/失败步骤 refund？