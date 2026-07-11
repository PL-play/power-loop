# Host Seams（3.14.0）— 接口级设计

> 目标：一次「宿主接缝」主题的 minor 发版，让重宿主（DeepTalk `agent` 服务）删掉三处
> 长期 workaround，并为 workflow 接入解锁两条前置能力。**全部可加性变更**：不带新参数时
> 行为逐字节不变；无 DB schema 变更（journal 增一个可选 key，向后兼容）。
>
> 新符号先以 **PROVISIONAL** 发布（顶层不进 `STABLE_API`），DeepTalk 验证一个迭代后在
> 后续 minor 提升为 STABLE。状态：**已实现（3.14.0，2026-07-11）**；实现与本设计的偏差：
> S1 guard 列表会传导给子 loop（孙辈同样进入 guard——实现时发现设计稿漏了这层，孙辈 spawn 的
> parent_loop 是子 loop）；S2 的 workflow 叶子 clamp 施加在引擎 spec 层而非 Executor 参数
> （协议签名不动，subprocess 叶子天然生效）。

对应的宿主痛点（均已在 DeepTalk 代码里核实）：

| 缝 | 痛点 / 现状 workaround |
|----|------------------------|
| S1 子运行环境态隔离 | `agent/app/tools/subagent.py::_run_spec_isolated` 手工 suspend/restore 三组 per-send hook 状态；每加一个有状态 hook 都要记得登记，漏一个=会话 57 级事故 |
| S2 per-send 工具集传导 | `subagent.py::_parent_allowed` contextvar + `main.py` 手工 set/reset；contextvar 在 detached 任务里失效，workflow 接入会再撞一次 |
| S3 子 agent 上下文策略 | `run_agent_spec` 的子 `AgentLoopConfig` 写死极简配置，重工具活的叶子无投影/折叠兜底，宿主只能 fork 建 loop 段 |
| S4 workflow 工具宿主注入 | `_handle_create_workflow` 写死默认 executor / 无预算，宿主想钳制只能整个 fork handler 连 schema 一起抄 |
| S5 timer 投递可插拔 | 投递写死 `loop.follow_up`，绕过宿主自己的运行管道（agent_run 簿记）；宿主因此完全没跑 TimerRunner，一切依赖 timer 的能力（含 workflow detached 唤醒）都是死的 |

---

## S1 — 子运行环境态隔离：child-run guards

### 接口

```python
# StatefulAgentLoop 新方法（对应 AgentHooks 的 name/replace/remove 风格）
ChildRunGuard = Callable[[], ContextManager[None]]   # 工厂：每次子运行调用一次

loop.register_child_run_guard(guard: ChildRunGuard, *, name: str | None = None) -> None
loop.remove_child_run_guard(name: str) -> bool
```

### 语义

- `run_agent_spec` 在**每次子运行**（`child_loop.send(...)` 段）外围，按注册顺序
  `ExitStack.enter_context(guard())` 进入全部 guard；子运行结束（含异常）逆序退出。
- guard 的典型实现是「快照父的 per-send 状态 → 子运行 → 恢复」。必须**可重入**
  （子的子 agent 会嵌套进入；token 式 suspend/restore 天然满足）。
- 仅同进程内联子运行需要（共享 hooks 对象 + 同 task contextvars 才有污染）；
  `SubprocessExecutor` 的叶子天然进程隔离，不适用、也不调用。
- workflow 的 `InProcessExecutor` 走 `run_agent_spec`，自动生效。

### 实现位置

`runtime/spec.py::run_agent_spec`（唯一 choke point）+ `StatefulAgentLoop` 上一个
`_child_run_guards: list[tuple[str | None, ChildRunGuard]]`。

### DeepTalk 删码

`_run_spec_isolated` 整个删除，改为 `loop_cache._build` 里三行
`loop.register_child_run_guard(...)`（reminders / pass_state / finalize 各一，
内部就是现有的 suspend/restore 对）。`subagent.py` 两处调用点改回直调 `run_agent_spec`。

### 测试

嵌套两层子 agent 下 guard 进出顺序与重入；guard 内 raise 不吞子结果；子运行异常时仍恢复。

---

## S2 — per-send 有效工具集传导给子孙

### 现状与缺口

`loop.send/follow_up` 已有 per-call `tools=`（名字序列或 ToolRegistry），只过滤**本次
父运行**；`run_agent_spec` 的子 agent 在 inherit 模式拿到父的**完整** registry —— 被禁
bash 的父，其子能跑 bash。缺的只是传导。

### 接口

```python
# power_loop/core/agent_context.py（新增，与 get_session_id 同族）
get_effective_tools() -> frozenset[str] | None    # None = 本次 send 无限制

# runtime/spec.py
run_agent_spec(..., inherit_send_filter: bool = True)
```

### 语义

- pipeline 在带 `tools=` 的 send/follow_up 期间设置该 contextvar
  （传 ToolRegistry 时取其全部名字），运行结束 reset。
- `run_agent_spec`（`inherit_send_filter=True` 默认）：子工具 =
  `filtered_registry(parent, spec.tools)` **∩** `get_effective_tools()`（为 None 则不裁）。
  即「子 ⊆ 父本次有效集」成为库级默认语义；`inherit_send_filter=False` 是显式逃生门。
- **workflow 提交时捕获**（解决 detached contextvar 失效）：`api.create_workflow`
  在构造 `Workflow` 时读取 `get_effective_tools()` 存为 `_allowed_tools`；引擎把它
  显式传给 executor 的每次 `run_agent`；`InProcessExecutor` 围绕 `run_agent_spec`
  以参数（非 contextvar）施加交集。
- **resume 一致性**：`journal.seed()` 新增可选 key `allowed_tools`（list[str] | null）；
  `resume_run/resume_detached` 读回并施加同一裁剪——否则 resume 会静默放宽权限。
  旧 journal 无此 key → None，行为同旧版。

### DeepTalk 删码

`subagent.py` 的 `_parent_allowed` contextvar、`set/reset_parent_allowed_tools`、
求交逻辑；`main.py` 两处 `set_parent_allowed_tools(allowed)` 包裹（`tools=allowed`
本来就在传，保留即可）。

### 测试

send(tools=[a,b]) 下 inherit 子只见 a,b；specify 子取交集；escape hatch；detached
workflow 叶子在父 send 结束后仍受提交时集合约束；resume 后集合不变。

---

## S3 — 子 agent `AgentLoopConfig` 宿主工厂

### 接口

```python
# AgentLoopConfig 新字段（与 representation / fold_strategy 同族的 config-pluggable seam）
subagent_config_factory: Callable[[AgentSpec, AgentLoopConfig], AgentLoopConfig] | None = None
```

### 语义

- `run_agent_spec` 照旧构建默认子 config（spec 派生字段 + 继承 retry_policy），然后若父
  config 带 factory：`child_config = factory(spec, default_child_config)`，**产物按原样使用**。
- 缝开在**宿主侧而非 spec 侧**是有意的：`AgentSpec` 是 LLM 作者写的，不让模型控制
  上下文策略。factory 拿到 spec 只是为了**按叶子分流**（如按 `spec.metadata`/name 给
  重工具叶子开 microcompact / 换 representation）。
- 注意事项进 docstring：默认 config 已带 `response_format`（结构化输出）与 spec 派生的
  system_prompt/model/预算字段，factory 若覆盖它们后果自负（`dataclasses.replace` 增量
  改是推荐姿势）。
- `spawn_agent` / `run_agent` / workflow `InProcessExecutor` 全部经 `run_agent_spec`，
  自动生效。

### 测试

factory 改 representation/microcompact 生效；不设 factory 与旧版逐字段一致；
factory 抛错 → 子运行失败并以工具错误面呈现（不炸父 loop）。

---

## S4 — `register_workflow_tools` 宿主注入点

### 接口

```python
def register_workflow_tools(
    registry, *, overwrite: bool = False,
    executor_factory: Callable[[StatefulAgentLoop, str | None], Executor] | None = None,
    budget_factory: Callable[[StatefulAgentLoop, str | None], SharedBudget | None] | None = None,
    spec_transform: Callable[[WorkflowSpec], WorkflowSpec] | None = None,
) -> None
```

### 语义

- 三者都按**每次工具调用**求值，实参 `(loop, parent_sid)` 来自当时的 agent_context。
- `_handle_create_workflow` 流程变为：解析校验 → `spec_transform(spec)`（可 raise
  `WorkflowSpecError`，聚合问题以工具错误串回给模型修复）→
  `create_workflow(spec, parent_loop=loop, executor=executor_factory(...),
  budget=budget_factory(...), parent_session_id=parent_sid)`。
- 不传任何参数 → 与现版逐字节相同。`workflow_status` 不变。
- resume 路径不需要新接口：`resume_run/resume_detached` 已接受 `executor=`/`budget=`，
  宿主启动恢复时用同一批 factory 的产物即可。

### DeepTalk 用法（不再 fork handler）

`executor_factory` 返回带能力钳制的自定义 Executor（工具交集之外的强制项：平台预算/
model 上限）；`budget_factory` 按 definition 配置造 `SharedBudget`。

### 测试

factory 被每次调用求值（非注册时）；spec_transform raise → 模型收到聚合错误；
detached + 自定义 executor 全程生效。

---

## S5 — TimerRunner 投递可插拔 + workflow 唤醒 helper 公开

### 接口

```python
TimerDelivery = Callable[[TimerFireCtx], Awaitable[None]]

TimerRunner(loop, *, delivery: TimerDelivery | None = None, ...)   # None = 现内置路径

# power_loop.workflow 新公开（现私有逻辑提取/改名）
async def claim_wake(store, parent_sid: str, run_id: str) -> bool   # 原子 claim；True=首次
def parse_workflow_wake(note: str | None) -> str | None             # 从 timer note 解析 run_id
```

### 语义

- **只替换最后一步**。scan / CAS claim / heartbeat / stale 恢复 / TIMER_FIRE hook /
  BREAK-SKIP-postpone 分流全部不变；走到内置
  `self._loop.follow_up(ctx.message, ...)` 处改为 `await (delivery or 内置)(ctx)`。
  `ctx` 携带 `session_id / timer_id / note / due_at / message`（hook 可能已重写 message）。
- delivery 正常返回 → `finish_firing_timer`（一次性完结/周期重臂，同现状）；
  delivery 抛错 → 走现有 `scan_once` 异常分支（重臂 +30s）。**delivery 必须幂等容忍
  at-least-once**（与现语义一致）。
- **wake 去重免费**：TIMER_FIRE hook 先于 delivery 运行，`register_wake_guard` 的
  SKIP 裁决对自定义投递同样生效——宿主 delivery 不需要自己去重。`claim_wake` 只服务
  完全不用 TimerRunner、自己轮 `store.due_timers()` 的宿主。
- `eager_wake=True` 的快路径仍直调 `loop.follow_up`（不经 TimerRunner）。自定义
  delivery 的宿主**不应开 eager_wake**——docstring 明示；不做自动阻断（组合无害，只是
  eager 那次绕过宿主管道）。

### DeepTalk 用法（= workflow 接入方案 A 的落点）

跑一个 `TimerRunner(loop, delivery=deeptalk_delivery)`；delivery 里
`parse_workflow_wake(ctx.note)` 判别 workflow 唤醒 → 调 api 内部接口造 dispatcher 风格
`agent.trigger`（lane C 同族），簿记（agent_run / cancel / finalize / 用量）全部天然齐全。
非 workflow timer 同样进自家管道，一个投递口统一。

### 测试

自定义 delivery 收到 hook 重写后的 message；delivery 抛错 → 重臂 +30s 且后续重投；
wake guard SKIP 时 delivery 不被调用；`claim_wake` 并发双 claim 只有一个 True。

---

## 版本与兼容

- 版本 `3.14.0`，CHANGELOG 五条分列（S1–S5），各附「宿主删码指引」一句。
- `STABLE_API` 元组**本次不增**；新符号（`get_effective_tools`、`claim_wake`、
  `parse_workflow_wake`、`ChildRunGuard`/`TimerDelivery` 类型别名）以 PROVISIONAL 发布。
  `StatefulAgentLoop.register_child_run_guard`、`AgentLoopConfig.subagent_config_factory`、
  `register_workflow_tools` 新 kwargs 是 STABLE 载体上的可加性成员，SemVer 无碍，
  CHANGELOG 标注。
- 存储：仅 journal blob 新增可选 `allowed_tools` key；无迁移。
- 发布后 DeepTalk 侧：repin `power-loop[all]>=3.14`，删除
  `_run_spec_isolated`（→ 三个 guard 注册）、`_parent_allowed` 全套（→ 已有 `tools=`）、
  workflow 接入直接走 S4/S5，不 fork、不自建轮询器。

## 实现顺序建议

S2 → S1 → S3（同在 `run_agent_spec` 一带，先做语义最重的）→ S4 → S5。
S4/S5 与前三条无耦合，可并行。
