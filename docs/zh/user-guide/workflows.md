# 动态工作流

[English](../../en/user-guide/workflows.md) | [用户手册](../index.md)

`power_loop.workflow` 是一个**可选**子模块，用于*代码驱动*的确定性多 Agent 编排：一份声明式的 `WorkflowSpec` JSON，其叶子节点是普通的子 Agent，由一个确定性引擎解释执行。它是对 `spawn_agent` / `run_agent`（[子代理](subagents.md)）那种*模型驱动*的即兴委托的补充，而非替代。

当你（开发者）预先就知道控制流（一条流水线、一次扇出、一个分支）时，使用工作流。当应该由*模型*决定调用谁时，使用即兴委托。

## spec

一个 `WorkflowSpec` 是一棵节点树。共有五种节点类型：

| 节点 | 作用 |
|---|---|
| `agent` | 运行一个子 Agent（一个叶子）；以 `id` 为键 |
| `sequence` | 按顺序运行 `steps` |
| `parallel` | 并发运行 `branches`（屏障式扇入） |
| `foreach` | 在一个运行时列表（`items` 或 `items_from`）上对 `body` 做映射，可选并发 |
| `branch` | 根据前序 agent 的输出选取一个 `case` |

```python
SPEC = {
    "name": "research_and_summarize",
    "input": "the Japanese tea ceremony",
    "budget": {"max_tokens": 200_000, "stop_at_remaining_pct": 5},
    "root": {
        "type": "sequence",
        "steps": [
            {"type": "agent", "id": "plan",
             "spec": {"name": "planner",
                      "system_prompt": 'Reply ONLY JSON: {"subtopics": ["...","...","..."]}'},
             "input": "Topic: {{input}}",
             "output_schema": {"name": "Plan", "schema": {
                 "type": "object", "required": ["subtopics"],
                 "properties": {"subtopics": {"type": "array", "items": {"type": "string"}}}}}},
            {"type": "foreach", "id": "research",
             "items_from": "plan.subtopics", "as": "subtopic",
             "parallel": True, "max_concurrency": 3,
             "body": {"type": "agent", "id": "researcher",
                      "spec": {"name": "researcher",
                               "system_prompt": "Write 2-3 factual sentences."},
                      "input": "Subtopic: {{subtopic}}"}},
            {"type": "agent", "id": "synthesize",
             "spec": {"name": "writer", "system_prompt": "Synthesize into one paragraph."},
             "inputs_from": ["research"]},
        ],
    },
}
```

节点之间的数据流：

- `{{input}}` / `{{var}}` —— 在节点的 `input` 中做模板替换。
- `output_schema` —— 让叶子产出结构化 JSON（经校验）；下游节点读取它。
- `items_from: "plan.subtopics"` —— `foreach` 从前序 agent 已解析的 payload 中读取一个键；`inputs_from: ["research"]` 则把前序结果作为文本喂入。

## 运行它

```python
from power_loop import StatefulAgentLoop, AgentLoopConfig
from power_loop.workflow import create_workflow

loop = StatefulAgentLoop(llm=make_llm(), db_path=":memory:",
                         config=AgentLoopConfig(system_prompt="orchestrator"))

wf = create_workflow(SPEC, parent_loop=loop)   # validates on creation
result = await wf.run()                         # deterministic, in-process

print(result.status)                            # completed | failed | budget_exceeded | cancelled
print(result.results["plan"].payload)           # parsed structured output
print(result.results["synthesize"].text)        # final sub-agent text
print(result.usage["total_tokens"])
```

`create_workflow` 会立即校验 spec；非法 spec 会抛出 `WorkflowSpecError`，**一次性**列出所有问题（这样编写该 spec 的 LLM 可以一次修复到位）。`result.results[node_id]` 是一个 `AgentResult`（`status`、`text`、`payload`、`usage`、`session_id`、`db_path`）。

参见 [example 27](../../../examples/27_dynamic_workflow.py)。

### 让模型来编写工作流

注册这些元工具，使主 Agent 能够自行产出一份 spec 并运行它：

```python
from power_loop.workflow import register_workflow_tools
register_workflow_tools(registry)   # adds create_workflow + workflow_status
```

（不要把它们授予某个工作流自己的叶子节点 —— 本层级拒绝嵌套运行。）

## 执行器

引擎通过一个 `Executor` 来运行叶子节点。开箱即用提供两种。

### 进程内（默认）

`InProcessExecutor` 通过 `run_agent_spec` 在父进程内运行每个叶子。快、简单，且共享父进程的 store。这正是 `wf.run()` 默认使用的执行器。

## Out-of-process (`SubprocessExecutor`)（进程外）

**让每个叶子运行在自己的操作系统进程里，对应自己的 SQLite 文件**，这样「每文件单写者」规则得以轻松成立，且一个崩溃 / 被杀 / 超时的叶子无法破坏编排器或它的兄弟节点。

```python
from power_loop.workflow import SubprocessExecutor, WorkerBootstrap, WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine

executor = SubprocessExecutor(
    bootstrap=WorkerBootstrap(llm_from_env=True),  # child rebuilds the provider from env
    runs_dir="/tmp/wf_runs",
    timeout_s=120,
    delete_on_success=False,   # keep each leaf db for inspection (GC via reap_runs)
)
result = await WorkflowEngine(loop, executor=executor, run_id="run1").run(
    WorkflowSpec.from_json(SPEC)
)
```

worker 仅凭**配置**（`WorkerBootstrap`）重建其依赖 —— 没有任何活动对象跨越进程边界。取消 = SIGTERM→SIGKILL；超时 / 崩溃 → `failed`（在 resume 时可重跑）。每个叶子的 db 会保留在磁盘上以便事后检查；用 `reap_runs(runs_dir, older_than_s=…)` 或 `cleanup_run(runs_dir, run_id)` 做 GC。要把每个叶子限制在沙箱里，可注入一个 [`WorkerLauncher`](sandboxing.md#workerlauncher--sandbox-a-workflow-leaf-process)。参见 [example 30](../../../examples/30_subprocess_isolation.py)。

## 分离执行 + 完成唤醒

在后台运行一个工作流，并让它在完成时唤醒父 Agent（父 Agent 通过 `pass_turn` 让出，再经由一个持久化定时器被重新唤醒）：

```python
from power_loop import TimerRunner
from power_loop.workflow import create_workflow, register_wake_guard, get_workflow, list_workflows

register_wake_guard(loop)                # dedupe the at-least-once wake timer
await TimerRunner(loop).start()          # REQUIRED: delivers the wake (see Timers)

wf = create_workflow(SPEC, parent_loop=loop, parent_session_id=parent_sid)
handle = await wf.start(detached=True)   # returns immediately
# inspect any time:
get_workflow(loop, parent_sid, handle.run_id, detail=True)   # status + per-step usage
list_workflows(loop, parent_sid)
```

宿主必须运行一个 `TimerRunner`，**并且**调用 `register_wake_guard(loop)`，否则父 Agent 永远不会被唤醒。进度会以 `source="workflow"` 的 `SYSTEM_LOG` 事件形式发布。

## 跨进程重启恢复

分离运行会把它的 spec 以及每一步的输出记入日志（journal）。如果进程在运行中途挂掉，把一个新的 loop（使用相同的 `db_path`）指向该 run id 并恢复 —— 已完成的步骤会被**重放**（不会再次调用其子 Agent），只有尚未完成的尾部才会重跑：

```python
from power_loop.workflow import resume_run, resume_detached

result = await resume_run(loop, parent_sid, run_id)        # sync
# or, to resume in the background and wake the parent:
handle = await resume_detached(loop, parent_sid, run_id)
```

如果原始运行使用了非默认的执行器或预算，则需重新提供 `executor=` / `budget=` —— 只有*数据*被记入日志；运行时对象来自活动进程。每个叶子会得到一个稳定的 `metadata["idempotency_key"] = f"{run_id}:{node_id}"`，因此带副作用的工具可以对重跑做去重。`foreach` 是原子的：它要么通过其记入日志的聚合结果进行重放，要么重跑整次扇出。

## 另见

- [子代理](subagents.md) —— 模型驱动的替代方案
- [沙箱](sandboxing.md) —— 隔离进程外叶子
- [定时器](timers.md) —— 分离运行背后的唤醒机制
- [黑板](blackboard.md) —— 协调对等 Agent
