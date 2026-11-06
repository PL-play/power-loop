# 异步编排与长任务

[English](../../en/user-guide/async-orchestration.md) | [用户指南](../index.md)

本文对应未发布的 10.0.0、设计 203 P3。宿主启动 `Runtime.worker`，通过
`RuntimeClient` 提交输入；worker 从持久输入重建执行环境。结果保存在 store，通知只负责唤醒。
共享执行还需要所有节点使用同一个 store、共享消息传输和共享 session 所有权协调器；默认进程内组件不能代替跨机器设施。

## 子任务与后台任务

`spawn_agent` 创建独立 child session，返回 `job_ref`、`input_ref`、`session_id`。
收据表示已受理。返回的 `JobRef/0` 可以直接等待：首次启动后读第 1 代结果；启动前失败或取消读第 0 代终态。
原启动输入首次 claim 时从 0 绑定到 1，此后不追随 resume。

子任务固定创建时的工具权限和配置，只允许一层委派。原主 session 后续 send 可以查询、nudge 或停止原子任务，
无需保留父 loop。nudge 只加入仍在执行的原代次；已中断任务使用显式 resume。
内联 `run_agent_spec` 使用同一持久创建路径再等待结果。等待保留 session 所有权并释放执行容量，因此容量 1 也可用。

`background_run(action="tool", tool=..., args=...)` 或 `action="run", command=...` 创建独立后台任务。
`check/list/wait/cancel` 使用原 `task_id`；wait 超时只结束等待。声明 `interrupt="background"` 且允许后台执行的工具，
开始时即创建独立任务；steer 到来后父只停止等待，原操作继续。实际工具费用归原后台 job 和原 root，受理操作不重复收费。

结果和父 inbox 引用在同一事务保存，提交后通知。运行中的父在轮次边界消费，已返回的父由 worker 启动后续处理。
重复通知不重复交付；明确取消的范围不会因迟到结果再次启动。子 Agent 完成前收集自己的后台结果，
在既有轮次预算内处理最终回答和一次输出修复。查询可取完整结果；模型通知可使用摘要。

父退出不会取消独立任务。远端 shell 只能查询或停止原资源；资源失联返回 interrupted/unknown，不能在新机器重跑命令。
不再使用 `_RUNNING`、`BackgroundManager`、`BackgroundRuntimeProjector` 或完成回调作为任务权威。

## 等待回答与续跑

`request_user_input` 保存 interaction、原 job、send 和轮次，然后释放执行容量。
用 `inspect(input_ref)` 读取 `waiting_input`；普通 send 不会替代回答。宿主通过原 job 引用回答：

```python
receipt = await runtime.client.answer(
    job_ref, interaction_id=interaction_id, wait_round=wait_round,
    value="blue", message_id="answer-1", actor=actor,
)
result = await runtime.client.wait(
    original_input_ref, actor=actor, deadline_ms=deadline_ms,
)
```

回答和对应 tool 消息原子写入。同值重复得到原回答状态；不同值、错误轮次、过期或取消后的新回答明确拒绝。
换节点回答仍保留原 job/generation、send 和累计轮次、费用。`StatefulAgentLoop.submit_input` 是同一入口的封装。

只有正代次 interrupted/unknown 且未取消的子任务可显式 resume；第 1 代续跑进入第 2 代，新 resume 输入绑定 2。
原 `JobRef/0` 和原 start 输入仍读 1。未绑定 resume 输入的撤销只 void 该输入，不改旧结果。
不会自动重放未知工具。取消 `JobRef/0` 仅限尚未启动；已经 claim 必须报 stale_generation，
而取消 `InputRef` 使用该输入原子绑定的实际代次。

## 保留与阶段边界

任务结束即释放进程内等待者和容量；检查点、额度分配和已结预算临时项沿用原定稿清理。
持久结果、原调用和描述保留到整棵任务树可按既有保留规则回收；父 send 返回不删除活跃子 session。

workflow driver/node 的统一运行时接入属于 P4，目前仍明确拒绝；定时器、巡检及业务宿主接入按后续阶段处理。
本页不提供用户绕过原管理范围直接控制任意子任务的新产品入口。历史调用格式不做运行时兼容。
