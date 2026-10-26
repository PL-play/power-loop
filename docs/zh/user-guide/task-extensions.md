# 异步任务扩展：复用 Hook 与 Event

任务、workflow 节点和 subagent 共用 power-loop 的扩展机制。扩展应基于类型化接口和已注册的工具契约，不按工具名称、平台模块名或业务类型推断权限。

## 选择扩展接口

| 需求 | 接口 | 边界 |
|---|---|---|
| 模型上下文、轮次策略 | `AgentHooks` 的 `LLM_BEFORE`、`ROUND_DECIDE` | 复用 typed context；不得绕过实际调用准入与预算 |
| 自然结束前补充收尾指导 | `COMPLETE_DECIDE` | 可以请求继续，但不代表所有子任务已完成；仍须经过运行时状态与预算检查 |
| UI、日志、耗时与 token 指标 | `AgentEventBus` | 订阅既有 LLM、工具、subagent、timer 事件；慢处理用异步订阅 |
| 监管收尾的观察通知 | `TASK_SUPERVISION_CLOSED` | 只有原始证据验证及事务成功返回后才发布 |
| 真实取消、外部结果回执、费用结算 | 现有运行时/资源/预算接口 | 保留持久化、幂等和恢复协议，不能用普通事件订阅回调代替 |

新策略优先接入已有 hook；新增观测点优先扩展已有 event 类型与 payload。不新建并列的 hook/event 总线，也不把数据库事务移入可被用户移除的 hook。

## 订阅监管收尾

在创建运行时前，显式传入宿主自己的总线。可以和 loop 共用同一个 bus：

```python
from power_loop import AgentEventBus, AgentEventType, TaskSupervisionClosedPayload
from power_loop.runtime.jobs import JobManager

bus = AgentEventBus(suppress_subscriber_errors=True)

async def refresh_task(event):
    payload: TaskSupervisionClosedPayload = event.data
    # 使用 (payload.store_id, payload.event_id) 去重；按原有权限重新读取任务。
    await my_ui.invalidate_task(payload.store_id, payload.job_id)

bus.subscribe(AgentEventType.TASK_SUPERVISION_CLOSED, refresh_task)
manager = JobManager(store, event_bus=bus)
# TaskRuntime(manager, ...); StatefulAgentLoop(..., event_bus=bus)
```

`JobManager` 默认不发布到全局 bus，避免不同宿主混用任务通知。宿主应在自己控制的租户/权限范围内配置总线；event bus 本身不是权限隔离层。

payload 只有 `store_id`、`job_id`、`generation`、`event_id`、`revision`；不含任务正文、租约、执行令牌、预算或内部证据。envelope 的 `session_id` 是本次执行验证的调用会话，`source` 为 `task_registry`。订阅者修改 payload 不会修改落库原件或方法返回值。

`finalize_supervision` 验证成功后，为本次请求的 job 发出通知。幂等重验可以重复发出同一 `event_id`；内部递归收尾的每一个后代不保证独立发出通知。它不是“每次状态变化恰好一次”的事件流。权限拒绝、证据不完整、generation 冲突、回滚以及提交回执未知时不发送；落库后进程崩溃也可能丢通知。需要可靠消费时，使用持久化任务记录/回执与原有恢复机制，不能凭“没收到事件”判断任务还在运行或重新执行。

此事件仅表示某一代监管证据已闭合，不承诺业务成功、所有将来代次结束或费用结清。事件不得直接驱动 workflow 下游放行、主 Agent 结果 ACK 或业务扣费。

## 回调失败与退出

- 发布发生在事务退出后，不持有任务控制锁等待订阅者。
- 同步 handler 必须短且不阻塞；I/O 用 `async def`。异步 handler 的生命周期沿用现有总线机制，由宿主负责退出时的处理。
- 同步订阅异常（包括直接抛出的 `CancelledError`）不会使已提交的 `finalize_supervision` 变为失败；真实事务/调用方取消仍由原运行时处理。
- 建议 `suppress_subscriber_errors=True`，这样普通异常不阻止同一通知的后续订阅者。严格 bus 保留原有“遇错停止分发”语义，但 JobManager 仍隔离该错误。不要把不能丢失的业务动作放进订阅者。
- 本接口不新增查询、表、DDL 或后台轮询，也不改变任务协议版本。未传 bus 的宿主行为保持原样。

## 原始模型响应的前置观察（B75）

`AgentEventType.LLM_RESPONSE_RECEIVED` 配合 `LlmResponseReceivedPayload`，在一次物理 `LLMService.complete` 返回后、缺失用量估算和原观察持久化之前发布。包含原 `call_id/round_index/attempt/model`、托管调用的 store/job/attempt/generation 身份，以及供应商实际提供的 `usage`、`cost_amount`、`response_id`。不包含输入、回复正文、credential、grant 或业务权限。缺失用量是空字典，金额/响应ID未知是 `None`，不自动估算或补零。

同步订阅者异常被隔离，修改事件数据不能改写原响应、采集包或观察记录。异步订阅者可能在原观察已保存之后才执行：这是一个尽力通知，不是可靠费用源、持久化回执或授权。若严格bus中的某个订阅者先失败，后续订阅者也不保证收到通知。原 `LLM_CALL_COMPLETED` 仍在观察保存尝试之后，`LLM_AFTER` 更晚。

有独立持久化需求时，实现下节的类型化端口。不要将可移除的hook或异步事件订阅者作为必经的财务事务。

## 可信响应采集端口

原 `BudgetConfig.provider` 可以额外实现 `power_loop.runtime.response_capture.ResponseCaptureProvider`：

```python
from power_loop.runtime.response_capture import ModelResponseCapture, ResponseCaptureReceipt

class MyBudgetProvider:
    # 原有 prepare/reserve/lookup/prepare_usage/apply_usage/release 等方法保留。
    async def capture_response(self, capture: ModelResponseCapture) -> ResponseCaptureReceipt:
        # 此处示意宿主自己的持久服务；先验证原 context/grant 的授权。
        # 必须原子按 capture_id 去重；同 ID 不同 fingerprint 必须拒绝。
        record = await self.evidence_store.put_if_absent(
            capture.capture_id, capture.fingerprint, capture,
        )
        # 只有实际 COMMIT 确认后返回。超时/丢 ACK 应抛错，不能假报成功。
        return ResponseCaptureReceipt(capture.capture_id, capture.fingerprint, record.id)
```

这是可信宿主配置的可选能力，不是模型可调用工具；普通工具恰好叫同一名称不会获得权限。不要求继承特定平台实现，不依赖平台API、SQL或调度器。宿主负责认证、独立证据存储和后续采集；SDK负责绑定、验证回执和实际任务生命周期。

调用顺序：

1. 原预算预留完成后，SDK在物理dispatch之前验证原策略、租约、context和grant，固定原回调与元数据。未实现端口时保持原行为。
2. 物理响应返回后保留原供应商字段。前置通知发布后，SDK将固定context、原reserved grant、原started_at_ms、当前received_at_ms及实际usage/cost_amount/response_id交给端口；调用端口前不再读取SDK数据库。时间均为宿主毫秒时钟，不能当供应商结算时间或要求两者在NTP变化时严格递增。
3. 等待原回执并核对capture_id/fingerprint；验证在实际任务内部进行，取消外层等待不会绕过验证。端口不得修改传入数据；回执前的数据变化同样拒绝。
4. 无论采集失败或回执无效，仍尝试正常模型观察/预算交接。采集回执本身不调用apply_usage，不核销额度，不生成执行退出或最终发票。

`capture_id`由原store_id+attempt_id确定，fingerprint覆盖整个采集包。`ModelResponseCapture`中的usage只接受已知token字段的非负整数，缺失保持缺失；cost_amount为有限非负十进制字符串，但不包含推断的币种或最终结算标志。字典复制隔离后续观察；宿主必须验证原grant和context，source/record ID本身不是认证。

采集和随后的普通观察使用原 `_host_call`/TaskRuntime 实际任务管理：有界等待超时不丢弃实际任务，`budget_control_tasks()`和shutdown仍能看到它们。取消调用方时后台原记录继续；关闭宿主前须按原生命周期等待或取消并核对实际退出。端口自取消/报错不能重调模型，原观察保存失败保留unknown；不能因收到采集回执就放行下游。

本轮同时防止响应返回后的处理或完成观察异常触发物理模型重试；严格的旧完成订阅仍可能向调用方报错，但不能因此重新购买一次请求。流结束观察失败不覆盖原结果/异常。

此端口并不保证零丢失：响应返回至独立存储首次COMMIT之间，硬崩溃仍可能没有副本；供应商未返回时也没有完整响应可采集。需要供应商可查询记录或可信对账补证；没有证据仍保持unknown。宿主从独立存储恢复后可使用既有 `stage_budget_usage_revision`/核账接口，且必须区分正常观察已入账、缺首次观察以及真正后续更正，不能对同一原请求重复扣费。

SDK无新增持久表、DDL或状态字段；schema10/protocol21不变。已有未接此端口的宿主不会自动获得独立采集能力，正式宿主存储/认证/巡检的接入须另行验收。

停机过程中若控制器退出时又移交了一个已经运行的原请求，运行时会在**最初的停机截止时间**内重新检查并等待该句柄；不会重新开始计时，也不会把仅取消等待者视为请求已退出。超时仍存活的句柄继续报告pending，`executor_status()`可观察后续真实退出。
