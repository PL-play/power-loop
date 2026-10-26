# 原费用的跨进程核对

宿主在进程重启后可以补交**已经记录的**模型用量，无需重新调用模型、重建 `Execution` 或认领旧 lease：

```python
# runtime 使用同一 store、tenant/principal、原 BudgetConfig manifest
# 和能处理原费用的可信 BudgetProvider；可以是尚未启动的新 boot。
receipt = await runtime.reconcile_budget_usage(
    original_session_id,
    job_id=original_send_job_id,
    attempt_id=original_attempt_id,
)
```

接口仅接受原标识。它先验证调用方权限、原预算策略及快照、模型 tracking/prepared/dispatched/finished 记录的关联。数据缺失或冲突时拒绝，不补造用量。没有实际返回但已保存 unknown 观察时，可补交该 unknown 事实，仍不证明供应商停止或费用为零。

两种恢复窗口共用原事务实现：

1. 模型观察已保存，预算投影尚未完成：从原观察投影计数一次，再准备原费用提交。
2. 提交已经冻结，API 或本地回执丢失：重发完全相同的提交与 revision；API 幂等，原计数不会再次应用。

同时恢复、事务回滚、重复调用和重启不生成新 attempt，也不重新 reserve/release。原任务取消、原 boot 退出、预算策略停用不影响保存已发生费用。接口不会自动启动 boot、恢复任务、写 assistant 消息、确认业务结果或关闭监管；返回 `usage_acknowledged` 也不意味着所有金额已结清。

供应商尚未返回的费用不能从此接口凭空取得。真实迟到发票/新 usage revision 仍须原供应商证据及既有 API 账务修订契约；不能传入人工猜测的金额替代记录。批量巡检与宿主调度由调用方按自己的权限范围组织；此接口不新增后台扫描器。重启后的宿主应继续保留原 provider/策略绑定，不能通过换钱包、报价或 scope“恢复”额度。

若宿主正在关闭当前 runtime，机械 HTTP 句柄仍受同一停机协议管理；不要在已经 shutdown 的实例上启动补交。用新的控制实例执行核对，并等其实际 HTTP 句柄退出后再关闭 store。取消等待不等于 API 未提交；后续仍核对原 submission。

## 恢复原策略与宿主停机

`model_budgets.recovery_config(manager, actor, job_id, attempt_id, provider_key=..., provider=...)`
从原策略表及独立事件校验后还原 `BudgetConfig`。宿主必须先以原历史授权验证 provider_key
和提供者；接口不创建新策略，不把 enabled=0 改为 1，也不重置 scope 计数。替换 scope、钱包
或提供者标识会被拒绝。

`runtime.budget_control_tasks()` 返回实际仍在运行的预算控制 Task；即使上层等待已超时或
取消，这些句柄仍保留。控制宿主应把它们计入自己的停机窗口，等待实际退出后再关闭传输和
共享 store；句柄退出不证明 API 没有提交。该接口不启动 boot，不返回任何执行令牌。

## 可信迟到证据与后续修订（task protocol 20）

宿主可用 `BudgetUsageEvidence` 和 `TaskRuntime.stage_budget_usage_revision` 登记原 attempt 的新证据。只有可信账务采集器可以使用；没有对应的模型工具。来源字段用于追溯，不会替宿主验证供应商签名、身份或账单真实性。

```python
from power_loop.runtime.budget_contracts import BudgetUsageEvidence

# confirmed_usage 保留全部已知 token 字段；金额等宿主事实放在 details。
evidence = BudgetUsageEvidence(
    evidence_id=stable_evidence_id,
    source=trusted_source_name,
    source_record_id=original_provider_record_id,
    token_quality="actual",
    usage=confirmed_usage,
    details=verified_host_facts,
    correction_of=previous_evidence_hash,  # 修改已有 actual 值时必填
    correction_reason=correction_reason,  # 与 correction_of 同时填写
)
staged = await runtime.stage_budget_usage_revision(
    original_session_id,
    job_id=original_send_job_id,
    attempt_id=original_attempt_id,
    expected_revision=previous_ack_revision,
    evidence=evidence,
)
receipt = await runtime.reconcile_budget_usage(
    original_session_id, job_id=original_send_job_id, attempt_id=original_attempt_id,
)
```

首次补充 actual 或同值确认可省略 `correction_of/correction_reason`。原观察的引用哈希是 `_hash(original_observation)`；后续引用为前一次 stage 返回的 `evidence_hash`。不要用 API 请求指纹冒充证据哈希。事件索引由 stage 回执的 `event_id` 提供；原始 source_record_id 与 evidence_id 也保存在该事件中。

原 `BudgetConfig.provider` 必须实现可选 `BudgetRevisionProvider` 接口的 `prepare_usage_revision(context, grant, previous_submission, previous_ack, evidence)`，返回同一 attempt、前一 revision+1 的 `BudgetSubmission`。准备方法不得执行付费动作或创建新预留；提交复用原 `apply_usage` 及 `BudgetAck` 幂等契约。宿主在准备时校验自己的金额、币种、价格和更正规则；SDK只处理通用 token 事实及原本地 scope。普通旧 provider 不会因为方法名相似的模型工具而获得这个能力。

前提与恢复规则：

- 必须已有原 tracking/prepared/dispatched/finished、预算观察及已确认的首次 submission/ACK。缺少首次观察或 ACK 时明确拒绝；先核对原提交。此接口不补造崩溃丢失的观察，也不负责向供应商自动查询。
- 每个 attempt 同时只接受一个待处理修订；并发相同 evidence_id/内容/expected_revision 返回同一个持久回执，不重复核销。不同快照、过时版本或未完成前一修订明确报错，不算受理。采集器必须保留源记录，核对后再决定如何处理，不能把拒绝当丢弃证据的许可。
- 登记事务保存独立证据事件、冻结前一 submission/ACK、核销所有原 scope 的 token 差额并设 pending。该事务不调用提供者；失败全部回滚。新 revision 不增加 used_calls，不改消息、原观察、原执行代次或资源退出状态。实际总量超过原上限会关闭新准入，后续更正不会擅自重新启用。
- unknown/estimated 不降级实际事实；已知 actual 字段不能消失。修改已确认数值需要引用原证据并保留原因。估算或缺少实际总量时不释放 token 风险；确认 token 也不能释放远端执行的 inflight 风险。
- 提交准备、API调用或本地ACK保存失败后 pending 保留。新控制实例重用原证据和冻结提交；旧准备回调及旧ACK不能覆盖新 revision。证据事件缺失时，即使 submission 已冻结，也拒绝继续提交。
- 原任务取消、预算停用不阻止合法核账；已经关闭的 runtime 不能再登记。取消等待不取消真实端口任务，仍通过 `budget_control_tasks()` 管理它们，实际退出前不关闭 store。

task schema仍是10；task protocol升到20，没有新增表或索引。旧协议必须显式维护迁移；普通打开不升级。协议19恢复器不知道修订语义，因此不能用它控制已有20数据。正式业务宿主的迟到采集与API修订适配须分别接入和验收，单独SDK测试不代表全链路上线。
