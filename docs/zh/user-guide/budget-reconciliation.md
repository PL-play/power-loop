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

- 后续修订必须已有原 tracking/prepared/dispatched 及已确认的前一 submission/ACK；首次来源通常为原 finished 观察，也可以是下节独立登记的首次财务证据。已有待确认的原提交必须先核对，不能用新证据替换它。本接口不补造崩溃丢失的执行观察，也不负责向供应商自动查询。
- 每个 attempt 同时只接受一个待处理修订；并发相同 evidence_id/内容/expected_revision 返回同一个持久回执，不重复核销。不同快照、过时版本或未完成前一修订明确报错，不算受理。采集器必须保留源记录，核对后再决定如何处理，不能把拒绝当丢弃证据的许可。
- 登记事务保存独立证据事件、冻结前一 submission/ACK、核销所有原 scope 的 token 差额并设 pending。该事务不调用提供者；失败全部回滚。后续 revision（≥2）不增加 used_calls，不改消息、原观察、原执行代次或资源退出状态。实际总量超过原上限会关闭新准入，后续更正不会擅自重新启用。
- unknown/estimated 不降级实际事实；已知 actual 字段不能消失。修改已确认数值需要引用原证据并保留原因。估算或缺少实际总量时不释放 token 风险；确认 token 也不能释放远端执行的 inflight 风险。
- 提交准备、API调用或本地ACK保存失败后 pending 保留。新控制实例重用原证据和冻结提交；旧准备回调及旧ACK不能覆盖新 revision。证据事件缺失时，即使 submission 已冻结，也拒绝继续提交。
- 原任务取消、预算停用不阻止合法核账；已经关闭的 runtime 不能再登记。取消等待不取消真实端口任务，仍通过 `budget_control_tasks()` 管理它们，实际退出前不关闭 store。

task schema仍是10；当前 task protocol 为21，没有新增表或索引。旧协议必须显式维护迁移；普通打开不升级。协议20不理解下节的独立首次财务证据，不能作为21数据的控制器。正式业务宿主的迟到采集与API修订适配须分别接入和验收，单独SDK测试不代表全链路上线。


## 原进程崩溃、没有返回观察时的首次财务证据

可信采集器已核实原供应商记录，但SDK只保留了原 tracking/prepared/dispatched 和预算grant、没有 finished 记录时，可调用同一登记接口，显式设置 `expected_revision=0`。它不是一次重试模型请求，也不创建虚假的 finished 事件。

```python
receipt = await control_runtime.stage_budget_usage_revision(
    original_session_id,
    job_id=original_send_job_id,
    attempt_id=original_attempt_id,
    expected_revision=0,
    evidence=verified_evidence,  # BudgetUsageEvidence，已在宿主完成来源认证
)
ack = await control_runtime.reconcile_budget_usage(
    original_session_id,
    job_id=original_send_job_id,
    attempt_id=original_attempt_id,
)
```

提供者必须另行实现可选 `BudgetInitialUsageProvider` 的
`prepare_initial_usage(context, grant, evidence, *, started_at_ms)`，返回原 attempt、revision=1 的 `BudgetSubmission`。旧 `BudgetRevisionProvider` 不会自动获得首次证据能力。原始开始时间由SDK传入；不能把核账时刻伪装成原调用或结束时间。宿主应记录费用证据来源及原授权/报价，但不能根据账单推断 returned/cancelled/停止成功。提交、ACK丢失恢复和实际端口句柄管理继续用原 `apply_usage`、`BudgetAck`、`budget_control_tasks()`，不需要新后台循环或消息工具。

- 首次财务登记保存独立 revision1 事件，并保留 `initial_usage_evidence` 指向该证据，后续修订不覆盖原来源。原事务只将 held_calls 转为 used_calls 一次；实际token总量已知时核销原token预留。未知或估算用量仍保留token风险。
- **inflight保持占用**，任务消息、原job状态、原grant、执行代次及控制权不变。费用ACK不构成执行退出证明，`require_quiescent`仍拒绝缺少原返回记录的任务，下游不能因此开始。
- 如果原模型稍后真实返回，原 finished 事件照常保存；预算投影只按这份真实返回释放一次inflight，不重复应用财务调用/token，也不覆盖独立财务证据。若返回后的预算投影再次崩溃，同一恢复入口会补齐执行投影。unknown观察不会释放inflight。
- 如果迟到真实观察与先前财务事实不一致，两者保留为不同来源。采集器核对后按原revision链提交明确更正；不能偷偷用后到的值覆盖已提交金额或token。
- 如果 finished 已存在，使用普通观察/原提交恢复路径；`expected_revision=0`拒绝替换它。已有pending或首次ACK未知也不能重新登记revision1，必须重交冻结原提交。初始证据没有前一版本，不能带correction_of。
- stage失败全部回滚；并发相同请求返回同一receipt，其他竞争快照明确拒绝。记录损坏/来源事件缺失时，即使提交已冻结也拒绝交付。采集器须保留原来源记录，拒绝不等于受理。

来源ID不是认证，SDK不自动向供应商取账单。生产宿主必须实现和验证新端口，并完成采集器授权、原请求匹配和金额规则。本SDK能力不能单独当作正式平台已支持崩溃账单采集；未知远端请求的停止责任继续保留。

## 在原观察之前保存响应副本

可选 `ResponseCaptureProvider.capture_response` 现已由SDK在原观察落库之前调用，详[任务扩展接口](task-extensions.md#可信响应采集端口)。原context/grant在dispatch前固定；响应返回后不依赖一次SDK数据库读取即可交给独立存储。采集回执与预算ACK不同，不自行登记首证、核销额度或证明退出。宿主恢复采集记录后须验证来源，按原attempt现有观察/修订状态选择正常核账或独立首证；已正常入账的同一响应不得重复结算。普通 `LLM_RESPONSE_RECEIVED` 事件仅用于观察。
