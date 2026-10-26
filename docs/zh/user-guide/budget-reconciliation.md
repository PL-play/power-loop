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
