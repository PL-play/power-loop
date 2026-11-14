# SDK 12：会话与输入

版本固定为 **12.0.0 / schema 17 / protocol 43 / descriptor 2**。当前为本地源码交付；没有发布、部署、业务切换或 P8 最终验收。

工厂按固定版本的 descriptor/profile 组装模型、工具、hooks 和资源。先启动 Runtime worker，再通过 `create_session` 创建会话，通过 `send(target, body, message_id=..., actor=...)` 发送输入。宿主提供可信 actor；模型参数不提供权限身份。

`TargetRef(store_id, session_id)` 定位会话；`InputRef(store_id, session_id, message_id)` 定位输入。Receipt 只确认受理；用 `wait(InputRef)` 等结果。同一个 message_id 的重试必须使用相同内容。普通输入在运行边界合并为一条 user 消息，每个边界读取最新 inbox_version。紧急输入使用 `priority="urgent"`，按工具的中断策略处理。

accept、claim、final 保留原子事务边界。session_state 保存执行状态和所有权；final 将结果写入普通 messages 并将输入标为终态。继续子会话使用普通 send。取消 InputRef 针对输入，取消 TargetRef 可以沿 linked 子会话传播；independent 子会话独立保留。结果通过 `result_message_id` 生成稳定输入 ID，不再另建结果账本。

子 Agent 是直接子会话；后台工具是 background 会话；workflow 创建调用方的 workflow_driver 子会话，节点是 agent 会话。图状态存于 driver 的 session_runtime_state。正常模型循环、格式修复、工具执行、上下文、hooks 和事件保持使用原实现。214 计费事件由宿主处理，SDK 不写扣费表。

默认单机不需要 Redis。共享执行需明确开启 shared_execution 并传入共享适配器；内置 DatabasePollingTransport 可用于共享数据库。参与节点共用 store/profile，node_id 各不相同。shutdown 有界等待；中断输入可由后续 worker 在所有权过期后恢复。

旧 schema/protocol/descriptor 直接拒绝。使用新库或新表前缀，当前 schema 可验证并重开；没有运行时迁移、双读双写或旧结果回退。`schema="tasks"` 是已有建库选项的名称，实际创建 schema 17 的会话表。

关闭与删除从 sessions 读取 lifecycle，并与受理共用 session_state 锁。运行、排队、有效所有权、未终态输入、未闭合工具和资源 guard 都会阻止删除。linked 子会话参与级联；independent 子会话保留。每批清理至多 1,000 行；部分删除保持 closing 并拒绝新输入。

保留清理类别为 sessions、messages、inbox、runtime、projections。活跃输入和开放会话上下文受到保护。记忆、默认定时器和黑板使用 session_runtime_state，不新增 schema 表。

可运行示例：[单机](../../examples/design215_single_process.py)、[共享](../../examples/design215_shared.py)。共享示例是单机两个 worker、独立连接，不能代替 P8 真实三节点验收。完整公开导出由 `STABLE_API` 和 packaging 测试冻结；详见 [完整接口与存储说明](../runtime-12.md)。
