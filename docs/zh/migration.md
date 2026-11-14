# SDK 12 切换边界

SDK 12 使用 schema 17、protocol 43、descriptor 2。使用新的 store ID 初始化空库；旧数据库、协议和描述符直接拒绝，不自动迁移，不保留兼容别名或双写。

注册 Runtime factory，使用冻结描述符和可信 actor 创建 session，再通过 RuntimeClient 提交输入。身份使用 TargetRef、InputRef、send_index 和消息 ID；执行所有权由 session worker 管理。

参见 [Runtime 12](runtime-12.md)、[入门](getting-started.md) 和可直接运行的 `examples/design215_single_process.py`、`examples/design215_shared.py`。
