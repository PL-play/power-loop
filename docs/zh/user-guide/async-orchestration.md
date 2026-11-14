# 异步编排与长任务

SDK 12.0.0 / schema 17 / protocol 43 / descriptor 2。

通过 Runtime 工厂装配模型、工具、hooks 和资源；调用 create_session 创建会话，再以稳定 message_id 调用 send。Receipt 表示已受理，wait(InputRef) 返回执行结果。子 Agent、后台工具和 workflow 都使用 session、InputRef 和普通结果输入；继续处理使用 send，取消使用 cancel。授权由宿主提供，工具参数不携带可信身份。

[Runtime 12](../../runtime-12.md)
