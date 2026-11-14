# 会话与输入引用

SDK 12.0.0 / schema 17 / protocol 43 / descriptor 2。

通过 Runtime 工厂装配模型、工具、hooks 和资源；调用 create_session 创建会话，再以稳定 message_id 调用 send。Receipt 表示已受理，wait(InputRef) 返回执行结果。子 Agent、后台工具和 workflow 都使用 session、InputRef 和普通结果输入；继续处理使用 send，取消使用 cancel。授权由宿主提供，工具参数不携带可信身份。

[Runtime 12](../../runtime-12.md)

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design215_single_process.py。

| 示例 | 行为 |
|---|---|
| [01_multi_turn_chat.py](../../../examples/01_multi_turn_chat.py) | 历史 |
| [11_cross_process_resume.py](../../../examples/11_cross_process_resume.py) | 重开 |
| [22_follow_up_steering.py](../../../examples/22_follow_up_steering.py) | 给运行中任务发消息 |
| [23_per_send_overrides.py](../../../examples/23_per_send_overrides.py) | 输入配置 |
| [12_retry_and_cancel.py](../../../examples/12_retry_and_cancel.py) | 重试与取消 |

```bash
.venv/bin/python examples/01_multi_turn_chat.py
```
