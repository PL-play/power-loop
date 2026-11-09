# 会话与输入引用

SDK 11.0.0 · 任务结构 15 · 协议 42 · 描述符结构 1。

[Runtime 契约](../../runtime-11.md) · [API](../api/index.md)

Runtime.open_session(descriptor, actor=actor) 返回包含 store 身份的 TargetRef。可信 actor 来自宿主认证。向该目标提交新一轮输入，保留 Receipt.input_ref，再用它等待或查询。JobRef 标识某一代执行，不能当作会话 ID。

输入只有 RuntimeClient.send：会话空闲时启动任务；任务运行中时，消息在下一轮并入该任务，连续发送不会拆成多个任务。`urgent=True` 另外跳过当前批次中尚未开始的工具调用；正在执行的工具和模型调用不会被打断。RequestOptions 在消息启动任务时固定。用 JobRef 给运行中的任务发消息；cancel(ref) 停止任务（未投递的输入作废，已投递的输入停止它并入的任务）；resume 继续原执行。RuntimeClient.messages(job_ref) 列出任务最近的消息（默认 20 条，每条截断到 500 字符）。已经结束的工作流节点保存子任务晚到结果，等待显式继续；投递不代表模型已经处理。

重启时使用保存的目标和引用，并重新装配同版本工厂。未知外部结果只查询原结果或明确返回 unknown，不自动重放。历史可通过 load_all_messages 和公开查询接口读取。关闭 facade 只关闭该入口；Runtime.aclose 排空 worker 并关闭其拥有的资源。删除会话和保留期清理是明确的存储操作，不等于关闭客户端。

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design203_single_process.py。

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
