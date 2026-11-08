# 会话与输入引用

SDK 11.0.0 · 任务结构 15 · 协议 42 · 描述符结构 1。

[Runtime 契约](../../runtime-11.md) · [API](../api/index.md)

Runtime.open_session(descriptor, actor=actor) 返回包含 store 身份的 TargetRef。可信 actor 来自宿主认证。向该目标提交新一轮输入，保留 Receipt.input_ref，再用它等待或查询。JobRef 标识某一代执行，不能当作会话 ID。

追加输入使用 RuntimeClient.append：忙时在检查点接收，空闲主会话可以唤醒。RequestOptions 在受理时固定。steer、nudge、cancel 指向原任务；answer 对应原交互；resume 继续原执行。已经结束的工作流节点保存子任务晚到结果，等待显式继续；投递不代表模型已经处理。

重启时使用保存的目标和引用，并重新装配同版本工厂。未知外部结果只查询原结果或明确返回 unknown，不自动重放。历史可通过 load_all_messages 和公开查询接口读取。关闭 facade 只关闭该入口；Runtime.aclose 排空 worker 并关闭其拥有的资源。删除会话和保留期清理是明确的存储操作，不等于关闭客户端。

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design203_single_process.py。

| 示例 | 行为 |
|---|---|
| [01_multi_turn_chat.py](../../../examples/01_multi_turn_chat.py) | 历史 |
| [11_cross_process_resume.py](../../../examples/11_cross_process_resume.py) | 重开 |
| [21_request_user_input.py](../../../examples/21_request_user_input.py) | 用户输入 |
| [22_follow_up_steering.py](../../../examples/22_follow_up_steering.py) | 追加与引导 |
| [23_per_send_overrides.py](../../../examples/23_per_send_overrides.py) | 输入配置 |
| [12_retry_and_cancel.py](../../../examples/12_retry_and_cancel.py) | 重试与取消 |

```bash
.venv/bin/python examples/01_multi_turn_chat.py
```
