# 快速开始

SDK 11.0.0 · 任务结构 15 · 协议 42 · 描述符结构 1。

[Runtime 契约](../../runtime-11.md) · [API](../api/index.md)

先打开 Store、注册工厂，再启动 Runtime.worker 并提交输入。工厂在执行节点取得模型、工具、hooks 和资源。描述符只保存版本化引用和可序列化配置，不保存活的 Python 对象或密钥。

下面的完整示例使用固定模型和 SQLite，不需要模型密钥、Redis 或消息中间件。实际调用模型时替换 build_demo 中的 FixedModel。Runtime.aclose 负责关闭。多轮对话复用 TargetRef，每条新输入使用新的 message_id；重试原输入使用原 ID。

RuntimeClient.send 受理后返回 Receipt；RuntimeClient.wait 返回持久化的 Result。StatefulAgentLoop(runtime=runtime) 是提交并等待的便捷入口。取消等待者不会取消原任务。

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design203_single_process.py。

| 示例 | 行为 |
|---|---|
| [design203_single_process.py](../../../examples/design203_single_process.py) | 完整本地装配 |
| [01_multi_turn_chat.py](../../../examples/01_multi_turn_chat.py) | 多轮历史 |
| [02_tool_calling.py](../../../examples/02_tool_calling.py) | 工具调用 |
| [08_streaming.py](../../../examples/08_streaming.py) | 流式输出 |
| [19_full_chatbot.py](../../../examples/19_full_chatbot.py) | 组合聊天示例 |

```bash
.venv/bin/python examples/design203_single_process.py
```
