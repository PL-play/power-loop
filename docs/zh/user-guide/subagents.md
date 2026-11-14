# 子 Agent 与任务归属

SDK 12.0.0 / schema 17 / protocol 43 / descriptor 2。

通过 Runtime 工厂装配模型、工具、hooks 和资源；调用 create_session 创建会话，再以稳定 message_id 调用 send。Receipt 表示已受理，wait(InputRef) 返回执行结果。子 Agent、后台工具和 workflow 都使用 session、InputRef 和普通结果输入；继续处理使用 send，取消使用 cancel。授权由宿主提供，工具参数不携带可信身份。

[Runtime 12](../../runtime-12.md)

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design215_single_process.py。

| 示例 | 行为 |
|---|---|
| [03_subagent_delegation.py](../../../examples/03_subagent_delegation.py) | 模型委派 |
| [06_declarative_subagent.py](../../../examples/06_declarative_subagent.py) | AgentSpec |
| [14_structured_card.py](../../../examples/14_structured_card.py) | 结构化结果 |
| [27_dynamic_workflow.py](../../../examples/27_dynamic_workflow.py) | 工作流归属 |

```bash
.venv/bin/python examples/03_subagent_delegation.py
```
