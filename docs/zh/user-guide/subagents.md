# 子 Agent 与任务归属

SDK 11.0.0 · 任务结构 15 · 协议 42 · 描述符结构 1。

[Runtime 契约](../../runtime-11.md) · [API](../api/index.md)

在工厂的 ToolRegistry 注册 spawn_agent。模型可见的 create 返回子任务受理引用，不等待模型执行。子任务拥有独立会话和 Runtime 任务，完成后投递父会话收件箱。等待不占唯一执行槽，容量为 1 时仍可运行。

AgentSpec 指定名称、system_prompt、工具白名单、模型参数、轮数限制和可选 output_schema，在受理前校验。工具白名单和深度限制在子节点生效；切换模型需要宿主声明 allowed_models。结构化输出必须解析校验，不能只因看起来像 JSON 就认定成功。

使用原 JobRef 查询、等待、引导、取消或继续。父任务中断不会暗中取消健康子任务，结果仍可查询。工作流节点显式继续沿用原归属和累计额度，不刷新为新 root。已经结束节点保存的晚到子结果，要到节点实际处理后才标记已消费。

工厂通过稳定引用重建工具、workspace 和 hooks，不能把捕获父 loop 的闭包或旧会话句柄当作部署资源。

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design203_single_process.py。

| 示例 | 行为 |
|---|---|
| [03_subagent_delegation.py](../../../examples/03_subagent_delegation.py) | 模型委派 |
| [06_declarative_subagent.py](../../../examples/06_declarative_subagent.py) | AgentSpec |
| [14_structured_card.py](../../../examples/14_structured_card.py) | 结构化结果 |
| [27_dynamic_workflow.py](../../../examples/27_dynamic_workflow.py) | 工作流归属 |

```bash
.venv/bin/python examples/03_subagent_delegation.py
```
