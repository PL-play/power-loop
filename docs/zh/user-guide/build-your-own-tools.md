# 扩展工具

SDK 11.0.0 · 任务结构 15 · 协议 42 · 描述符结构 1。

[Runtime 契约](../../runtime-11.md) · [API](../api/index.md)

用 JSON 输入结构定义 ToolDefinition，把同步或异步处理器注册到 ToolRegistry，再由执行工厂返回该注册表。RequestOptions.tools 是工具白名单，必需工具缺失时明确失败。处理器接收已校验参数，错误输出应可理解，不能虚构成功结果。

普通无状态工具可以执行本地计算。拥有外部任务、费用或受管资源的工具需要明确 ToolRuntimeContract 和对应宿主后端。不要捕获旧 loop 或调用其私有方法。执行身份来自可信 runtime 上下文，不从模型参数取得 tenant 或 session。

MCP 需安装 mcp extra，配置实际传输、注册发现的工具定义，并随执行资源关闭连接。编号示例启动本地 stdio 服务并实际调用。依赖不可用时明确报错，不静默回退。

业务权限和额度检查仍在权威 API。未知外部动作只按原身份查询或返回 unknown，不能因本地请求失败自动重放。

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design203_single_process.py。

| 示例 | 行为 |
|---|---|
| [02_tool_calling.py](../../../examples/02_tool_calling.py) | 定义与处理器 |
| [37_custom_retrieval_tool.py](../../../examples/37_custom_retrieval_tool.py) | 检索 |
| [38_mcp_tools.py](../../../examples/38_mcp_tools.py) | 本地 MCP |
| [41_custom_async_tool.py](../../../examples/41_custom_async_tool.py) | 异步处理器 |
| [42_build_your_own_tools.py](../../../examples/42_build_your_own_tools.py) | 受管工具契约 |

```bash
.venv/bin/python examples/02_tool_calling.py
```
