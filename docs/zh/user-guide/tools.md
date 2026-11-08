# 工具与内置能力

SDK 11.0.0 · 任务结构 15 · 协议 42 · 描述符结构 1。

[Runtime 契约](../../runtime-11.md) · [API](../api/index.md)

工厂提供 ToolRegistry 和明确的 RuntimeEnv。create_default_tool_registry 装配选择的内置工具，但不会自动授权所有资源。RequestOptions.tools 限制本次调用工具，required_profiles 声明部署必需能力。校验错误和能力缺失明确可见。

工作区工具需要指定 workspace、skills 和 shell 后端。background_run 使用当前 Runtime 任务接口及原进程控制器；workflow 使用明确的 workflow_file_io。StoreBlackboard 在数据库内原子更新；提供业务 API 后端时仍以 API 为权威。schedule_wakeup 是唯一 SDK 定时工具，由 TimerBackend 和就绪 driver 支持。Runtime 启动默认 store driver，纯 Client 不启动；显式宿主定时后端必须带 driver，不回退本地表。

注册表支持同步函数、异步函数和可调用对象。直接调用注册表适用于独立普通工具；Runtime 绑定工具必须处于执行上下文。工具结果保留文本、图片和错误格式，正常前后 hooks、实际工具计价及白名单继续生效。

## 可执行示例

从 SDK 源码目录使用已安装依赖的环境运行。完整程序包含资源清理；编号示例共用 examples/_runtime.py。依赖宿主的示例明确列出所需服务。不依赖该辅助文件的完整装配见 design203_single_process.py。

| 示例 | 行为 |
|---|---|
| [20_default_tools.py](../../../examples/20_default_tools.py) | 默认工具 |
| [26_timers.py](../../../examples/26_timers.py) | 定时器 |
| [27_dynamic_workflow.py](../../../examples/27_dynamic_workflow.py) | 工作流 |
| [29_shared_blackboard.py](../../../examples/29_shared_blackboard.py) | 黑板 |
| [38_mcp_tools.py](../../../examples/38_mcp_tools.py) | MCP |
| [42_build_your_own_tools.py](../../../examples/42_build_your_own_tools.py) | 工具契约 |

```bash
.venv/bin/python examples/20_default_tools.py
```

workflow 的 logs 操作读取原节点历史。source="messages" 读取原消息，source="projection" 读取模型视图；可按 message_id/message_ids、seq 或 send_index 定位，并用 cursor、limit、char_offset、max_chars 限制结果。output_file 通过工厂明确提供的 workflow_file_io 在授权工作区导出 JSONL，不重新执行节点。
