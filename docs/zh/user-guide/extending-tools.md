# 用自己的工具扩展 power-loop

[English](../../en/user-guide/extending-tools.md) | [用户指南](../index.md)

power-loop 是**内核**,不是开箱即用平台:它不带连接器、不带向量库、不带 SaaS 集成。这是刻意的——那些会带来依赖、固化的取舍和持续维护,不该塞进约 5k 行的内核。你通过 `ToolRegistry` 自带工具,要接外部系统就写一个薄工具或走 [MCP](#外部系统mcp)。本页就是配方。

## 配方:定义 + handler + 注册

一个工具是两样东西按同名绑定——`ToolDefinition`(模型看到的 JSON Schema)和 handler(你的代码):

```python
from power_loop import ToolDefinition, ToolRegistry

def search_docs(query: str, top_k: int = 1) -> str:
    ...                      # 你的领域逻辑——DB、HTTP、向量库,随你
    return "results…"

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="search_docs",
        description="检索知识库,返回最相关的笔记。",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
        required_params=("query",),    # handler 运行前做缺参校验
    ),
    search_docs,
)

loop = StatefulAgentLoop(llm=llm, tool_registry=registry)
```

可运行示例:[`examples/37_custom_retrieval_tool.py`](../../../examples/37_custom_retrieval_tool.py)。

## 同步 vs 异步 handler

- **同步** handler 由 `invoke_async` 放到工作线程(`asyncio.to_thread`)跑,所以阻塞调用(DB、子进程、HTTP)不会卡事件循环;contextvars(运行时 env、会话身份)会传入线程。
- **`async def`** handler 在循环上跑——用于本就异步的 I/O(`httpx`、异步 DB 驱动)。

模型只看到 JSON Schema;返回值(字符串)成为 tool 消息。校验(`required_params`)在 handler 之前跑,所以你可以假定声明的入参都在。

## 按调用白名单

不必新建注册表就能限制某次运行可用的工具:

```python
await loop.send("…", session_id=sid, tools=["search_docs"])   # 模型只看到这些
```

`tools=` 接受名字列表(经 `registry.subset` 从 loop 注册表白名单)或直接一个 `ToolRegistry`。

## 外部系统:MCP

对住在另一个进程/服务里的工具,连接器路径是 **[Model Context Protocol](https://modelcontextprotocol.io)**,而不是捆绑集成。`power_loop.contrib.mcp` 把 MCP server 的工具映射成 power-loop 工具:

```python
from power_loop.contrib.mcp import StdioMCPClient, register_mcp_tools   # power-loop[mcp]

client = await StdioMCPClient("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"]).connect()
await register_mcp_tools(registry, client, prefix="fs.")   # MCP 工具 → ToolDefinition
...
await client.aclose()
```

接入点是小小的 `MCPToolSource` Protocol(`list_tools` / `call_tool`),所以 `mcp` SDK 是可选的——在任意客户端(HTTP/SSE、测试里的假实现)上实现该 Protocol 即可。`register_mcp_tools` 把每个工具的 `inputSchema` 直接映射成 `ToolDefinition`。

## 为什么不捆绑连接器?

你没写的连接器,是你没选的依赖、改不了的 schema、修不了的故障。内核给你**接缝**(`ToolRegistry`)和一条标准远程路径(MCP);上面 30 秒的配方通常比接一个通用连接器代码还少——而且是你自己的。
