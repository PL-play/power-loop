"""38 · MCP 工具 / MCP tools (ECO-1)

What you learn / 你将学到
--------------------------
- 用 `contrib.mcp` 把一个**真实 MCP server** 的工具接进 Agent:`StdioMCPClient` 连上 server,
  `register_mcp_tools` 把它的工具映射成 power-loop `ToolDefinition`,Agent 像调本地工具一样调它。
  / Wire a REAL MCP server's tools into the agent: StdioMCPClient connects, register_mcp_tools
  maps the server's tools to ToolDefinitions, and the agent calls them like local tools.
- 接入点是 `MCPToolSource` Protocol,所以 `mcp` SDK 是可选的(`power-loop[mcp]`);本例临时起一个
  极小的 MCPServer stdio server(`python -c …`)演示真实往返。
  / The seam is the MCPToolSource Protocol; this demo spins up a tiny real MCPServer stdio server.

Run / 运行 (needs `pip install 'power-loop[mcp]'`)
--------------------------------------------------------
    python examples/38_mcp_tools.py
"""

from __future__ import annotations

import asyncio
import sys
import time

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService, ToolRegistry
from power_loop.contrib.mcp import StdioMCPClient, register_mcp_tools
from power_loop.runtime.pricing import Free

# A tiny real MCP server, launched as a subprocess over stdio (python -c …).
_MCP_SERVER = """
from mcp.server.mcpserver import MCPServer
s = MCPServer("demo")

@s.tool()
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

s.run()
"""


class ArithmeticModel(LLMService):
    async def complete(self, request, **kwargs):
        results = [m['content'] for m in request.messages if m.get('role') == 'tool']
        if results:
            assert '42' in results[-1]
            return LLMResponse(raw_text='42')
        return LLMResponse(raw_text='', tool_calls=[{'id':'add','type':'function','function':{
            'name':'mcp_add','arguments':'{"a":21,"b":21}'}}])
    async def close(self):
        pass

async def main():
    client = await StdioMCPClient(sys.executable, ['-c', _MCP_SERVER]).connect()
    try:
        registry = ToolRegistry()
        names = await register_mcp_tools(registry, client, pricing=Free('local arithmetic MCP server'), prefix='mcp_')
        assert names == ['mcp_add']
        async with demo_runtime(ArithmeticModel(), registry) as (runtime, loop, target, actor):
            result = await loop.send(target, 'Compute 21 + 21', message_id='add', actor=actor,
                deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed' and result.final_text == '42'
            print(names, result.final_text)
            return result.final_text
    finally:
        await client.aclose()

if __name__ == '__main__':
    asyncio.run(main())
