"""38 · MCP 工具 / MCP tools (ECO-1)

What you learn / 你将学到
--------------------------
- 用 `contrib.mcp` 把一个**真实 MCP server** 的工具接进 Agent:`StdioMCPClient` 连上 server,
  `register_mcp_tools` 把它的工具映射成 power-loop `ToolDefinition`,Agent 像调本地工具一样调它。
  / Wire a REAL MCP server's tools into the agent: StdioMCPClient connects, register_mcp_tools
  maps the server's tools to ToolDefinitions, and the agent calls them like local tools.
- 接入点是 `MCPToolSource` Protocol,所以 `mcp` SDK 是可选的(`power-loop[mcp]`);本例临时起一个
  极小的 FastMCP stdio server(`python -c …`)演示真实往返。
  / The seam is the MCPToolSource Protocol; this demo spins up a tiny real FastMCP stdio server.

Run / 运行 (needs `pip install 'power-loop[mcp,openai]'`)
--------------------------------------------------------
    python examples/38_mcp_tools.py
"""

from __future__ import annotations

import asyncio
import sys

from _helpers import make_llm

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.contrib.mcp import StdioMCPClient, register_mcp_tools

# A tiny real MCP server, launched as a subprocess over stdio (python -c …).
_MCP_SERVER = """
from mcp.server.fastmcp import FastMCP
s = FastMCP("demo")

@s.tool()
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

s.run()
"""


async def main() -> str:
    # Connect StdioMCPClient to the real server, then expose its tools to the agent.
    client = await StdioMCPClient(sys.executable, ["-c", _MCP_SERVER]).connect()
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=128),
            config=AgentLoopConfig(
                system_prompt="Use the available tools to compute; answer with the number.",
                max_rounds=4,
            ),
        )
        names = await register_mcp_tools(loop.tool_registry or _new_registry(loop), client, prefix="mcp.")
        sid = loop.new_session()
        result = await loop.send("What is 21 + 21? Use the add tool.", session_id=sid, tools=names)
        summary = f"registered {names} | answer={result.final_text!r}"
        print(summary)
        return summary
    finally:
        await client.aclose()


def _new_registry(loop: StatefulAgentLoop):
    from power_loop import ToolRegistry

    loop.tool_registry = ToolRegistry()
    return loop.tool_registry


if __name__ == "__main__":
    asyncio.run(main())
