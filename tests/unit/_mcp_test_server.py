"""A minimal real MCP server (FastMCP, stdio) used by test_mcp_real_server.py.

Launched as a subprocess (`python _mcp_test_server.py`) so the test exercises
StdioMCPClient end-to-end over a real stdio transport. The leading underscore keeps
pytest from collecting it as a test module.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("power-loop-test-server")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@server.tool()
def greet(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    server.run()  # stdio transport by default
