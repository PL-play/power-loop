"""ECO-1, end-to-end: StdioMCPClient against a REAL MCP server.

Spawns a real FastMCP stdio server subprocess (`_mcp_test_server.py`), connects the
shipped StdioMCPClient to it, registers its tools into a ToolRegistry, and invokes them
over the real transport — exercising the whole path (the `mcp` SDK + our adapter), not a
fake source. Skipped if the `mcp` SDK isn't installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from power_loop.contrib.mcp import (  # noqa: E402
    MCPToolSource,
    StdioMCPClient,
    register_mcp_tools,
)
from power_loop.tools.registry import ToolRegistry  # noqa: E402

pytestmark = pytest.mark.unit

_SERVER = str(Path(__file__).parent / "_mcp_test_server.py")


async def test_stdio_client_roundtrip_against_real_server() -> None:
    client = await StdioMCPClient(sys.executable, [_SERVER]).connect()
    try:
        assert isinstance(client, MCPToolSource)

        # list_tools over the real server returns the server's tools + their schemas
        specs = {s.name: s for s in await client.list_tools()}
        assert {"add", "greet"} <= set(specs)
        assert "a" in specs["add"].input_schema.get("properties", {})

        # register into a ToolRegistry and invoke through it — a real stdio round-trip
        registry = ToolRegistry()
        names = await register_mcp_tools(registry, client, prefix="mcp.")
        assert "mcp.add" in names and "mcp.greet" in names

        add_out = await registry.invoke_async("mcp.add", {"a": 2, "b": 3})
        assert "5" in add_out, add_out
        greet_out = await registry.invoke_async("mcp.greet", {"name": "Ada"})
        assert "Hello, Ada!" in greet_out, greet_out

        # the registered definition carries the server's inputSchema + required params
        add_def = registry.get("mcp.add")
        assert add_def is not None
        assert set(add_def.definition.required_params) == {"a", "b"}
    finally:
        await client.aclose()


async def test_register_then_direct_call_proxies_remote_name() -> None:
    """A prefixed local tool proxies to the unprefixed remote name on the real server."""
    client = await StdioMCPClient(sys.executable, [_SERVER]).connect()
    try:
        registry = ToolRegistry()
        await register_mcp_tools(registry, client, prefix="srv.")
        out = await registry.invoke_async("srv.greet", {"name": "Bob"})
        assert out == "Hello, Bob!"
    finally:
        await client.aclose()
