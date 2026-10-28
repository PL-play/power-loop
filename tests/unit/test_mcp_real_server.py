"""ECO-1, end-to-end: StdioMCPClient against a REAL MCP server.

Spawns a real FastMCP stdio server subprocess (`_mcp_test_server.py`), connects the
shipped StdioMCPClient to it, registers its tools into a ToolRegistry, and invokes them
over the real transport — exercising the whole path (the `mcp` SDK + our adapter), not a
fake source. Skipped if the `mcp` SDK isn't installed.
"""

from __future__ import annotations

import asyncio
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
        names = await register_mcp_tools(registry, client, prefix="mcp_")
        assert "mcp_add" in names and "mcp_greet" in names

        add_out = await registry.invoke_async("mcp_add", {"a": 2, "b": 3})
        assert "5" in add_out, add_out
        greet_out = await registry.invoke_async("mcp_greet", {"name": "Ada"})
        assert "Hello, Ada!" in greet_out, greet_out

        # the registered definition carries the server's inputSchema + required params
        add_def = registry.get("mcp_add")
        assert add_def is not None
        assert set(add_def.definition.required_params) == {"a", "b"}
    finally:
        await client.aclose()


async def test_register_then_direct_call_proxies_remote_name() -> None:
    """A prefixed local tool proxies to the unprefixed remote name on the real server
    (a dotted prefix is made portable: ``srv.`` → ``srv_``)."""
    client = await StdioMCPClient(sys.executable, [_SERVER]).connect()
    try:
        registry = ToolRegistry()
        await register_mcp_tools(registry, client, prefix="srv.")
        out = await registry.invoke_async("srv_greet", {"name": "Bob"})
        assert out == "Hello, Bob!"
    finally:
        await client.aclose()


@pytest.mark.parametrize("failure", ["exit", "cancel"])
async def test_failed_connect_closes_original_server_before_return(monkeypatch, failure):
    original = asyncio.create_subprocess_exec
    processes = []

    async def capture(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    program = "raise SystemExit(3)" if failure == "exit" else "import time; time.sleep(60)"
    client = StdioMCPClient(sys.executable, ["-c", program])
    try:
        if failure == "cancel":
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(.5):
                    await client.connect()
        else:
            from mcp.shared.exceptions import McpError
            with pytest.raises(McpError):
                await client.connect()
        assert len(processes) == 1 and processes[0].returncode is not None
        assert client._stack is None and client._session is None
        await client.aclose()
    finally:
        # Use retained original process handles if the regression reappears.
        for process in processes:
            if process.returncode is None:
                process.kill()
                await process.wait()
