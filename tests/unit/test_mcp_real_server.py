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

from power_loop.runtime.pricing import Free

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
        names = await register_mcp_tools(registry, client, prefix="mcp_", pricing=Free("test_fixture"))
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
        await register_mcp_tools(registry, client, prefix="srv.", pricing=Free("test_fixture"))
        out = await registry.invoke_async("srv_greet", {"name": "Bob"})
        assert out == "Hello, Bob!"
    finally:
        await client.aclose()


@pytest.mark.parametrize("failure", ["exit", "cancel"])
async def test_failed_connect_closes_original_server_before_return(monkeypatch, failure, record_property):
    import mcp.client.stdio as stdio

    original = stdio._create_platform_compatible_process
    processes = []
    connecting_task = asyncio.current_task()
    cancellation = None

    async def capture(*args, **kwargs):
        nonlocal cancellation
        process = await original(*args, **kwargs)
        processes.append(process)
        record_property("owned_process_pid", process.pid)
        if failure == "cancel":
            cancellation = asyncio.get_running_loop().call_later(.1, connecting_task.cancel)
        return process

    monkeypatch.setattr(stdio, "_create_platform_compatible_process", capture)
    program = "raise SystemExit(3)" if failure == "exit" else "import time; time.sleep(60)"
    client = StdioMCPClient(sys.executable, ["-c", program])
    try:
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await client.connect()
        else:
            import mcp.shared.exceptions as errors
            error_class = getattr(errors, "MCPError", getattr(errors, "McpError", None))
            assert error_class is not None
            with pytest.raises(error_class):
                await client.connect()
        assert len(processes) == 1 and processes[0].returncode is not None
        assert client._stack is None and client._session is None
        await client.aclose()
    finally:
        if cancellation is not None:
            cancellation.cancel()
            connecting_task.uncancel()
        # Use retained original process handles if the regression reappears.
        for process in processes:
            if process.returncode is None:
                process.kill()
                await process.wait()
