"""Failed/cancelled connection owns and unwinds every acquired MCP context."""
import asyncio
import sys
from contextlib import asynccontextmanager
from types import ModuleType

import pytest

from power_loop.contrib.mcp import StdioMCPClient

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("stage", ["transport", "session", "initialize", "cancel", "success"])
async def test_connection_unwinds_in_original_task_and_can_reconnect(monkeypatch, stage):
    events = []
    owner = asyncio.current_task()

    @asynccontextmanager
    async def transport(params):
        assert asyncio.current_task() is owner
        events.append("transport_enter")
        if stage == "transport":
            raise OSError("transport unavailable")
        try:
            yield object(), object()
        finally:
            assert asyncio.current_task() is owner
            events.append("transport_exit")

    class Session:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            events.append("session_enter")
            if stage == "session":
                raise OSError("session unavailable")
            return self

        async def __aexit__(self, *args):
            assert asyncio.current_task() is owner
            events.append("session_exit")

        async def initialize(self):
            if stage == "initialize":
                raise OSError("initialization unavailable")
            if stage == "cancel":
                raise asyncio.CancelledError()

    sdk, stdio = ModuleType("mcp"), ModuleType("mcp.client.stdio")
    sdk.ClientSession = Session
    sdk.StdioServerParameters = lambda **kwargs: kwargs
    stdio.stdio_client = transport
    monkeypatch.setitem(sys.modules, "mcp", sdk)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio)
    client = StdioMCPClient("example")
    if stage == "success":
        assert await client.connect() is client
        with pytest.raises(RuntimeError, match="already connected"):
            await client.connect()
        assert events == ["transport_enter", "session_enter"]
        await client.aclose()
    else:
        with pytest.raises(asyncio.CancelledError if stage == "cancel" else OSError):
            await client.connect()
    assert client._stack is None and client._session is None
    assert events == {
        "transport": ["transport_enter"],
        "session": ["transport_enter", "session_enter", "transport_exit"],
    }.get(stage, ["transport_enter", "session_enter", "session_exit", "transport_exit"])
    stage = "success"
    assert await client.connect() is client
    await client.aclose()
    previous = events.copy()
    await client.aclose()
    assert events == previous
