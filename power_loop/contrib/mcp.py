"""MCP (Model Context Protocol) tool adapter (ECO-1).

Surfaces tools from an MCP server as power-loop :class:`ToolDefinition`s, so an agent can
call them through the normal :class:`ToolRegistry` — the same convention as the built-in
``register_*_tools`` helpers.

The integration point is the small :class:`MCPToolSource` Protocol (``list_tools`` +
``call_tool``), so the official ``mcp`` SDK is **not** a hard dependency: implement the
Protocol over any client. :func:`register_mcp_tools` maps each tool's JSON ``inputSchema``
straight to a ``ToolDefinition`` and registers an async handler that proxies the call.

A default :class:`StdioMCPClient` (lazy-imports the ``mcp`` SDK — ``power-loop[mcp]``)
talks to a stdio MCP server::

    from power_loop.contrib.mcp import StdioMCPClient, register_mcp_tools

    client = await StdioMCPClient("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"]).connect()
    names = await register_mcp_tools(registry, client, prefix="fs.")
    ...
    await client.aclose()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry

__all__ = ["McpToolSpec", "MCPToolSource", "register_mcp_tools", "StdioMCPClient"]

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass
class McpToolSpec:
    """One MCP tool: its name, description, and JSON-Schema input shape."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: dict(_EMPTY_SCHEMA))


@runtime_checkable
class MCPToolSource(Protocol):
    """Anything that can enumerate + call MCP tools. Implement this over any client
    (the shipped :class:`StdioMCPClient`, an HTTP/SSE client, or a fake in tests)."""

    async def list_tools(self) -> list[McpToolSpec]: ...
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


async def register_mcp_tools(
    registry: ToolRegistry,
    source: MCPToolSource,
    *,
    prefix: str = "",
    overwrite: bool = False,
) -> list[str]:
    """Register every tool from ``source`` into ``registry`` (names optionally
    ``prefix``-ed to avoid collisions). Each tool's ``inputSchema`` becomes the
    ``ToolDefinition`` schema (its ``required`` list drives missing-param validation),
    and the handler proxies the call to ``source.call_tool``. Returns the registered
    tool names. Async because enumerating an MCP server is a network round-trip."""
    registered: list[str] = []
    for spec in await source.list_tools():
        schema = spec.input_schema or dict(_EMPTY_SCHEMA)
        required = tuple(schema.get("required", []) or ())
        local_name = f"{prefix}{spec.name}"

        def _make_handler(remote_name: str):
            async def _handler(**kwargs: Any) -> str:
                return await source.call_tool(remote_name, dict(kwargs))

            return _handler

        registry.register(
            ToolDefinition(
                name=local_name,
                description=spec.description or f"MCP tool {spec.name}",
                input_schema=schema,
                required_params=required,
            ),
            _make_handler(spec.name),
            overwrite=overwrite,
        )
        registered.append(local_name)
    return registered


class StdioMCPClient:
    """Default :class:`MCPToolSource` over a stdio MCP server, via the ``mcp`` SDK
    (``power-loop[mcp]``). Call :meth:`connect` before use and :meth:`aclose` after.

    The ``mcp`` SDK is imported lazily, so this module is importable without the extra;
    only constructing/connecting requires it."""

    def __init__(self, command: str, args: list[str] | None = None, *, env: dict[str, str] | None = None) -> None:
        self._command = command
        self._args = list(args or [])
        self._env = env
        self._session: Any = None
        self._stack: Any = None

    async def connect(self) -> StdioMCPClient:
        try:
            from contextlib import AsyncExitStack

            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "StdioMCPClient needs the MCP SDK: pip install 'power-loop[mcp]'"
            ) from exc
        self._stack = AsyncExitStack()
        params = StdioServerParameters(command=self._command, args=self._args, env=self._env)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def list_tools(self) -> list[McpToolSpec]:
        resp = await self._session.list_tools()
        return [
            McpToolSpec(
                name=t.name,
                description=getattr(t, "description", "") or "",
                input_schema=dict(getattr(t, "inputSchema", None) or _EMPTY_SCHEMA),
            )
            for t in resp.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._session.call_tool(name, arguments)
        return _content_to_text(getattr(result, "content", None))

    async def aclose(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None


def _content_to_text(content: Any) -> str:
    """Flatten an MCP tool result's content blocks to text (best-effort)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        parts.append(text if isinstance(text, str) else str(block))
    return "\n".join(parts)
