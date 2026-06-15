"""ECO-1: the MCP tool adapter maps an MCPToolSource into the ToolRegistry.

Tested against a fake MCPToolSource (no `mcp` SDK dependency) — that pins the mapping
(inputSchema → ToolDefinition, required-param validation, call proxying, name prefixing),
which is the part that regresses. StdioMCPClient just implements this Protocol over the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop.contracts.errors import ToolValidationError
from power_loop.contrib.mcp import MCPToolSource, McpToolSpec, register_mcp_tools
from power_loop.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


@dataclass
class _FakeSource:
    specs: list[McpToolSpec]
    calls: list[tuple[str, dict]] = field(default_factory=list)

    async def list_tools(self) -> list[McpToolSpec]:
        return self.specs

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return f"{name}({arguments})"


def _specs() -> list[McpToolSpec]:
    return [
        McpToolSpec(
            name="search",
            description="search the web",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        ),
        McpToolSpec(name="ping"),  # no schema → defaults
    ]


def test_fake_source_satisfies_protocol() -> None:
    assert isinstance(_FakeSource(_specs()), MCPToolSource)


async def test_register_maps_schema_and_required() -> None:
    reg = ToolRegistry()
    src = _FakeSource(_specs())
    names = await register_mcp_tools(reg, src)
    assert names == ["search", "ping"]
    search = reg.get("search")
    assert search is not None
    assert search.definition.required_params == ("q",)
    assert search.definition.input_schema["properties"]["q"]["type"] == "string"
    assert search.is_async  # the proxy handler is async


async def test_invocation_proxies_to_source() -> None:
    reg = ToolRegistry()
    src = _FakeSource(_specs())
    await register_mcp_tools(reg, src)
    out = await reg.invoke_async("search", {"q": "tea"})
    assert out == "search({'q': 'tea'})"
    assert src.calls == [("search", {"q": "tea"})]


async def test_missing_required_param_is_rejected() -> None:
    reg = ToolRegistry()
    await register_mcp_tools(reg, _FakeSource(_specs()))
    with pytest.raises(ToolValidationError):
        await reg.invoke_async("search", {})  # missing required "q"


async def test_prefix_avoids_name_collisions() -> None:
    reg = ToolRegistry()
    reg2_src = _FakeSource(_specs())
    names = await register_mcp_tools(reg, reg2_src, prefix="mcp.")
    assert names == ["mcp.search", "mcp.ping"]
    assert reg.has("mcp.search") and not reg.has("search")
    # the proxy still calls the REMOTE (unprefixed) name
    await reg.invoke_async("mcp.ping", {})
    assert reg2_src.calls[-1][0] == "ping"


def test_stdio_client_importable_without_mcp_sdk() -> None:
    """The module + class import without the mcp SDK; only connect() needs the extra."""
    from power_loop.contrib.mcp import StdioMCPClient

    client = StdioMCPClient("echo", ["hi"])  # construction must not import mcp
    assert client._command == "echo"
