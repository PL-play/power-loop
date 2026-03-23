from __future__ import annotations

from power_loop.tools.registry import ToolRegistry, build_registry


def create_default_tool_registry() -> ToolRegistry:
    from power_loop.tools.default_manifest import DEFAULT_TOOL_DEFINITIONS
    from power_loop.tools.default_tools import DEFAULT_TOOL_HANDLERS

    return build_registry(DEFAULT_TOOL_DEFINITIONS, DEFAULT_TOOL_HANDLERS)


__all__ = [
    "ToolRegistry",
    "build_registry",
    "create_default_tool_registry",
]
