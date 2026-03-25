from __future__ import annotations

from typing import Sequence

from power_loop.tools.registry import ToolRegistry, build_registry


def create_default_tool_registry(
    *,
    preset: str | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> ToolRegistry:
    """Create a :class:`ToolRegistry` pre-loaded with default tools.

    All three filter arguments are optional and forwarded to
    :func:`~power_loop.tools.default_manifest.get_tool_definitions`.

    Args:
        preset: ``"core"`` (bash/read/write/edit/patch/glob/grep/skill),
                ``"explore"`` (read-only subset), or ``"full"`` (all 11 tools).
                Defaults to ``"full"`` when *include* is also ``None``.
        include: Explicit tool names to register (overrides *preset*).
        exclude: Tool names to drop from the selected set.

    Examples::

        # All default tools
        reg = create_default_tool_registry()

        # Only core coding tools
        reg = create_default_tool_registry(preset="core")

        # Cherry-pick
        reg = create_default_tool_registry(include=["bash", "read_file", "grep"])

        # Full minus background tasks
        reg = create_default_tool_registry(exclude=["background_run", "check_background"])
    """
    from power_loop.tools.default_manifest import get_tool_definitions
    from power_loop.tools.default_tools import DEFAULT_TOOL_HANDLERS

    definitions = get_tool_definitions(preset=preset, include=include, exclude=exclude)
    return build_registry(definitions, DEFAULT_TOOL_HANDLERS)


__all__ = [
    "ToolRegistry",
    "build_registry",
    "create_default_tool_registry",
]
