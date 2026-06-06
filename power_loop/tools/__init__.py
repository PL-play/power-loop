from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from pathlib import Path
from typing import Any

from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools.registry import ToolRegistry, build_registry

_WORKSPACE_TOOL_NAMES = {
    "bash",
    "read_file",
    "write_file",
    "edit_file",
    "apply_patch",
    "glob",
    "grep",
    "background_run",
}


def create_default_tool_registry(
    *,
    preset: str | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    workspace_dir: str | Path | None = None,
    home_dir: str | Path | None = None,
    skills_dir: str | Path | None = None,
) -> ToolRegistry:
    """Create a :class:`ToolRegistry` pre-loaded with default tools.

    All three filter arguments are optional and forwarded to
    :func:`~power_loop.tools.default_manifest.get_tool_definitions`.

    Args:
        preset: ``"core"`` (bash/read/write/edit/patch/glob/grep/skill/input),
                ``"explore"`` (read-only subset), or ``"full"`` (all 12 tools).
                Defaults to ``"full"`` when *include* is also ``None``.
        include: Explicit tool names to register (overrides *preset*).
        exclude: Tool names to drop from the selected set.
        workspace_dir: Required for filesystem/search/shell/background tools.
            If omitted, ``POWER_LOOP_WORKSPACE`` is used. No implicit
            current-working-directory fallback is applied.
        home_dir: Optional runtime home for allowlisted ``@home`` paths.
            If omitted, ``POWER_LOOP_HOME`` is used when present.
        skills_dir: Optional default skills directory for ``load_skill``.
            If omitted, ``POWER_LOOP_SKILLS_DIR`` is used when present.

    Examples::

        # All default tools (requires an explicit workspace)
        reg = create_default_tool_registry(workspace_dir="/path/to/project")

        # Only core coding tools
        reg = create_default_tool_registry(preset="core", workspace_dir="/path/to/project")

        # Cherry-pick
        reg = create_default_tool_registry(include=["bash", "read_file", "grep"], workspace_dir="/path/to/project")

        # Full minus background tasks
        reg = create_default_tool_registry(
            exclude=["background_run", "check_background"],
            workspace_dir="/path/to/project",
        )
    """
    from power_loop.tools.default_manifest import get_tool_definitions
    from power_loop.tools.default_tools import DEFAULT_TOOL_HANDLERS

    definitions = get_tool_definitions(preset=preset, include=include, exclude=exclude)
    names = {definition.name for definition in definitions}
    runtime_env = RuntimeEnv.from_env(
        workspace_dir=workspace_dir,
        home_dir=home_dir,
        skills_dir=skills_dir,
    )
    if names & _WORKSPACE_TOOL_NAMES:
        runtime_env.require_workspace_dir()
    return build_registry(definitions, _bind_handlers(DEFAULT_TOOL_HANDLERS, runtime_env))


def _bind_handlers(handlers: dict[str, Any], runtime_env: RuntimeEnv) -> dict[str, Any]:
    bound: dict[str, Any] = {}
    for name, handler in handlers.items():
        bound[name] = _bind_handler(handler, runtime_env)
    return bound


def _bind_handler(handler: Any, runtime_env: RuntimeEnv) -> Any:
    @wraps(handler)
    def _wrapped(**kwargs: Any) -> Any:
        with runtime_env_context(runtime_env):
            return handler(**kwargs)

    return _wrapped


__all__ = [
    "ToolRegistry",
    "build_registry",
    "create_default_tool_registry",
]
