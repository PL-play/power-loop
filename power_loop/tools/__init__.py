from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any

from power_loop.runtime.env import RuntimeEnv, get_runtime_env, runtime_env_context
from power_loop.tools.default_manifest import get_tool_definitions
from power_loop.tools.default_tools import DEFAULT_TOOL_HANDLERS
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
    shell_backend: Any | None = None,
    blackboard: Any | None = None,
    blackboard_id: str | None = None,
    bind: bool = True,
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
        bind: When true (default), bind handlers to one ``RuntimeEnv`` now.
            When false, return handlers that resolve the current
            ``runtime_env_context`` at invocation time.

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
    definitions = get_tool_definitions(preset=preset, include=include, exclude=exclude)
    if not bind:
        # Unbound: handlers read the current RuntimeEnv at call time; the caller
        # supplies it per call via runtime_env_context(...). No eager workspace.
        return build_registry(definitions, DEFAULT_TOOL_HANDLERS)
    names = {definition.name for definition in definitions}
    runtime_env = RuntimeEnv.from_env(
        workspace_dir=workspace_dir,
        home_dir=home_dir,
        skills_dir=skills_dir,
    )
    # Bind an explicit ShellBackend/Blackboard if the caller passed one; otherwise the
    # bound handlers inherit whatever the outer runtime_env_context injects (H5.1).
    if shell_backend is not None or blackboard is not None or blackboard_id is not None:
        runtime_env = replace(
            runtime_env,
            shell_backend=shell_backend if shell_backend is not None else runtime_env.shell_backend,
            blackboard=blackboard if blackboard is not None else runtime_env.blackboard,
            blackboard_id=blackboard_id if blackboard_id is not None else runtime_env.blackboard_id,
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
    # H5.1: bind the registry's PATHS (workspace/home/skills) but do NOT clobber an
    # outer runtime_env_context's injected ShellBackend/Blackboard. A host that wraps a
    # send() in `runtime_env_context(shell_backend=sandbox, ...)` must still have its
    # sandbox honored; previously the bind reset the contextvar to a path-only snapshot
    # (shell_backend=None), silently defeating the injection. The bound env wins only
    # where it set a value explicitly.
    def _merged() -> RuntimeEnv:
        outer = get_runtime_env()
        return replace(
            runtime_env,
            shell_backend=runtime_env.shell_backend or outer.shell_backend,
            blackboard=runtime_env.blackboard or outer.blackboard,
            blackboard_id=runtime_env.blackboard_id or outer.blackboard_id,
        )

    # An async handler returns a coroutine — the env context MUST be held across its
    # await, not just while the coroutine object is constructed (else the contextvar is
    # reset before the body runs and the handler sees an empty env). So bind async and
    # sync handlers with matching wrappers.
    if inspect.iscoroutinefunction(handler):
        @wraps(handler)
        async def _wrapped_async(**kwargs: Any) -> Any:
            with runtime_env_context(_merged()):
                return await handler(**kwargs)

        return _wrapped_async

    @wraps(handler)
    def _wrapped(**kwargs: Any) -> Any:
        with runtime_env_context(_merged()):
            return handler(**kwargs)

    return _wrapped


__all__ = [
    "ToolRegistry",
    "build_registry",
    "create_default_tool_registry",
    "get_tool_definitions",
    "DEFAULT_TOOL_HANDLERS",
]
