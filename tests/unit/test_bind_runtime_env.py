"""H5.1: binding a default tool registry must not shadow an outer-injected
ShellBackend/Blackboard from runtime_env_context."""

from __future__ import annotations

from pathlib import Path

import pytest

from power_loop.runtime.env import RuntimeEnv, get_runtime_env, runtime_env_context
from power_loop.tools import _bind_handler, create_default_tool_registry

pytestmark = pytest.mark.unit


def test_bound_handler_honors_outer_injected_backend() -> None:
    """A path-only bound env must inherit the outer context's ShellBackend instead
    of resetting it to None (the injection footgun)."""
    bound_env = RuntimeEnv(workspace_dir=Path("/tmp/ws"))  # no shell_backend
    sentinel = object()
    seen: dict = {}

    def probe(**kw) -> str:
        env = get_runtime_env()
        seen["backend"] = env.shell_backend
        seen["workspace"] = env.workspace_dir
        return "ok"

    wrapped = _bind_handler(probe, bound_env)
    with runtime_env_context(RuntimeEnv(shell_backend=sentinel)):  # type: ignore[arg-type]
        wrapped()

    assert seen["backend"] is sentinel             # outer injection honored …
    assert str(seen["workspace"]) == "/tmp/ws"     # … and the bound path still applies


def test_bound_explicit_backend_wins_over_outer() -> None:
    explicit = object()
    bound_env = RuntimeEnv(workspace_dir=Path("/tmp/ws"), shell_backend=explicit)  # type: ignore[arg-type]
    outer = object()
    seen: dict = {}

    wrapped = _bind_handler(lambda **kw: seen.__setitem__("b", get_runtime_env().shell_backend), bound_env)
    with runtime_env_context(RuntimeEnv(shell_backend=outer)):  # type: ignore[arg-type]
        wrapped()

    assert seen["b"] is explicit  # the registry's explicit backend wins


def test_create_default_tool_registry_accepts_injection_params() -> None:
    """The new shell_backend/blackboard/blackboard_id params are accepted and bound."""
    backend = object()
    reg = create_default_tool_registry(
        include=["read_file"], workspace_dir="/tmp/ws", shell_backend=backend,  # type: ignore[arg-type]
    )
    assert reg is not None  # constructs without error; backend is bound into handlers
