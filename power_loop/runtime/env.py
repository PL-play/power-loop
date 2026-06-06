from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path


class RuntimeEnvError(RuntimeError):
    """Raised when an opt-in runtime feature is missing required paths."""


@dataclass(frozen=True)
class RuntimeEnv:
    """Paths used by opt-in runtime features such as default tools and skills."""

    workspace_dir: Path | None = None
    home_dir: Path | None = None
    skills_dir: Path | None = None

    @classmethod
    def from_env(
        cls,
        *,
        workspace_dir: str | Path | None = None,
        home_dir: str | Path | None = None,
        skills_dir: str | Path | None = None,
    ) -> RuntimeEnv:
        return cls(
            workspace_dir=_resolve_optional_path(workspace_dir, "POWER_LOOP_WORKSPACE"),
            home_dir=_resolve_optional_path(home_dir, "POWER_LOOP_HOME"),
            skills_dir=_resolve_optional_path(skills_dir, "POWER_LOOP_SKILLS_DIR"),
        )

    def require_workspace_dir(self) -> Path:
        if self.workspace_dir is None:
            raise RuntimeEnvError(
                "POWER_LOOP_WORKSPACE is required for default filesystem, search, shell, and background tools. "
                "Pass workspace_dir=... to create_default_tool_registry(...) or set POWER_LOOP_WORKSPACE."
            )
        return self.workspace_dir

    def require_home_dir(self) -> Path:
        if self.home_dir is None:
            raise RuntimeEnvError(
                "POWER_LOOP_HOME is required for runtime cache/home access. "
                "Pass home_dir=... to create_default_tool_registry(...) or set POWER_LOOP_HOME."
            )
        return self.home_dir

    def require_skills_dir(self) -> Path:
        if self.skills_dir is None:
            raise RuntimeEnvError(
                "POWER_LOOP_SKILLS_DIR is required for default skill loading. "
                "Pass skills_dir=..., set AgentLoopConfig.skills_dir, or set POWER_LOOP_SKILLS_DIR."
            )
        return self.skills_dir

    @property
    def home_rw_allowlist(self) -> tuple[Path, ...]:
        home = self.require_home_dir()
        roots = [home / ".cache", home / "logs"]
        if self.skills_dir is not None:
            roots.append(self.skills_dir)
        return tuple(roots)


def _resolve_optional_path(explicit: str | Path | None, env_key: str) -> Path | None:
    raw = explicit if explicit is not None else (os.environ.get(env_key) or "").strip()
    if raw is None or str(raw).strip() == "":
        return None
    return Path(raw).expanduser().resolve()


_DEFAULT_RUNTIME_ENV = RuntimeEnv.from_env()
_CURRENT_RUNTIME_ENV: ContextVar[RuntimeEnv] = ContextVar("power_loop_runtime_env", default=_DEFAULT_RUNTIME_ENV)


def get_runtime_env() -> RuntimeEnv:
    return _CURRENT_RUNTIME_ENV.get()


@contextmanager
def runtime_env_context(env: RuntimeEnv) -> Iterator[None]:
    token = _CURRENT_RUNTIME_ENV.set(env)
    try:
        yield
    finally:
        _CURRENT_RUNTIME_ENV.reset(token)


def _is_in_home_rw_allowlist(path: Path, env: RuntimeEnv) -> bool:
    resolved = path.resolve()
    for allowed_root in env.home_rw_allowlist:
        try:
            if resolved.is_relative_to(allowed_root.resolve()):
                return True
        except Exception:
            continue
    return False


def safe_path(p: str, purpose: str = "rw", *, env: RuntimeEnv | None = None) -> Path:
    runtime_env = env if env is not None else get_runtime_env()
    workspace_dir = runtime_env.require_workspace_dir()

    raw_input = (p or "").strip()
    if not raw_input:
        raise ValueError("Path is required")

    if raw_input.startswith("@workspace/"):
        candidate = (workspace_dir / raw_input[len("@workspace/") :]).resolve()
    elif raw_input.startswith("@home/"):
        candidate = (runtime_env.require_home_dir() / raw_input[len("@home/") :]).resolve()
    else:
        raw = Path(raw_input).expanduser()
        if raw.is_absolute():
            candidate = raw.resolve()
        else:
            candidate = (workspace_dir / raw).resolve()

    if candidate.is_relative_to(workspace_dir):
        return candidate

    if runtime_env.home_dir is not None and candidate.is_relative_to(runtime_env.home_dir):
        if _is_in_home_rw_allowlist(candidate, runtime_env):
            return candidate
        raise ValueError(
            f"Access to POWER_LOOP_HOME is restricted. Workspace is at {workspace_dir}, "
            f"home is at {runtime_env.home_dir}. Use relative paths, @workspace/<path>, "
            "or an allowlisted @home/.cache, @home/logs, or skills path."
        )

    raise ValueError(
        f"Path escapes allowed directories: {p}. "
        f"Workspace: {workspace_dir}. Use relative paths or @workspace/<path>."
    )


__all__ = [
    "RuntimeEnv",
    "RuntimeEnvError",
    "get_runtime_env",
    "runtime_env_context",
    "safe_path",
]
