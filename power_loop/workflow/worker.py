"""Isolated worker core for an out-of-process executor (Phase 0).

The hard question for a subprocess executor is *not* the plumbing — it is whether
a sub-agent's live dependencies (the LLM service, the tool registry) can be
**rebuilt from serializable config alone**, with NO reference to the parent
process's live objects, and run an agent loop against its **own** database. If
that holds, each sub-agent can get its own SQLite file (one writer per file, no
shared-write problem) and the supervisor inspects the result + the private trace
afterward.

:func:`run_spec_isolated` is that core. It is written so it can run unchanged
either in-process (Phase 0 / embedding) or inside a spawned worker process later
(Phase 1) — it only ever touches:

* a :class:`WorkerBootstrap` (how to build llm + tools from config), and
* a ``db_path`` it owns exclusively.

It deliberately does NOT import or accept a ``parent_loop`` / shared ``store`` /
shared ``event_bus`` — proving the boundary is clean. The returned dict matches
``run_agent_spec``'s shape (plus ``db_path``) so it drops into the ``Executor``
seam.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from power_loop.contracts.errors import PowerLoopError

if TYPE_CHECKING:
    from llm_client.interface import LLMService
    from power_loop.tools.registry import ToolRegistry

__all__ = ["WorkerBootstrap", "WorkerBootstrapError", "run_spec_isolated"]


class WorkerBootstrapError(PowerLoopError):
    """Raised when a worker cannot rebuild its dependencies from config."""


@dataclass
class WorkerBootstrap:
    """How an isolated worker rebuilds its live dependencies from config.

    Two paths, mirroring the two lifetimes this core must serve:

    * **Serializable** (real subprocess): ``llm_from_env`` + ``tool_preset`` +
      ``workspace_dir`` — pure data, safe to ship across a process boundary so the
      worker rebuilds the LLM via ``create_llm_service_from_env`` and tools via
      ``create_default_tool_registry``.
    * **Factories** (in-process / tests / embedding): ``llm_factory`` /
      ``registry_factory`` — direct callables (NOT serializable). Phase 0 uses
      these to prove the mechanism without standing up a real provider.

    Factories take precedence when set.
    """

    # serializable path
    llm_from_env: bool = False
    provider_prefix: str | None = None
    tool_preset: str | None = None       # None → no tools (safe default for an isolated leaf)
    workspace_dir: str | None = None
    home_dir: str | None = None
    # in-process / test path (not serializable)
    llm_factory: Callable[[], LLMService] | None = None
    registry_factory: Callable[[], ToolRegistry] | None = None

    def build_llm(self) -> LLMService:
        if self.llm_factory is not None:
            return self.llm_factory()
        if self.llm_from_env:
            from power_loop.runtime.provider import create_llm_service_from_env

            if self.provider_prefix:
                return create_llm_service_from_env(prefix=self.provider_prefix)
            return create_llm_service_from_env()
        raise WorkerBootstrapError(
            "WorkerBootstrap has no LLM source: set llm_factory or llm_from_env=True"
        )

    def build_registry(self, whitelist: list[str] | None) -> ToolRegistry | None:
        """Build the tool registry, then narrow it to the spec's whitelist."""
        from power_loop.runtime.spec import filtered_registry

        if self.registry_factory is not None:
            reg = self.registry_factory()
        elif self.tool_preset is not None:
            from power_loop.tools import create_default_tool_registry

            reg = create_default_tool_registry(
                preset=self.tool_preset,
                workspace_dir=self.workspace_dir,
                home_dir=self.home_dir,
            )
        else:
            return None  # no tools at all
        return filtered_registry(reg, whitelist)


async def run_spec_isolated(
    spec: Any,
    user_input: str,
    *,
    bootstrap: WorkerBootstrap,
    db_path: str,
    max_spawn_depth: int | None = None,
) -> dict[str, Any]:
    """Run one ``AgentSpec`` to completion against an isolated DB at ``db_path``.

    Rebuilds llm + tools from ``bootstrap`` only — no parent loop, no shared
    store, no shared event bus. The DB file is left on disk after the run so the
    supervisor can inspect the sub-agent's full trace (open it read-only; WAL
    allows reads even while another connection writes). Returns a dict shaped
    like ``run_agent_spec`` plus ``db_path``.
    """
    from power_loop.agent.stateful_loop import StatefulAgentLoop
    from power_loop.agent.types import AgentLoopConfig
    from power_loop.runtime.session_store import MAX_SPAWN_DEPTH
    from power_loop.runtime.spec import AgentSpec

    spec = spec if isinstance(spec, AgentSpec) else AgentSpec.from_json(spec)

    response_format = None
    if spec.output_schema:
        from power_loop.runtime.structured import StructuredOutputSpec

        response_format = StructuredOutputSpec(
            name=str(spec.output_schema.get("name") or "Output"),
            schema=spec.output_schema.get("schema") or spec.output_schema,
        ).to_openai_response_format()

    config = AgentLoopConfig(
        system_prompt=spec.system_prompt,
        max_rounds=int(spec.max_rounds),
        max_tokens=int(spec.max_tokens),
        temperature=float(spec.temperature),
        model=spec.model,
        response_format=response_format,
    )

    llm = bootstrap.build_llm()
    registry = bootstrap.build_registry(spec.tools)

    # Owns its store (db_path); no parent objects of any kind are referenced.
    loop = StatefulAgentLoop(
        llm=llm,
        db_path=db_path,
        config=config,
        tool_registry=registry,
        max_spawn_depth=max_spawn_depth if max_spawn_depth is not None else MAX_SPAWN_DEPTH,
    )
    try:
        sid = loop.new_session(metadata={"spec_name": spec.name, "isolated": True})
        result = await loop.send(user_input, session_id=sid)
        return {
            "session_id": sid,
            "status": result.status,
            "final_text": result.final_text,
            "rounds": result.rounds,
            "depth": 1,
            "usage": dict(result.usage or {}),
            "db_path": db_path,
        }
    finally:
        # Close the connection but keep the file: the supervisor inspects it later.
        loop.close()
