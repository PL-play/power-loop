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

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from power_loop.contracts.errors import PowerLoopError

if TYPE_CHECKING:
    from llm_client.interface import LLMService
    from power_loop.tools.registry import ToolRegistry

__all__ = [
    "WorkerBootstrap",
    "WorkerBootstrapError",
    "WorkerJob",
    "run_spec_isolated",
    "run_job",
    "encode_result",
    "decode_result",
    "main",
]

# Frame marker so the parent can pick the result line out of a worker's stdout
# even if logging or prints leaked onto the same stream. Plain ASCII on purpose:
# control chars like RS (0x1e) are line boundaries to str.splitlines().
RESULT_SENTINEL = "__POWERLOOP_WF_RESULT__ "


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


# ── serialization: bootstrap + job + result framing (Phase 1 IPC contract) ────

# Only these WorkerBootstrap fields survive a process boundary (factories cannot).
_BOOTSTRAP_SERIALIZABLE = (
    "llm_from_env", "provider_prefix", "tool_preset", "workspace_dir", "home_dir",
)


def _bootstrap_to_dict(b: WorkerBootstrap) -> dict[str, Any]:
    if b.llm_factory is not None or b.registry_factory is not None:
        raise WorkerBootstrapError(
            "cannot serialize a WorkerBootstrap with factories for a subprocess; "
            "use the config path (llm_from_env / tool_preset) instead"
        )
    if not b.llm_from_env:
        raise WorkerBootstrapError(
            "a subprocess WorkerBootstrap needs llm_from_env=True (the worker rebuilds "
            "its provider from env/config)"
        )
    return {k: getattr(b, k) for k in _BOOTSTRAP_SERIALIZABLE}


def _bootstrap_from_dict(d: dict[str, Any]) -> WorkerBootstrap:
    return WorkerBootstrap(**{k: d.get(k) for k in _BOOTSTRAP_SERIALIZABLE if k in d})


# Attach as methods for ergonomics.
WorkerBootstrap.to_serializable_dict = _bootstrap_to_dict  # type: ignore[attr-defined]
WorkerBootstrap.from_dict = staticmethod(_bootstrap_from_dict)  # type: ignore[attr-defined]


@dataclass
class WorkerJob:
    """One unit of out-of-process work: everything a worker needs, as pure data."""

    spec: dict[str, Any]
    user_input: str
    db_path: str
    bootstrap: dict[str, Any] = field(default_factory=dict)
    max_spawn_depth: int | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "spec": self.spec,
                "user_input": self.user_input,
                "db_path": self.db_path,
                "bootstrap": self.bootstrap,
                "max_spawn_depth": self.max_spawn_depth,
            }
        )

    @classmethod
    def from_json(cls, payload: str | dict[str, Any]) -> WorkerJob:
        d = json.loads(payload) if isinstance(payload, str) else dict(payload)
        return cls(
            spec=d["spec"],
            user_input=d.get("user_input", ""),
            db_path=d["db_path"],
            bootstrap=d.get("bootstrap") or {},
            max_spawn_depth=d.get("max_spawn_depth"),
        )


def _maybe_fault_inject(job: WorkerJob) -> None:
    """Test-only fault injection: hard-crash the worker for a named node.

    Gated behind ``POWER_LOOP_TEST_CRASH_NODES`` (comma-separated node ids), which
    production never sets. ``os._exit`` kills the process before it writes a result
    frame, simulating a worker that died mid-run — the orchestrator records the
    leaf as failed and the resume tier re-runs exactly it.
    """
    import os

    nodes = os.environ.get("POWER_LOOP_TEST_CRASH_NODES", "").strip()
    if not nodes:
        return
    nid = (job.spec.get("metadata") or {}).get("workflow_node_id")
    if nid and nid in {n.strip() for n in nodes.split(",") if n.strip()}:
        os._exit(13)


async def run_job(job: WorkerJob) -> dict[str, Any]:
    """Execute a job in this process: rebuild the bootstrap and run the spec."""
    _maybe_fault_inject(job)
    bootstrap = _bootstrap_from_dict(job.bootstrap)
    return await run_spec_isolated(
        job.spec, job.user_input,
        bootstrap=bootstrap, db_path=job.db_path, max_spawn_depth=job.max_spawn_depth,
    )


def encode_result(obj: dict[str, Any]) -> str:
    """Frame a result envelope as a single sentinel-prefixed line."""
    return RESULT_SENTINEL + json.dumps(obj)


def decode_result(stdout: str) -> dict[str, Any]:
    """Pull the result envelope out of a worker's stdout (last framed line wins)."""
    # Split on real newlines only; the JSON payload escapes its own newlines, and
    # str.splitlines() would also break on exotic separators inside the text.
    for line in reversed(stdout.split("\n")):
        if line.startswith(RESULT_SENTINEL):
            return json.loads(line[len(RESULT_SENTINEL):])
    raise WorkerBootstrapError("no result frame found in worker output")


def main(argv: list[str] | None = None, *, stdin: Any = None, stdout: Any = None) -> int:
    """Worker process entrypoint: read a job from stdin, run it, frame the result.

    ``python -m power_loop.workflow.worker`` runs this. ``stdin``/``stdout`` are
    injectable so the entrypoint can be exercised in-process by tests.
    """
    import asyncio
    import sys
    import traceback

    src = stdin if stdin is not None else sys.stdin
    out = stdout if stdout is not None else sys.stdout
    try:
        job = WorkerJob.from_json(src.read())
        result = asyncio.run(run_job(job))
        envelope: dict[str, Any] = {"ok": True, "result": result}
    except BaseException as exc:  # noqa: BLE001 — a worker must always report, never hang
        envelope = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    out.write(encode_result(envelope) + "\n")
    out.flush()
    return 0 if envelope.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
