"""Declarative subagent spec (M1.8).

The parent agent emits an :class:`AgentSpec` (as JSON or dict) describing
*what kind of* subagent to run, and :func:`run_agent_spec` materializes it as
a child :class:`StatefulAgentLoop` session.

This is the in-process complement of the imperative :mod:`spawn_agent` tool:

* ``spawn_agent`` — a tool the LLM calls in turn-form ("delegate this task").
* ``run_agent`` (meta-tool) — accepts a full :class:`AgentSpec` JSON for
  cases where the parent wants explicit control over the child's tool
  whitelist / model / max_rounds / system prompt.

Both paths reuse this module's :func:`run_agent_spec` so depth-guard,
parent-linking, lifecycle cleanup and tool-whitelist filtering live in one
place.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field, fields
from typing import Any

from power_loop.agent.types import AgentLoopConfig
from power_loop.contracts.errors import SpecValidationError
from power_loop.contracts.event_payloads import (
    SubagentCompletedPayload,
    SubagentLimitPayload,
    SubagentTaskStartPayload,
    SubagentTextPayload,
)
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.runtime.cancellation import CancellationLike
from power_loop.runtime.store.store import MAX_SPAWN_DEPTH
from power_loop.runtime.store.types import SubagentLifecycle
from power_loop.tools.registry import ToolRegistry

# Lifecycle events published from run_agent_spec carry this source so sinks can
# attribute them to a sub-agent (vs the parent's own pipeline events).
SUBAGENT_EVENT_SOURCE = "subagent"

__all__ = [
    "AgentSpec",
    "AgentSpecError",
    "filtered_registry",
    "run_agent_spec",
]


class AgentSpecError(SpecValidationError):
    """Raised when an :class:`AgentSpec` payload fails strict validation.

    Inherits from :class:`~power_loop.contracts.errors.SpecValidationError`
    so ``except PowerLoopError`` catches it alongside all other library errors.
    """
    # Keep ValueError-compatible interface for callers that already do
    # ``except ValueError`` or ``except AgentSpecError``.
    pass


@dataclass(frozen=True)
class AgentSpec:
    """Strict-schema description of a one-shot subagent run.

    All fields are explicit; unknown JSON keys are rejected so a hallucinated
    payload from the parent LLM surfaces early.
    """

    name: str
    system_prompt: str
    tools: list[str] | None = None      # whitelist; None = inherit all from parent
    max_rounds: int = 8
    max_tokens: int = 4000
    temperature: float = 0.0
    model: str | None = None      # per-subagent model override (None = inherit the service default)
    output_schema: dict[str, Any] | None = None   # {name, schema}; enforces structured output
    lifecycle: str = SubagentLifecycle.EPHEMERAL.value
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── validation ────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise AgentSpecError("AgentSpec.name must be a non-empty string")
        if not self.system_prompt or not isinstance(self.system_prompt, str):
            raise AgentSpecError("AgentSpec.system_prompt must be a non-empty string")
        if self.tools is not None:
            if not isinstance(self.tools, list) or not all(isinstance(x, str) for x in self.tools):
                raise AgentSpecError("AgentSpec.tools must be a list[str] or None")
        if not 1 <= int(self.max_rounds) <= 50:
            raise AgentSpecError("AgentSpec.max_rounds must be in [1, 50]")
        try:
            SubagentLifecycle(self.lifecycle)
        except ValueError as exc:
            raise AgentSpecError(f"unknown lifecycle: {self.lifecycle}") from exc
        if self.output_schema is not None and not isinstance(self.output_schema, dict):
            raise AgentSpecError("AgentSpec.output_schema must be an object or None")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSpec:
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise AgentSpecError(f"unknown AgentSpec field(s): {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @classmethod
    def from_json(cls, payload: str | dict[str, Any]) -> AgentSpec:
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise AgentSpecError(f"AgentSpec JSON parse error: {exc}") from exc
        else:
            data = dict(payload)
        if not isinstance(data, dict):
            raise AgentSpecError("AgentSpec payload must decode to an object")
        return cls.from_dict(data)


# ── helpers ──────────────────────────────────────────────────────────────


def filtered_registry(
    parent_registry: ToolRegistry | None, whitelist: list[str] | None
) -> ToolRegistry | None:
    """Return a registry containing only the whitelisted tools from ``parent``.

    * ``parent_registry=None`` → ``None``
    * ``whitelist=None`` → return the parent registry as-is (full inherit)
    * Names missing from the parent registry are silently skipped (subagent
      simply doesn't see them).
    """
    if parent_registry is None:
        return None
    if whitelist is None:
        return parent_registry
    sub = ToolRegistry()
    for name in whitelist:
        existing = parent_registry.get(name)
        if existing is None:
            continue
        sub.register(existing.definition, existing.handler, overwrite=True)
    return sub


def _publish_subagent_event(
    parent_loop: Any,
    parent_sid: str | None,
    event_type: AgentEventType,
    payload: Any,
) -> None:
    """Best-effort publish of a sub-agent lifecycle event on the parent's bus.

    The event is attributed to ``parent_sid`` (so it surfaces in the parent /
    host event stream) and tagged ``source="subagent"``. Observability must
    never break a run, so all errors are swallowed.
    """
    try:
        parent_loop.event_bus.publish(
            AgentEvent(
                type=event_type,
                data=payload,
                session_id=parent_sid,
                source=SUBAGENT_EVENT_SOURCE,
            )
        )
    except Exception:  # noqa: BLE001 — observability must never break a run
        pass


async def run_agent_spec(
    spec: AgentSpec | dict[str, Any] | str,
    user_input: str,
    *,
    parent_loop: Any,
    spawn_tool_call_id: str | None = None,
    stop_event: CancellationLike = None,
) -> dict[str, Any]:
    """Materialize ``spec`` as a child session under ``parent_loop`` and run it.

    Returns a dict with ``final_text``, ``status``, ``rounds``, ``session_id``,
    ``depth`` — easy for the parent LLM to consume as a tool result.

    ``stop_event`` (any :data:`CancellationLike`: a ``CancellationToken``,
    ``threading.Event`` / ``asyncio.Event``, or ``() -> bool``) is forwarded to
    the child's run loop. When it fires, the child stops at the next round /
    cancellation checkpoint and returns ``status="cancelled"`` — the way an
    orchestrator (e.g. a workflow) tears down in-flight sub-agents.

    Lifecycle:
      * ``EPHEMERAL`` (default) → child session is deleted on success;
        preserved on non-completed status (``hit_round_limit`` / ``cancelled``)
        so the parent can debug.
      * ``LINKED`` → child kept; cascade-deleted when parent is closed.
      * ``DETACHED`` → child kept, lives independently.
    """
    if not isinstance(spec, AgentSpec):
        spec = AgentSpec.from_json(spec)

    parent_sid = None
    # parent_loop must be a StatefulAgentLoop; we accept Any to avoid a cycle.
    from power_loop.core.agent_context import get_session_id
    parent_sid = get_session_id()

    # Pre-check depth so we don't even create the session row if it'll bounce.
    # The effective ceiling lives on the store (configurable); fall back to the
    # module default for any custom store that predates the attribute.
    store = await parent_loop._ensure_store()
    max_depth = getattr(store, "max_spawn_depth", MAX_SPAWN_DEPTH)
    if parent_sid is not None:
        parent_row = await store.get_session(parent_sid)
        if parent_row is not None and parent_row.spawn_depth + 1 > max_depth:
            return {
                "session_id": None,
                "status": "rejected",
                "final_text": (
                    f"spawn rejected — depth {parent_row.spawn_depth + 1} "
                    f"exceeds max {max_depth}"
                ),
                "rounds": 0,
                "depth": parent_row.spawn_depth + 1,
            }

    response_format = None
    if spec.output_schema:
        from power_loop.runtime.structured import StructuredOutputSpec

        response_format = StructuredOutputSpec(
            name=str(spec.output_schema.get("name") or "Output"),
            schema=spec.output_schema.get("schema") or spec.output_schema,
        ).to_openai_response_format()

    child_config = AgentLoopConfig(
        system_prompt=spec.system_prompt,
        max_rounds=int(spec.max_rounds),
        max_tokens=int(spec.max_tokens),
        temperature=float(spec.temperature),
        model=spec.model,
        response_format=response_format,
    )

    child_sid = await store.create_session(
        system_prompt=spec.system_prompt,
        config={
            "spec_name": spec.name,
            "max_rounds": child_config.max_rounds,
            "max_tokens": child_config.max_tokens,
            "temperature": child_config.temperature,
            "model": spec.model,
        },
        parent_session_id=parent_sid,
        spawn_tool_call_id=spawn_tool_call_id,
        lifecycle=SubagentLifecycle(spec.lifecycle),
        # Stamp spec_name into METADATA (not just config) so the blackboard author label is the spec
        # name, not the raw session UUID (subagent-coordination-3): blackboard._author reads metadata.
        metadata={**dict(spec.metadata), "spec_name": spec.name},
    )

    child_row = await store.get_session(child_sid)
    depth = child_row.spawn_depth if child_row is not None else 1
    _publish_subagent_event(
        parent_loop, parent_sid, AgentEventType.SUBAGENT_TASK_START,
        SubagentTaskStartPayload(
            task=user_input,
            preset=str(spec.metadata.get("preset") or "core"),
            sub_session_id=child_sid,
            depth=depth,
        ),
    )

    sub_registry = filtered_registry(parent_loop.tool_registry, spec.tools)

    # Build a sibling loop sharing the same store + registry-subset.
    from power_loop.agent.stateful_loop import StatefulAgentLoop

    child_loop = StatefulAgentLoop(
        llm=parent_loop.llm,
        store=store,
        config=child_config,
        tool_registry=sub_registry,
        hooks=parent_loop._runner.hooks,
        event_bus=parent_loop._runner.event_bus,
    )
    try:
        result = await child_loop.send(user_input, session_id=child_sid, stop_event=stop_event)
    except BaseException:
        # A raised child run (cancellation, provider error, bug) would otherwise leak the EPHEMERAL
        # child session — the cleanup below is only reached on a normal return (subagent-coordination-2).
        if spec.lifecycle == SubagentLifecycle.EPHEMERAL.value:
            with contextlib.suppress(Exception):
                await store.close_session(child_sid, cascade=True)
        raise

    # ── Sub-agent lifecycle events (source="subagent", on the parent stream) ──
    # TEXT only when the child produced an answer; LIMIT for the round-limit
    # failure mode it specifically models; COMPLETED is the terminal marker
    # always emitted last.
    if result.status == "completed":
        _publish_subagent_event(
            parent_loop, parent_sid, AgentEventType.SUBAGENT_TEXT,
            SubagentTextPayload(
                sub_session_id=child_sid, status=result.status,
                rounds=result.rounds, final_text=result.final_text or "",
            ),
        )
    elif result.status == "hit_round_limit":
        _publish_subagent_event(
            parent_loop, parent_sid, AgentEventType.SUBAGENT_LIMIT,
            SubagentLimitPayload(sub_session_id=child_sid, max_rounds=int(spec.max_rounds)),
        )
    _publish_subagent_event(
        parent_loop, parent_sid, AgentEventType.SUBAGENT_COMPLETED,
        SubagentCompletedPayload(
            sub_session_id=child_sid, status=result.status,
            rounds=result.rounds, final_text=result.final_text or "",
        ),
    )

    # Lifecycle cleanup.
    if spec.lifecycle == SubagentLifecycle.EPHEMERAL.value:
        if result.status == "completed":
            await store.close_session(child_sid, cascade=True)
            returned_sid: str | None = None
        else:
            # Preserve for debugging.
            returned_sid = child_sid
    else:
        returned_sid = child_sid

    return {
        "session_id": returned_sid,
        "status": result.status,
        "final_text": result.final_text,
        "rounds": result.rounds,
        "depth": depth,
        "usage": dict(result.usage or {}),
    }
