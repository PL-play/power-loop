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

import json
import re
from dataclasses import dataclass, field, fields, replace
from typing import Any

from power_loop.agent.types import AgentLoopConfig
from power_loop.contracts.errors import SpecValidationError
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.runtime.store.types import SubagentLifecycle
from power_loop.tools.registry import ToolRegistry

# Lifecycle events published from run_agent_spec carry this source so sinks can
# attribute them to a sub-agent (vs the parent's own pipeline events).
SUBAGENT_EVENT_SOURCE = "subagent"

__all__ = [
    "AgentSpec",
    "AgentSpecError",
    "filtered_registry",
    "normalize_output_schema",
    "output_response_format",
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
    # None = 跟父 loop 一样（推荐）。**这一项不是「答案能多长」，思考内容也算在里面**：
    # 开了思考的模型被压在一个小预算里，会把预算全花在思考上、正文一个字都没有 =「空回复」，
    # 而那一轮重试还会再空一次（真实事故：子 agent 3000、父 40000，子运行整个降级）。
    max_tokens: int | None = None
    temperature: float = 0.0
    model: str | None = None      # per-subagent model override (None = inherit the service default)
    # {name, schema, strict?}; enforces structured output. strict defaults to True — a schema that
    # doesn't follow strict-mode rules (all keys required, no extra keys) opts out with strict: false.
    output_schema: dict[str, Any] | None = None
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
        if int(self.max_rounds) < 1:
            raise AgentSpecError("AgentSpec.max_rounds must be >= 1")
        try:
            SubagentLifecycle(self.lifecycle)
        except ValueError as exc:
            raise AgentSpecError(f"unknown lifecycle: {self.lifecycle}") from exc
        if self.output_schema is not None and not isinstance(self.output_schema, dict):
            raise AgentSpecError("AgentSpec.output_schema must be an object or None")
        if self.output_schema is not None and not isinstance(
            self.output_schema.get("strict", True), bool
        ):
            raise AgentSpecError("AgentSpec.output_schema.strict must be a boolean")

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

# OpenAI json_schema 的 name 只收 ^[A-Za-z0-9_-]{1,64}$；agent 起的名字可能是中文，原样发给原生模型会 400。
_SCHEMA_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def normalize_output_schema(
    value: Any, *, default_name: str = "Output", default_strict: bool = True
) -> dict[str, Any]:
    """Normalize a structured-output declaration to ``{"name", "schema", "strict"}``.

    Accepts ``{name?, schema, strict?}``, a bare JSON Schema, or either one as a JSON string
    (models sometimes send an object argument as a string). The root must be ``type: object`` —
    structured output only returns objects. ``strict`` is what the declaration says, else
    ``default_strict``; pass False for schemas an LLM wrote (they rarely follow strict-mode
    rules, and a native json_schema provider 400s on them). Raises :class:`AgentSpecError`.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise AgentSpecError(f"output_schema 不是合法的 JSON：{exc}") from None
    if not isinstance(value, dict) or not value:
        raise AgentSpecError("output_schema 要是一个 JSON Schema 对象")
    if isinstance(value.get("schema"), dict):
        name, schema = value.get("name"), value["schema"]
        strict = value.get("strict", default_strict)
    else:
        name, schema, strict = None, value, default_strict
    if schema.get("type") != "object":
        raise AgentSpecError(
            'output_schema 的根必须是 {"type": "object", ...}（结构化输出只收对象）'
        )
    if not isinstance(strict, bool):
        raise AgentSpecError("output_schema.strict 只能是 true 或 false")
    name = _SCHEMA_NAME_RE.sub("_", str(name or default_name))[:64] or default_name
    return {"name": name, "schema": schema, "strict": strict}


def output_response_format(output_schema: dict[str, Any]) -> dict[str, Any]:
    """``AgentSpec.output_schema`` (``{name, schema, strict?}`` or a bare schema) → the
    OpenAI-compatible ``response_format``. ``strict`` defaults to True."""
    from power_loop.runtime.structured import StructuredOutputSpec

    return StructuredOutputSpec(
        name=str(output_schema.get("name") or "Output"),
        schema=output_schema.get("schema") or output_schema,
        strict=bool(output_schema.get("strict", True)),
    ).to_openai_response_format()


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


def _child_config(spec: AgentSpec, parent_loop: Any) -> AgentLoopConfig:
    response_format = output_response_format(spec.output_schema) if spec.output_schema else None

    # Inherit the parent's transport resilience: the child reuses the parent's
    # LLM client (same connection pool), and a delegation often fires after a
    # long tool phase — right when pooled keep-alive connections have gone
    # stale. Without the parent's retry policy a single httpx.ReadError on the
    # child's FIRST streaming call killed the whole delegation ("Error: ").
    parent_config = getattr(parent_loop, "config", None)
    from power_loop.runtime.tool_context import active_engine
    context_policy = getattr(active_engine.get(), "config", getattr(parent_config, "tool_context", None))
    child_config = AgentLoopConfig(
        system_prompt=spec.system_prompt,
        max_rounds=int(spec.max_rounds),
        # max_tokens=None → 跟父：子运行和父用的是同一个模型（model=None 时），
        # 父的预算容得下它的思考，子的也该容得下。
        max_tokens=int(spec.max_tokens) if spec.max_tokens else int(
            getattr(parent_config, "max_tokens", 0) or 4000),
        temperature=float(spec.temperature),
        model=spec.model or getattr(parent_config, "model", None),
        response_format=response_format,
        retry_policy=getattr(parent_config, "retry_policy", None),
        tool_context=context_policy,
        image_retention_rounds=getattr(parent_config, "image_retention_rounds", 1),
    )
    # A child starts with its parent's context policy, but owns its mutable representation
    # and per-session archive/fuse state. Hosts may resolve model-specific limits below.
    import copy
    context_fields = (
        "representation", "fold_strategy", "max_context_rows", "context_budget_tokens",
        "context_checkpoint_tokens", "insend_distill_tokens", "insend_distill_batch",
        "insend_distill_hot_tail", "insend_distill_keep_tools", "tool_result_max_chars",
        "microcompact_enabled", "microcompact_size_limit", "microcompact_hot_tail",
        "microcompact_spill_dir", "context_strategy", "context_cut_tokens",
    )
    child_config = replace(child_config, **{
        name: copy.copy(getattr(parent_config, name)) for name in context_fields
        if parent_config is not None and hasattr(parent_config, name) and hasattr(child_config, name)
    })
    # HOST config seam (subagent_config_factory, PROVISIONAL 3.14): the parent's
    # config may rewrite the child's default config — e.g. hand heavy leaves a
    # context strategy (representation/fold/microcompact). The factory's output
    # is used AS IS; it received the spec-derived default, so an increment via
    # dataclasses.replace is the intended usage. A raising factory fails THIS
    # delegation (surfaced as the tool error), never the parent loop.
    _config_factory = getattr(parent_config, "subagent_config_factory", None)
    if _config_factory is not None:
        child_config = _config_factory(spec, child_config)
        if not isinstance(child_config, AgentLoopConfig):
            raise TypeError(
                "subagent_config_factory must return an AgentLoopConfig "
                f"(got {type(child_config).__name__})"
            )

    # The host factory may adjust context/model settings, not widen delegation.
    return replace(child_config, delegation_policy=getattr(parent_config, "delegation_policy", "legacy"))




def _model_service(parent_loop, spec, override=None):
    binding = spec.metadata.get('model_binding')
    if binding is None:
        return parent_loop.llm if override is None else override
    services = getattr(parent_loop.task_runtime, 'model_services', {})
    service = services.get(binding)
    if service is None or override is not None and override is not service:
        from power_loop.agent.runtime_types import RuntimeRequestError
        raise RuntimeRequestError('resource_unavailable', f'model_binding_unavailable: {binding}')
    return service


async def run_agent_spec(spec, user_input, *, parent_loop, execution_role="subagent"):
    """Create an independent child and await its durable first result."""
    from power_loop.agent.runtime_types import RuntimeRequestError
    if execution_role != "subagent":
        raise RuntimeRequestError("feature_unavailable", "submit workflow nodes through RuntimeClient.workflow")
    if not isinstance(spec, AgentSpec):
        spec = AgentSpec.from_json(spec)
    from .delegation import run_inline
    return await run_inline(spec, user_input, parent_loop=parent_loop)
