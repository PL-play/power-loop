"""WorkflowSpec — a declarative JSON DSL for deterministic multi-agent workflows.

This is the *code-driven* counterpart to the *model-driven* delegation already
provided by ``spawn_agent`` / ``run_agent``: instead of letting the parent LLM
decide each delegation ad-hoc, a :class:`WorkflowSpec` describes a deterministic
control-flow graph (sequence / parallel / foreach / branch) whose leaves are
ordinary sub-agents (each an :class:`~power_loop.runtime.spec.AgentSpec`). The
spec itself is *dynamic* in two ways:

1. the parent LLM can **author it at runtime** (emit the JSON, same idiom as
   emitting an ``AgentSpec`` for ``run_agent``); and
2. its control constructs (``foreach`` over runtime data, ``branch`` on a prior
   step's output) expand dynamically while the deterministic engine interprets
   it.

It is **not** a static precompiled DAG and **not** a router LLM — the engine
just interprets the spec deterministically; the only LLM calls are the leaf
sub-agents.

Validation philosophy mirrors :class:`AgentSpec`: strict dataclasses, unknown
JSON keys rejected, *all* problems aggregated and raised at once from
:meth:`WorkflowSpec.from_json` so a hallucinated payload surfaces every issue in
one shot (see :class:`WorkflowSpecError`).

The node types in this first (minimal) tier:

* ``agent``    — run one sub-agent (an ``AgentSpec`` payload).
* ``sequence`` — run child nodes in order.
* ``parallel`` — run child branches concurrently (barrier; fan-in).
* ``foreach``  — map a node template over a runtime list (optionally concurrent).
* ``branch``   — pick one child node by a prior step's value.

Deferred to later tiers (documented, not yet implemented): ``while`` / ``until``,
``wait_timer``, ``human_gate``, detached execution + completion callbacks, and
orchestration-level resume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from power_loop.contracts.errors import SpecValidationError
from power_loop.runtime.spec import AgentSpec, AgentSpecError

__all__ = [
    "WorkflowSpecError",
    "AgentNode",
    "SequenceNode",
    "ParallelNode",
    "ForeachNode",
    "BranchNode",
    "WorkflowNode",
    "WorkflowBudget",
    "WorkflowSpec",
]


class WorkflowSpecError(SpecValidationError):
    """Raised when a :class:`WorkflowSpec` payload fails strict validation.

    Inherits :class:`~power_loop.contracts.errors.SpecValidationError` so
    ``except PowerLoopError`` catches it alongside the rest of the library.

    Unlike a single-message error, this aggregates *all* problems found while
    parsing the spec; :attr:`problems` holds them individually and ``str(err)``
    renders them as a bulleted list.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        body = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"invalid WorkflowSpec — {len(self.problems)} problem(s):\n{body}")


# ── node dataclasses ────────────────────────────────────────────────────────
#
# Each node is a frozen dataclass carrying its ``type`` discriminator. They are
# built by the recursive ``_parse_node`` below, which is where validation lives
# (so construction stays cheap and the public surface is ``WorkflowSpec``).


@dataclass(frozen=True)
class AgentNode:
    """Run a single sub-agent.

    ``spec`` is materialized into an :class:`AgentSpec`. ``input`` is a template
    (``{{var}}`` placeholders) for the user message sent to the sub-agent;
    ``inputs_from`` names earlier node ids whose output text is appended as
    context. ``output_schema`` (``{"name", "schema"}``) makes the engine parse
    the sub-agent's final text into a structured payload that downstream
    ``items_from`` / ``branch.on`` references can read.
    """

    type: str
    id: str
    spec: AgentSpec
    input: str = "{{input}}"
    inputs_from: tuple[str, ...] = ()
    output_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class SequenceNode:
    type: str
    steps: tuple[WorkflowNode, ...]
    id: str | None = None


@dataclass(frozen=True)
class ParallelNode:
    type: str
    branches: tuple[WorkflowNode, ...]
    id: str | None = None
    max_concurrency: int = 5
    on_error: str = "halt"  # "halt" | "continue"


@dataclass(frozen=True)
class ForeachNode:
    """Map ``body`` over a runtime list.

    The list comes from ``items_from`` (``"node_id.key"`` into a prior agent's
    parsed payload) or a literal ``items`` list. Each element binds to the
    ``as`` variable, available as ``{{as}}`` inside ``body``.
    """

    type: str
    as_var: str
    body: WorkflowNode
    id: str | None = None
    items_from: str | None = None
    items: tuple[Any, ...] | None = None
    parallel: bool = True
    max_concurrency: int = 5
    on_error: str = "halt"


@dataclass(frozen=True)
class BranchNode:
    """Pick one child by a prior step's value (``on`` = ``"node_id.key"``)."""

    type: str
    on: str
    cases: dict[str, WorkflowNode]
    id: str | None = None
    default: WorkflowNode | None = None


WorkflowNode = AgentNode | SequenceNode | ParallelNode | ForeachNode | BranchNode

_NODE_TYPES = {"agent", "sequence", "parallel", "foreach", "branch"}
_ON_ERROR = {"halt", "continue"}


@dataclass(frozen=True)
class WorkflowBudget:
    """Optional shared token ceiling across all sub-agents of a run.

    ``stop_at_remaining_pct`` lets the engine stop *spawning new* steps once the
    pool is nearly exhausted (soft enforcement; in-flight steps finish).
    """

    max_tokens: int
    stop_at_remaining_pct: float = 0.0


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    root: WorkflowNode
    input: str = ""
    budget: WorkflowBudget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── construction / validation ──────────────────────────────────────────

    @classmethod
    def from_json(cls, payload: str | dict[str, Any]) -> WorkflowSpec:
        """Build (and fully validate) a WorkflowSpec from JSON / a dict.

        Aggregates every problem and raises a single :class:`WorkflowSpecError`.
        """
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise WorkflowSpecError([f"JSON parse error: {exc}"]) from exc
        else:
            data = payload
        if not isinstance(data, dict):
            raise WorkflowSpecError(["payload must decode to an object"])

        problems: list[str] = []
        allowed = {"name", "root", "input", "budget", "metadata"}
        unknown = set(data) - allowed
        if unknown:
            problems.append(f"unknown top-level key(s): {sorted(unknown)}")

        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            problems.append("'name' must be a non-empty string")
            name = "workflow"

        wf_input = data.get("input", "")
        if not isinstance(wf_input, str):
            problems.append("'input' must be a string")
            wf_input = ""

        budget = None
        if "budget" in data and data["budget"] is not None:
            budget = _parse_budget(data["budget"], problems)

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            problems.append("'metadata' must be an object")
            metadata = {}

        root = None
        if "root" not in data:
            problems.append("missing required key 'root'")
        else:
            declared_ids: set[str] = set()
            _collect_agent_ids(data["root"], declared_ids)
            root = _parse_node(data["root"], "root", problems, declared_ids)

        if problems:
            raise WorkflowSpecError(problems)

        assert root is not None  # guaranteed when no problems
        return cls(name=name, root=root, input=wf_input, budget=budget, metadata=dict(metadata))

    # ── serialization (round-trips through from_json) ───────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to the JSON shape ``from_json`` accepts.

        Used to persist the spec in a run's journal so a workflow can be resumed
        across a process restart without the caller re-supplying it. The result
        round-trips: ``WorkflowSpec.from_json(spec.to_dict())`` rebuilds an
        equivalent spec.
        """
        out: dict[str, Any] = {"name": self.name, "root": _node_to_dict(self.root)}
        if self.input:
            out["input"] = self.input
        if self.budget is not None:
            out["budget"] = {
                "max_tokens": self.budget.max_tokens,
                "stop_at_remaining_pct": self.budget.stop_at_remaining_pct,
            }
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


# ── serialization helpers ───────────────────────────────────────────────────


def _agent_spec_to_dict(spec: AgentSpec) -> dict[str, Any]:
    """The declared AgentSpec fields, as a dict from_json round-trips."""
    from dataclasses import asdict

    return asdict(spec)


def _node_to_dict(node: WorkflowNode) -> dict[str, Any]:
    if isinstance(node, AgentNode):
        out: dict[str, Any] = {
            "type": "agent",
            "id": node.id,
            "spec": _agent_spec_to_dict(node.spec),
        }
        if node.input != "{{input}}":
            out["input"] = node.input
        if node.inputs_from:
            out["inputs_from"] = list(node.inputs_from)
        if node.output_schema is not None:
            out["output_schema"] = node.output_schema
        return out
    if isinstance(node, SequenceNode):
        out = {"type": "sequence", "steps": [_node_to_dict(s) for s in node.steps]}
        if node.id is not None:
            out["id"] = node.id
        return out
    if isinstance(node, ParallelNode):
        out = {
            "type": "parallel",
            "branches": [_node_to_dict(b) for b in node.branches],
            "max_concurrency": node.max_concurrency,
            "on_error": node.on_error,
        }
        if node.id is not None:
            out["id"] = node.id
        return out
    if isinstance(node, ForeachNode):
        out = {
            "type": "foreach",
            "as": node.as_var,
            "body": _node_to_dict(node.body),
            "parallel": node.parallel,
            "max_concurrency": node.max_concurrency,
            "on_error": node.on_error,
        }
        if node.id is not None:
            out["id"] = node.id
        if node.items_from is not None:
            out["items_from"] = node.items_from
        if node.items is not None:
            out["items"] = list(node.items)
        return out
    if isinstance(node, BranchNode):
        out = {
            "type": "branch",
            "on": node.on,
            "cases": {k: _node_to_dict(v) for k, v in node.cases.items()},
        }
        if node.id is not None:
            out["id"] = node.id
        if node.default is not None:
            out["default"] = _node_to_dict(node.default)
        return out
    raise TypeError(f"cannot serialize node of type {type(node).__name__}")


# ── parsing helpers ─────────────────────────────────────────────────────────


def _parse_budget(data: Any, problems: list[str]) -> WorkflowBudget | None:
    if not isinstance(data, dict):
        problems.append("'budget' must be an object")
        return None
    unknown = set(data) - {"max_tokens", "stop_at_remaining_pct"}
    if unknown:
        problems.append(f"'budget' has unknown key(s): {sorted(unknown)}")
    mt = data.get("max_tokens")
    if not isinstance(mt, int) or isinstance(mt, bool) or mt <= 0:
        problems.append("'budget.max_tokens' must be a positive integer")
        return None
    pct = data.get("stop_at_remaining_pct", 0.0)
    if not isinstance(pct, (int, float)) or isinstance(pct, bool) or not 0.0 <= float(pct) <= 100.0:
        problems.append("'budget.stop_at_remaining_pct' must be a number in [0, 100]")
        pct = 0.0
    return WorkflowBudget(max_tokens=int(mt), stop_at_remaining_pct=float(pct))


def _collect_agent_ids(data: Any, out: set[str]) -> None:
    """Pre-walk the (raw) tree gathering every agent-node ``id`` for ref checks."""
    if not isinstance(data, dict):
        return
    if data.get("type") == "agent" and isinstance(data.get("id"), str):
        out.add(data["id"])
    for key in ("steps", "branches"):
        for child in data.get(key) or []:
            _collect_agent_ids(child, out)
    if isinstance(data.get("body"), dict):
        _collect_agent_ids(data["body"], out)
    if isinstance(data.get("default"), dict):
        _collect_agent_ids(data["default"], out)
    for child in (data.get("cases") or {}).values():
        _collect_agent_ids(child, out)


def _ref_target(ref: str) -> str:
    """``"plan.subtopics"`` -> ``"plan"`` (the node id portion)."""
    return ref.split(".", 1)[0]


def _parse_node(
    data: Any, path: str, problems: list[str], ids: set[str]
) -> WorkflowNode | None:
    if not isinstance(data, dict):
        problems.append(f"{path}: node must be an object")
        return None
    ntype = data.get("type")
    if ntype not in _NODE_TYPES:
        problems.append(
            f"{path}: 'type' must be one of {sorted(_NODE_TYPES)} (got {ntype!r})"
        )
        return None

    if ntype == "agent":
        return _parse_agent(data, path, problems)
    if ntype == "sequence":
        return _parse_sequence(data, path, problems, ids)
    if ntype == "parallel":
        return _parse_parallel(data, path, problems, ids)
    if ntype == "foreach":
        return _parse_foreach(data, path, problems, ids)
    if ntype == "branch":
        return _parse_branch(data, path, problems, ids)
    return None  # unreachable


def _check_unknown(data: dict, allowed: set[str], path: str, problems: list[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        problems.append(f"{path}: unknown key(s): {sorted(unknown)}")


def _parse_agent(data: dict, path: str, problems: list[str]) -> AgentNode | None:
    _check_unknown(
        data, {"type", "id", "spec", "input", "inputs_from", "output_schema"}, path, problems
    )
    node_id = data.get("id")
    if not isinstance(node_id, str) or not node_id.strip():
        problems.append(f"{path}: agent node requires a non-empty string 'id'")
        node_id = node_id if isinstance(node_id, str) else "?"

    spec_obj: AgentSpec | None = None
    if "spec" not in data:
        problems.append(f"{path}: agent node requires 'spec'")
    else:
        try:
            spec_obj = AgentSpec.from_json(data["spec"])
        except (AgentSpecError, TypeError, ValueError) as exc:
            # AgentSpec.from_json raises TypeError when a required field
            # (name / system_prompt) is missing; fold it into our aggregate.
            problems.append(f"{path}.spec: {exc}")

    inp = data.get("input", "{{input}}")
    if not isinstance(inp, str):
        problems.append(f"{path}: 'input' must be a string template")
        inp = "{{input}}"

    inputs_from = data.get("inputs_from", [])
    if not (isinstance(inputs_from, list) and all(isinstance(x, str) for x in inputs_from)):
        problems.append(f"{path}: 'inputs_from' must be a list[str]")
        inputs_from = []

    out_schema = data.get("output_schema")
    if out_schema is not None and not isinstance(out_schema, dict):
        problems.append(f"{path}: 'output_schema' must be an object {{name, schema}}")
        out_schema = None

    if spec_obj is None:
        return None
    return AgentNode(
        type="agent",
        id=node_id,
        spec=spec_obj,
        input=inp,
        inputs_from=tuple(inputs_from),
        output_schema=out_schema,
    )


def _parse_sequence(data: dict, path: str, problems: list[str], ids: set[str]) -> SequenceNode | None:
    _check_unknown(data, {"type", "id", "steps"}, path, problems)
    steps_raw = data.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        problems.append(f"{path}: 'steps' must be a non-empty list")
        return None
    steps = [_parse_node(s, f"{path}.steps[{i}]", problems, ids) for i, s in enumerate(steps_raw)]
    if any(s is None for s in steps):
        return None
    return SequenceNode(type="sequence", id=data.get("id"), steps=tuple(s for s in steps if s))


def _parse_parallel(data: dict, path: str, problems: list[str], ids: set[str]) -> ParallelNode | None:
    _check_unknown(data, {"type", "id", "branches", "max_concurrency", "on_error"}, path, problems)
    branches_raw = data.get("branches")
    if not isinstance(branches_raw, list) or not branches_raw:
        problems.append(f"{path}: 'branches' must be a non-empty list")
        return None
    mc = _int_ge1(data.get("max_concurrency", 5), f"{path}.max_concurrency", problems)
    on_err = _on_error(data.get("on_error", "halt"), path, problems)
    branches = [
        _parse_node(b, f"{path}.branches[{i}]", problems, ids) for i, b in enumerate(branches_raw)
    ]
    if any(b is None for b in branches):
        return None
    return ParallelNode(
        type="parallel",
        id=data.get("id"),
        branches=tuple(b for b in branches if b),
        max_concurrency=mc,
        on_error=on_err,
    )


def _parse_foreach(data: dict, path: str, problems: list[str], ids: set[str]) -> ForeachNode | None:
    _check_unknown(
        data,
        {"type", "id", "items_from", "items", "as", "body", "parallel", "max_concurrency", "on_error"},
        path,
        problems,
    )
    as_var = data.get("as")
    if not isinstance(as_var, str) or not as_var.strip():
        problems.append(f"{path}: foreach requires a non-empty string 'as'")
        as_var = "item"

    items_from = data.get("items_from")
    items = data.get("items")
    if items_from is None and items is None:
        problems.append(f"{path}: foreach requires either 'items_from' or 'items'")
    if items_from is not None and items is not None:
        problems.append(f"{path}: foreach cannot have both 'items_from' and 'items'")
    if items_from is not None:
        if not isinstance(items_from, str):
            problems.append(f"{path}: 'items_from' must be a string 'node_id.key'")
        elif _ref_target(items_from) not in ids:
            problems.append(
                f"{path}: 'items_from' references unknown agent node "
                f"'{_ref_target(items_from)}' (known: {sorted(ids)})"
            )
    if items is not None and not isinstance(items, list):
        problems.append(f"{path}: 'items' must be a list")
        items = None

    mc = _int_ge1(data.get("max_concurrency", 5), f"{path}.max_concurrency", problems)
    par = data.get("parallel", True)
    if not isinstance(par, bool):
        problems.append(f"{path}: 'parallel' must be a boolean")
        par = True
    on_err = _on_error(data.get("on_error", "halt"), path, problems)

    body = None
    if "body" not in data:
        problems.append(f"{path}: foreach requires 'body'")
    else:
        body = _parse_node(data["body"], f"{path}.body", problems, ids)
    if body is None:
        return None
    return ForeachNode(
        type="foreach",
        id=data.get("id"),
        as_var=as_var,
        items_from=items_from if isinstance(items_from, str) else None,
        items=tuple(items) if isinstance(items, list) else None,
        body=body,
        parallel=par,
        max_concurrency=mc,
        on_error=on_err,
    )


def _parse_branch(data: dict, path: str, problems: list[str], ids: set[str]) -> BranchNode | None:
    _check_unknown(data, {"type", "id", "on", "cases", "default"}, path, problems)
    on = data.get("on")
    if not isinstance(on, str) or not on.strip():
        problems.append(f"{path}: branch requires a non-empty string 'on' ('node_id.key')")
        on = "?"
    elif _ref_target(on) not in ids:
        problems.append(
            f"{path}: 'on' references unknown agent node '{_ref_target(on)}' (known: {sorted(ids)})"
        )
    cases_raw = data.get("cases")
    cases: dict[str, WorkflowNode] = {}
    if not isinstance(cases_raw, dict) or not cases_raw:
        problems.append(f"{path}: branch requires a non-empty 'cases' object")
    else:
        for key, child in cases_raw.items():
            parsed = _parse_node(child, f"{path}.cases[{key!r}]", problems, ids)
            if parsed is not None:
                cases[str(key)] = parsed
    default = None
    if data.get("default") is not None:
        default = _parse_node(data["default"], f"{path}.default", problems, ids)
    if not cases:
        return None
    return BranchNode(type="branch", id=data.get("id"), on=on, cases=cases, default=default)


def _int_ge1(value: Any, path: str, problems: list[str]) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        problems.append(f"{path} must be an integer >= 1 (got {value!r})")
        return 1
    return value


def _on_error(value: Any, path: str, problems: list[str]) -> str:
    if value not in _ON_ERROR:
        problems.append(f"{path}: 'on_error' must be one of {sorted(_ON_ERROR)} (got {value!r})")
        return "halt"
    return value
