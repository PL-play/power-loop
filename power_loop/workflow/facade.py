"""Shared model submission boundary, with explicit trusted host ports.

The low-level SDK can still await a Workflow. Model submissions always return
an admission receipt. No platform import, model-selected executor, or duplicate
live-handle registry belongs here.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any

from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_effective_tools, get_session_id
from power_loop.runtime.execution_roles import require_orchestration

from .api import create_workflow
from .engine import in_workflow
from .result import WorkflowRunHandle
from .spec import WorkflowSpec, WorkflowSpecError

log = logging.getLogger(__name__)
_ACTIONS = ("help", "validate", "create", "status", "list", "cancel", "leaves", "logs", "nudge", "steer")
_MAX_FILE_BYTES = 1024 * 1024


class WorkflowToolInputError(ValueError):
    """A public, library-owned input code; arbitrary host errors stay private."""


class WorkflowPolicyError(WorkflowSpecError):
    """Keep host policy failures distinct without a second validation path."""


@dataclass(frozen=True)
class WorkflowToolPorts:
    """Host-selected capabilities; never populated from model arguments.

    load_spec(file, args) may return a spec/dict or an awaitable. Policy and
    resource factories are synchronous and run in the invoking context. View
    observers may be async, are bounded, and cannot invalidate admission.
    Completion is the actual engine callback, not a presentation observer.
    """

    load_spec: Callable[..., Any] | None = None
    spec_transform: Callable[[WorkflowSpec], WorkflowSpec] | None = None
    executor_factory: Callable[..., Any] | None = None
    budget_factory: Callable[..., Any] | None = None
    file_io_factory: Callable[..., Any] | None = None
    allowed_tools_factory: Callable[..., Any] | None = None
    on_complete: Callable[..., Any] | None = None
    on_compiled: Callable[..., Any] | None = None
    on_accepted: Callable[..., Any] | None = None
    allow_inline_spec: bool = True
    observer_timeout_s: float = 2.0
    resource_adapter: tuple[str, str] | None = None

    def __post_init__(self):
        if type(self.allow_inline_spec) is not bool:
            raise TypeError("allow_inline_spec must be a bool")
        value = self.observer_timeout_s
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not 0 < value <= 30):
            raise WorkflowToolInputError("invalid_workflow_observer_timeout")
        if self.resource_adapter is not None:
            if (type(self.resource_adapter) is not tuple or len(self.resource_adapter) != 2 or
                    any(type(value) is not str or not value.strip() for value in self.resource_adapter)):
                raise WorkflowToolInputError("invalid_workflow_resource_adapter")
            if any(value is not None for value in (self.executor_factory, self.budget_factory, self.file_io_factory, self.on_complete)):
                raise WorkflowToolInputError("workflow_resource_ports_conflict")


@dataclass(frozen=True)
class PreparedWorkflow:
    spec: WorkflowSpec
    allowed_tools: frozenset[str] | None


@dataclass(frozen=True)
class WorkflowAdmission:
    spec: WorkflowSpec
    handle: WorkflowRunHandle
    receipt: dict[str, Any]


def _read_workspace(ref):
    from power_loop.runtime.env import get_runtime_env, safe_path

    if not isinstance(ref, str) or not ref.strip():
        raise WorkflowToolInputError("workflow_file_required")
    env = get_runtime_env()
    path = safe_path(ref, "r", env=env)
    # Workflow includes are workspace files. The general filesystem tool's
    # extra home/read-only roots do not implicitly grant workflow imports.
    if not path.is_relative_to(env.require_workspace_dir().resolve()):
        raise WorkflowToolInputError("workflow_file_outside_workspace")
    with path.open("rb") as source:
        raw = source.read(_MAX_FILE_BYTES + 1)
    if len(raw) > _MAX_FILE_BYTES:
        raise WorkflowToolInputError("workflow_file_too_large")
    return raw.decode("utf-8")


def _load_workspace_spec(ref, args):
    if not isinstance(ref, str) or not ref.endswith((".yaml", ".yml")):
        raise WorkflowToolInputError("workflow_yaml_file_required")
    return WorkflowSpec.from_yaml(_read_workspace(ref), args=args, read_workspace=_read_workspace)


async def prepare_workflow(arguments: Mapping[str, Any], ports: WorkflowToolPorts) -> PreparedWorkflow:
    """Freeze inputs, compile and apply the same policy for create/validate."""
    options = copy.deepcopy(dict(arguments))
    if set(options) - {"spec", "file", "args", "input", "detached", "wait", "sync"}:
        raise WorkflowToolInputError("invalid_workflow_parameters")
    # Capture before a host loader can await, and never let a host whitelist
    # accidentally widen the actual calling send's effective capabilities.
    ambient = get_effective_tools()
    captured = None if ambient is None else frozenset(ambient)
    has_spec, has_file = options.get("spec") is not None, options.get("file") is not None
    if has_spec == has_file:
        raise WorkflowToolInputError("workflow_requires_exactly_one_spec_or_file")
    if has_spec and not ports.allow_inline_spec:
        raise WorkflowToolInputError("workflow_file_required")
    args = options.get("args")
    if args is not None and not isinstance(args, dict):
        raise WorkflowToolInputError("workflow_args_must_be_object")
    if has_spec and args is not None:
        raise WorkflowToolInputError("workflow_args_require_file")
    if has_file:
        ref = options["file"]
        if not isinstance(ref, str) or not ref.strip():
            raise WorkflowToolInputError("workflow_file_required")
        payload = (ports.load_spec or _load_workspace_spec)(ref, args or {})
        if inspect.isawaitable(payload):
            payload = await payload
    else:
        payload = options["spec"]
    # Round-trip even a host-owned WorkflowSpec so mutations of its nested
    # dictionaries after this boundary cannot alter submitted assets/specs.
    spec = WorkflowSpec.from_json(copy.deepcopy(payload.to_dict() if isinstance(payload, WorkflowSpec) else payload))
    if "input" in options:
        if not isinstance(options["input"], str):
            raise WorkflowToolInputError("workflow_input_must_be_string")
        spec = replace(spec, input=options["input"])
    if ports.spec_transform is not None:
        try:
            spec = ports.spec_transform(spec)
        except WorkflowSpecError as exc:
            raise WorkflowPolicyError(exc.problems) from exc
        if not isinstance(spec, WorkflowSpec):
            raise TypeError("spec_transform must return a WorkflowSpec")
        spec = WorkflowSpec.from_json(copy.deepcopy(spec.to_dict()))
    if ports.allowed_tools_factory is not None:
        allowed = ports.allowed_tools_factory(get_current_loop(), get_session_id())
        if allowed is not None:
            if isinstance(allowed, (str, bytes)):
                raise TypeError("allowed_tools_factory must return tool names, not a string")
            allowed = frozenset(allowed)
            if any(not isinstance(name, str) or not name for name in allowed):
                raise TypeError("allowed_tools_factory returned invalid tool names")
            captured = allowed if captured is None else captured & allowed
    return PreparedWorkflow(spec, captured)


async def _observe(callback, ports, *args):
    if callback is None:
        return None
    try:
        async with asyncio.timeout(ports.observer_timeout_s):
            result = callback(*args)
            if inspect.isawaitable(result):
                await result
    except Exception:
        log.warning("workflow presentation observer failed", exc_info=True)
        return "presentation_observer_failed"
    return None


async def submit_workflow(arguments: Mapping[str, Any], ports: WorkflowToolPorts | None = None) -> WorkflowAdmission:
    """Admit once and return without awaiting business completion.

    Exceptions from actual storage/admission remain errors; the engine's durable
    operation handles uncertain ACKs. This layer never starts replacement work.
    """
    ports = ports or WorkflowToolPorts()
    options = copy.deepcopy(dict(arguments))
    loop, sid = get_current_loop(), get_session_id()
    if loop is None or not sid:
        raise WorkflowToolInputError("active_session_required")
    await require_orchestration(loop, "workflow", sid=sid)
    if in_workflow():
        raise WorkflowToolInputError("nested_workflow_forbidden")
    if any(key in options for key in ("wait", "sync")) or ("detached" in options and options["detached"] is not True):
        raise WorkflowToolInputError("workflow_async_only")
    # A control-only pipeline lane must never authorize source loading, resource
    # factories or presentation hooks for a new execution.
    from power_loop.runtime.task_runtime import check_dispatch
    await check_dispatch(sid)
    prepared = await prepare_workflow(options, ports)
    warning = await _observe(ports.on_compiled, ports, copy.deepcopy(prepared.spec.to_dict()))
    resource_binding = None
    if ports.resource_adapter is not None:
        from power_loop.runtime.jobs import JobError
        from power_loop.runtime.workflow_resources import WorkflowResourceScope, execution_contract
        runtime = getattr(loop, "task_runtime", None)
        resources = getattr(runtime, "workflow_resources", None)
        if resources is None:
            raise JobError("workflow_resource_registry_required")
        bound = await resources.prepare(*ports.resource_adapter,
            scope=WorkflowResourceScope(runtime.tenant_id, runtime.principal_id, sid),
            contract_sha256=execution_contract(prepared.spec, prepared.allowed_tools))
        resource_binding = bound.binding
    wf = create_workflow(prepared.spec, parent_loop=loop, parent_session_id=sid,
        executor=ports.executor_factory(loop, sid) if ports.executor_factory else None,
        budget=ports.budget_factory(loop, sid) if ports.budget_factory else None,
        file_io=ports.file_io_factory(loop, sid) if ports.file_io_factory else None,
        allowed_tools=prepared.allowed_tools, resource_binding=resource_binding)
    handle = await wf.start(detached=True, on_complete=ports.on_complete)
    if not isinstance(handle, WorkflowRunHandle):
        raise TypeError("asynchronous workflow admission must return a WorkflowRunHandle")
    native = getattr(loop, "task_runtime", None) is not None
    receipt = {"version": 1, "backend": "task_registry" if native else "legacy",
        "status": "accepted", "run_id": handle.run_id, "workflow": prepared.spec.name,
        "job_id": handle.job_id, "operation_id": handle.operation_id,
        "completion_delivery": "owner_inbox" if native else ("host_callback" if ports.on_complete else "durable_timer_requires_runner"),
        "view_warnings": [warning] if warning else []}
    accepted = WorkflowAdmission(WorkflowSpec.from_json(copy.deepcopy(prepared.spec.to_dict())), handle, copy.deepcopy(receipt))
    warning = await _observe(ports.on_accepted, ports, accepted)
    if warning:
        receipt["view_warnings"].append(warning)
    return WorkflowAdmission(prepared.spec, handle, receipt)


def workflow_tool_definition(*, allow_inline_spec=True, description_suffix=""):
    properties = {
        "action": {"type": "string", "enum": list(_ACTIONS)},
        "file": {"type": "string", "description": "Workspace YAML file; host may also grant skill: references."},
        "args": {"type": "object"}, "input": {"type": "string"},
        "run_id": {"type": "string"}, "detail": {"type": "boolean"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        "max_chars": {"type": "integer", "minimum": 1024, "maximum": 32000},
        "cursor": {"type": "string"},
        "reason": {"type": "string", "maxLength": 1000},
        "node": {"type": "string", "description": "One unambiguous existing node ID for logs/cancel/nudge/steer. Only cancel may omit all node selectors for the whole workflow."},
        "node_job_id": {"type": "string", "description": "Exact native workflow-node job when an ID has multiple routes. Mutually exclusive with node."},
        "expected_generation": {"type": "integer", "minimum": 0, "description": "Native cancellation only; legacy generations are unavailable."},
        "sid": {"type": "string", "description": "Exact session from leaves. Native nudge/steer require run_id and a node/node_job_id or this saved sid; optional sid further constrains the node."},
        "message": {"type": "string", "minLength": 1, "maxLength": 64000, "description": "nudge/steer: correction for the original node. Native control preserves its saved tools and rounds; do not pass rounds or force."},
        "source": {"type": "string", "enum": ["messages", "projection"], "description": "Native logs: saved original messages or projections."},
        "ids": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"oneOf": [{"type": "integer"}, {"type": "string"}]}, "description": "Native logs: inclusive seq range for messages, ID range for projection."},
        "seq": {"oneOf": [{"type": "integer"}, {"type": "string"}], "description": "Native logs: one original seq coordinate."},
        "id": {"oneOf": [{"type": "integer"}, {"type": "string"}], "description": "Native logs: one projection ID."},
        "message_id": {"oneOf": [{"type": "integer"}, {"type": "string"}], "description": "Native logs: one global raw-message ID. Decimal strings preserve precision."},
        "message_ids": {"type": "array", "minItems": 1, "maxItems": 100, "items": {"oneOf": [{"type": "integer"}, {"type": "string"}]}},
        "send_index": {"oneOf": [{"type": "integer"}, {"type": "string"}], "description": "Native logs: optional narrower send filter; cannot widen a send job's authority."},
        "char_offset": {"type": "integer", "minimum": 0, "description": "Native logs: next_char_offset for one record."},
        "output_file": {"type": "string", "description": "Native logs: export originals as JSONL to a new workspace file. Omit cursor/char_offset."},
        "tail": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Legacy logs preview only: latest assistant rounds, default 40. Native logs use limit/coordinates."},
        "verbose": {"type": "boolean", "description": "Legacy logs preview only: longer argument/result excerpts."},
    }
    if allow_inline_spec:
        properties["spec"] = {"type": "object", "description": "WorkflowSpec JSON. Select exactly one spec or file."}
    return ToolDefinition(name="workflow", description=(
        "Create and inspect deterministic agent workflows. create always admits asynchronous work; "
        "the returned accepted receipt is not its result. Use validate for compilation/policy checks without execution. "
        "list/status read saved tasks; status without run_id lists owned workflows. They do not start agents. Results require actual completion and owner consumption. "
        "cancel requests stopping the original workflow or a selected node; acceptance is not physical stop. "
        "Native nudge/steer send a correction to the original node through its owning engine; an accepted receipt is not processing/completion. "
        "They preserve the original contract and never start a substitute session. Old stores require a host control adapter. "
        "leaves lists owned node/session bindings; legacy responses identify their compatibility scope. "
        "logs reads retained node history without running agents: native uses run_id plus node/node_job_id; "
        "legacy accepts an owned sid or run_id plus node. Use limit/cursor/max_chars or exact coordinates, "
        "or legacy-only tail/verbose for a round preview. "
        "Main agents may orchestrate; nested workflows are forbidden. " + description_suffix),
        input_schema={"type": "object", "properties": properties, "additionalProperties": False},
        required_params=(), interrupt="background")


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


async def run_workflow_tool(arguments: Mapping[str, Any], *, ports: WorkflowToolPorts | None = None) -> str:
    """Library action router. Host ports are supplied by registration only."""
    ports = ports or WorkflowToolPorts()
    options = copy.deepcopy(dict(arguments))
    action = options.pop("action", "help")
    if action not in _ACTIONS:
        return _dump({"version": 1, "error": {"code": "invalid_workflow_action"}})
    allowed = {"create": {"spec", "file", "args", "input", "wait", "sync", "detached"},
        "validate": {"spec", "file", "args", "input"}, "help": set(),
        "status": {"run_id", "detail", "max_chars"}, "list": {"limit", "max_chars", "cursor"},
        "cancel": {"run_id", "reason", "expected_generation", "node", "node_job_id"},
        "nudge": {"run_id", "node", "node_job_id", "sid", "message"},
        "steer": {"run_id", "node", "node_job_id", "sid", "message"},
        "leaves": {"run_id", "limit", "max_chars", "cursor"},
        "logs": {"run_id", "node", "node_job_id", "sid", "source", "ids", "seq", "id", "message_id", "message_ids",
            "send_index", "limit", "cursor", "char_offset", "max_chars", "output_file", "tail", "verbose"}}[action]
    if set(options) - allowed:
        return _dump({"version": 1, "error": {"code": "invalid_workflow_parameters"}})
    try:
        if action == "help":
            return _dump({"version": 1, "actions": list(_ACTIONS), "execution": "asynchronous", "source": "file_or_spec" if ports.allow_inline_spec else "file"})
        loop, sid = get_current_loop(), get_session_id()
        if loop is None or not sid:
            raise WorkflowToolInputError("active_session_required")
        await require_orchestration(loop, "workflow", sid=sid)
        if action == "create":
            return _dump((await submit_workflow(options, ports)).receipt)
        if action == "validate":
            from power_loop.runtime.task_runtime import check_dispatch

            from .tool import _spec_summary
            await check_dispatch(sid)
            prepared = await prepare_workflow(options, ports)
            return _dump({"version": 1, "status": "valid", "summary": _spec_summary(prepared.spec)})
        if action == "cancel":
            return _dump(await _cancel(loop, sid, options))
        if action in ("nudge", "steer"):
            return _dump(await _nudge(loop, sid, options))
        if action in ("leaves", "logs"):
            return _dump(await _history(loop, sid, action, options))
        return _dump(await _inspect(loop, sid, action, options))
    except WorkflowSpecError as exc:
        return _dump({"version": 1, "error": {"code": "invalid_workflow_spec", "problems": list(exc.problems)}})
    except ValueError as exc:
        # Known contract codes only. Filesystem/decode/backend details are not
        # model-visible; arbitrary host exceptions must not leak credentials.
        from power_loop.runtime.jobs import JobError
        roles = {"execution_role_unavailable", "execution_role_conflict", "execution_role_forbids_workflow", "invalid_delegation_policy"}
        code = (getattr(exc, "code", str(exc)) if isinstance(exc, (WorkflowToolInputError, JobError)) or str(exc) in roles
            else "workflow_control_requires_reconciliation" if action in ("nudge", "steer") else "workflow_input_rejected")
        result = {"version": 1, "error": {"code": code}}
        if code in ("stop_control_unconfirmed", "stop_requires_reconciliation", "workflow_control_requires_reconciliation"):
            result["control_state"] = "unknown"
        return _dump(result)
    except Exception:
        result = {"version": 1, "error": {"code": "workflow_admission_requires_reconciliation" if action == "create" else
            "stop_requires_reconciliation" if action == "cancel" else
            "workflow_control_requires_reconciliation" if action in ("nudge", "steer") else "workflow_backend_unavailable"}}
        if action in ("cancel", "nudge", "steer"):
            result["control_state"] = "unknown"
        return _dump(result)


async def _owned_native_workflow(loop, sid, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{8,64}", run_id):
        raise WorkflowToolInputError("workflow_run_id_required")
    manager, actor = loop.task_runtime.manager, loop.task_runtime.actor(sid)
    operation = await manager.get_operation(actor, "op_"+run_id)
    job = await manager.get_job(actor, operation.job_id, include_result=False)
    if job.kind != "workflow" or job.created_by_session_id != sid or operation.action != "create":
        raise WorkflowToolInputError("workflow_run_not_owned")
    return job


async def _nudge(loop, sid, options):
    if getattr(loop, "task_runtime", None) is None:
        raise WorkflowToolInputError("legacy_workflow_control_requires_host")
    from power_loop.runtime.task_runtime import check_dispatch

    from .control import nudge
    await check_dispatch(sid)
    workflow = await _owned_native_workflow(loop, sid, options.get("run_id"))
    return await nudge(loop, sid, workflow, options)


async def _cancel(loop, sid, options):
    from power_loop.runtime.task_control import stop_from_current_call

    run_id = options.get("run_id")
    if getattr(loop, "task_runtime", None) is not None:
        job = await _owned_native_workflow(loop, sid, run_id)
        target = await _native_cancel_target(loop.task_runtime, sid, job, options)
        return {"run_id": run_id, **await stop_from_current_call(loop, sid, target,
            reason=options.get("reason", "Stopped by request"), expected_generation=options.get("expected_generation"))}
    from .runner import request_owned_node_stop, request_owned_stop
    if "node" in options and "node_job_id" in options:
        raise WorkflowToolInputError("ambiguous_workflow_node_selector")
    if "node_job_id" in options:
        raise WorkflowToolInputError("legacy_node_job_unavailable")
    if "node" in options:
        return await request_owned_node_stop(loop, sid, run_id, options["node"],
            reason=options.get("reason", "Stopped by request"), expected_generation=options.get("expected_generation"))
    return await request_owned_stop(loop, sid, run_id, reason=options.get("reason", "Stopped by request"),
        expected_generation=options.get("expected_generation"))


async def _native_cancel_target(runtime, sid, workflow, options):
    from .control import select_native_target
    return await select_native_target(runtime, sid, workflow, options)


async def _history(loop, sid, action, options):
    from power_loop.runtime import task_history, task_query, task_sessions

    runtime = getattr(loop, "task_runtime", None)
    if runtime is None:
        from power_loop.runtime.jobs import JobError

        from .legacy_history import leaves, logs
        try:
            return await (leaves(loop, sid, options) if action == "leaves" else logs(loop, sid, options))
        except JobError:
            raise
        except Exception as exc:
            raise WorkflowToolInputError("legacy_workflow_history_backend_unavailable") from exc
    workflow = await _owned_native_workflow(loop, sid, options.get("run_id"))
    budget = options.get("max_chars", 8000)
    if type(budget) is not int or not 1024 <= budget <= 32000:
        raise WorkflowToolInputError("invalid_query_max_chars")
    if action == "logs":
        if "tail" in options or "verbose" in options:
            raise WorkflowToolInputError("native_workflow_logs_use_limit_and_coordinates")
        if "node" not in options and "node_job_id" not in options:
            raise WorkflowToolInputError("workflow_history_node_required")
        node = await _native_cancel_target(runtime, sid, workflow, options)
        return await task_history.read_history(runtime.manager, runtime.actor(sid), node,
            session_id=options.get("sid"), **{k: v for k, v in options.items() if k not in ("run_id", "node", "node_job_id", "sid")})
    listing = await task_query.list_jobs(runtime.manager, runtime.actor(sid), root_job_id=workflow.job_id,
        kinds=["workflow_node"], status="all", limit=options.get("limit", 20), cursor=options.get("cursor"), max_chars=max(1024, budget-512))
    nodes = []
    for row in listing["nodes"]:
        if row["context_only"]:
            continue
        binding = await task_sessions.list_sessions(runtime.manager, runtime.actor(sid), row["job_id"])
        nodes.append({key: row[key] for key in ("job_id", "parent_job_id", "title", "generation", "status", "outcome", "resource_state")} |
            {"bindings": binding["bindings"], "reason": binding["reason"]})
    result = {"version": 1, "backend": "task_registry", "run_id": options["run_id"], "workflow_job_id": workflow.job_id,
        "nodes": nodes, "next_cursor": listing["next_cursor"], "truncated": listing["truncated"],
        "snapshot_at": listing["snapshot_at"], "execution_authority": False}
    if len(_dump(result)) > budget:
        raise WorkflowToolInputError("workflow_history_budget_too_small_reduce_limit")
    return result


async def _inspect(loop, sid, action, options):
    from .introspect import get_workflow, list_workflows

    budget = options.get("max_chars", 8000)
    if type(budget) is not int or not 1024 <= budget <= 32000:
        raise WorkflowToolInputError("invalid_query_max_chars")
    if "detail" in options and type(options["detail"]) is not bool:
        raise WorkflowToolInputError("invalid_workflow_detail")
    if action == "status" and options.get("run_id") in (None, ""):
        action = "list"
        options = {key: value for key, value in options.items() if key not in ("run_id", "detail")}
    runtime = getattr(loop, "task_runtime", None)
    if runtime is not None:
        from power_loop.runtime import task_query
        actor, manager = runtime.actor(sid), runtime.manager
        if action == "list":
            result = {"backend": "task_registry", **await task_query.list_jobs(manager, actor, kinds=["workflow"], status="all",
                limit=options.get("limit", 20), cursor=options.get("cursor"), max_chars=max(1024, budget-64))}
            if len(_dump(result)) > budget:
                raise WorkflowToolInputError("query_result_too_small")
            return result
        job = await _owned_native_workflow(loop, sid, options.get("run_id"))
        result = {"backend": "task_registry", **await task_query.get_job(manager, actor, job.job_id, max_chars=max(1024, budget-64))}
        if len(_dump(result)) > budget:
            raise WorkflowToolInputError("query_result_too_small")
        return result
    base = {"version": 1, "backend": "legacy", "state_authority": "compatibility_journal", "physical_state_confirmed": False}
    if action == "list":
        if options.get("cursor") is not None:
            raise WorkflowToolInputError("legacy_workflow_cursor_unavailable")
        limit = options.get("limit", 20)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise WorkflowToolInputError("invalid_query_limit")
        rows = await list_workflows(loop, sid)
        result = {**base, "runs": rows[:limit], "truncated": len(rows) > limit}
    else:
        run_id = options.get("run_id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{8,64}", run_id):
            raise WorkflowToolInputError("workflow_run_id_required")
        detail = options.get("detail", False)
        if type(detail) is not bool:
            raise WorkflowToolInputError("invalid_workflow_detail")
        row = await get_workflow(loop, sid, run_id, detail=detail)
        if row is None:
            raise WorkflowToolInputError("workflow_run_not_found")
        result = {**base, "run": row}
    if len(_dump(result)) > budget:
        raise WorkflowToolInputError("query_result_too_small")
    return result


def workflow_runtime_tool(handler, *, key="sdk.workflow"):
    """Declare this router's implemented native actions for a trusted host wrapper."""
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

    return FunctionRuntimeTool(handler, ToolRuntimeContract(key, background_allowed=False, allowed_roles=("main",)),
        policy=lambda arguments: ToolDispatchPolicy(
            native_admitted=arguments.get("action", "help") in _ACTIONS,
            control_only=arguments.get("action", "help") in ("help", "status", "list", "cancel", "leaves", "logs")))


def register_workflow_tool(registry, *, ports=None, overwrite=False, description_suffix=""):
    """Opt-in unified entry; no preset or role allowlist is broadened."""
    selected = ports or WorkflowToolPorts()

    @wraps(run_workflow_tool)
    async def handler(**arguments):
        return await run_workflow_tool(arguments, ports=selected)

    registry.register(workflow_tool_definition(allow_inline_spec=selected.allow_inline_spec,
        description_suffix=description_suffix), workflow_runtime_tool(handler), overwrite=overwrite)
