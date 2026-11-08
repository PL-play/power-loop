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
from power_loop.runtime.pricing import Free

from .engine import in_workflow
from .runner import WorkflowRunHandle
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
    tool-grant factories are synchronous and run in the invoking context. View
    observers may be async, are bounded, and cannot invalidate admission.
    Completion is read from the durable workflow result.
    """

    load_spec: Callable[..., Any] | None = None
    spec_transform: Callable[[WorkflowSpec], WorkflowSpec] | None = None
    allowed_tools_factory: Callable[..., Any] | None = None
    on_compiled: Callable[..., Any] | None = None
    on_accepted: Callable[..., Any] | None = None
    allow_inline_spec: bool = True
    observer_timeout_s: float = 2.0

    def __post_init__(self):
        if type(self.allow_inline_spec) is not bool:
            raise TypeError("allow_inline_spec must be a bool")
        value = self.observer_timeout_s
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 30
        ):
            raise WorkflowToolInputError("invalid_workflow_observer_timeout")


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
    if set(options) - {"spec", "file", "args", "input"}:
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
    from power_loop.agent.runtime_types import RequestOptions, TargetRef, plain
    from power_loop.runtime.child_tasks import tool_actor

    from .runner import submit

    binding, actor = await tool_actor(loop)
    client = loop._runtime_worker.runtime.client
    handle = await submit(
        client,
        TargetRef(loop.store.store_id, sid),
        prepared.spec,
        message_id="workflow:" + actor.tool_call_id,
        actor=actor,
        descriptor=loop._runtime_worker_context.descriptor,
        parent=binding.lease,
        options=RequestOptions(tools=sorted(prepared.allowed_tools) if prepared.allowed_tools is not None else None),
    )
    receipt = {
        "version": 1,
        "status": "accepted",
        "run_id": handle.run_id,
        "workflow": prepared.spec.name,
        "job_id": handle.job_id,
        "job_ref": plain(handle.job_ref),
        "input_ref": plain(handle.input_ref),
        "operation_id": handle.operation_id,
        "completion_delivery": "owner_inbox",
        "view_warnings": [warning] if warning else [],
    }

    accepted = WorkflowAdmission(
        WorkflowSpec.from_json(copy.deepcopy(prepared.spec.to_dict())), handle, copy.deepcopy(receipt)
    )
    warning = await _observe(ports.on_accepted, ports, accepted)
    if warning:
        receipt["view_warnings"].append(warning)
    return WorkflowAdmission(prepared.spec, handle, receipt)


def workflow_tool_definition(*, allow_inline_spec=True, description_suffix=""):
    properties = {
        "action": {"type": "string", "enum": list(_ACTIONS)},
        "file": {"type": "string"},
        "args": {"type": "object"},
        "input": {"type": "string"},
        "run_id": {"type": "string"},
        "node": {"type": "string"},
        "node_job_id": {"type": "string"},
        "message": {"type": "string"},
        "reason": {"type": "string"},
        "expected_generation": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        "cursor": {"type": "string"},
        "max_chars": {"type": "integer", "minimum": 1024, "maximum": 32000},
        "detail": {"type": "boolean"},
        "source": {"type": "string", "enum": ["messages", "projection"]},
        "message_id": {"type": "string"},
        "message_ids": {"type": "array", "items": {"type": "string"}},
        "seq": {"type": "string"},
        "id": {"type": "string"},
        "send_index": {"type": "integer", "minimum": 1},
        "char_offset": {"type": "integer", "minimum": 0},
        "output_file": {"type": "string"},
    }
    if allow_inline_spec:
        properties["spec"] = {"type": "object"}
    return ToolDefinition(
        name="workflow",
        pricing=Free("composite"),
        description=(
            "Create asynchronous workflows; inspect saved tasks, leaves and messages; cancel or steer an exact node. "
            "Accepted is a receipt, not completion. Results arrive in the owner inbox. " + description_suffix
        ),
        input_schema={"type": "object", "properties": properties, "additionalProperties": False},
    )


def _dump(value):
    return json.dumps(value, ensure_ascii=False, default=str)


async def _owned_native_workflow(loop, sid, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{8,64}", run_id):
        raise WorkflowToolInputError("workflow_run_id_required")
    manager = loop.task_runtime.manager
    event = await manager.db.fetchone(
        f"SELECT job_id FROM {manager.events} WHERE event_id=? AND action='create'", ("op_" + run_id,)
    )
    if event is None:
        raise WorkflowToolInputError("workflow_run_not_found")
    job = await manager.get_job(loop.task_runtime.actor(sid), event["job_id"], include_result=False)
    if job.kind != "workflow":
        raise WorkflowToolInputError("workflow_run_not_found")
    return job


async def run_workflow_tool(arguments=None, *, ports=None, **kwargs):
    from power_loop.agent.runtime_types import JobRef, RuntimeRequestError, plain
    from power_loop.runtime import task_history, task_query
    from power_loop.runtime.child_tasks import tool_actor
    from power_loop.runtime.jobs import JobError

    ports = ports or WorkflowToolPorts()
    options = dict(arguments or kwargs)
    action = options.pop("action", "help")
    try:
        if action not in _ACTIONS:
            raise WorkflowToolInputError("invalid_workflow_action")
        if action == "help":
            return _dump({"actions": list(_ACTIONS), "execution": "asynchronous"})
        loop, sid = get_current_loop(), get_session_id()
        if loop is None or not sid:
            raise WorkflowToolInputError("active_session_required")
        await require_orchestration(loop, "workflow", sid=sid)
        if action == "create":
            return _dump((await submit_workflow(options, ports)).receipt)
        if action == "validate":
            from .tool import _spec_summary

            prepared = await prepare_workflow(options, ports)
            return _dump({"status": "valid", "summary": _spec_summary(prepared.spec)})
        manager, actor = loop.task_runtime.manager, loop.task_runtime.actor(sid)
        if action == "list" or action == "status" and not options.get("run_id"):
            value = await task_query.list_jobs(
                manager,
                actor,
                kinds=["workflow"],
                status="all",
                limit=options.get("limit", 20),
                cursor=options.get("cursor"),
                max_chars=options.get("max_chars", 8000),
            )
        else:
            job = await _owned_native_workflow(loop, sid, options.get("run_id"))
            if action == "status":
                value = await task_query.get_job(manager, actor, job.job_id, max_chars=options.get("max_chars", 8000))
            elif action == "leaves":
                rows = await manager.db.fetchall(
                    f"SELECT job_id,session_id,generation,title,status,outcome FROM {manager.jobs} "
                    "WHERE parent_job_id=? AND kind='workflow_node' ORDER BY job_id LIMIT 101",
                    (job.job_id,),
                )
                value = {"nodes": rows[:100], "truncated": len(rows) > 100}
            else:
                from .control import select_native_target

                selected = await select_native_target(loop.task_runtime, sid, job, options)
                node = await manager.get_job(actor, selected, include_result=False)
                if action == "logs":
                    value = await task_history.read_history(manager, actor, selected, **{
                        key: options[key] for key in ("source", "message_id", "message_ids", "seq", "id",
                            "send_index", "limit", "cursor", "char_offset", "max_chars", "output_file")
                        if key in options
                    })
                else:
                    _, trusted = await tool_actor(loop)
                    client = loop._runtime_worker.runtime.client
                    ref = JobRef(loop.store.store_id, selected, options.get("expected_generation", node.generation))
                    if action == "cancel":
                        value = plain(
                            await client.cancel(
                                ref, actor=trusted, scope="subtree", reason=options.get("reason", "cancelled")
                            )
                        )
                    else:
                        value = plain(
                            await getattr(client, action)(
                                ref, options["message"], actor=trusted, message_id=action + ":" + trusted.tool_call_id
                            )
                        )
        return _dump(value)
    except (WorkflowToolInputError, WorkflowSpecError, JobError, RuntimeRequestError) as exc:
        return _dump({"error": {"code": getattr(exc, "code", str(exc))}})


_CONTROL_ACTIONS = ("help", "status", "list", "cancel", "leaves", "logs")




def workflow_runtime_tool(handler, *, key="sdk.workflow"):
    """Declare this router's implemented native actions for a trusted host wrapper."""
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

    return FunctionRuntimeTool(
        handler,
        ToolRuntimeContract(key, background_allowed=False, allowed_roles=("main",)),
        policy=lambda arguments: ToolDispatchPolicy(
            native_admitted=arguments.get("action", "help") in _ACTIONS,
            control_only=arguments.get("action", "help") in _CONTROL_ACTIONS,
        ),
    )


def register_workflow_tool(registry, *, ports=None, overwrite=False, description_suffix=""):
    """Opt-in unified entry; no preset or role allowlist is broadened."""
    selected = ports or WorkflowToolPorts()

    @wraps(run_workflow_tool)
    async def handler(**arguments):
        return await run_workflow_tool(arguments, ports=selected)

    registry.register(
        workflow_tool_definition(allow_inline_spec=selected.allow_inline_spec, description_suffix=description_suffix),
        workflow_runtime_tool(handler),
        overwrite=overwrite,
    )
