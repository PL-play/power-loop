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
_ACTIONS = ("help", "validate", "create", "status", "list", "cancel", "leaves", "logs", "send")
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
    prepared = await prepare_workflow(options, ports)
    warning = await _observe(ports.on_compiled, ports, copy.deepcopy(prepared.spec.to_dict()))
    from power_loop.agent.runtime_types import RequestOptions, TargetRef, plain
    from power_loop.runtime.child_tasks import tool_actor

    from .runner import submit

    binding, actor = await tool_actor(loop)
    del binding
    client = loop._runtime_worker.runtime.client
    handle = await submit(
        client,
        TargetRef(loop.store.store_id, sid),
        prepared.spec,
        message_id="workflow:" + actor.tool_call_id,
        actor=actor,
        descriptor=loop._runtime_worker_context.descriptor,
        options=RequestOptions(tools=sorted(prepared.allowed_tools) if prepared.allowed_tools is not None else None),
    )
    receipt = {
        "version": 1,
        "status": "accepted",
        "run_id": handle.run_id,
        "workflow": prepared.spec.name,
        "target_ref": plain(handle.target),
        "input_ref": plain(handle.input_ref),
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
        "node_session_id": {"type": "string"},
        "message": {"type": "string"},
        "urgent": {"type": "boolean"},
        "reason": {"type": "string"},
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
        "preview_chars": {"type": "integer", "minimum": 100, "maximum": 4000},
    }
    if allow_inline_spec:
        properties["spec"] = {"type": "object"}
    return ToolDefinition(
        name="workflow",
        pricing=Free("composite"),
        description=(
            "Create asynchronous workflows; inspect saved tasks, leaves and messages; cancel or send a message to an exact node. "
            "logs lists a node's latest messages (default 20, at most 50, each shortened to preview_chars, default 500); "
            "source=projection lists its compact projected history; seq/id reads one record in full. "
            "Accepted is a receipt, not completion. Results arrive in the owner inbox. " + description_suffix
        ),
        input_schema={"type": "object", "properties": properties, "additionalProperties": False},
    )


def _dump(value):
    return json.dumps(value, ensure_ascii=False, default=str)


async def _owned_native_workflow(loop, sid, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"inp_[a-f0-9]{56}", run_id):
        raise WorkflowToolInputError("workflow_run_id_required")
    row = await loop.store._db.fetchone(
        f"SELECT * FROM {loop.store.t.sessions} WHERE parent_session_id=? "
        "AND session_kind='workflow_driver' AND start_input_message_id=?",
        (sid, run_id),
    )
    if row is None:
        raise WorkflowToolInputError("workflow_run_not_found")
    from . import journal

    saved = await journal.read(loop.store, sid, run_id)
    if saved is None:
        raise WorkflowToolInputError("workflow_run_not_found")
    from power_loop.agent.runtime_types import InputRef, TargetRef, TrustedActor

    actor = TrustedActor(row["tenant_id"], row["principal_id"], sid)
    target = TargetRef(loop.store.store_id, row["session_id"])
    await loop._runtime_worker.runtime.client.inspect(
        InputRef(target.store_id, target.session_id, run_id), actor=actor,
    )
    return row, saved, actor


def _select_node(saved, options):
    from power_loop.agent.runtime_types import TargetRef

    has_name = options.get("node") is not None
    has_session = options.get("node_session_id") is not None
    if has_name and has_session:
        raise WorkflowToolInputError("ambiguous_workflow_node_selector")
    driver = TargetRef(**saved["driver_target"])
    if not has_name and not has_session:
        return driver, None
    nodes = saved["nodes"]
    if has_session:
        matches = [entry for entry in nodes if entry["session_id"] == options["node_session_id"]]
    else:
        name = options["node"]
        if not isinstance(name, str) or not name.strip():
            raise WorkflowToolInputError("invalid_workflow_node")
        matches = [entry for entry in nodes if entry["path"] == name]
    targets = {entry["session_id"] for entry in matches}
    if not targets:
        raise WorkflowToolInputError("workflow_node_not_found")
    if len(targets) != 1:
        raise WorkflowToolInputError("workflow_node_ambiguous_choose_session")
    session_id = next(iter(targets))
    entry = max(matches, key=lambda item: (item["attempt"], item["continuation_index"]))
    return TargetRef(saved["driver_target"]["store_id"], session_id), entry


def _parse_int_option(value: Any, code: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise WorkflowToolInputError(code)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise WorkflowToolInputError(code)
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise WorkflowToolInputError(code) from exc
    else:
        parsed = value
    if minimum is not None and parsed < minimum:
        raise WorkflowToolInputError(code)
    return parsed


async def _read_logs(loop, target, actor, options):
    """Bounded raw/projected session reads with no task/history authority."""
    source = options.get("source", "messages")
    if source not in {"messages", "projection"}:
        raise WorkflowToolInputError("invalid_workflow_history_source")
    limit_raw = options.get("limit")
    limit = min(_parse_int_option(limit_raw, "invalid_workflow_limit", minimum=1), 50) if limit_raw is not None else 20
    preview_raw = options.get("preview_chars")
    preview = _parse_int_option(preview_raw, "invalid_workflow_preview_chars", minimum=1) if preview_raw is not None else 500
    output_file = options.get("output_file")
    if output_file is not None and (not isinstance(output_file, str) or not output_file.strip()):
        raise WorkflowToolInputError("workflow_export_path_invalid")
    if source == "projection" and options.get("seq") is not None:
        raise WorkflowToolInputError("invalid_workflow_logs_filter")

    where, params = ["session_id=?"], [target.session_id]
    filter_cols = (
        (("seq", "seq"), ("id", "id"), ("message_id", "id"), ("send_index", "send_index"))
        if source == "messages"
        else (("id", "id"), ("message_id", "id"), ("send_index", "send_index"))
    )
    for key, column in filter_cols:
        if options.get(key) is not None:
            where.append(column + "=?")
            params.append(_parse_int_option(options[key], f"invalid_workflow_{key}", minimum=1))
    if "message_ids" in options and options["message_ids"] is not None:
        raw_ids = options["message_ids"]
        if not isinstance(raw_ids, list) or not raw_ids or len(raw_ids) > 100:
            raise WorkflowToolInputError("invalid_workflow_message_ids")
        ids = [_parse_int_option(value, "invalid_workflow_message_ids", minimum=1) for value in raw_ids]
        where.append("id IN (" + ",".join("?" for _ in ids) + ")")
        params.extend(ids)
    has_cursor = options.get("cursor") is not None
    cursor = _parse_int_option(options["cursor"], "invalid_workflow_cursor", minimum=0) if has_cursor else None
    latest = not has_cursor and not any(
        options.get(key) is not None for key in ("seq", "id", "message_id", "message_ids")
    )

    client = loop._runtime_worker.runtime.client
    await client.inspect(target, actor=actor)
    store = loop.store
    table = store.t.messages if source == "messages" else store.t.project_messages
    coordinate = "seq" if source == "messages" else "id"
    if latest:
        rows = await store._db.fetchall(
            f"SELECT * FROM {table} WHERE {' AND '.join(where)} "
            f"ORDER BY {coordinate} DESC LIMIT ?",
            [*params, limit],
        )
        page_rows = list(reversed(rows))
        has_more = False
    else:
        selected = list(where)
        query_params = list(params)
        if cursor is not None:
            selected.append(f"{coordinate}>?")
            query_params.append(cursor)
        rows = await store._db.fetchall(
            f"SELECT * FROM {table} WHERE {' AND '.join(selected)} "
            f"ORDER BY {coordinate} ASC LIMIT ?",
            [*query_params, limit + 1],
        )
        has_more = len(rows) > limit
        page_rows = rows[:limit]
    records = []
    for row in page_rows:
        item = dict(row)
        for key in ("id", "seq", "send_index", "source_seq_lo", "source_seq_hi"):
            if item.get(key) is not None:
                item[key] = str(item[key])
        if output_file:
            records.append(item)
        else:
            content = item.get("content") or item.get("content_json") or item.get("rendered_text")
            records.append({
                key: item[key] for key in ("id", "seq", "send_index", "role", "name", "kind")
                if key in item
            } | {"preview": str(content or "")[:preview]})
    if output_file:
        from power_loop.runtime.env import RuntimeEnvError, get_runtime_env, safe_path

        try:
            path = safe_path(output_file, "w", env=get_runtime_env())
        except (ValueError, RuntimeEnvError) as exc:
            raise WorkflowToolInputError("workflow_export_path_invalid") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in records))
        return {"status": "exported", "path": output_file, "records": len(records)}
    return {
        "status": "ok" if records else "unavailable",
        "source": source,
        "records": records,
        "next_cursor": str(page_rows[-1][coordinate]) if has_more and page_rows else None,
    }


async def run_workflow_tool(arguments=None, *, ports=None, **kwargs):
    from power_loop.agent.runtime_types import RuntimeRequestError, plain, sha256_canonical
    from power_loop.runtime.child_tasks import tool_actor

    from . import journal

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
        if action == "create":
            # The public submission boundary performs the same role check.
            return _dump((await submit_workflow(options, ports)).receipt)
        await require_orchestration(loop, "workflow", sid=sid)
        if action == "validate":
            from .tool import _spec_summary

            prepared = await prepare_workflow(options, ports)
            return _dump({"status": "valid", "summary": _spec_summary(prepared.spec)})
        if action == "list" or action == "status" and not options.get("run_id"):
            run_ids = await journal.list_run_ids(loop.store, sid)
            offset = (
                _parse_int_option(options["cursor"], "invalid_workflow_cursor", minimum=0)
                if options.get("cursor") is not None
                else 0
            )
            limit = (
                min(_parse_int_option(options["limit"], "invalid_workflow_limit", minimum=1), 50)
                if options.get("limit") is not None
                else 20
            )
            selected = run_ids[offset:offset + limit]
            value = {
                "runs": [await journal.read(loop.store, sid, run_id) for run_id in selected],
                "next_cursor": str(offset + len(selected)) if offset + len(selected) < len(run_ids) else None,
            }
        else:
            _row, saved, actor = await _owned_native_workflow(loop, sid, options.get("run_id"))
            if action == "status":
                value = saved
            elif action == "leaves":
                value = {"nodes": saved["nodes"][:100], "truncated": len(saved["nodes"]) > 100}
            else:
                selected, entry = _select_node(saved, options)
                client = loop._runtime_worker.runtime.client
                if action == "logs":
                    value = await _read_logs(loop, selected, actor, options)
                elif action == "cancel":
                    if entry is None:
                        value = plain(await client.cancel(
                            selected, actor=actor, scope="subtree",
                            reason=options.get("reason", "cancelled"),
                        ))
                    else:
                        value = plain(await client.cancel(
                            selected, actor=actor, scope="subtree",
                            reason=options.get("reason", "cancelled"),
                        ))
                else:
                    if entry is None:
                        raise WorkflowToolInputError("workflow_node_required")
                    if not isinstance(options.get("message"), str):
                        raise WorkflowToolInputError("workflow_message_required")
                    _binding, trusted = await tool_actor(loop)
                    message_id = "inp_" + sha256_canonical([
                        "workflow_node_send", trusted.call_ref.to_dict(), trusted.tool_call_id,
                    ])[:56]
                    value = plain(await client.send(
                        selected, options["message"], message_id=message_id, actor=trusted,
                        priority="urgent" if options.get("urgent") is True else "normal",
                    ))
        return _dump(value)
    except (WorkflowToolInputError, WorkflowSpecError, RuntimeRequestError) as exc:
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
            control_only=(
                arguments.get("action", "help") in _CONTROL_ACTIONS
                and not (
                    arguments.get("action", "help") == "logs"
                    and arguments.get("output_file") not in (None, "")
                )
            ),
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
