"""Library-owned jobs schema and handlers; hosts may localize presentation."""
from __future__ import annotations

import json

from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime.jobs import JobError
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

JOBS_DEFINITION = ToolDefinition(
    name="jobs",
    description=("Inspect your authorized tasks and descendants, including saved history and result/message references. "
        "list defaults to active obligations, which can include a finished outcome with unresolved resources or supervision. "
        "get reads one job_id; view=operations lists its saved operation call/receipt references, "
        "view=results lists canonical result versions. view=attempts reads bounded physical model-call observations for one send job, without billing settlement. "
        "Select event_id for operations/results to read one saved event in character chunks, "
        "or output_file to export saved events as new workspace JSONL. Reads never acknowledge or execute them. "
        "stop requests durable cancellation for one job_id and its supervised subtree; acceptance is not physical stop. "
        "Use separate stop calls for separate roots. Never stop unrelated work. Legacy stores expose limited live best-effort information only."),
    input_schema={"type": "object", "properties": {
        "action": {"type": "string", "enum": ["list", "get", "stop"]},
        "job_id": {"type": "string", "description": "get/stop: a task ID or an authorized unambiguous alias."},
        "ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20,
            "description": "Legacy stop compatibility. Native stop uses one job_id; a single-element ids array is accepted as an alias."},
        "root_job_id": {"type": "string"},
        "scope": {"type": "string", "enum": ["self", "descendants"]},
        "status": {"oneOf": [{"type": "string", "enum": ["active", "terminal", "all"]}, {"type": "array", "items": {"type": "string"}}]},
        "kinds": {"type": "array", "items": {"type": "string"}},
        "billing_status": {"type": "string"},
        "created_after": {"type": "integer"}, "created_before": {"type": "integer"},
        "max_depth": {"type": "integer", "minimum": 0, "maximum": 63},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        "max_chars": {"type": "integer", "minimum": 1024, "maximum": 32000},
        "cursor": {"type": "string"},
        "view": {"type": "string", "enum": ["summary", "operations", "results", "attempts"], "description": "get only; attempts reads one send job's model observations (limit/cursor/max_chars only)."},
        "event_id": {"type": "string", "description": "get with view=operations/results: one exact event from the selected job."},
        "char_offset": {"type": "integer", "minimum": 0, "description": "get event_id: returned next_char_offset for the next chunk."},
        "output_file": {"type": "string", "description": "get view=operations/results: export selected saved events as JSONL to a new workspace file."},
        "reason": {"type": "string", "maxLength": 1000},
        "expected_generation": {"type": "integer", "minimum": 0},
    }, "required": ["action"], "additionalProperties": False},
    required_params=("action",),
)


def _json(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _error(code):
    result = {"version": 1, "error": {"code": code}}
    if code in ("stop_control_unconfirmed", "stop_requires_reconciliation"):
        result["control_state"] = "unknown"
    return result


def _ids(job_id, ids):
    if job_id is not None and ids is not None:
        raise JobError("ambiguous_stop_target")
    values = [job_id] if job_id is not None else ids
    if (not isinstance(values, list) or not 1 <= len(values) <= 20 or
            any(not isinstance(v, str) or not v.strip() or len(v) > 255 for v in values)):
        raise JobError("stop_target_required")
    return list(dict.fromkeys(v.strip() for v in values))


def _budget(options):
    value = options.get("max_chars", 8000)
    if type(value) is not int or not 1024 <= value <= 32000:
        raise JobError("invalid_query_max_chars")
    return value


async def _resolve(manager, actor, value, *, metadata_only=False):
    from functools import partial

    get_job = partial(manager.get_job, include_result=False) if metadata_only else manager.get_job
    if not isinstance(value, str) or not value:
        raise JobError("job_id_required")
    if value.startswith("job_"):
        await get_job(actor, value)
        return value
    if ":" not in value:
        operation = await manager.get_operation(actor, value if value.startswith("op_") else "op_"+value)
        job = await get_job(actor, operation.job_id)
        if job.kind not in ("workflow", "subagent"):
            raise JobError("job_alias_kind_mismatch")
        return job.job_id
    kind, _, tail = value.partition(":")
    if kind not in ("wf", "sub") or not tail:
        raise JobError("unknown_job_alias")
    operation = await manager.get_operation(actor, "op_"+tail)
    job = await get_job(actor, operation.job_id)
    if job.kind != {"wf": "workflow", "sub": "subagent"}[kind]:
        raise JobError("job_alias_kind_mismatch")
    return job.job_id


async def _native(loop, sid, action, options):
    from power_loop.runtime import task_control, task_query

    runtime = loop.task_runtime
    manager, actor = runtime.manager, runtime.actor(sid)
    if action == "list":
        allowed = {"root_job_id", "scope", "status", "kinds", "billing_status", "created_after", "created_before", "max_depth", "limit", "max_chars", "cursor"}
        if set(options) - allowed:
            raise JobError("invalid_jobs_parameters")
        if options.get("root_job_id") is not None:
            options["root_job_id"] = await _resolve(manager, actor, options["root_job_id"])
        budget = _budget(options)
        options["max_chars"] = max(1024, budget - 64)
        result = {"backend": "task_registry", **await task_query.list_jobs(manager, actor, **options)}
        if len(_json(result)) > budget:
            raise JobError("query_result_too_small")
        return result
    if action == "get":
        if set(options) - {"job_id", "max_chars", "view", "event_id", "char_offset", "limit", "cursor", "output_file"}:
            raise JobError("invalid_jobs_parameters")
        view = options.get("view", "summary")
        if view not in ("summary", "operations", "results", "attempts"):
            raise JobError("invalid_jobs_parameters")
        target = await _resolve(manager, actor, options.get("job_id"), metadata_only=view != "summary")
        budget = _budget(options)
        if view == "attempts":
            if set(options) - {"job_id", "view", "limit", "cursor", "max_chars"}:
                raise JobError("invalid_jobs_parameters")
            from power_loop.runtime.model_attempt_query import list_attempts
            return await list_attempts(manager, actor, target, **{k: v for k, v in options.items() if k not in ("job_id", "view")})
        if view in ("operations", "results"):
            from power_loop.runtime.task_events import read_events
            return await read_events(manager, actor, target, **{k: v for k, v in options.items() if k != "job_id"})
        if view != "summary" or set(options) - {"job_id", "max_chars", "view"}:
            raise JobError("invalid_jobs_parameters")
        effective = max(1024, budget - 64)
        result = {"backend": "task_registry", **await task_query.get_job(manager, actor, target, max_chars=effective)}
        if len(_json(result)) > budget:
            raise JobError("query_result_too_small")
        return result
    if set(options) - {"job_id", "ids", "reason", "expected_generation"}:
        raise JobError("invalid_jobs_parameters")
    targets = _ids(options.get("job_id"), options.get("ids"))
    if len(targets) != 1:
        raise JobError("native_stop_requires_single_job_id")
    target = await _resolve(manager, actor, targets[0])
    return await task_control.stop_from_current_call(loop, sid, target,
        reason=options.get("reason", "Stopped by request"), expected_generation=options.get("expected_generation"))


def resolve_legacy_job_id(value, known):
    if value in known:
        return value
    hits = [k for k in known if k.partition(":")[2] == value]
    return hits[0] if len(hits) == 1 else value


async def _legacy(loop, sid, action, options):
    options = dict(options)
    if action == "get" and options.get("view") == "summary":
        options.pop("view")
    if options.get("status") == "active":
        options.pop("status")
    if set(options) - {"job_id", "ids", "limit", "max_chars"}:
        raise JobError("task_registry_required")
    limit, budget = options.get("limit", 20), options.get("max_chars", 8000)
    if type(limit) is not int or not 1 <= limit <= 100 or type(budget) is not int or not 1024 <= budget <= 32000:
        raise JobError("invalid_jobs_limit")
    rows = [r for r in await loop.list_jobs(sid) if r["kind"] != "foreground"]
    known = {r["id"] for r in rows}
    base = {"version": 1, "backend": "legacy", "history_available": False, "hierarchy_available": False,
        "control_guarantee": "legacy_best_effort"}
    if action == "list":
        if options.get("job_id") is not None or options.get("ids") is not None:
            raise JobError("invalid_jobs_parameters")
        result = {**base, "nodes": [], "matched_count": 0, "truncated": False, "next_cursor": None}
        for row in rows[:limit]:
            trial = {**result, "nodes": [*result["nodes"], row], "matched_count": len(result["nodes"]) + 1}
            if len(_json(trial)) + 30 > budget:
                break
            result = trial
        result["truncated"] = len(result["nodes"]) < len(rows)
        return result
    if action == "get":
        if options.get("ids") is not None:
            raise JobError("invalid_jobs_parameters")
        if not isinstance(options.get("job_id"), str) or not options["job_id"]:
            raise JobError("job_id_required")
        jid = resolve_legacy_job_id(options["job_id"], known)
        row = next((r for r in rows if r["id"] == jid), None)
        if row is None:
            raise JobError("legacy_history_unavailable")
        result = {**base, "node": row}
        if len(_json(result)) > budget:
            raise JobError("query_result_too_small")
        return result
    wanted = _ids(options.get("job_id"), options.get("ids"))
    results = []
    for wanted_id in wanted:
        jid = resolve_legacy_job_id(wanted_id, known)
        if jid not in known:
            results.append({"id": jid, "result": "unknown"})
        else:
            results.append(await loop.stop_job(sid, jid))
    return {**base, "results": results}


async def run_jobs(action="", **kwargs):
    """Uniform JSON handler. Library code owns authorization/control semantics."""
    loop, sid = get_current_loop(), get_session_id()
    if loop is None or not sid:
        return _json(_error("active_session_required"))
    if action not in ("list", "get", "stop"):
        return _json(_error("invalid_jobs_action"))
    options = {k: v for k, v in kwargs.items() if v is not None}
    try:
        result = await (_native(loop, sid, action, options) if getattr(loop, "task_runtime", None) is not None else _legacy(loop, sid, action, options))
        return _json(result)
    except JobError as exc:
        return _json(_error(exc.code))
    except Exception:
        return _json(_error("stop_requires_reconciliation" if action == "stop" else "jobs_backend_unavailable"))


JOBS_HANDLER = FunctionRuntimeTool(
    run_jobs, ToolRuntimeContract("sdk.jobs"),
    policy=lambda arguments: ToolDispatchPolicy(native_admitted=True, control_only=True),
)


def register_jobs_tools(registry, *, overwrite=False, definition=None):
    """Explicit opt-in registration, following host include/role allowlists."""
    selected = definition or JOBS_DEFINITION
    if selected.name != "jobs" or selected.input_schema != JOBS_DEFINITION.input_schema:
        raise ValueError("jobs definition may localize description, not replace the schema")
    registry.register(selected, JOBS_HANDLER, overwrite=overwrite)
