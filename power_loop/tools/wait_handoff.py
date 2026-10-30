"""Explicit SDK registration for finite-wait inspection and main decisions."""

from __future__ import annotations

from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime import wait_decisions
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobError
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import _SEND

WAIT_HANDOFF_DEFINITION = ToolDefinition(pricing=Free("local"),
    name="wait_handoff",
    description=("Review an existing task's deferred continuation request. inspect reads its original request "
        "and current time; field=note or continuation supports char_offset/max_chars for full text. "
        "decide records continue, defer or cancel for the original owning executor. defer requires a future "
        "next_review_at_ms inside the original deadline. A recorded decision is not a completed task, "
        "dispatched nudge or confirmed physical stop. Never create replacement work. Main session only."),
    input_schema={"type": "object", "properties": {
        "action": {"type": "string", "enum": ["inspect", "decide"]},
        "event_id": {"type": "string"},
        "decision": {"type": "string", "enum": ["continue", "defer", "cancel"]},
        "reason": {"type": "string", "maxLength": 1000},
        "next_review_at_ms": {"type": "integer", "minimum": 1},
        "field": {"type": "string", "enum": ["summary", "note", "continuation"]},
        "char_offset": {"type": "integer", "minimum": 0},
        "max_chars": {"type": "integer", "minimum": 256, "maximum": 16000},
    }, "required": ["action", "event_id"], "additionalProperties": False},
    required_params=("action", "event_id"),
)


async def run_wait_handoff(action="", **options):
    from power_loop.runtime.jobs import _json
    try:
        loop, sid, execution = get_current_loop(), get_session_id(), _SEND.get()
        if loop is None or execution is None or execution.runtime is not getattr(loop, "task_runtime", None) or execution.actor.session_id != sid:
            raise JobError("task_send_required")
        manager, actor = execution.runtime.manager, execution.actor
        allowed = ({"event_id", "field", "char_offset", "max_chars"} if action == "inspect" else
                   {"event_id", "decision", "reason", "next_review_at_ms"} if action == "decide" else set())
        if action not in ("inspect", "decide") or set(options) - allowed or "event_id" not in options:
            raise JobError("invalid_wait_handoff_arguments")
        if action == "decide" and "decision" not in options:
            raise JobError("invalid_wait_handoff_arguments")
        source = await manager.get_job(actor, execution.lease.job_id)
        if source.execution_role != "main":
            raise JobError("role_forbids_wait_decision")
        if action == "inspect":
            return _json(await wait_decisions.inspect(manager, actor, **options))
        seq, call_id = current_invocation(sid)
        if seq is None or not call_id:
            raise JobError("persisted_tool_call_required")
        request = {**options, "caller": execution.lease, "call_ref": await manager.store.message_ref(sid, seq),
                   "tool_call_id": call_id}
        try:
            operation, replayed = await wait_decisions.record_decision(manager, actor, **request)
        except Exception:
            operation = await wait_decisions.recall_decision(manager, actor, **request)
            if operation is None:
                raise
            replayed = True
        return _json({**operation.result["response"], "operation_id": operation.operation_id, "replayed": replayed})
    except JobError as exc:
        result = {"version": 1, "error": {"code": exc.code}}
        if exc.code.endswith(("evidence_corrupt", "evidence_missing", "receipt_missing")):
            result["control_state"] = "unknown"
        return _json(result)
    except Exception:
        return _json({"version": 1, "error": {"code": "wait_handoff_requires_reconciliation"}, "control_state": "unknown"})


def register_wait_handoff_tool(registry, *, overwrite=False):
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

    registry.register(WAIT_HANDOFF_DEFINITION, FunctionRuntimeTool(
        run_wait_handoff,
        ToolRuntimeContract("sdk.wait-review", allowed_roles=("main",), capabilities=("task.wait-review.v1",)),
        policy=lambda arguments: ToolDispatchPolicy(native_admitted=True, control_only=arguments.get("action") == "inspect"),
    ), overwrite=overwrite)
