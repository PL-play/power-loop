"""Workflow adapters for the opt-in task runtime; no platform dependency.

Execution evidence precedes the legacy journal/inbox/wake. This does not yet
provide B2's final completion decision or B3's cross-process resume protocol.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .invocation import current_invocation
from .jobs import JobError
from .spec import _CHILD_ADMISSION
from .task_runtime import _SEND, Execution

log = logging.getLogger(__name__)
ROUTE_KEY = "\x00task-route"
_WORKFLOW: ContextVar[Execution | None] = ContextVar("power_loop_task_workflow", default=None)


@dataclass
class NodeExecution:
    binding: Execution
    route: list
    node_id: str


_NODE: ContextVar[NodeExecution | None] = ContextVar("power_loop_task_node", default=None)


def current_workflow(runtime):
    binding = _WORKFLOW.get()
    return binding if binding is not None and binding.runtime is runtime else None


def node_metadata(runtime):
    node = _NODE.get()
    if node is None or node.binding.runtime is not runtime:
        raise JobError("task_node_admission_required")
    return {"task_owner_job_id": node.binding.lease.job_id, "task_node_route": node.route}


def validate_new_workflow(spec, *, eager_wake=False):
    from power_loop.workflow.engine import WorkflowEngine
    from power_loop.workflow.spec import LoopNode
    if eager_wake:
        raise JobError("task_adapter_unavailable:eager_workflow_wake")
    def check(node):
        if isinstance(node, LoopNode) and node.continue_nodes:
            raise JobError("task_adapter_unavailable:workflow_session_rebind")
        for child in WorkflowEngine._children(node):
            check(child)
        if getattr(node, "fallback", None) is not None:
            check(node.fallback)
    check(spec.root)


async def admit_workflow(runtime, sid, request):
    execution = _SEND.get()
    if execution is None or execution.runtime is not runtime or execution.actor.session_id != sid:
        raise JobError("task_send_required")
    assistant_seq, call_id = current_invocation(sid)
    if assistant_seq is None or not call_id:
        raise JobError("persisted_tool_call_required")
    return await runtime.manager.register_job(execution.actor, kind="workflow", title=request["spec"]["name"][:200],
        parent=execution.lease, independent=True,
        call_ref=await runtime.manager.store.message_ref(sid, assistant_seq), tool_call_id=call_id,
        request=request)


@asynccontextmanager
async def node_execution(runtime, engine, node, env, driver_sid, *, cancel_token=None):
    from power_loop.workflow.spec import _node_to_dict

    workflow = current_workflow(runtime)
    if workflow is None or workflow is not engine._task_execution:
        raise JobError("task_workflow_admission_required")
    route = list(env.get(ROUTE_KEY, []))
    key = hashlib.sha256(json.dumps([node.id, route], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    actor = runtime.actor(driver_sid)
    from .node_resume import _PREPARED
    prepared = _PREPARED.get()
    restore_token = None
    if prepared is None and getattr(engine, "_restore_operation_id", None):
        import secrets
        from types import SimpleNamespace

        from .spec import workflow_continuation_loop
        from .workflow_restore_execution import authority, claim_node
        token = secrets.token_hex(16)
        cancelled = None
        try:
            lease, sid = await claim_node(runtime, workflow, node.id, driver_sid, token)
        except BaseException as exc:
            # A local ACK loss may be reconciled only with our exact claim.
            source = engine._restore_request["plan"]["nodes"][node.id]
            row = await runtime.manager._row(runtime.manager.db, source["lease"]["job_id"])
            grant, _ = await authority(runtime.manager, runtime.manager.db, row, source["binding"]["session_id"])
            if grant["token"] != token:
                raise
            from .jobs import JobLease
            lease, sid = JobLease(**grant["node"]), grant["session_id"]
            if isinstance(exc, asyncio.CancelledError):
                cancelled = exc
        source = engine._restore_request["plan"]["nodes"][node.id]
        restored_binding = Execution(runtime, actor, lease)
        try:
            if cancelled is not None:
                raise cancelled
            child = await workflow_continuation_loop(engine._loop, sid, expected_contract=source["contract"])
            try:
                await runtime.manager.bind_node_session(actor, lease, sid)
            except Exception:
                await runtime.manager.check_execution(actor, lease)
                grant, _ = await authority(runtime.manager, runtime.manager.db,
                    await runtime.manager._row(runtime.manager.db, lease.job_id), sid)
                if grant["token"] != token:
                    raise JobError("workflow_node_authority_mismatch") from None
        except BaseException as exc:
            from power_loop.workflow.result import AgentResult
            result = AgentResult(node_id=node.id, session_id=sid, text="", status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", error=str(exc))
            saved = await runtime.record_scope(restored_binding, result.to_dict())
            result.status, result.usage = saved["status"], saved.get("usage", {})
            engine._results[node.id] = result
            raise
        prepared = SimpleNamespace(engine=engine, node_id=node.id, binding=restored_binding,
            sid=sid, child=child, contract=source["contract"], pending=False, restored=True)
    if prepared is not None:
        if prepared.engine is not engine or prepared.node_id != node.id or prepared.binding.actor != actor:
            raise JobError("node_resume_execution_mismatch")
        binding = prepared.binding
        if getattr(prepared, "restored", False):
            engine._task_nodes[prepared.sid] = binding
            restore_token = _PREPARED.set(prepared)
    else:
        registration = await runtime.manager.register_job(actor, kind="workflow_node", parent=workflow.lease,
            system_key="node:" + key, title=node.id[:200], request={"node": _node_to_dict(node), "route": route})
        if registration.replayed:
            raise JobError("task_node_requires_reconciliation")
        binding = await runtime.claim(actor, registration)

    async def admitted(sid):
        await runtime.manager.bind_node_session(actor, binding.lease, sid)
        engine._task_nodes[sid] = binding

    token = _NODE.set(NodeExecution(binding, route, node.id))
    admission_token = _CHILD_ADMISSION.set(admitted)
    try:
        async with runtime.execution(binding, owner=True, cancel_token=cancel_token):
            yield binding
    except BaseException as exc:
        if not binding.stop_renewal.is_set() or getattr(exc, "code", None) in ("required_execution_unsettled", "required_result_unhandled", "required_input_unhandled"):
            saved = await runtime.record_scope(binding, {**getattr(exc, "execution_result", {}), "node_id": node.id,
                "status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", "error_type": type(exc).__name__})
            prior = engine._results.get(node.id)
            if prior is not None:
                prior.status = saved["status"]
                prior.error = str(exc)
                prior.usage = saved.get("usage", prior.usage)
        raise
    finally:
        for sid, active in list(engine._task_nodes.items()):
            if active is binding:
                engine._task_nodes.pop(sid, None)
        _CHILD_ADMISSION.reset(admission_token)
        _NODE.reset(token)
        if restore_token is not None:
            _PREPARED.reset(restore_token)


async def collect_node(runtime, loop, raw, stop_event, *, timeout_s=1800, max_rounds=30):
    """Consume required child results on their original model session, bounded."""
    from power_loop.workflow.admission import idle

    from . import delegation
    from .spec import workflow_continuation_loop

    sid = raw.get("session_id")
    if not sid:
        return raw
    store = runtime.manager.store
    deadline = asyncio.get_running_loop().time() + timeout_s
    remaining = max_rounds
    while True:
        async with idle():
            await delegation.await_children(store, sid, stop_event, max(0, deadline - asyncio.get_running_loop().time()))
        if raw.get("status") != "completed" or stop_event.is_cancelled():
            return raw
        pending = (await store.inbox_counts(sid))["pending"]
        if not pending and not raw.get("new_input_after_pass"):
            return raw
        if remaining <= 0:
            raise JobError("node_settlement_round_limit")
        child = await workflow_continuation_loop(loop, sid)
        if raw.get("new_input_after_pass"):
            work = child.follow_up("Process the new input already in this session and finish the original task. Do not repeat completed actions.",
                                   session_id=sid, stop_event=stop_event, max_rounds=remaining)
        else:
            work = child.flush_follow_ups(sid, stop_event=stop_event, max_rounds=remaining)
        more = await asyncio.wait_for(work, max(0, deadline - asyncio.get_running_loop().time()))
        if more is None:
            raise JobError("node_settlement_missing_result")
        remaining -= max(1, int(more.rounds))
        usage = dict(raw.get("usage") or {})
        for key, value in (more.usage or {}).items():
            usage[key] = usage.get(key, 0) + value
        raw = {**raw, "status": more.status, "final_text": more.final_text,
               "rounds": int(raw.get("rounds") or 0) + int(more.rounds), "usage": usage,
               "new_input_after_pass": bool(more.new_input_after_pass)}


async def ensure_node_answer(runtime, loop, raw, stop_event, *, timeout_s=30):
    """A completed text node must actually deliver text, not only pass_turn.reason."""
    from types import SimpleNamespace

    from .subagent_output import ensure_answer
    from .task_runtime import _OWNER

    binding = _OWNER.get()
    if (raw.get("status") != "completed" or (raw.get("final_text") or "").strip() or
            binding is None or binding.runtime is not runtime or binding.lease.generation != 1):
        return raw
    sid = raw.get("session_id")
    if stop_event.is_cancelled():
        return {**raw, "status": "cancelled"}
    key = "node_answer:" + str(sid)
    if not sid or binding.resource_evidence.get(key):
        return {**raw, "status": "failed", "error": "node_output_missing"}
    # The original engine owns this task. Mark before any await so re-entry
    # cannot allocate a second supplement. Restart admission remains closed.
    binding.resource_evidence[key] = True
    running = SimpleNamespace(task_binding=binding, answer_attempted=False, token=stop_event)
    try:
        value = await ensure_answer(loop, running, sid, raw, asyncio.get_running_loop().time()+timeout_s)
    except Exception as exc:
        return {**raw, "status": "failed", "error": "node_output_completion_failed:"+type(exc).__name__}
    if value.get("error") == "subagent_output_missing":
        value = {**value, "error": "node_output_missing"}
    if value.get("status") == "completed" and not (value.get("final_text") or "").strip():
        return {**value, "status": "cancelled" if stop_event.is_cancelled() else "failed", "error": "node_output_missing"}
    return value


async def execute_background(loop, parent_sid, run_id, registration, build_engine, run_spec, on_complete):
    from power_loop.workflow import runner
    from power_loop.workflow.result import WorkflowResult

    runtime = loop.task_runtime

    binding = None

    async def deliver(result, saved):
        from .workflow_results import project
        result.status = saved["status"]
        await project(loop, runtime.actor(parent_sid), registration.job.job_id,
                      binding.lease.generation if binding else 0, lease=binding.lease if binding else None)
        if on_complete is not None and saved["status"] != "cancelled" and run_id not in runner._STOPPED_RUNS:
            from .workflow_results import run_callback
            await run_callback(loop, runtime.actor(parent_sid), registration.job.job_id,
                               binding.lease.generation if binding else 0, on_complete)

    def failed(exc):
        return WorkflowResult(name=run_spec.name,
            status="cancelled" if isinstance(exc, asyncio.CancelledError) or getattr(exc, "code", None) == "cancel_requested" else "failed",
            errors=[f"{type(exc).__name__}: {exc}"])

    try:
        binding = await runtime.claim(runtime.actor(parent_sid), registration)
    except BaseException as exc:
        try:
            result = failed(exc)
            saved = await runtime.manager.record_unstarted_result(runtime.actor(parent_sid), registration.job.job_id,
                outcome=result.status, result=result.to_dict())
            await deliver(result, saved)
        except BaseException:
            # A claim may have committed before losing its ACK. Only generation
            # zero can close as unstarted; do not invent stop/completion proof.
            log.exception("workflow claim requires reconciliation: %s", run_id)
        return
    binding.resource_evidence["workflow_run_id"] = run_id
    engine = None
    try:
        async with runtime.execution(binding, context=_WORKFLOW):
            try:
                import inspect
                candidate = build_engine()
                engine = await candidate if inspect.isawaitable(candidate) else candidate
                engine._task_execution = binding
                runner._ENGINES[run_id] = engine
                result = await engine.run(run_spec)
            except BaseException as exc:
                result = engine._build_result(run_spec, failed(exc).status) if engine is not None else failed(exc)
                result.errors.append(f"{type(exc).__name__}: {exc}")
            saved = await runtime.record_scope(binding, result.to_dict())
            # Canonical result is already committed before compatibility views.
            await deliver(result, saved)
    except BaseException as exc:
        # An ambiguous result commit/projected completion must never be rewritten.
        if not binding.stop_renewal.is_set() or getattr(exc, "code", None) in ("required_execution_unsettled", "required_result_unhandled", "required_input_unhandled"):
            try:
                result = failed(exc)
                saved = await runtime.record_scope(binding, result.to_dict())
                await deliver(result, saved)
                return
            except BaseException:
                log.exception("workflow failure evidence requires reconciliation: %s", run_id)
        log.exception("workflow execution/delivery requires reconciliation: %s", run_id)
    finally:
        if runner._ENGINES.get(run_id) is engine:
            runner._ENGINES.pop(run_id, None)
