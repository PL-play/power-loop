"""Owned asynchronous native restoration, preserving workflow/node identities."""
from __future__ import annotations

import asyncio
import json
import logging
import secrets

from .invocation import current_invocation
from .jobs import JobError, JobLease, _hash
from .task_runtime import _SEND, Execution

log = logging.getLogger(__name__)
_TASKS = set()


async def start(loop, parent_sid, run_id, *, continuation, executor=None, budget=None, force=False, file_io=None,
                eager_wake=False, on_complete=None):
    """Legacy flags cannot bypass native admission, original contracts or roles."""
    from power_loop.workflow import runner
    from power_loop.workflow.result import AgentResult, SharedBudget, WorkflowRunHandle
    from power_loop.workflow.spec import WorkflowSpec

    from .workflow_restore_execution import settle
    from .workflow_resume import admit, claim_dispatch, dispatch_id
    from .workflow_tasks import _WORKFLOW
    source = _SEND.get()
    runtime = loop.task_runtime
    if source is None or source.runtime is not runtime or source.actor.session_id != parent_sid:
        raise JobError("task_adapter_unavailable:workflow_resume")
    if executor is not None or budget is not None or file_io is not None or eager_wake or on_complete is not None:
        raise JobError("workflow_restore_adapter_unsupported")
    if not isinstance(continuation, tuple) or len(continuation) != 3:
        raise JobError("workflow_restore_target_required")
    node_id, message, rounds = continuation
    if type(rounds) is not int or rounds < 1:
        raise JobError("invalid_continuation_rounds")
    seq, call_id = current_invocation(parent_sid)
    if seq is None or not call_id:
        raise JobError("persisted_tool_call_required")
    manager, store = runtime.manager, runtime.manager.store
    ref = await store.message_ref(parent_sid, seq)
    ready = asyncio.get_running_loop().create_future()
    ready.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)

    async def owned():
        handle = binding = engine = operation = None
        claimed = False
        try:
            creation = await manager.get_operation(source.actor, "op_"+run_id)
            job = await manager.get_job(source.actor, creation.job_id)
            if creation.action != "create" or job.kind != "workflow" or job.created_by_session_id != parent_sid:
                raise JobError("workflow_restore_target_mismatch")
            old = await manager.operation_for_call(source.actor, ref, call_id)
            if old is not None:
                request = old.result.get("request", {})
                if (old.result.get("mode") != "workflow_continue" or old.job_id != job.job_id or
                        request.get("node_id") != node_id or request.get("message") != message or request.get("requested_rounds") != rounds):
                    raise JobError("operation_conflict")
                prior = runtime._workflow_handles.get(old.operation_id) or runner.get_run_handle(run_id)
                if prior is None or prior.operation_id != old.operation_id:
                    raise JobError("workflow_admission_requires_reconciliation")
                ready.set_result(prior)
                return
            prior = runner.get_run_handle(run_id)
            if prior is not None and not prior.task.done():
                await asyncio.wait_for(asyncio.shield(prior.task), 5)
            plan = await manager.get_workflow_restore_plan(source.actor, job.job_id, generation=job.generation, node_id=node_id, message=message)
            maximum = int(plan["nodes"][node_id]["contract"]["spec"]["max_rounds"])
            if rounds > maximum:
                raise JobError("continuation_rounds_exceed_original_contract")
            args = dict(caller=source.lease, call_ref=ref, tool_call_id=call_id, generation=job.generation,
                node_id=node_id, message=message, executor_id=runtime.executor_id, boot_id=runtime.boot_id,
                plan_hash=_hash(plan), lease_ms=runtime.lease_ms, rounds=rounds)
            try:
                operation, repeated = await admit(manager, source.actor, job.job_id, **args)
            except Exception:
                operation = await manager.operation_for_call(source.actor, ref, call_id)
                if operation is None or operation.result.get("mode") != "workflow_continue" or operation.result["request"]["plan_hash"] != _hash(plan):
                    raise
                repeated = False
            if repeated:
                prior = runtime._workflow_handles.get(operation.operation_id)
                if prior is None:
                    raise JobError("workflow_admission_requires_reconciliation")
                ready.set_result(prior)
                return
            request = operation.result["request"]
            binding = Execution(runtime, source.actor, JobLease(**request["target"]))
            dispatch_token = secrets.token_hex(16)
            try:
                claimed = await claim_dispatch(manager, source.actor, operation.operation_id, caller=source.lease, token=dispatch_token)
            except Exception:
                saved = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (dispatch_id(operation.operation_id),))
                if saved is None or json.loads(saved["payload_json"]).get("token") != dispatch_token:
                    raise
                claimed = True
            if not claimed:
                raise JobError("workflow_restore_dispatch_unowned")
            from power_loop.runtime.cancellation import CancellationToken
            cancel = CancellationToken()
            handle = WorkflowRunHandle(run_id=run_id, task=asyncio.current_task(), cancel_token=cancel,
                job_id=job.job_id, operation_id=operation.operation_id)
            handle._task_runtime = runtime
            from types import SimpleNamespace
            handle._task_registration = SimpleNamespace(job=job)
            handle._task_generation = binding.lease.generation
            runner._HANDLES[run_id] = handle
            runner._LIVE_RUN_IDS.add(run_id)
            runner._SETTLED_RUNS.discard(run_id)
            runner._STOPPED_RUNS.discard(run_id)
            runner.remember_handle(runtime, handle)
            ready.set_result(handle)
            binding.resource_evidence["workflow_run_id"] = run_id
            async with runtime.execution(binding, context=_WORKFLOW):
                spec = WorkflowSpec.from_json(request["plan"]["spec"])
                pool = None
                if spec.budget:
                    pool = SharedBudget(spec.budget.max_tokens, stop_at_remaining_pct=spec.budget.stop_at_remaining_pct)
                    pool.commit(request["prior_usage"])
                replay = {nid: AgentResult(**request["plan"]["nodes"][nid]["result"]) for nid in request["plan"]["replay"]}
                engine = runner.WorkflowEngine(loop, budget=pool, stop_event=cancel, run_id=run_id,
                    allowed_tools=frozenset(request["plan"]["allowed_tools"]) if request["plan"]["allowed_tools"] is not None else None,
                    on_step=runner.make_on_step(store, parent_sid, run_id), on_node_start=runner.make_on_node_start(store, parent_sid, run_id),
                    replay=replay, resume_sessions={nid: request["plan"]["nodes"][nid]["binding"]["session_id"] for nid in request["affected"]})
                engine._task_execution = binding
                engine._restore_operation_id = operation.operation_id
                engine._restore_request = request
                engine._replay_no_charge.update(replay)
                for nid in request["affected"]:
                    limit = int(request["plan"]["nodes"][nid]["contract"]["spec"]["max_rounds"])
                    if nid == node_id:
                        limit = min(rounds, limit)
                    prompt = message if nid == node_id else "Re-evaluate your original task using the updated upstream results."
                    engine._resume_instructions[nid] = (prompt, limit)
                runner._ENGINES[run_id] = engine
                try:
                    result = await engine.run(spec)
                except BaseException as exc:
                    result = engine._build_result(spec, "cancelled" if isinstance(exc, asyncio.CancelledError) or cancel.is_cancelled() else "failed")
                    result.errors.append(type(exc).__name__+": "+str(exc))
                try:
                    await settle(runtime, binding, result.status)
                except Exception:
                    from .workflow_restore_execution import get, identity, require_settled
                    try:
                        async with manager._write() as tx:
                            row = await manager._authorize(tx, source.actor, job.job_id)
                            from .workflow_resume import execution
                            settled_op, _ = await execution(manager, tx, row)
                            await require_settled(manager, tx, row, settled_op, success=result.status == "completed")
                            receipt = await get(manager, tx, identity(manager, "settled", job.job_id, binding.lease.generation), job.job_id, "workflow_restore_settled")
                            if receipt["status"] != result.status:
                                raise JobError("workflow_restore_reservation_unsettled") from None
                    except Exception as problem:
                        # A negative result does not close unresolved supervision.
                        if result.status == "completed":
                            result.status = "failed"
                        result.errors.append("Restoration settlement remains unresolved: "+type(problem).__name__)
                await runtime.record_scope(binding, result.to_dict())
                from .workflow_results import project
                await project(loop, source.actor, job.job_id, binding.lease.generation, lease=binding.lease)
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            if binding is not None and not claimed:
                # Admission survived but dispatch responsibility is unresolved.
                # Request cancellation without inventing physical-stop proof.
                try:
                    await manager.request_cancel(source.actor, binding.lease.job_id,
                        system_key="restore-undispatched:"+operation.operation_id, reason="Restoration dispatch responsibility unresolved")
                except BaseException:
                    log.exception("undispatched restore cancellation requires reconciliation")
            if binding is not None and claimed:
                try:
                    from .result_revisions import result_id
                    canonical = await manager.db.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=? AND event_type='result'",
                        (result_id(binding.lease.job_id, binding.lease.generation),))
                    if canonical is not None:
                        from .workflow_results import project
                        await project(loop, source.actor, binding.lease.job_id, binding.lease.generation, lease=binding.lease)
                    else:
                        from power_loop.workflow.result import WorkflowResult
                        status = "cancelled" if isinstance(exc, asyncio.CancelledError) or (handle is not None and handle.cancel_token.is_cancelled()) else "failed"
                        negative = engine._build_result(WorkflowSpec.from_json(operation.result["request"]["plan"]["spec"]), status) if engine is not None else WorkflowResult(
                            name=operation.result["request"]["plan"]["spec"]["name"], status=status)
                        negative.errors.append("Restoration stopped before verified completion: "+type(exc).__name__)
                        await runtime.record_scope(binding, negative.to_dict())
                        from .workflow_results import project
                        await project(loop, source.actor, binding.lease.job_id, binding.lease.generation, lease=binding.lease)
                except BaseException:
                    log.exception("restore completion/cancellation evidence requires reconciliation")
                    try:
                        await manager.request_cancel(source.actor, binding.lease.job_id,
                            system_key="restore-unsettled:"+operation.operation_id, reason="Restoration requires reconciliation: "+type(exc).__name__)
                    except BaseException:
                        log.exception("restore cancellation intent requires reconciliation")
            if handle is not None:
                log.exception("restored workflow requires reconciliation: %s", run_id)
        finally:
            if binding is not None and claimed:
                try:
                    from .workflow_controller import physical
                    from .workflow_restore_execution import driver_proof
                    async with manager._write() as tx:
                        row = await manager._authorize(tx, source.actor, binding.lease.job_id)
                        _, request, _ = await driver_proof(manager, tx, row, binding.lease.generation)
                        sid = request["driver_sid"]
                        if await tx.fetchone(f"SELECT session_id FROM {store.t.sessions} WHERE session_id=?", (sid,)):
                            await physical(manager, tx, {**row, "generation": binding.lease.generation}, sid, run_id)
                        else:
                            sid = None
                    if sid is not None:
                        await loop.close_session(sid, cascade=False)
                except Exception:
                    log.debug("restored driver cleanup requires reconciliation", exc_info=True)
            if engine is not None and runner._ENGINES.get(run_id) is engine:
                runner._ENGINES.pop(run_id, None)
            if handle is not None and runner._HANDLES.get(run_id) is handle:
                runner._HANDLES.pop(run_id, None)
                runner._LIVE_RUN_IDS.discard(run_id)
                runner._SETTLED_RUNS.discard(run_id)
                runner._STOPPED_RUNS.discard(run_id)

    coro = owned()
    try:
        task = asyncio.create_task(coro, name="workflow-restore-"+run_id)
    except BaseException:
        coro.close()
        raise
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return await asyncio.shield(ready)
