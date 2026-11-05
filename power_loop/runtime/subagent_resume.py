"""Controlled new executions of a successfully settled standalone child."""
from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
from dataclasses import asdict

from .jobs import JobError, JobLease, _hash, _json
from .jobs import verify_stored as _verify_stored


@_verify_stored
async def admit(manager, actor, job_id, *, caller, call_ref, tool_call_id, session_id, message,
                expected_generation, executor_id, boot_id, contract, lease_ms=30000):
    if not isinstance(message, str) or not message.strip() or len(message) > 64000 or "\x00" in message:
        raise JobError("invalid_nudge_message")
    if isinstance(expected_generation, bool) or not isinstance(expected_generation, int) or not 1 <= expected_generation < 9223372036854775807:
        raise JobError("invalid_result_generation")
    if not all(isinstance(v, str) and 0 < len(v) <= 64 for v in (executor_id, boot_id)):
        raise JobError("invalid_executor")
    if isinstance(lease_ms, bool) or not isinstance(lease_ms, int) or not 1 <= lease_ms <= 300000:
        raise JobError("invalid_lease_duration")
    if not isinstance(caller, JobLease):
        raise JobError("execution_lease_required")
    contract = json.loads(_json(contract))
    contract_hash = _hash(contract)
    target = JobLease(job_id, executor_id, boot_id, expected_generation+1)
    request = {"target": asdict(target), "session_id": session_id, "message": message,
               "previous_generation": expected_generation, "lease_ms": lease_ms, "contract_hash": contract_hash, "caller": asdict(caller)}
    from .subagent_admission import admit as accept
    return await accept(manager, actor, request, call_ref, tool_call_id)


async def claim_dispatch(manager, actor, operation_id, *, caller, token):
    if not isinstance(token, str) or not token or len(token) > 64:
        raise JobError("invalid_dispatch_token")
    if not isinstance(caller, JobLease):
        raise JobError("execution_lease_required")
    from .resume_dispatch import claim
    return await claim(manager, actor, operation_id, caller=caller, token=token)


async def resume_locked(loop, source, call_ref, call_id, sid, message, *, on_complete=None, finalize=None, admission=None):
    """Caller holds the original session gate. Replays never dispatch here."""
    from . import delegation
    from .nudges import receipt
    from .original_owner_review import invoke
    from .spec import workflow_continuation_loop
    from .task_runtime import Execution

    runtime = loop.task_runtime
    manager, store = runtime.manager, runtime.manager.store
    rows = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='subagent' AND session_id=? AND created_by_session_id=? "
        "AND tenant_id=? AND principal_id=?", (sid, source.actor.session_id, source.actor.tenant_id, source.actor.principal_id))
    if len(rows) != 1:
        raise JobError("task_adapter_unavailable:subagent_resume")
    job = await manager.get_job(source.actor, rows[0]["job_id"])
    # Build the saved child contract before mutating task state. Never substitute
    # parent tools/persona for a missing or broken original child contract.
    contract = await store.get_runtime_state(sid, "workflow_execution_contract")
    child = await workflow_continuation_loop(loop, sid, expected_contract=contract)
    if child._session_busy(sid):
        raise JobError("resume_session_busy")
    args = dict(caller=source.lease, call_ref=call_ref, tool_call_id=call_id, session_id=sid, message=message,
        expected_generation=job.generation, executor_id=runtime.executor_id, boot_id=runtime.boot_id, contract=contract, lease_ms=runtime.lease_ms)
    try:
        operation, repeated = await manager.resume_subagent(source.actor, job.job_id, **args)
    except Exception:
        from .subagent_admission import recover_call
        operation = await recover_call(manager, source.actor, call_ref, call_id, session_id=sid, message=message)
        if operation is None or operation.result.get("mode") != "continue" or operation.result["request"]["session_id"] != sid or operation.result["request"]["message"] != message:
            raise
        repeated = False  # no dispatch has been attempted by this invocation
    if repeated:
        return receipt(operation, True)
    target = JobLease(**operation.result["request"]["target"])
    token = secrets.token_hex(16)
    try:
        claimed = await manager.claim_resume_dispatch(source.actor, operation.operation_id, caller=source.lease, token=token)
    except Exception:
        old = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", ("dispatch_"+operation.operation_id,))
        if old is None or json.loads(old["payload_json"]).get("token") != token:
            raise
        claimed = True
    if not claimed:
        return receipt(operation, True)
    binding = Execution(runtime, runtime.actor(sid), target)
    before = await store.history_bounds(sid)
    after_seq = before[1] if before else 0
    await manager.check_execution(binding.actor, binding.lease)
    run = delegation.Running(operation.result["run_id"], token=delegation.CancellationToken(),
        parent=source.actor.session_id, title=job.title, task_binding=binding, admission=admission)
    run.projection_expected = child.config.projection_representation is not None
    delegation._RUNNING[(id(store), sid)] = run

    async def execute():
        acquired = False
        try:
            async with runtime.execution(binding, owner=True, cancel_token=run.token):
                try:
                    if admission is not None:
                        run.phase = "queued"
                        acquired = await delegation._acquire_admission(run)
                    run.phase = "running"
                    with contextlib.ExitStack() as guards:
                        for _, guard in getattr(loop, "_child_run_guards", ()):
                            guards.enter_context(guard())
                        result = await invoke(child,sid,child.flush_follow_ups,sid, stop_event=run.token, max_rounds=child.config.max_rounds)
                    if result is None:
                        raise JobError("resume_input_missing")
                    rounds = int(result.rounds)
                    if result.status == "completed" and not (result.final_text or "").strip():
                        remaining = int(child.config.max_rounds or 3)-rounds
                        if remaining > 0:
                            with contextlib.ExitStack() as guards:
                                for _, guard in getattr(loop, "_child_run_guards", ()):
                                    guards.enter_context(guard())
                                result = await invoke(child,sid,child.follow_up,
                                    "Your previous turn did not contain a final answer. There is no outstanding delegated task. "
                                    "Use the work already completed in this session and return the requested final answer as assistant text. "
                                    "Do not repeat completed actions or call pass_turn.", session_id=sid,
                                    stop_event=run.token, max_rounds=remaining, tools=[])
                            rounds += int(result.rounds)
                        if result.status == "completed" and not (result.final_text or "").strip():
                            raise JobError("resume_output_missing")
                    raw = {"status": result.status, "final_text": result.final_text, "rounds": rounds,
                           "usage": result.usage, "new_input_after_pass": result.new_input_after_pass}
                except BaseException as exc:
                    raw = {"status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", "error": str(exc)}
                finally:
                    if acquired:
                        delegation._release_admission(run)
                await delegation._complete(loop, run.parent, sid, run, raw, after_seq, "linked", finalize, on_complete)
        except BaseException as exc:
            await delegation._complete(loop, run.parent, sid, run,
                {"status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", "error": str(exc)},
                after_seq, "linked", None, on_complete)

    coroutine = execute()
    try:
        run.task = asyncio.create_task(coroutine, name="subagent-"+run.run_id)
    except BaseException:
        coroutine.close()
        delegation._RUNNING.pop((id(store), sid), None)
        await runtime.record_subagent(binding, {"status": "failed", "error": "resume dispatch failed before execution"})
        raise
    delegation._retain(run.task)
    runtime.track_task(run.task, cancel_token=run.token, job_id=binding.lease.job_id)
    return receipt(operation, False)
