"""Controlled new executions of a successfully settled standalone child."""
from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
from dataclasses import asdict

from .jobs import JobError, JobLease, _hash, _json, _operation
from .result_revisions import execution_id, result_id


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
    key = manager._origin_key(actor, "nudge", call_ref, tool_call_id, None)
    fingerprint = _hash([request, call_ref.to_dict()])
    async with manager._write() as tx:
        old = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE operation_key=?", (key,))
        if old:
            await manager._authorize(tx, actor, old["job_id"])
            if old["request_hash"] != fingerprint:
                raise JobError("operation_conflict")
            return _operation(old), True
        await manager._validate_call(tx, actor, call_ref, tool_call_id)
        for action in ("create", "steer"):
            if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE operation_key=?",
                (manager._origin_key(actor, action, call_ref, tool_call_id, None),)):
                raise JobError("operation_conflict")
        source = await manager._fence(tx, actor, caller)
        source_create = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create'", (caller.job_id,))
        send_index = json.loads(source_create["payload_json"]).get("request", {}).get("send_index") if source_create else None
        if (source["kind"] != "send" or source["session_id"] != actor.session_id or source["execution_role"] != "main" or
                isinstance(send_index, bool) or not isinstance(send_index, int) or send_index < 1 or send_index != call_ref.send_index):
            raise JobError("resume_caller_mismatch")
        job = await manager._authorize(tx, actor, job_id)
        if (job["kind"] != "subagent" or job["execution_role"] != "subagent" or job["control_owner_job_id"] is not None or
                job["created_by_session_id"] != actor.session_id or job["session_id"] != session_id):
            raise JobError("resume_target_mismatch")
        await manager._supervision_barriers(tx, job)
        if job["generation"] != expected_generation:
            raise JobError("resume_generation_changed")
        from .finalization import require_resumable
        await require_resumable(manager, tx, job)
        if (job["outcome"] != "succeeded" or job["resource_state"] != "stopped" or job["intake"] != "closed"):
            raise JobError("resume_target_unsettled")
        from .processing import require_handled
        try:
            await require_handled(manager, tx, job)
        except JobError as exc:
            raise JobError("resume_result_unhandled") from exc
        await manager._require_scope_ready(tx, job)
        session = await tx.fetchone(f"SELECT parent_session_id,metadata_json FROM {manager.store.t.sessions} WHERE session_id=?", (session_id,))
        meta = json.loads(session["metadata_json"] or "{}") if session else {}
        if (session is None or session["parent_session_id"] != actor.session_id or
                meta.get("execution_role") != "subagent" or not meta.get("standalone_subagent")):
            raise JobError("resume_session_unavailable")
        saved_contract = await tx.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
            (session_id, "workflow_execution_contract"))
        if (saved_contract is None or not isinstance(contract, dict) or contract.get("version") != 1 or
                _hash(json.loads(saved_contract["value_json"])) != contract_hash):
            raise JobError("resume_contract_mismatch")
        state = await tx.fetchone(f"SELECT pending_json FROM {manager.store.t.session_state} WHERE session_id=?", (session_id,))
        if state is not None and state["pending_json"]:
            raise JobError("resume_tools_pending")
        pending = await tx.fetchone(f"SELECT item_id FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND status IN ('pending','claimed')", (session_id,))
        if pending is not None:
            raise JobError("resume_input_pending")
        from .execution_membership import require_current, start
        await require_current(manager, tx, job)
        now = manager.now()
        from .executor_registry import ExecutorIdentity, require_accepting
        executor = await require_accepting(manager, tx, ExecutorIdentity(actor.tenant_id, actor.principal_id, executor_id, boot_id),
            required=job["executor_key"] is not None or source["executor_key"] is not None)
        await tx.execute(f"UPDATE {manager.jobs} SET generation=?,executor_id=?,boot_id=?,executor_key=?,lease_until_ms=?,"
            "status='running',supervision_status='open',intake='open',resource_state='running',result_status='none',outcome=NULL,result_json=NULL,"
            "version=version+1,updated_at=? WHERE job_id=?",
            (target.generation, executor_id, boot_id, executor["executor_key"] if executor else None, now+lease_ms, now, job_id))
        payload = {"job_id": job_id, "session_id": session_id, "mode": "continue", "status": "accepted",
                   "generation": target.generation, "result_event_id": result_id(job_id, target.generation), "request": request}
        operation = await manager._insert_operation(tx, job_id=job_id, version=job["version"]+1,
            actor=actor, action="nudge", key=key, fingerprint=fingerprint, call_ref=call_ref,
            tool_call_id=tool_call_id, payload=payload)
        payload["run_id"] = operation.operation_id[3:]
        await tx.execute(f"UPDATE {manager.events} SET payload_json=?,delivery_status='pending_delivery' WHERE event_id=?",
            (_json(payload), operation.operation_id))
        cols = ("session_id", "item_id", "kind", "mode", "status", "content", "created_at")
        if not await tx.execute(manager.db.dialect.insert_ignore(manager.store.t.follow_up_queue, cols),
                (session_id, "nudge:"+operation.operation_id, "relay", "steer", "pending", message, now)):
            raise JobError("nudge_inbox_conflict")
        await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
            (execution_id(job_id, target.generation), job_id, "execution", job["version"]+1,
             _json({"operation_id": operation.operation_id, "run_id": payload["run_id"], "target": request["target"]}), "not_applicable", now))
        await start(manager, tx, await manager._row(tx, job_id))
        from .finalization import reopen
        await reopen(manager, tx, job, asdict(target), operation.operation_id)
        return _operation(await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=?", (operation.operation_id,))), False


async def claim_dispatch(manager, actor, operation_id, *, caller, token):
    if not isinstance(token, str) or not token or len(token) > 64:
        raise JobError("invalid_dispatch_token")
    if not isinstance(caller, JobLease):
        raise JobError("execution_lease_required")
    event_id = "dispatch_"+operation_id
    async with manager._write() as tx:
        op = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_id=? AND action='nudge' AND event_type='operation'", (operation_id,))
        if op is None or op["caller_session_id"] != actor.session_id:
            raise JobError("resume_dispatch_mismatch")
        job = await manager._authorize(tx, actor, op["job_id"])
        payload = json.loads(op["payload_json"])
        if payload.get("mode") != "continue" or payload["request"].get("caller") != asdict(caller):
            raise JobError("resume_dispatch_mismatch")
        old = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (event_id,))
        if old:
            return json.loads(old["payload_json"])["token"] == token
        source = await manager._fence(tx, actor, caller)
        if source["kind"] != "send" or source["session_id"] != actor.session_id or source["execution_role"] != "main":
            raise JobError("resume_caller_mismatch")
        await manager._fence(tx, actor, JobLease(**payload["request"]["target"]))
        if job["kind"] == "workflow_node":
            from .node_resume import validate_execution
            await validate_execution(manager, tx, job, payload["request"]["session_id"])
        # The claim belongs to the same originating model send, not another
        # later send that happens to reuse the original principal/SID.
        create = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create'", (caller.job_id,))
        if json.loads(create["payload_json"])["request"]["send_index"] != int(json.loads(op["call_ref_json"])["send_index"]):
            raise JobError("resume_caller_mismatch")
        await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
            (event_id, job["job_id"], "dispatch", job["version"], _json({"token": token, "operation_id": operation_id}), "claimed", manager.now()))
        return True


async def resume_locked(loop, source, call_ref, call_id, sid, message, *, on_complete=None, finalize=None, admission=None):
    """Caller holds the original session gate. Replays never dispatch here."""
    from . import delegation
    from .nudges import receipt
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
    if child._lock_for(sid).locked():
        raise JobError("resume_session_busy")
    args = dict(caller=source.lease, call_ref=call_ref, tool_call_id=call_id, session_id=sid, message=message,
        expected_generation=job.generation, executor_id=runtime.executor_id, boot_id=runtime.boot_id, contract=contract, lease_ms=runtime.lease_ms)
    try:
        operation, repeated = await manager.resume_subagent(source.actor, job.job_id, **args)
    except Exception:
        operation = await manager.operation_for_call(source.actor, call_ref, call_id)
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
        parent=source.actor.session_id, title=job.title, task_binding=binding)
    run.projection_expected = child.config.projection_representation is not None
    delegation._RUNNING[(id(store), sid)] = run

    async def execute():
        acquired = False
        try:
            async with runtime.execution(binding, owner=True, cancel_token=run.token):
                try:
                    if admission is not None:
                        run.phase = "queued"
                        await admission.acquire()
                        acquired = True
                    run.phase = "running"
                    with contextlib.ExitStack() as guards:
                        for _, guard in getattr(loop, "_child_run_guards", ()):
                            guards.enter_context(guard())
                        result = await child.flush_follow_ups(sid, stop_event=run.token, max_rounds=child.config.max_rounds)
                    if result is None:
                        raise JobError("resume_input_missing")
                    rounds = int(result.rounds)
                    if result.status == "completed" and not (result.final_text or "").strip():
                        remaining = int(child.config.max_rounds or 3)-rounds
                        if remaining > 0:
                            with contextlib.ExitStack() as guards:
                                for _, guard in getattr(loop, "_child_run_guards", ()):
                                    guards.enter_context(guard())
                                result = await child.follow_up(
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
                        admission.release()
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
