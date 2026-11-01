"""Success-only leaf continuation owned by the original, still-active workflow."""
from __future__ import annotations

import asyncio
import json
import secrets
from contextvars import ContextVar
from dataclasses import asdict
from types import SimpleNamespace

from .jobs import JobError, JobLease, _hash, _json
from .jobs import verify_stored as _verify_stored
from .result_revisions import execution_id

_PREPARED = ContextVar("power_loop_prepared_node_resume", default=None)


def selection(spec, node_id):
    from power_loop.workflow.continuation import plan_continuation, walk, within
    from power_loop.workflow.spec import AgentNode, ParallelNode, SequenceNode
    plan = plan_continuation(spec, node_id)
    occurrences = dict(walk(spec.root))
    ancestors = [occurrences[plan.target_path[:i]] for i in range(len(plan.target_path))]
    node = occurrences[plan.target_path]
    if (not ancestors or not isinstance(ancestors[-1], ParallelNode) or
            any(not isinstance(n, (SequenceNode, ParallelNode)) for n in ancestors) or
            not isinstance(node, AgentNode) or node.fallback is not None or
            any(p < plan.target_path and not within(plan.target_path, p) for p in plan.affected_paths)):
        raise JobError("node_resume_scope_unsupported")
    return node, ancestors, plan


@_verify_stored
async def admit(manager, actor, job_id, *, caller, workflow, call_ref, tool_call_id,
                node, message, previous_result, contract, plan, executor_id, boot_id, lease_ms=30000):
    if not isinstance(message, str) or not message.strip() or len(message) > 64000 or "\x00" in message:
        raise JobError("invalid_nudge_message")
    if not isinstance(caller, JobLease) or not isinstance(workflow, JobLease):
        raise JobError("execution_lease_required")
    if not all(isinstance(v, str) and 0 < len(v) <= 64 for v in (executor_id, boot_id)):
        raise JobError("invalid_executor")
    if isinstance(lease_ms, bool) or not isinstance(lease_ms, int) or not 1 <= lease_ms <= 300000:
        raise JobError("invalid_lease_duration")
    node, contract, previous_result, plan = json.loads(_json([node, contract, previous_result, plan]))
    generation = node.get("generation", 1)
    if isinstance(generation, bool) or not isinstance(generation, int) or not 1 <= generation < 2**63-1:
        raise JobError("invalid_result_generation")
    target = JobLease(job_id, executor_id, boot_id, generation+1)
    request = {"target": asdict(target), "caller": asdict(caller), "workflow": asdict(workflow),
        "session_id": node["session_id"], "message": message, "previous_generation": generation,
        "node": {**node, "generation": target.generation}, "contract_hash": _hash(contract),
        "previous_result_hash": _hash(previous_result), "plan": plan, "lease_ms": lease_ms}
    from .node_admission import admit as admit_node
    return await admit_node(manager, actor, request, call_ref, tool_call_id)


async def validate_execution(manager, tx, job, sid, *, dispatch=False):
    row = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND event_type='execution' AND job_id=?",
        (execution_id(job["job_id"], job["generation"]), job["job_id"]))
    if row is None:
        raise JobError("task_adapter_unavailable:workflow_node_resume")
    execution = json.loads(row["payload_json"])
    op = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND event_type='operation' AND action='nudge' AND job_id=?",
        (execution["operation_id"], job["job_id"]))
    payload = json.loads(op["payload_json"]) if op else {}
    request = payload.get("request", {})
    expected = asdict(JobLease(job["job_id"], job["executor_id"], job["boot_id"], job["generation"]))
    from .node_inputs import descriptor
    if (payload.get("mode") != "continue" or _hash(request.get("target")) != _hash(expected) or
            _hash(execution.get("target")) != _hash(expected) or request.get("session_id") != sid or
            _hash(request.get("node")) != _hash(await descriptor(manager, tx, job, sid)) or
            request.get("workflow") != execution.get("workflow") or request["workflow"]["job_id"] != job["control_owner_job_id"]):
        raise JobError("node_resume_execution_mismatch")
    await manager._fence(tx, controller_actor(job), JobLease(**request["workflow"]))
    if dispatch and not await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=? AND job_id=? AND event_type='dispatch' AND delivery_status='claimed'",
            ("dispatch_"+execution["operation_id"], job["job_id"])):
        raise JobError("node_resume_dispatch_required")


def controller_actor(job):
    from .jobs import JobActor
    return JobActor(job["tenant_id"], job["principal_id"], job["created_by_session_id"])


def reserve(loop, engine, source, ref, call_id, node_id, sid, message, *, expected_job=None):
    """Register real owned work before the first admission await; return its receipt future."""
    from power_loop.workflow.continuation import walk, within
    from power_loop.workflow.engine import _IN_WORKFLOW, InProcessExecutor
    from power_loop.workflow.result import AgentResult

    from .nudges import receipt
    from .spec import workflow_continuation_loop
    from .task_runtime import Execution
    from .workflow_tasks import _WORKFLOW, ROUTE_KEY

    runtime, manager = loop.task_runtime, loop.task_runtime.manager
    if (engine is None or not engine._accept_continuations or engine._cancel.is_cancelled() or
            engine._task_execution is None or engine._task_execution.stop_renewal.is_set() or
            type(engine._executor) is not InProcessExecutor):
        raise JobError("task_adapter_unavailable:workflow_node_resume")
    node, ancestors, plan = selection(engine._spec, node_id)
    frame = engine._frames.get(id(node))
    prior = engine._results.get(node_id)
    if (not frame or frame["active"] or prior is None or prior.status != "completed" or prior.session_id != sid or
            any(id(n) in engine._continuation_barriers for n in [*ancestors, node]) or
            not engine._frames.get(id(ancestors[-1]), {}).get("active") or
            (plan.invalidated - {node_id}) & engine._results.keys()):
        raise JobError("node_resume_scope_unavailable")
    for path, candidate in walk(engine._spec.root):
        if (path in plan.affected_paths and engine._frames.get(id(candidate), {}).get("active") and
                not within(plan.target_path, path)):
            raise JobError("node_resume_consumer_active")
    old = engine._repairs.get(id(node))
    if old is not None:
        if not old.done():
            raise JobError("node_resume_in_progress")
        old.result()
    workflow = engine._task_execution
    previous_result, frozen_plan = prior.to_dict(), plan.to_dict()
    ready = asyncio.get_running_loop().create_future()
    ready.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)

    async def repair():
        operation, binding, owned = None, None, False
        try:
            # Resolve by full physical descriptor, never by node ID/latest journal.
            matches = []
            from .node_inputs import descriptor
            async with manager._read() as reader:
                rows = await reader.fetchall(f"SELECT * FROM {manager.jobs} WHERE parent_job_id=? AND kind='workflow_node' AND tenant_id=? AND principal_id=?",
                    (workflow.lease.job_id, source.actor.tenant_id, source.actor.principal_id))
                for job in rows:
                    try:
                        data = await descriptor(manager, reader, job, sid)
                    except JobError:
                        continue
                    if data["node_id"] == node_id and data["workflow_run_id"] == engine._run_id:
                        matches.append((job, data))
            if len(matches) != 1:
                raise JobError("node_input_binding_mismatch")
            job, data = matches[0]
            if expected_job is not None and (job["job_id"], job["generation"]) != expected_job:
                raise JobError("workflow_node_generation_changed")
            contract = await manager.store.get_runtime_state(sid, "workflow_execution_contract")
            child = await workflow_continuation_loop(loop, sid, expected_contract=contract)
            if child._lock_for(sid).locked():
                raise JobError("resume_session_busy")
            try:
                operation, replayed = await manager.resume_workflow_node(source.actor, job["job_id"], caller=source.lease,
                    workflow=workflow.lease, call_ref=ref, tool_call_id=call_id, node=data, message=message,
                    previous_result=previous_result, contract=contract, plan=frozen_plan,
                    executor_id=runtime.executor_id, boot_id=runtime.boot_id, lease_ms=runtime.lease_ms)
            except Exception:
                from .node_admission import recover_call
                saved = await recover_call(manager, source.actor, ref, call_id, session_id=sid, message=message,
                    node_key=(engine._run_id, node_id), expected_job=(job["job_id"], job["generation"]))
                request = saved.result.get("request", {}) if saved else {}
                if (saved is None or saved.action != "nudge" or saved.result.get("mode") != "continue" or saved.job_id != job["job_id"] or
                        request.get("session_id") != sid or request.get("message") != message or
                        request.get("previous_generation") != job["generation"] or request.get("workflow") != asdict(workflow.lease) or
                        request.get("caller") != asdict(source.lease) or _hash(request.get("node")) != _hash({**data, "generation": job["generation"]+1}) or
                        _hash(request.get("plan")) != _hash(frozen_plan) or request.get("contract_hash") != _hash(contract) or
                        request.get("previous_result_hash") != _hash(previous_result)):
                    raise
                operation = saved
                replayed = False
            if replayed:
                ready.set_result(receipt(operation, True))
                return prior
            binding = Execution(runtime, runtime.actor(frame["driver"]), JobLease(**operation.result["request"]["target"]))
            token = secrets.token_hex(16)
            try:
                claimed = await manager.claim_resume_dispatch(source.actor, operation.operation_id, caller=source.lease, token=token)
            except Exception:
                row = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", ("dispatch_"+operation.operation_id,))
                if row is None or json.loads(row["payload_json"]).get("token") != token:
                    raise
                claimed = True
            if not claimed:
                raise JobError("node_resume_dispatch_unowned")
            owned = True
            prepared = SimpleNamespace(engine=engine, node_id=node_id, binding=binding, sid=sid, child=child, contract=contract, pending=True)
            wt, pt, guard = _WORKFLOW.set(workflow), _PREPARED.set(prepared), _IN_WORKFLOW.set(True)
            try:
                async with runtime.execution(binding, owner=True):
                    await manager.bind_node_session(binding.actor, binding.lease, sid)
                    engine._task_nodes[sid] = binding
                    ready.set_result(receipt(operation, False))
                    engine._resume_sessions[node_id] = sid
                    engine._resume_instructions[node_id] = (message, child.config.max_rounds)
                    engine._results.pop(node_id, None)
                    engine._replay.pop(node_id, None)
                    await engine._emit_step(AgentResult(node_id=node_id, status="pending", text="", session_id=sid))
                    return await engine._exec(node, {**frame["env"], ROUTE_KEY: data["route"] or []}, frame["driver"])
            finally:
                _IN_WORKFLOW.reset(guard)
                _PREPARED.reset(pt)
                _WORKFLOW.reset(wt)
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            if owned and binding is not None:
                # Exiting execution always retires renewal, including an
                # exception before model admission. That is not a result
                # checkpoint. Preserve an already committed result, otherwise
                # append the original failure through the normal proof gates.
                current = await manager.get_job(binding.actor, binding.lease.job_id)
                if current.generation == binding.lease.generation and current.result_status != "none":
                    raise
                failed = AgentResult(node_id=node_id, status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", text="", session_id=sid, error=str(exc))
                saved = await runtime.record_scope(binding, failed.to_dict())
                failed.status, failed.usage = saved["status"], saved.get("usage", {})
                engine._results[node_id] = failed
                engine._frames[id(node)]["result"] = failed
                await engine._emit_step(failed)
                return failed
            if operation is not None:
                raise  # Ambiguous admission/exit must block workflow success.
            return prior
        finally:
            if binding is not None and engine._task_nodes.get(sid) is binding:
                engine._task_nodes.pop(sid, None)

    coro = repair()
    try:
        task = asyncio.create_task(coro, name="workflow-node-resume-"+node_id)
    except BaseException:
        coro.close()
        ready.cancel()
        raise
    engine._repairs[id(node)] = task
    runtime.track_task(task)
    return ready
