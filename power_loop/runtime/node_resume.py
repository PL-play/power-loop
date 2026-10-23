"""Success-only leaf continuation owned by the original, still-active workflow."""
from __future__ import annotations

import asyncio
import json
import secrets
from contextvars import ContextVar
from dataclasses import asdict
from types import SimpleNamespace

from .jobs import JobError, JobLease, _hash, _json, _operation
from .result_revisions import execution_id, result_id

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
        origin = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND action='create'", (caller.job_id,))
        send_index = json.loads(origin["payload_json"]).get("request", {}).get("send_index") if origin else None
        if (source["kind"] != "send" or source["session_id"] != actor.session_id or source["execution_role"] != "main" or
                isinstance(send_index, bool) or not isinstance(send_index, int) or send_index != call_ref.send_index):
            raise JobError("resume_caller_mismatch")
        wf = await manager._fence(tx, actor, workflow)
        job = await manager._authorize(tx, actor, job_id)
        if (wf["kind"] != "workflow" or wf["created_by_session_id"] != actor.session_id or
                job["kind"] != "workflow_node" or job["execution_role"] != "workflow_node" or
                job["parent_job_id"] != wf["job_id"] or job["control_owner_job_id"] != wf["job_id"] or
                job["created_by_session_id"] != wf["session_id"]):
            raise JobError("resume_target_mismatch")
        from .workflow_controller import for_node
        await for_node(manager, tx, job)
        await manager._supervision_barriers(tx, job)
        await manager._control_owners(tx, job)
        if job["generation"] != generation:
            raise JobError("resume_generation_changed")
        if (job["status"] != "settling" or job["outcome"] != "succeeded" or job["resource_state"] != "stopped" or
                job["intake"] != "closed" or job["supervision_status"] != "open"):
            raise JobError("resume_target_unsettled")
        from .node_inputs import descriptor, is_closed
        if _hash(await descriptor(manager, tx, job, node["session_id"])) != _hash(node) or not await is_closed(manager, tx, job, node["session_id"]):
            raise JobError("node_input_binding_mismatch")
        canonical = await tx.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=? AND job_id=? AND event_type='result'",
            (result_id(job_id, generation), job_id))
        if (canonical is None or _hash(json.loads(canonical["payload_json"])["result"]) != _hash(previous_result) or
                _hash(json.loads(job["result_json"])) != _hash(previous_result)):
            raise JobError("node_resume_result_mismatch")
        wf_op = await tx.fetchone(f"SELECT event_id,payload_json FROM {manager.events} WHERE job_id=? AND action='create'", (wf["job_id"],))
        from power_loop.workflow import journal
        from power_loop.workflow.spec import WorkflowSpec
        if wf_op is None or wf_op["event_id"][3:] != node["workflow_run_id"]:
            raise JobError("workflow_run_mismatch")
        frozen = json.loads(wf_op["payload_json"])["request"]["spec"]
        _, _, expected_plan = selection(WorkflowSpec.from_json(frozen), node["node_id"])
        if _hash(expected_plan.to_dict()) != _hash(plan):
            raise JobError("node_resume_plan_mismatch")
        journal_row = await tx.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key=?",
            (actor.session_id, journal.run_key(node["workflow_run_id"])))
        j = json.loads(journal_row["value_json"]) if journal_row else {}
        steps = [s for s in j.get("steps", []) if s.get("node_id") == node["node_id"]]
        fields = ("node_id", "status", "session_id", "text", "payload", "usage", "error")
        if (j.get("status") != "running" or len(steps) != 1 or
                _hash({k: steps[0].get(k) for k in fields}) != _hash({k: previous_result.get(k) for k in fields})):
            raise JobError("node_resume_journal_mismatch")
        await manager._require_scope_ready(tx, job)
        saved = await tx.fetchone(f"SELECT value_json FROM {manager.store.t.session_runtime_state} WHERE session_id=? AND state_key='workflow_execution_contract'", (node["session_id"],))
        if not isinstance(contract, dict) or contract.get("version") != 1 or saved is None or _hash(json.loads(saved["value_json"])) != _hash(contract):
            raise JobError("resume_contract_mismatch")
        state = await tx.fetchone(f"SELECT pending_json FROM {manager.store.t.session_state} WHERE session_id=?", (node["session_id"],))
        if state is not None and state["pending_json"]:
            raise JobError("resume_tools_pending")
        if await tx.fetchone(f"SELECT item_id FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND status IN ('pending','claimed')", (node["session_id"],)):
            raise JobError("resume_input_pending")
        from .execution_membership import require_current, start
        await require_current(manager, tx, job)
        now = manager.now()
        await tx.execute(f"UPDATE {manager.jobs} SET generation=?,executor_id=?,boot_id=?,lease_until_ms=?,status='running',"
            "intake='open',resource_state='running',result_status='none',outcome=NULL,result_json=NULL,version=version+1,updated_at=? WHERE job_id=?",
            (target.generation, executor_id, boot_id, now+lease_ms, now, job_id))
        payload = {"job_id": job_id, "session_id": node["session_id"], "run_id": node["workflow_run_id"],
            "mode": "continue", "status": "accepted", "generation": target.generation,
            "result_event_id": result_id(job_id, target.generation), "request": request}
        op = await manager._insert_operation(tx, job_id=job_id, version=job["version"]+1, actor=actor, action="nudge",
            key=key, fingerprint=fingerprint, call_ref=call_ref, tool_call_id=tool_call_id, payload=payload)
        await tx.execute(f"UPDATE {manager.events} SET delivery_status='pending_delivery' WHERE event_id=?", (op.operation_id,))
        for event_id, kind, data in ((execution_id(job_id, target.generation), "execution", {"operation_id": op.operation_id,
                "run_id": node["workflow_run_id"], "target": request["target"], "workflow": request["workflow"]}),
                (manager._node_session_event_id(job_id, node["session_id"], target.generation), "node_session", request["node"])):
            await tx.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
                (event_id, job_id, kind, job["version"]+1, _json(data), "not_applicable", now))
        cols = ("session_id", "item_id", "kind", "mode", "status", "content", "created_at")
        if not await tx.execute(manager.db.dialect.insert_ignore(manager.store.t.follow_up_queue, cols),
                (node["session_id"], "nudge:"+op.operation_id, "relay", "steer", "pending", message, now)):
            raise JobError("nudge_inbox_conflict")
        await start(manager, tx, await manager._row(tx, job_id))
        return op, False


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
            rows = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE parent_job_id=? AND kind='workflow_node' AND tenant_id=? AND principal_id=?",
                (workflow.lease.job_id, source.actor.tenant_id, source.actor.principal_id))
            # Resolve by full physical descriptor, never by node ID/latest journal.
            matches = []
            from .node_inputs import descriptor
            for job_id in {row["job_id"] for row in rows}:
                job = await manager.get_job(source.actor, job_id)
                if job.parent_job_id != workflow.lease.job_id:
                    continue
                try:
                    data = await descriptor(manager, manager.db, asdict(job), sid)
                except JobError:
                    continue
                if data["node_id"] == node_id and data["workflow_run_id"] == engine._run_id:
                    matches.append((job, data))
            if len(matches) != 1:
                raise JobError("node_input_binding_mismatch")
            job, data = matches[0]
            if expected_job is not None and (job.job_id, job.generation) != expected_job:
                raise JobError("workflow_node_generation_changed")
            contract = await manager.store.get_runtime_state(sid, "workflow_execution_contract")
            child = await workflow_continuation_loop(loop, sid, expected_contract=contract)
            if child._lock_for(sid).locked():
                raise JobError("resume_session_busy")
            try:
                operation, replayed = await manager.resume_workflow_node(source.actor, job.job_id, caller=source.lease,
                    workflow=workflow.lease, call_ref=ref, tool_call_id=call_id, node=data, message=message,
                    previous_result=previous_result, contract=contract, plan=frozen_plan,
                    executor_id=runtime.executor_id, boot_id=runtime.boot_id, lease_ms=runtime.lease_ms)
            except Exception:
                saved = await manager.operation_for_call(source.actor, ref, call_id)
                request = saved.result.get("request", {}) if saved else {}
                if (saved is None or saved.action != "nudge" or saved.result.get("mode") != "continue" or saved.job_id != job.job_id or
                        request.get("session_id") != sid or request.get("message") != message or
                        request.get("previous_generation") != job.generation or request.get("workflow") != asdict(workflow.lease) or
                        request.get("caller") != asdict(source.lease) or _hash(request.get("node")) != _hash({**data, "generation": job.generation+1}) or
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
            if owned and binding is not None and not binding.stop_renewal.is_set():
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
    return ready
