"""Opt-in execution adapters for the experimental task store.

Main sends and standalone, linked subagent creates only. This is deliberately
not enabled by opening a v10 database or by a model argument. Other async
adapters fail closed until they participate in the same durable protocol.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

from .invocation import current_invocation
from .jobs import JobActor, JobError, JobLease, JobManager, JobRegistration


@dataclass(frozen=True)
class Execution:
    runtime: TaskRuntime
    actor: JobActor
    lease: JobLease
    stop_renewal: asyncio.Event = field(default_factory=asyncio.Event, compare=False, repr=False)
    resource_evidence: dict = field(default_factory=dict, compare=False, repr=False)


_SEND: ContextVar[Execution | None] = ContextVar("power_loop_task_send", default=None)
_OWNER: ContextVar[Execution | None] = ContextVar("power_loop_task_owner", default=None)


def reject_unregistered_adapter(name: str, loop=None) -> None:
    from power_loop.core.agent_context import get_current_loop
    loop = loop or get_current_loop()
    if getattr(loop, "task_runtime", None) is not None:
        raise JobError("task_adapter_unavailable:" + name)


async def check_dispatch(session_id: str | None) -> None:
    execution = _SEND.get()
    if execution is not None and execution.actor.session_id == session_id:
        if execution.resource_evidence.get("resource_state") == "unknown":
            raise JobError("unresolved_prior_attempt")
        await execution.runtime.manager.check_execution(execution.actor, execution.lease)


async def check_tool_dispatch(session_id, tool_name, registry, tool_args=None):
    """Only the library-owned query/cancel handler may bypass a spent lease.

    A tool name is not authority. Trusted host formatting wrappers must preserve
    the library handler with functools.wraps and must remain control-only.
    """
    execution = _SEND.get()
    if execution is None or execution.actor.session_id != session_id:
        await check_dispatch(session_id)
        return
    import inspect

    from power_loop.tools.default_tools import _h_recall_send
    from power_loop.tools.jobs import run_jobs
    from power_loop.workflow.facade import run_workflow_tool
    from power_loop.workflow.tool import _handle_workflow_status
    registered = registry.get(tool_name)
    actual = inspect.unwrap(registered.handler) if registered is not None else None
    safe = actual is run_jobs or actual is _handle_workflow_status
    if actual is _h_recall_send and isinstance(tool_args, dict) and tool_args.get("job_id") is not None:
        safe = True
    if actual is run_workflow_tool and isinstance(tool_args, dict):
        safe = tool_args.get("action", "help") in ("help", "status", "list", "cancel", "leaves", "logs")
    if safe:
        from .task_control import _source
        seq, call_id = current_invocation(session_id)
        if seq is None or not call_id:
            raise JobError("persisted_tool_call_required")
        manager = execution.runtime.manager
        ref = await manager.store.message_ref(session_id, seq)
        async with manager._write() as tx:
            await _source(manager, tx, execution.actor, execution.lease, ref, call_id)
        return
    await check_dispatch(session_id)


async def prepare_model_processing(pipeline, messages, round_index, *, provider_call=True):
    execution = _SEND.get()
    if execution is None or execution.actor.session_id != pipeline.session_id:
        return
    execution.resource_evidence["processing_snapshot_id"] = None
    if not provider_call:
        return
    # Projection/rewritten rows do not get credited just because a result is
    # somewhere in the raw transcript. Match identities and actual raw content.
    included = {id(message) for message in messages}
    seqs = getattr(pipeline.sink, "_history_seqs", [])
    actual = [{"seq": seq, "role": message.get("role"), "content": message.get("content")}
              for message, seq in zip(pipeline.history, seqs, strict=False)
              if seq is not None and id(message) in included and isinstance(message.get("content"), str)]
    execution.resource_evidence["processing_snapshot_id"] = await execution.runtime.manager.prepare_processing(
        execution.actor, execution.lease, round_index=round_index, send_index=pipeline.send_index, messages=actual)


def mark_unconfirmed_resource() -> None:
    execution = _SEND.get()
    if execution is not None:
        execution.resource_evidence["resource_state"] = "unknown"


async def bind_tool_receipt(store, sid: str, assistant_seq: int | None, call_id: str, receipt_seq: int) -> None:
    execution = _SEND.get()
    if (execution is None or execution.runtime.manager.store is not store or
            execution.actor.session_id != sid or assistant_seq is None):
        return
    manager = execution.runtime.manager
    operation = await manager.operation_for_call(execution.actor, await store.message_ref(sid, assistant_seq), call_id)
    if operation is not None:
        await manager.reconcile_receipt(execution.actor, operation.operation_id, await store.message_ref(sid, receipt_seq))


class TaskRuntime:
    """Trusted host configuration, shared by loops using the same opened store."""

    def __init__(self, manager: JobManager, *, tenant_id: str, principal_id: str,
                 executor_id: str, boot_id: str, lease_ms: int = 30000, workflow_resources=None):
        JobActor(tenant_id, principal_id, "validation")
        if not all(isinstance(v, str) and 0 < len(v) <= 64 for v in (executor_id, boot_id)):
            raise JobError("invalid_executor")
        if isinstance(lease_ms, bool) or not isinstance(lease_ms, int) or not 300 <= lease_ms <= 300000:
            raise JobError("invalid_lease_duration")
        if workflow_resources is not None:
            from .workflow_resources import WorkflowResourceRegistry
            if type(workflow_resources) is not WorkflowResourceRegistry:
                raise JobError("workflow_resource_registry_invalid")
        self.manager = manager
        self.tenant_id, self.principal_id = tenant_id, principal_id
        self.executor_id, self.boot_id, self.lease_ms = executor_id, boot_id, lease_ms
        self.workflow_resources = workflow_resources
        self._workflow_handles: dict = {}
        self._local_executions: dict = {}

    def actor(self, sid: str) -> JobActor:
        return JobActor(self.tenant_id, self.principal_id, sid)

    async def claim(self, actor: JobActor, registration: JobRegistration) -> Execution:
        lease = await self.manager.claim_job(actor, registration.job.job_id, executor_id=self.executor_id,
            boot_id=self.boot_id, expected_version=registration.job.version, lease_ms=self.lease_ms)
        return Execution(self, actor, lease)

    @asynccontextmanager
    async def execution(self, binding: Execution, *, owner: bool = False, context=None, cancel_token=None):
        """Renew only while this attempt runs; a failed renewal stops dispatch."""
        await self.manager.check_execution(binding.actor, binding.lease)
        variable = context if context is not None else _OWNER if owner else _SEND
        token = variable.set(binding)
        task = asyncio.current_task()
        local = {"binding": binding, "task": task, "signalled": False, "cancel_token": cancel_token}
        local_key = id(local)
        lost = []

        async def renew():
            try:
                while True:
                    try:
                        await asyncio.wait_for(binding.stop_renewal.wait(), self.lease_ms / 3000)
                        return
                    except TimeoutError:
                        pass
                    await self.manager.renew_execution(binding.actor, binding.lease, lease_ms=self.lease_ms)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not binding.stop_renewal.is_set():
                    # A durable stop or another nested scope already cancelled
                    # this task. Let its actual-exit/usage cleanup finish.
                    if local["signalled"] or task.cancelling():
                        return
                    if cancel_token is not None:
                        # A logical node can share the parent task. Stop its
                        # own operation, never the task hosting the parent.
                        if not isinstance(exc, JobError) or exc.code != "cancel_requested":
                            lost.append(exc)
                        local["signalled"] = True
                        cancel_token.cancel("task supervision lost")
                    else:
                        if not isinstance(exc, JobError) or exc.code != "cancel_requested":
                            lost.append(exc)
                        task.cancel("task supervision lost")

        heartbeat = asyncio.create_task(renew(), name="job-lease-" + binding.lease.job_id)
        self._local_executions[local_key] = local
        try:
            yield binding
        except asyncio.CancelledError:
            if lost:
                raise JobError("task_supervision_lost") from lost[0]
            raise
        else:
            if lost:
                # A lease/backend failure is not a user-tolerated node cancel.
                raise JobError("task_supervision_lost") from lost[0]
        finally:
            self._local_executions.pop(local_key, None)
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            variable.reset(token)

    @asynccontextmanager
    async def model_send(self, loop, sid: str, send_index: int, *, cancel_token=None):
        from .execution_roles import session_role
        if await loop.ensure_store() is not self.manager.store:
            raise JobError("task_store_mismatch")
        actor = self.actor(sid)
        owner = _OWNER.get()
        registering_actor = actor
        if owner is not None and (owner.runtime is not self or owner.actor.session_id != sid):
            if owner.runtime is self:
                job = await self.manager.get_job(owner.actor, owner.lease.job_id)
                session = await self.manager.store.get_session(sid)
                restored = False
                if job.kind == "workflow_node" and session is not None and session.parent_session_id != owner.actor.session_id:
                    from .workflow_restore_execution import controls_node
                    async with self.manager._write() as tx:
                        restored = await controls_node(self.manager, tx, await self.manager._row(tx, job.job_id), owner.actor, sid)
                if (job.kind == "workflow_node" and session is not None and
                        (session.parent_session_id == owner.actor.session_id or restored) and
                        session.metadata.get("task_owner_job_id") == job.job_id):
                    # A logical node controls all its physical retry sessions.
                    # Registration checks the durable node_session admission,
                    # while model calls retain the actual child actor/SID.
                    registering_actor = owner.actor
                else:
                    owner = None
            else:
                owner = None
        if owner is None and await session_role(self.manager.store, sid) != "main":
            raise JobError("task_owner_required")
        registration = await self.manager.register_job(registering_actor, kind="send", session_id=sid,
            parent=owner.lease if owner else None,
            system_key=f"send:{sid}:{send_index}" if registering_actor != actor else f"send:{send_index}",
            request={"send_index": send_index, "model_attempt_protocol": 1})
        if registration.replayed:
            # No automatic replay of a possibly side-effecting pending attempt.
            raise JobError("task_send_requires_reconciliation")
        binding = await self.claim(actor, registration)
        evidence = {"exit": "interrupted", "session_id": sid, "send_index": send_index}
        try:
            async with self.execution(binding, cancel_token=cancel_token):
                from .model_attempts import enable
                await enable(binding)
                yield evidence
                snapshot = binding.resource_evidence.get("processing_snapshot_id")
                if snapshot and evidence.get("exit") == "completed" and not evidence.get("new_input_after_pass"):
                    last = await self.manager.db.fetchone(f"SELECT MAX(seq) AS seq FROM {self.manager.store.t.messages} "
                        "WHERE session_id=? AND role='assistant' AND send_index=?", (sid, send_index))
                    if last is None or last["seq"] is None:
                        raise JobError("processing_response_missing")
                    decision_args = {"snapshot_id": snapshot,
                        "response_ref": await self.manager.store.message_ref(sid, last["seq"]),
                        "reason": "Original session completed its registered processing send with no unresolved tools."}
                    try:
                        decisions = await self.manager.commit_processing_decision(actor, binding.lease, **decision_args)
                    except Exception:
                        decisions = await self.manager.recall_processing_decision(actor, binding.lease, **decision_args)
                        if decisions is None:
                            raise
                        evidence["processing_decision_ack_reconciled"] = True
                    evidence["processing_decisions"] = decisions
        except BaseException as exc:
            evidence["exit"] = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            evidence["error_type"] = type(exc).__name__
            raise
        finally:
            if binding.resource_evidence.get("model_attempts_inflight"):
                binding.resource_evidence["resource_state"] = "unknown"
            last = await self.manager.db.fetchone(f"SELECT MAX(seq) AS seq FROM {self.manager.store.t.messages} "
                "WHERE session_id=? AND role='assistant' AND send_index=?", (sid, send_index))
            if last is not None and last["seq"] is not None:
                evidence["last_assistant_ref"] = (await self.manager.store.message_ref(sid, last["seq"])).to_dict()
            await self.manager.close_model_execution(actor, binding.lease, evidence=evidence,
                resource_state=binding.resource_evidence.get("resource_state", "stopped"))
            from .finalization import after_send
            try:
                await after_send(self.manager, actor, binding.lease)
            except Exception:
                # The model exit is already durable. A failed mechanical
                # close leaves supervision open for reconciliation; it must
                # not turn that completed business send into a retry.
                logging.getLogger(__name__).exception(
                    "supervision remains open after model exit: %s", binding.lease.job_id)

    async def admit_subagent(self, sid: str, spec, task: str) -> JobRegistration:
        execution = _SEND.get()
        if execution is None or execution.runtime is not self or execution.actor.session_id != sid:
            raise JobError("task_send_required")
        if spec.lifecycle != "linked":
            raise JobError("task_adapter_unavailable:ephemeral_subagent")
        assistant_seq, call_id = current_invocation(sid)
        if assistant_seq is None or not call_id:
            raise JobError("persisted_tool_call_required")
        role = await self.manager.get_job(execution.actor, execution.lease.job_id)
        return await self.manager.register_job(execution.actor, kind="subagent", title=spec.name,
            parent=execution.lease, independent=role.execution_role == "main",
            call_ref=await self.manager.store.message_ref(sid, assistant_seq), tool_call_id=call_id,
            request={"spec": asdict(spec), "input": task})

    async def bind_subagent(self, parent: str, registration: JobRegistration, sid: str) -> Execution:
        bound = await self.manager.bind_session(self.actor(parent), registration.job.job_id, sid)
        # The child's SID now authorizes its own model sends and execution evidence.
        return await self.claim(self.actor(sid), JobRegistration(bound, registration.operation, False))

    async def record_subagent(self, binding: Execution, result: dict) -> dict:
        return await self.record_scope(binding, result)

    async def record_scope(self, binding: Execution, result: dict) -> dict:
        # The execution has stopped before its result is committed/projected.
        # A slow compatibility write must not renew a finished lease or turn a
        # successful canonical result into a spurious heartbeat failure.
        binding.stop_renewal.set()
        result = dict(result)
        recorded_usage = await self.manager.execution_usage(binding.actor, binding.lease.job_id, generation=binding.lease.generation)
        # Empty is an authoritative execution ledger too. A replayed result's
        # old rollup must never become this generation's newly incurred usage.
        result["usage"] = recorded_usage
        outcome = {"completed": "succeeded", "skipped": "succeeded", "cancelled": "cancelled",
                   "timed_out": "timed_out", "budget_exceeded": "budget_exceeded"}.get(result.get("status"), "failed")
        unknown = await self.manager.db.fetchone(f"SELECT job_id FROM {self.manager.jobs} "
            "WHERE control_owner_job_id=? AND resource_state IN ('unknown','running') LIMIT 1", (binding.lease.job_id,))
        try:
            await self.manager.record_execution_result(binding.actor, binding.lease, outcome=outcome,
                result=result, resource_state="unknown" if unknown else "stopped", require_children_stopped=True)
        except JobError as exc:
            from .executor_registry import FENCE_ERRORS
            if exc.code in FENCE_ERRORS or exc.code in ("required_execution_unsettled", "required_result_unhandled", "required_input_unhandled"):
                exc.execution_result = result
            raise
        return (await self.manager.get_job(binding.actor, binding.lease.job_id)).result

    def receipt(self, registration: JobRegistration) -> dict:
        return {"job_id": registration.job.job_id, "operation_id": registration.operation.operation_id,
                "session_id": registration.job.session_id, "run_id": registration.operation.operation_id[3:],
                "mode": "create", "status": "accepted", "replayed": registration.replayed}
