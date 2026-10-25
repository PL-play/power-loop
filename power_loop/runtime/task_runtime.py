"""Opt-in execution adapters for the experimental task store.

Main sends and standalone, linked subagent creates only. This is deliberately
not enabled by opening a v10 database or by a model argument. Other async
adapters fail closed until they participate in the same durable protocol.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
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
_TASK_OWNERS: ContextVar[tuple] = ContextVar("power_loop_task_owners", default=())


def _check_task_owner(runtime):
    task = asyncio.current_task()
    for task_ref, owner in _TASK_OWNERS.get():
        if (
            task_ref() is task
            and owner is not runtime
            and (owner._lifecycle is not None or runtime._lifecycle is not None)
        ):
            raise JobError("executor_task_owner_conflict")


def _enter_task_owner(runtime):
    _check_task_owner(runtime)
    task = asyncio.current_task()
    current = tuple((ref, owner) for ref, owner in _TASK_OWNERS.get() if ref() is task)
    return _TASK_OWNERS.set((*current, (weakref.ref(task), runtime)))


def reject_unregistered_adapter(name: str, loop=None) -> None:
    from power_loop.core.agent_context import get_current_loop

    loop = loop or get_current_loop()
    if getattr(loop, "task_runtime", None) is not None:
        raise JobError("task_adapter_unavailable:" + name)


async def check_dispatch(session_id: str | None) -> None:
    execution = _SEND.get()
    if execution is not None and execution.actor.session_id == session_id:
        execution.runtime.check_local_admission()
        if execution.resource_evidence.get("resource_state") == "unknown":
            raise JobError("unresolved_prior_attempt")
        await execution.runtime.manager.check_execution(execution.actor, execution.lease)
        from .managed_effects import require_source_resolved
        async with execution.runtime.manager._write() as tx:
            await require_source_resolved(execution.runtime.manager, tx,
                await execution.runtime.manager._row(tx, execution.lease.job_id))


async def check_tool_dispatch(session_id, tool_name, registry, tool_args=None):
    """Only the library-owned query/cancel handler may bypass a spent lease.

    A tool name is not authority. Trusted host formatting wrappers must preserve
    the library handler with functools.wraps and must remain control-only.
    """
    execution = _SEND.get()
    if execution is None or execution.actor.session_id != session_id:
        await check_dispatch(session_id)
        return registry.get(tool_name)
    from .tool_contracts import ToolContractError
    from .tool_contracts import policy as tool_policy
    registered = registry.get(tool_name)
    try:
        declared = tool_policy(registered, tool_args or {}, runtime=execution.runtime)
    except ToolContractError as exc:
        raise JobError(str(exc)) from exc
    from .tool_contracts import resolve
    implementation = resolve(registered)
    if implementation is not None:
        async with execution.runtime.manager._write() as tx:
            source = await execution.runtime.manager._authorize(tx, execution.actor, execution.lease.job_id)
            if source["execution_role"] not in implementation.runtime_contract.allowed_roles:
                raise JobError("execution_role_forbids_tool")
    if declared.control_only:
        from .task_control import _source

        seq, call_id = current_invocation(session_id)
        if seq is None or not call_id:
            raise JobError("persisted_tool_call_required")
        manager = execution.runtime.manager
        ref = await manager.store.message_ref(session_id, seq)
        async with manager._write() as tx:
            await _source(manager, tx, execution.actor, execution.lease, ref, call_id)
        return registered
    from .model_budgets import _policy as budget_policy

    async with execution.runtime.manager._write() as tx:
        policy = await budget_policy(execution.runtime, tx)
    if policy is not None:
        if not declared.native_admitted:
            raise JobError("budget_tool_adapter_unavailable:" + tool_name)
    await check_dispatch(session_id)
    return registered


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
    actual = [
        {"seq": seq, "role": message.get("role"), "content": message.get("content")}
        for message, seq in zip(pipeline.history, seqs, strict=False)
        if seq is not None and id(message) in included and isinstance(message.get("content"), str)
    ]
    execution.resource_evidence["processing_snapshot_id"] = await execution.runtime.manager.prepare_processing(
        execution.actor, execution.lease, round_index=round_index, send_index=pipeline.send_index, messages=actual
    )


def mark_unconfirmed_resource() -> None:
    execution = _SEND.get()
    if execution is not None:
        execution.resource_evidence["resource_state"] = "unknown"


def preserve_model_usage(session_id, usage):
    """Retain this pipeline's counters even when a later checkpoint raises."""
    execution = _SEND.get()
    if execution is not None and execution.actor.session_id == session_id:
        execution.resource_evidence["model_usage_totals"] = dict(usage)


async def bind_tool_receipt(store, sid: str, assistant_seq: int | None, call_id: str, receipt_seq: int) -> None:
    execution = _SEND.get()
    if (
        execution is None
        or execution.runtime.manager.store is not store
        or execution.actor.session_id != sid
        or assistant_seq is None
    ):
        return
    manager = execution.runtime.manager
    operation = await manager.operation_for_call(execution.actor, await store.message_ref(sid, assistant_seq), call_id)
    if operation is not None:
        await manager.reconcile_receipt(
            execution.actor, operation.operation_id, await store.message_ref(sid, receipt_seq)
        )


class TaskRuntime:
    """Trusted host configuration, shared by loops using the same opened store."""

    def __init__(
        self,
        manager: JobManager,
        *,
        tenant_id: str,
        principal_id: str,
        executor_id: str,
        boot_id: str,
        lease_ms: int = 30000,
        workflow_resources=None,
        boot_contract_hash: str | None = None,
        budget_config=None,
        require_host_send_steps: bool = False,
    ):
        JobActor(tenant_id, principal_id, "validation")
        if not all(isinstance(v, str) and 0 < len(v) <= 64 for v in (executor_id, boot_id)):
            raise JobError("invalid_executor")
        if isinstance(lease_ms, bool) or not isinstance(lease_ms, int) or not 300 <= lease_ms <= 300000:
            raise JobError("invalid_lease_duration")
        if type(require_host_send_steps) is not bool:
            raise JobError("invalid_host_send_step_requirement")
        self.require_host_send_steps = self._original_host_step_requirement = require_host_send_steps
        if workflow_resources is not None:
            from .workflow_resources import WorkflowResourceRegistry

            if type(workflow_resources) is not WorkflowResourceRegistry:
                raise JobError("workflow_resource_registry_invalid")
        if budget_config is not None:
            from .budget_contracts import BudgetConfig

            if type(budget_config) is not BudgetConfig or boot_contract_hash is None:
                raise JobError("budget_runtime_requires_trusted_boot")
        self.budget_config = budget_config
        self._original_budget_config = budget_config
        self._budget_handles = {}
        self._adapter_tasks = set()
        self.manager = manager
        self.tenant_id, self.principal_id = tenant_id, principal_id
        self.executor_id, self.boot_id, self.lease_ms = executor_id, boot_id, lease_ms
        self.workflow_resources = workflow_resources
        self._workflow_handles: dict = {}
        self._local_executions: dict = {}
        self._model_executions: dict = {}
        from .executor_lifecycle import ExecutorLifecycle

        self._lifecycle = ExecutorLifecycle(self, boot_contract_hash) if boot_contract_hash is not None else None

    def check_local_admission(self):
        if self._lifecycle is not None:
            self._lifecycle.check_admission()

    async def ensure_started(self):
        if self._lifecycle is not None:
            await self._lifecycle.start()

    async def start(self):
        if self._lifecycle is None:
            raise JobError("executor_lifecycle_not_configured")
        await self._lifecycle.start()

    async def shutdown(self, *, timeout_s=30.0, grace_s=1.0, reason="host shutdown"):
        if self._lifecycle is None:
            raise JobError("executor_lifecycle_not_configured")
        return await self._lifecycle.shutdown(timeout_s=timeout_s, grace_s=grace_s, reason=reason)

    async def executor_status(self):
        if self._lifecycle is None:
            raise JobError("executor_lifecycle_not_configured")
        return await self._lifecycle.status()

    def track_task(self, task, *, cancel_token=None, job_id=None):
        if not isinstance(task, asyncio.Task):
            raise JobError("executor_task_handle_required")
        # Keep original mechanical writes reachable even for a legacy runtime
        # without the optional automatic boot lifecycle.
        self._adapter_tasks.add(task)

        def observe(done):
            self._adapter_tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(observe)
        if self._lifecycle is not None:
            self._lifecycle.track_task(task, cancel_token=cancel_token, job_id=job_id)

    def actor(self, sid: str) -> JobActor:
        return JobActor(self.tenant_id, self.principal_id, sid)

    def root_context(self, session_id, *, namespace, payload):
        """Bind trusted per-root provenance before entering the main send."""
        from .root_context import bind
        return bind(self, session_id, namespace=namespace, payload=payload)

    def host_send_step(self, session_id, *, namespace, plan_hash, step_id):
        """Bind a trusted host processing step inside its original root scope."""
        from .host_send_steps import bind
        return bind(self, session_id, namespace=namespace, plan_hash=plan_hash, step_id=step_id)

    async def claim(self, actor: JobActor, registration: JobRegistration) -> Execution:
        _check_task_owner(self)
        await self.ensure_started()
        lease = await self.manager.claim_job(
            actor,
            registration.job.job_id,
            executor_id=self.executor_id,
            boot_id=self.boot_id,
            expected_version=registration.job.version,
            lease_ms=self.lease_ms,
        )
        return Execution(self, actor, lease)

    @asynccontextmanager
    async def execution(self, binding: Execution, *, owner: bool = False, context=None, cancel_token=None):
        """Renew only while this attempt runs; a failed renewal stops dispatch."""
        _check_task_owner(self)
        await self.ensure_started()
        self.check_local_admission()
        await self.manager.check_execution(binding.actor, binding.lease)
        variable = context if context is not None else _OWNER if owner else _SEND
        token = variable.set(binding)
        owner_token = _enter_task_owner(self)
        task = asyncio.current_task()
        local = {"binding": binding, "task": task, "signalled": False, "cancel_token": cancel_token}
        local_key = id(local)
        lost = []

        def supervision_lost(exc):
            lost.append(exc)
            # The body may catch cancellation and persist its result before
            # this context manager exits. Preserve the real cause immediately.
            binding.resource_evidence["supervision_error"] = getattr(exc, "code", type(exc).__name__)

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
                            supervision_lost(exc)
                        from .executor_lifecycle import signal_token

                        local["signalled"] = signal_token(cancel_token, "task supervision lost")
                        if not local["signalled"]:
                            # A malformed/read-only host token cannot leave an
                            # unconfirmed executor running indefinitely. It is
                            # a supervision failure, not tolerated node cancel.
                            supervision_lost(JobError("task_cancel_signal_unavailable"))
                            local["signalled"] = bool(task.cancel("task supervision lost"))
                    else:
                        if not isinstance(exc, JobError) or exc.code != "cancel_requested":
                            supervision_lost(exc)
                        task.cancel("task supervision lost")

        heartbeat = None
        try:
            coroutine = renew()
            try:
                heartbeat = asyncio.create_task(coroutine, name="job-lease-" + binding.lease.job_id)
            except BaseException:
                coroutine.close()
                raise
            self._local_executions[local_key] = local
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
            try:
                if heartbeat is not None:
                    heartbeat.cancel()
                    await asyncio.gather(heartbeat, return_exceptions=True)
            finally:
                variable.reset(token)
                _TASK_OWNERS.reset(owner_token)

    @asynccontextmanager
    async def model_send(self, loop, sid: str, send_index: int, *, cancel_token=None):
        owner_token = _enter_task_owner(self)
        local = None
        try:
            await self.ensure_started()
            local = {"task": asyncio.current_task(), "cancel_token": cancel_token, "signalled": False}
            self._model_executions[id(local)] = local
            async with self._model_send(loop, sid, send_index, cancel_token=cancel_token) as evidence:
                yield evidence
        finally:
            if local is not None:
                self._model_executions.pop(id(local), None)
            _TASK_OWNERS.reset(owner_token)

    @asynccontextmanager
    async def _model_send(self, loop, sid: str, send_index: int, *, cancel_token=None):
        from .execution_roles import session_role

        if (type(self.require_host_send_steps) is not bool
                or self.require_host_send_steps != self._original_host_step_requirement):
            raise JobError("host_send_step_configuration_changed")
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
                if (
                    job.kind == "workflow_node"
                    and session is not None
                    and session.parent_session_id != owner.actor.session_id
                ):
                    from .workflow_restore_execution import controls_node

                    async with self.manager._write() as tx:
                        restored = await controls_node(
                            self.manager, tx, await self.manager._row(tx, job.job_id), owner.actor, sid
                        )
                if (
                    job.kind == "workflow_node"
                    and session is not None
                    and (session.parent_session_id == owner.actor.session_id or restored)
                    and session.metadata.get("task_owner_job_id") == job.job_id
                ):
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
        request = {"send_index": send_index, "model_attempt_protocol": 1}
        if owner is None:
            from .root_context import for_send
            provenance = for_send(self, sid)
            if provenance is not None:
                request["host_context"] = provenance
            from .host_send_steps import FIELD
            from .host_send_steps import for_send as step_for_send
            step = step_for_send(self, sid)
            if step is None and self.require_host_send_steps:
                raise JobError("host_send_step_required")
            if step is not None:
                request[FIELD] = step
        registration = await self.manager.register_job(
            registering_actor,
            kind="send",
            session_id=sid,
            parent=owner.lease if owner else None,
            system_key=f"send:{sid}:{send_index}" if registering_actor != actor else f"send:{send_index}",
            request=request,
        )
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
                    last = await self.manager.db.fetchone(
                        f"SELECT MAX(seq) AS seq FROM {self.manager.store.t.messages} "
                        "WHERE session_id=? AND role='assistant' AND send_index=?",
                        (sid, send_index),
                    )
                    if last is None or last["seq"] is None:
                        raise JobError("processing_response_missing")
                    decision_args = {
                        "snapshot_id": snapshot,
                        "response_ref": await self.manager.store.message_ref(sid, last["seq"]),
                        "reason": "Original session completed its registered processing send with no unresolved tools.",
                    }
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
            evidence.setdefault("usage", dict(binding.resource_evidence.get("model_usage_totals", {})))
            raise
        finally:
            if binding.resource_evidence.get("model_attempts_inflight"):
                binding.resource_evidence["resource_state"] = "unknown"
            last = await self.manager.db.fetchone(
                f"SELECT MAX(seq) AS seq FROM {self.manager.store.t.messages} "
                "WHERE session_id=? AND role='assistant' AND send_index=?",
                (sid, send_index),
            )
            if last is not None and last["seq"] is not None:
                evidence["last_assistant_ref"] = (await self.manager.store.message_ref(sid, last["seq"])).to_dict()
            await self.manager.close_model_execution(
                actor,
                binding.lease,
                evidence=evidence,
                resource_state=binding.resource_evidence.get("resource_state", "stopped"),
            )
            from .finalization import after_send

            try:
                await after_send(self.manager, actor, binding.lease)
            except Exception:
                # The model exit is already durable. A failed mechanical
                # close leaves supervision open for reconciliation; it must
                # not turn that completed business send into a retry.
                logging.getLogger(__name__).exception(
                    "supervision remains open after model exit: %s", binding.lease.job_id
                )

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
        return await self.manager.register_job(
            execution.actor,
            kind="subagent",
            title=spec.name,
            parent=execution.lease,
            independent=role.execution_role == "main",
            call_ref=await self.manager.store.message_ref(sid, assistant_seq),
            tool_call_id=call_id,
            request={"spec": asdict(spec), "input": task, "completion_window_protocol": 1},
        )

    async def bind_subagent(self, parent: str, registration: JobRegistration, sid: str) -> Execution:
        bound = await self.manager.bind_session(self.actor(parent), registration.job.job_id, sid)
        # The child's SID now authorizes its own model sends and execution evidence.
        return await self.claim(self.actor(sid), JobRegistration(bound, registration.operation, False))

    async def record_subagent(self, binding: Execution, result: dict) -> dict:
        return await self.record_scope(binding, result)

    async def record_scope(self, binding: Execution, result: dict) -> dict:
        try:
            return await self._record_scope(binding, result)
        except asyncio.CancelledError as exc:
            # Capture before any read/write can be interrupted. Usage is still
            # reconstructed from the original ledger by the failure writer.
            if not isinstance(getattr(exc, "execution_result", None), dict):
                exc.execution_result = dict(result)
            raise

    async def _record_scope(self, binding: Execution, result: dict) -> dict:
        # The execution has stopped before its result is committed/projected.
        # A slow compatibility write must not renew a finished lease or turn a
        # successful canonical result into a spurious heartbeat failure.
        binding.stop_renewal.set()
        result = dict(result)
        recorded_usage = await self.manager.execution_usage(
            binding.actor, binding.lease.job_id, generation=binding.lease.generation
        )
        # Empty is an authoritative execution ledger too. A replayed result's
        # old rollup must never become this generation's newly incurred usage.
        result["usage"] = recorded_usage
        supervision_error = binding.resource_evidence.get("supervision_error")
        if supervision_error is not None:
            result.update(execution_status=result.get("status"), status="failed", supervision_error=supervision_error)
        outcome = {
            "completed": "succeeded",
            "skipped": "succeeded",
            "cancelled": "cancelled",
            "timed_out": "timed_out",
            "budget_exceeded": "budget_exceeded",
        }.get(result.get("status"), "failed")
        unknown = await self.manager.db.fetchone(
            f"SELECT job_id FROM {self.manager.jobs} "
            "WHERE control_owner_job_id=? AND resource_state IN ('unknown','running') LIMIT 1",
            (binding.lease.job_id,),
        )
        try:
            await self.manager.record_execution_result(
                binding.actor,
                binding.lease,
                outcome=outcome,
                result=result,
                resource_state="unknown" if unknown else "stopped",
                require_children_stopped=True,
            )
        except JobError as exc:
            from .executor_registry import FENCE_ERRORS

            if exc.code in FENCE_ERRORS or exc.code in (
                "required_execution_unsettled",
                "required_result_unhandled",
                "required_input_unhandled",
            ):
                exc.execution_result = result
            raise
        except asyncio.CancelledError as exc:
            # The original completed output is still evidence if a containing
            # workflow heartbeat cancels this task during its result write.
            exc.execution_result = result
            raise
        return (await self.manager.get_job(binding.actor, binding.lease.job_id)).result

    def receipt(self, registration: JobRegistration) -> dict:
        return {
            "job_id": registration.job.job_id,
            "operation_id": registration.operation.operation_id,
            "session_id": registration.job.session_id,
            "run_id": registration.operation.operation_id[3:],
            "mode": "create",
            "status": "accepted",
            "replayed": registration.replayed,
        }
