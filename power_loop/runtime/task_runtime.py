"""Opt-in execution adapters for the experimental task store.

Main sends, host continuations and standalone, linked subagent creates. This is deliberately
not enabled by opening a v10 database or by a model argument. Other async
adapters fail closed until they participate in the same durable protocol.
"""

from __future__ import annotations

import asyncio
import weakref
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .invocation import current_invocation
from .jobs import JobActor, JobError, JobLease, JobManager, JobRegistration


@dataclass(frozen=True)
class Execution:
    runtime: TaskRuntime
    actor: JobActor
    lease: JobLease
    stop_renewal: asyncio.Event = field(default_factory=asyncio.Event, compare=False, repr=False)
    resource_evidence: dict = field(default_factory=dict, compare=False, repr=False)
    consumed_inputs: dict = field(default_factory=dict, compare=False, repr=False)
    observed_inputs: set = field(default_factory=set, compare=False, repr=False)


_SEND: ContextVar[Execution | None] = ContextVar("power_loop_task_send", default=None)
_OWNER: ContextVar[Execution | None] = ContextVar("power_loop_task_owner", default=None)
_TASK_OWNERS: ContextVar[tuple] = ContextVar("power_loop_task_owners", default=())


@dataclass
class SessionSend:
    """The session-native execution identity of one running send (design 215).

    No job, generation, lease upgrade or supervision: the claim transaction already
    installed the owner fence, and ``final`` writes the result message plus the input
    terminal states. The loop only appends messages and advances this in-memory record.
    """

    store: object
    target: object
    batch: object
    ownership: object
    stop_renewal: asyncio.Event = field(default_factory=asyncio.Event, compare=False, repr=False)
    consumed_inputs: list = field(default_factory=list, compare=False, repr=False)
    usage_totals: dict = field(default_factory=dict, compare=False, repr=False)
    error_code: str | None = field(default=None, compare=False, repr=False)
    resource_evidence: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def session_id(self):
        return self.target.session_id

    @property
    def send_index(self):
        return self.batch.send_index


_SESSION_SEND: ContextVar[SessionSend | None] = ContextVar("power_loop_session_send", default=None)


def current_session_send(store=None, session_id=None) -> SessionSend | None:
    send = _SESSION_SEND.get()
    if send is None:
        return None
    if store is not None and send.store is not store:
        return None
    if session_id is not None and send.session_id != session_id:
        return None
    return send


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


async def check_dispatch(session_id: str | None, *, invocation=None) -> None:
    # A live session send trusts its own committed writes; ownership is enforced by
    # the conditional message/state writes, not a per-dispatch read (AGENTS.md 热路径).
    if current_session_send(session_id=session_id) is not None:
        return
    execution = _SEND.get()
    if execution is not None and execution.actor.session_id == session_id:
        execution.runtime.check_local_admission()
        if execution.resource_evidence.get("resource_state") == "unknown":
            raise JobError("unresolved_prior_attempt")
        if execution.resource_evidence.get("_control_state") is not None:
            from .control_runtime import current
            control = current(execution)
            control.require_admission()
            if control.managed_check:
                from .managed_effects import require_source_resolved
                manager = execution.runtime.manager
                async with manager._write() as tx:
                    await require_source_resolved(manager, tx, await manager._row(tx, execution.lease.job_id),
                                                  invocation=invocation)
                if invocation is None:
                    control.managed_check = False
            return
        await execution.runtime.manager.check_execution(execution.actor, execution.lease)
        from .managed_effects import require_source_resolved
        async with execution.runtime.manager._write() as tx:
            await require_source_resolved(execution.runtime.manager, tx,
                await execution.runtime.manager._row(tx, execution.lease.job_id), invocation=invocation)


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
    from .control_runtime import current, source
    from .tool_contracts import resolve
    current(execution)
    implementation = resolve(registered)
    if implementation is not None:
        if source(execution)["execution_role"] not in implementation.runtime_contract.allowed_roles:
            raise JobError("execution_role_forbids_tool")
    if declared.control_only:
        from .task_control import _source

        seq, call_id = current_invocation(session_id)
        if seq is None or not call_id:
            raise JobError("persisted_tool_call_required")
        manager = execution.runtime.manager
        from .control_runtime import flush_binding
        await flush_binding(execution)
        ref = await manager.store.message_ref(session_id, seq)
        async with manager._write() as tx:
            await _source(manager, tx, execution.actor, execution.lease, ref, call_id)
        return registered
    await check_dispatch(session_id, invocation=current_invocation(session_id))
    return registered


async def observe_model_inputs(pipeline, messages, round_index, *, provider_call=True):
    return


def mark_unconfirmed_resource() -> None:
    send = current_session_send()
    if send is not None:
        send.resource_evidence["resource_state"] = "unknown"
    execution = _SEND.get()
    if execution is not None:
        execution.resource_evidence["resource_state"] = "unknown"


def preserve_model_usage(session_id, usage):
    """Retain this pipeline's counters even when a later checkpoint raises."""
    send = current_session_send(session_id=session_id)
    if send is not None:
        send.usage_totals = dict(usage)
    execution = _SEND.get()
    if execution is not None and execution.actor.session_id == session_id:
        execution.resource_evidence["model_usage_totals"] = dict(usage)


def atomic_pending_required(store, sid):
    # A store-backed session send always records assistant tool_calls and their
    # pending ids in one atomic append (design 215 §8.3); no job binding is needed.
    if current_session_send(store, sid) is not None:
        return True
    execution = _SEND.get()
    return (execution is not None and execution.runtime.manager.store is store
            and execution.actor.session_id == sid)


def tool_receipt_required(store, sid: str, assistant_seq: int | None, call_id: str) -> bool:
    """Only a known operation-free call may clear pending in its message append.

    A missing in-memory operation list is uncertain (for example a failed return
    checkpoint), so retain the durable marker until the receipt lookup finishes.
    Session-native sends have no job operations: the tool message clears pending.
    """
    if current_session_send(store, sid) is not None:
        return False
    execution = _SEND.get()
    if (execution is None or execution.runtime.manager.store is not store
            or execution.actor.session_id != sid or assistant_seq is None):
        return False
    return execution.resource_evidence.get("_control_tool_operations", {}).get((assistant_seq, call_id)) != []


async def bind_tool_receipt(store, sid: str, assistant_seq: int | None, call_id: str, receipt_seq: int) -> None:
    if current_session_send(store, sid) is not None:
        return
    execution = _SEND.get()
    if (
        execution is None
        or execution.runtime.manager.store is not store
        or execution.actor.session_id != sid
        or assistant_seq is None
    ):
        return
    manager = execution.runtime.manager
    known = execution.resource_evidence.get("_control_tool_operations", {}).get((assistant_seq, call_id))
    if known == []:
        return
    if known is not None:
        from .control_runtime import flush_binding
        await flush_binding(execution)
    operation = await manager.operation_for_call(execution.actor, await store.message_ref(sid, assistant_seq), call_id)
    if operation is not None:
        await manager.reconcile_receipt(
            execution.actor, operation.operation_id, await store.message_ref(sid, receipt_seq)
        )
        from .tool_call_plans import finish_receipt
        await finish_receipt(manager, execution.actor, await store.message_ref(sid, assistant_seq), call_id)


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
        boot_contract_hash: str | None = None,
        require_host_send_steps: bool = False,
        checkpoint_policy=None,
    ):
        JobActor(tenant_id, principal_id, "validation")
        if not all(isinstance(v, str) and 0 < len(v) <= 64 for v in (executor_id, boot_id)):
            raise JobError("invalid_executor")
        if isinstance(lease_ms, bool) or not isinstance(lease_ms, int) or not 300 <= lease_ms <= 300000:
            raise JobError("invalid_lease_duration")
        if type(require_host_send_steps) is not bool:
            raise JobError("invalid_host_send_step_requirement")
        self.require_host_send_steps = self._original_host_step_requirement = require_host_send_steps
        from .execution_state import CheckpointPolicy
        if checkpoint_policy is not None and type(checkpoint_policy) is not CheckpointPolicy:
            raise JobError("invalid_checkpoint_policy")
        self.checkpoint_policy = checkpoint_policy or CheckpointPolicy()
        self._adapter_tasks = set()
        self.manager = manager
        self.tenant_id, self.principal_id = tenant_id, principal_id
        self.executor_id, self.boot_id, self.lease_ms = executor_id, boot_id, lease_ms
        self._local_executions: dict = {}
        self._model_executions: dict = {}
        from .executor_lifecycle import ExecutorLifecycle

        self._lifecycle = ExecutorLifecycle(self, boot_contract_hash) if boot_contract_hash is not None else None

    def check_local_admission(self):
        from .session_ownership import require_current_ownership
        require_current_ownership(self.manager.store)
        if self._lifecycle is not None:
            self._lifecycle.check_admission()

    async def ensure_started(self):
        if self._lifecycle is not None:
            self._lifecycle.check_admission()
        async with asyncio.timeout(2):
            await self._flush_owned_checkpoints(failed_only=True)
        if self._lifecycle is not None:
            await self._lifecycle.start()

    async def flush_control_checkpoints(self, *, timeout_s=2):
        if type(timeout_s) not in (int, float) or not 0 <= timeout_s <= 60:
            raise JobError("invalid_controller_checkpoint_timeout")
        if timeout_s:
            async with asyncio.timeout(timeout_s):
                await self._flush_owned_checkpoints()
        return {"execution_checkpoint_pending": self._owned_checkpoint_pending()}

    def _owned_checkpoint_pending(self):
        from .control_runtime import current
        return sum(bool(current(binding)._pending or current(binding)._dirty)
                   for binding in self.manager._execution_controls.values() if binding.runtime is self)

    async def _flush_owned_checkpoints(self, *, failed_only=False):
        from .control_runtime import current, flush_binding
        for binding in tuple(self.manager._execution_controls.values()):
            if binding.runtime is self and (not failed_only or current(binding).checkpoint_failed):
                await flush_binding(binding)

    async def start(self):
        if self._lifecycle is None:
            raise JobError("executor_lifecycle_not_configured")
        await self.ensure_started()

    async def shutdown(self, *, timeout_s=30.0, grace_s=1.0, reason="host shutdown"):
        if self._lifecycle is None:
            raise JobError("executor_lifecycle_not_configured")
        result = await self._lifecycle.shutdown(timeout_s=timeout_s, grace_s=grace_s, reason=reason)
        return self._completion_status(result)

    def _completion_status(self, result):
        result['execution_checkpoint_pending'] = self._owned_checkpoint_pending()
        if result['execution_checkpoint_pending']:
            result['store_safe_to_close'] = False
        return result

    async def executor_status(self):
        if self._lifecycle is None:
            raise JobError("executor_lifecycle_not_configured")
        return self._completion_status(await self._lifecycle.status())

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
        )
        return Execution(self, actor, lease)

    @asynccontextmanager
    async def execution(self, binding: Execution, *, owner: bool = False, context=None, cancel_token=None, initialize_model=False):
        """Bind one job attempt; only its session coordinator renews execution."""
        _check_task_owner(self)
        if (not isinstance(binding, Execution) or binding.runtime is not self
                or (binding.actor.tenant_id, binding.actor.principal_id) != (self.tenant_id, self.principal_id)
                or (binding.lease.executor_id, binding.lease.boot_id) != (self.executor_id, self.boot_id)):
            raise JobError("execution_binding_mismatch")
        if (owner or context is not None) and binding.stop_renewal.is_set():
            raise JobError("controller_execution_exited")
        await self.ensure_started()
        self.check_local_admission()
        if (owner or context is not None) and binding.stop_renewal.is_set():
            raise JobError("controller_execution_exited")
        if initialize_model:
            from .model_attempts import enable
            await enable(binding)  # One admission transaction owns all checks.
        else:
            await self.manager.check_execution(binding.actor, binding.lease)
        variable = context if context is not None else _OWNER if owner else _SEND
        token = variable.set(binding)
        owner_token = _enter_task_owner(self)
        task = asyncio.current_task()
        local = {"binding": binding, "task": task, "signalled": False, "cancel_token": cancel_token}
        local_key = id(local)
        try:
            self._local_executions[local_key] = local
            yield binding
        finally:
            self._local_executions.pop(local_key, None)
            variable.reset(token)
            _TASK_OWNERS.reset(owner_token)


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
