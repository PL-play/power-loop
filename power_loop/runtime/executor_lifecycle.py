"""Opt-in host boot startup and bounded local shutdown.

The durable registry remains authoritative. Local handles only allow this process
to signal its own work; their disappearance never proves remote/provider stop.
"""

from __future__ import annotations

import asyncio
import math

from . import executor_registry as registry
from .jobs import JobError


def signal_token(token, reason):
    """A present token may be read-only. Count only an observed stop signal."""
    if token is None:
        return False
    try:
        token.cancel(reason)
        return bool(token.is_cancelled())
    except Exception:
        return False


def _observe(task):
    if not task.cancelled():
        task.exception()


def _create_task(coroutine, *, name):
    try:
        task = asyncio.create_task(coroutine, name=name)
    except BaseException:
        coroutine.close()
        raise
    task.add_done_callback(_observe)
    return task


class ExecutorLifecycle:
    def __init__(self, runtime, contract_hash):
        if not registry._digest(contract_hash):
            raise JobError("invalid_executor_contract")
        self.runtime = runtime
        self.manager = runtime.manager
        self.store = runtime.manager.store
        self.store_id = self.store.store_id
        self.database = self.manager.db
        self.tables = (self.manager.prefix, self.manager.jobs, self.manager.events)
        self.identity = registry.ExecutorIdentity(
            runtime.tenant_id, runtime.principal_id, runtime.executor_id, runtime.boot_id
        )
        self.contract_hash = contract_hash
        self.closed = False
        self.failure = None
        self._start_task = None
        self._heartbeat_task = None
        self._shutdown_task = None
        self._shutdown_deadline = None
        self._shutdown_grace = None
        self._shutdown_reason = None
        self._control_task = None
        self._version = None
        self._owned_tasks = {}
        self._shutdown_waiters = set()

    def check_admission(self):
        try:
            self._check_manager_binding()
        except JobError:
            self.closed = True
            self.failure = "executor_binding_mismatch"
        if (
            self.runtime.manager is not self.manager
            or self.manager.store is not self.store
            or self.store.store_id != self.store_id
            or any(
                getattr(self.runtime, key) != getattr(self.identity, key)
                for key in ("tenant_id", "principal_id", "executor_id", "boot_id")
            )
        ):
            self.closed = True
            self.failure = "executor_binding_mismatch"
        if self.closed:
            raise JobError(self.failure or "executor_not_accepting")

    async def start(self):
        self.check_admission()
        if self._start_task is None:
            try:
                self._start_task = _create_task(self._startup(), name="executor-start-" + self.identity.boot_id)
            except BaseException:
                self.closed = True
                self.failure = "executor_startup_failed"
                self._signal(self._locals(), force=False)
                raise
        # Cancelling one caller does not revoke an already accepted registration.
        await asyncio.shield(self._start_task)
        self.check_admission()

    async def _idempotent(self, function, **kwargs):
        self._check_manager_binding()
        try:
            return await function(self.manager, self.identity, **kwargs)
        except JobError:
            raise
        except Exception:
            # These registry ports recall the exact original operation. No
            # business work/provider call is repeated after an uncertain ACK.
            self._check_manager_binding()
            return await function(self.manager, self.identity, **kwargs)

    def _check_manager_binding(self):
        if (
            self.manager.store is not self.store
            or self.store.store_id != self.store_id
            or self.manager.db is not self.database
            or (self.manager.prefix, self.manager.jobs, self.manager.events) != self.tables
        ):
            raise JobError("executor_binding_mismatch")

    async def _startup(self):
        try:
            await self._idempotent(
                registry.register, contract_hash=self.contract_hash, startup_lease_ms=self.runtime.lease_ms
            )
            self.check_admission()
            await self._idempotent(
                registry.activate, expected_version=1, system_key="runtime-startup", lease_ms=self.runtime.lease_ms
            )
            self.check_admission()
            current = await registry.get(self.manager, self.identity)
            if not current["admission_open"]:
                raise JobError("executor_not_accepting")
            self._version = current["version"]
            self._heartbeat_task = _create_task(self._heartbeat(), name="executor-heartbeat-" + self.identity.boot_id)
        except BaseException:
            self.closed = True
            self.failure = "executor_startup_failed"
            self._signal(self._locals(), force=False)
            raise

    async def _heartbeat(self):
        try:
            while not self.closed:
                await asyncio.sleep(self.runtime.lease_ms / 3000)
                if self.closed:
                    return
                self.check_admission()
                receipt = await self._idempotent(
                    registry.heartbeat,
                    expected_version=self._version,
                    system_key="runtime-heartbeat:" + str(self._version),
                    lease_ms=self.runtime.lease_ms,
                )
                self._version = receipt["executor"]["version"]
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failure = "executor_supervision_lost"
            # Do not await ourselves while shutting down the heartbeat.
            self._begin_shutdown(5.0, 0.25, "executor heartbeat unavailable")

    def track_task(self, task, *, cancel_token=None, job_id=None):
        """Retain adapter tasks, including admission queues and exit delivery."""
        if not isinstance(task, asyncio.Task):
            raise JobError("executor_task_handle_required")
        if task.done():
            return
        existing = self._owned_tasks.get(task)
        value = {"task": task, "cancel_token": cancel_token, "job_id": job_id, "signalled": False}
        if existing is not None:
            if existing["cancel_token"] is not cancel_token or existing["job_id"] != job_id:
                raise JobError("executor_task_binding_conflict")
            return
        self._owned_tasks[task] = value
        task.add_done_callback(lambda done: self._owned_tasks.pop(done, None))
        if self.closed:
            # Late handle registration cannot reopen admission. The durable
            # dispatch gate still checks the original boot before paid work.
            self._signal([value], force=False)

    @staticmethod
    def _validate_shutdown(timeout_s, grace_s, reason):
        for value in (timeout_s, grace_s):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 300:
                raise JobError("invalid_executor_shutdown_timeout")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000 or "\x00" in reason:
            raise JobError("invalid_executor_drain_reason")

    def _begin_shutdown(self, timeout_s, grace_s, reason):
        self._validate_shutdown(timeout_s, grace_s, reason)
        self.closed = True
        if self._shutdown_deadline is None:
            self._shutdown_deadline = asyncio.get_running_loop().time() + timeout_s
            self._shutdown_grace = grace_s
            self._shutdown_reason = reason
        if self._shutdown_task is not None and self._shutdown_task.done() and self._control_task is None:
            # Task allocation failed before any durable drain RPC was created.
            # An explicit retry may create that same mechanical operation, with
            # the original deadline/reason and no repeated business execution.
            self._shutdown_task = None
        if self._shutdown_task is None:
            try:
                self._shutdown_task = _create_task(
                    self._shutdown(self._shutdown_deadline, self._shutdown_grace, self._shutdown_reason),
                    name="executor-shutdown-" + self.identity.boot_id,
                )
            except BaseException:
                self.failure = self.failure or "executor_shutdown_failed"
                self._signal(self._locals(), force=False)
                raise
        return self._shutdown_task

    async def shutdown(self, *, timeout_s=30.0, grace_s=1.0, reason="host shutdown"):
        # The first call owns the original deadline/reason. Repeated callers
        # observe that operation, never a fresh shutdown budget or control key.
        waiter = asyncio.current_task()
        self._shutdown_waiters.add(waiter)
        try:
            return await asyncio.shield(self._begin_shutdown(timeout_s, grace_s, reason))
        finally:
            self._shutdown_waiters.discard(waiter)

    async def status(self):
        """Reobserve late control/exit without extending the original stop window."""
        if not self.closed:
            try:
                self.check_admission()
            except JobError:
                pass
        boot, error, receipt = None, None, None
        try:
            self._check_manager_binding()
            boot = await registry.get(self.manager, self.identity)
        except Exception as exc:
            error = exc.code if isinstance(exc, JobError) else type(exc).__name__
        control = self._control_task
        if control is not None and control.done():
            try:
                receipt = control.result()
            except BaseException as exc:
                error = exc.code if isinstance(exc, JobError) else type(exc).__name__
        locals_ = {v["task"] for v in self._locals() if not v["task"].done()}
        pending = locals_ | {
            t
            for t in (self._start_task, self._heartbeat_task, self._control_task, self._shutdown_task)
            if t is not None and not t.done()
        }
        confirmed = (
            receipt is not None and boot is not None and boot["state"] == "draining" and not boot["admission_open"]
        )
        return {
            "admission_open": not self.closed and boot is not None and boot["admission_open"],
            "boot": boot,
            "durable_drain_confirmed": confirmed,
            "control_error": error,
            "local_failure": self.failure,
            "control_pending": control is not None and not control.done(),
            "local_tasks_pending": len(locals_),
            "pending_handles": len(pending),
            "store_safe_to_close": confirmed and not pending and error is None,
            "physical_stop_confirmed": False,
            "billing_settled": False,
        }

    def _locals(self):
        return [
            *self.runtime._local_executions.values(),
            *self.runtime._model_executions.values(),
            *self._owned_tasks.values(),
        ]

    def _signal(self, entries, *, force):
        signals, tokens, tasks = 0, set(), {}
        for entry in entries:
            task = entry["task"]
            if task.done():
                continue
            tasks.setdefault(task, []).append(entry)
            token = entry.get("cancel_token")
            if token is not None and id(token) not in tokens and not entry.get("signalled"):
                tokens.add(id(token))
                try:
                    already_cancelled = token.is_cancelled()
                except Exception:
                    already_cancelled = False
                if signal_token(token, "executor shutdown"):
                    entry["signalled"] = True
                    signals += not already_cancelled
        for task, members in tasks.items():
            if task is asyncio.current_task() or task in self._shutdown_waiters or task.done() or task.cancelling():
                continue
            if force or not any(v.get("signalled") for v in members):
                # Whole-boot shutdown owns every retained member of this
                # runtime. Individual node cancellation uses task_control.
                signals += bool(task.cancel("executor shutdown"))
                for member in members:
                    member["signalled"] = True
        return signals

    async def _drain(self, reason):
        if self._start_task is not None:
            try:
                await asyncio.shield(self._start_task)
            except Exception:
                pass  # registration may have committed before startup failed
        return await self._idempotent(registry.drain, system_key="runtime-shutdown", reason=reason)

    async def _shutdown(self, deadline, grace_s, reason):
        clock = asyncio.get_running_loop().time
        try:
            self._control_task = _create_task(self._drain(reason), name="executor-drain-" + self.identity.boot_id)
        except BaseException:
            self.failure = self.failure or "executor_shutdown_failed"
            self._signal(self._locals(), force=False)
            raise
        # A hung store cannot consume an unbounded stop window. Retain this
        # exact control task after the deadline, rather than inventing no-commit.
        await asyncio.wait({self._control_task}, timeout=min(1.0, max(0.0, deadline - clock())))
        entries = self._locals()
        signals = self._signal(entries, force=False)
        heartbeat = self._heartbeat_task
        if heartbeat is not None and not heartbeat.done() and not heartbeat.cancelling():
            heartbeat.cancel()
        current = asyncio.current_task()
        tasks = {v["task"] for v in entries if v["task"] is not current and not v["task"].done()}
        if tasks:
            waiting = tasks - self._shutdown_waiters
            if waiting:
                await asyncio.wait(waiting, timeout=min(grace_s, max(0.0, deadline - clock())))
        # Refresh handles admitted while the durable operation was in flight.
        entries = self._locals()
        signals += self._signal(entries, force=True)
        tasks |= {v["task"] for v in entries if v["task"] is not current and not v["task"].done()}
        if heartbeat is not None:
            tasks.add(heartbeat)
        if not self._control_task.done():
            tasks.add(self._control_task)
        if tasks:
            waiting = tasks - self._shutdown_waiters
            if waiting:
                await asyncio.wait(waiting, timeout=max(0.0, deadline - clock()))
        # A controller can exit by transferring an already-running request.
        # Re-read after every wait and spend only the ORIGINAL remaining stop
        # window on those handles. A single final snapshot can otherwise report
        # pending work while there is still time to observe its actual exit.
        while True:
            entries = self._locals()
            tasks |= {v["task"] for v in entries if v["task"] is not current and not v["task"].done()}
            waiting = {t for t in tasks if not t.done()} - self._shutdown_waiters
            remaining = deadline - clock()
            if not waiting or remaining <= 0:
                break
            signals += self._signal(entries, force=True)
            await asyncio.wait(waiting, timeout=remaining)
        receipt, error = None, None
        if self._control_task.done():
            try:
                receipt = self._control_task.result()
            except BaseException as exc:
                error = exc.code if isinstance(exc, JobError) else type(exc).__name__
        # A controller can hand its still-live request to this runtime while
        # the final bounded wait observes that controller exit. Re-read local
        # ownership before declaring the store safe; the old task set cannot
        # prove that no newly retained request exists.
        entries = self._locals()
        tasks |= {v["task"] for v in entries if v["task"] is not current and not v["task"].done()}
        pending = [t for t in tasks if not t.done()]
        return {
            "admission_open": False,
            "durable_drain_confirmed": receipt is not None,
            "drain_receipt": receipt,
            "control_error": error,
            "local_signals": signals,
            "local_tasks_pending": sum(t in {v["task"] for v in entries} for t in pending),
            "control_pending": not self._control_task.done(),
            "pending_handles": len(pending),
            "store_safe_to_close": not pending and receipt is not None,
            "physical_stop_confirmed": False,
            "billing_settled": False,
        }
