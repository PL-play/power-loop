"""Bounded execution scheduling with an independent control listener."""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace

from power_loop.agent.follow_up import render_inbox_rows
from power_loop.agent.sink import _encode_content, _meta_with_content_encoding
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.core.events import AgentEventBus
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .runtime_store import RuntimeInputStore, now_ms
from .runtime_transport import SubscriptionSpec, TransportPacket
from .runtime_types import (
    InputRef,
    JobRef,
    RequestOptions,
    Result,
    RuntimeDescriptor,
    RuntimeRequestError,
    ShutdownResult,
    TargetRef,
    canonical,
    plain,
)

log = logging.getLogger(__name__)


@dataclass
class _ControlInputRead:
    input_ref: InputRef
    hint_consumed: bool


@dataclass
class _SessionExecution:
    store_id: str
    session_id: str
    node_id: str
    boot_id: str
    input_ref: InputRef
    options: RequestOptions = field(default_factory=RequestOptions)
    job_ref: JobRef | None = None
    epoch: int = 0
    lease: object = None
    stop: CancellationToken = field(default_factory=CancellationToken)
    append_refs: dict = field(default_factory=dict)
    claiming_refs: frozenset[InputRef] = field(default_factory=frozenset)
    control_read: _ControlInputRead | None = None
    append_overflow: bool = False
    append_cursor: int = 0
    loop: object = None
    closing: bool = False
    task: object = None
    shutdown_interrupted: bool = False
    idle_users: int = 0
    output: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=256))

    actor: object = None
    parent_job_id: str | None = None
    root_job_id: str | None = None
    resources: object = None
    descriptor: object = None
    expires_at_ms: int | None = None

    def snapshot(self):
        from power_loop.core.execution_context import ExecutionContext
        return ExecutionContext(self.store_id, self.session_id, self.node_id, self.boot_id,
            self.input_ref, self.job_ref, self.parent_job_id, self.root_job_id,
            self.epoch, self.lease.token if self.lease else None, self.actor, self.options, self.resources, self.stop,
            self.descriptor.host_context if self.descriptor else {})


class _InputStart:
    def __init__(self, worker, context, record, send_index):
        self.worker, self.context, self.record, self.send_index = worker, context, record, send_index
        self.structured_output, self.output_error = None, None
        self.owner = None
        self.parent_result = None

    @property
    def child_job_id(self):
        return (self.record.row["target_job_id"] if self.record.meta["operation"] == "resume"
                else self.record.row["delivered_job_id"])

    async def claim_owner(self, runtime, tx):
        if not self.child_job_id:
            return None
        from power_loop.runtime.jobs import JobRegistration, _record
        row = await runtime.manager._row(tx, self.child_job_id)
        if self.record.meta["operation"] == "resume":
            from power_loop.runtime.subagent_resume import resume_claim
            self.owner = await resume_claim(runtime, tx, self.record, row)
            return self.owner
        self.owner = await runtime.claim(runtime.actor(self.context.session_id),
            JobRegistration(_record(row), None, False))
        return self.owner

    async def prepare(self, tx):
        port, ctx = self.worker.inputs, self.context
        records = await port.claim_inputs(tx, TargetRef(ctx.store_id, ctx.session_id), ctx, 1, [ctx.input_ref])
        if not records:
            raise RuntimeRequestError("input_not_startable")
        self.record = records[0]
        if self.record.meta["operation"] == "result_ready":
            if await port.discard_cancelled_result(tx, self.record):
                raise RuntimeRequestError("origin_cancelled")
            self.record = await port.result_input(self.record, tx)

    async def bind(self, tx, binding):
        port, ctx, store = self.worker.inputs, self.context, self.worker.runtime.store
        self.model_ref = JobRef(ctx.store_id, binding.lease.job_id, binding.lease.generation)
        lease = self.owner.lease if self.owner else binding.lease
        ref = JobRef(ctx.store_id, lease.job_id, lease.generation)
        await port.bind_input_job(tx, ctx.input_ref, ref, ctx, claim_token=self.record.row["claim_token"])
        state = await port.db.dialect.lock_state(tx, port.state, ctx.session_id)
        ctx.lease.require_owned()
        seq = state["next_seq"]
        self.next_seq = seq + 1
        body = json.loads(self.record.row["content"])
        role, name = "user", None
        content, structured = (body, False) if isinstance(body, str) else _encode_content(body.get("content"))
        if not isinstance(body, str):
            role, name = body.get("role", "user"), body.get("name")
        input_meta = ({"inbox": {"item_ids": [ctx.input_ref.message_id], "kinds": [self.record.row["kind"]]}}
            if self.record.row["kind"] != "user" else {})
        meta = _meta_with_content_encoding(input_meta, structured=structured)
        if self.record.meta["options"].get("metadata"):
            meta = dict(meta or {}, send=self.record.meta["options"]["metadata"])
        timestamp = now_ms()
        await tx.execute(f"INSERT INTO {store.t.messages} "
            "(session_id,seq,role,name,content,state,meta_json,send_index,created_at) VALUES (?,?,?,?,?,'active',?,?,?)",
            (ctx.session_id, seq, role, name, content, canonical(meta or {}), self.send_index, timestamp))
        await tx.execute(f"UPDATE {port.state} SET next_seq=next_seq+1,inbox_version=inbox_version+1,runtime_version=runtime_version+1 "
                         "WHERE session_id=?", (ctx.session_id,))
        await tx.execute(port.db.dialect.upsert(store.t.session_runtime_state, ("session_id", "state_key"),
                         ("value_json", "updated_at")),
                         (ctx.session_id, "send_index", canonical(self.send_index), timestamp))
        await tx.execute(f"UPDATE {port.q} SET status='delivered',delivered_seq=?,delivered_send_index=?,delivered_at=?,meta_json=?,"
                         "claim_token=NULL,claimed_at=NULL,claim_owner_epoch=NULL WHERE session_id=? AND item_id=?",
                         (seq, self.send_index, timestamp, canonical(self.record.meta), ctx.session_id, ctx.input_ref.message_id))
        ctx.job_ref = ref
        source = binding.resource_evidence.get("_control_source", {})
        ctx.parent_job_id = source.get("parent_job_id")
        ctx.root_job_id = source.get("root_job_id", ref.job_id)
        binding.resource_evidence["_runtime_send_index"] = self.send_index

    def committed(self):
        holding = self.worker.runtime.store._holding(self.context.session_id)
        holding.rows.pop("state", None)
        holding.next_seq = self.next_seq
        holding.runtime["send_index"] = self.send_index
        holding.change_snapshot = None

    async def prepare_finish(self, tx):
        self.result_route = await self.worker.inputs._lock_result_sessions(tx, self.context.job_ref)
        self.context.lease.require_owned()

    def completed(self, result):
        response_format = self.context.options.response_format
        if result.status == "completed" and "child_spec" in self.context.descriptor.config and not result.final_text.strip():
            # Keep the child's actual answer from this input when its last
            # model turn is blank. No history read or supplier call is needed.
            result.final_text = next((item["text"].strip() for item in reversed(result.texts)
                                      if item["text"].strip()), "")
            if not result.final_text:
                result.status = "failed"
                self.output_error = "subagent_output_missing"
        if result.status != "completed" or not response_format or response_format.get("type") not in {"json_object", "json_schema"}:
            return
        from power_loop.runtime.structured import StructuredOutputError, parse_structured
        schema = (plain(response_format["json_schema"]["schema"])
                  if response_format["type"] == "json_schema" else None)
        try:
            value = parse_structured(result.final_text, schema=schema)
            canonical(value)  # Model output, like request JSON, must be finite.
            self.structured_output = value
        except StructuredOutputError as exc:
            result.status = "failed"
            self.output_error = "structured_output: " + exc.reason
        except ValueError:
            result.status = "failed"
            self.output_error = "structured_output: invalid_json"

    def public_result(self, evidence):
        from power_loop.runtime.send_receipt import KEY
        ctx = self.context
        result = evidence.get(KEY, {}).get("result", {})
        status = result.get("status", evidence["exit"])
        error = self.output_error or evidence.get("error_type")
        if status in {"degraded", "timed_out"}:
            error = result.get("final_text") or status
            status = "failed"
        if status not in {"completed", "failed", "cancelled", "interrupted", "unknown"}:
            error = status
            status = "interrupted"
        return Result(ctx.job_ref, ctx.input_ref, status, result.get("final_text", ""),
                       result.get("usage", evidence.get("usage", {})), self.structured_output, error,
                       evidence.get("error_code"))

    async def finish(self, tx, binding, evidence):
        ctx = self.context
        value = self.public_result(evidence)
        deadline = self.worker._wait_deadlines.pop(ctx.job_ref, None)
        if deadline is not None:
            deadline.cancel()
        status = value.status
        if self.owner:
            manager = binding.runtime.manager
            row = await manager._row(tx, ctx.job_ref.job_id)
            manager._check_proof(row, self.owner.lease)
            if row["kind"] == "send":
                await manager.close_model_execution(self.owner.actor, self.owner.lease,
                    evidence={**evidence, "_public_result": plain(value)})
                if self.result_route["parent_session_id"] is not None:
                    self.parent_result = await self.worker.inputs._commit_result(tx, ctx.job_ref, value, ctx, self.result_route)
                return
            outcome = {"completed": "succeeded", "cancelled": "cancelled",
                       "interrupted": "interrupted"}.get(status, "failed")
            if row["cancel_requested_at"] is not None:
                outcome, value = "cancelled", replace(value, status="cancelled")
            await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET status='settling',outcome=?,"
                "resource_state='stopped',intake='closed',version=version+1,updated_at=? "
                "WHERE job_id=? AND generation=? AND version=?",
                (outcome, manager.now(), row["job_id"], row["generation"], row["version"]), obligations=True)
        if self.owner:
            self.parent_result = await self.worker.inputs._commit_result(
                tx, ctx.job_ref, value, ctx, self.result_route)

    async def notify_finished(self):
        if self.parent_result:
            ref = self.parent_result
            await self.worker.runtime.client._notify(TargetRef(ref.store_id, ref.session_id),
                ref.message_id, ref, ["work", "activation"])


class RuntimeWorker:
    def __init__(self, runtime):
        self.runtime = runtime
        self.inputs = RuntimeInputStore(runtime.store)
        self._subscriptions, self._listeners = [], []
        self._bindings = []
        self._pending = {}
        self._active = {}
        self._tasks = set()
        self._waiting = set()
        self._wait_deadlines = {}
        self._capacity = asyncio.Semaphore(runtime.capacity)
        self._wake = asyncio.Event()
        self._started, self._closing, self._closed = False, False, False
        self._lifecycle_lock = asyncio.Lock()
        self._recovery_cursor = None
        self._job_recovery_cursor = ""
        self._overflow = False
        self._guard_monitor = None
        self.last_error = None
        self.shutdown_result = ShutdownResult(True)

    async def start(self):
        async with self._lifecycle_lock:
            await self._start()

    async def _start(self):
        if self._closed:
            raise RuntimeRequestError("session_closed")
        if self._started:
            return
        r = self.runtime
        try:
            preflight = getattr(r.factory, "preflight", None)
            if preflight is not None:
                await preflight(r.required_profiles)
            elif r.required_profiles:
                raise RuntimeRequestError("capability_unavailable", "factory preflight is required")
            if r.process_guard is not None:
                await r.process_guard.acquire()
            for channel in ("activation", "work", "control"):
                binding = SubscriptionSpec(r.namespace, r.store.store_id, channel, r.node_id + "/" + r.boot_id)
                sub = await r.transport.subscribe(binding)
                self._bindings.append(binding)
                self._subscriptions.append(sub)
                await sub.ready()
            # Ready precedes recovery; dispatch begins only after the bounded initial page.
            await self.recover()
            if r.timer_driver is not None:
                await r.timer_driver.start()
                if not r.timer_driver.status().ready:
                    raise RuntimeRequestError("capability_unavailable", "timer driver is not ready")
        except BaseException:
            if r.timer_driver is not None:
                await r.timer_driver.stop()
            for sub in self._subscriptions:
                await sub.aclose()
            self._subscriptions.clear()
            self._bindings.clear()
            if r.process_guard is not None:
                await r.process_guard.release()
            raise
        self._started = True
        self._listeners = [asyncio.create_task(self._listen(s), context=contextvars.Context()) for s in self._subscriptions]
        self._dispatcher = asyncio.create_task(self._dispatch(), context=contextvars.Context())
        self._recovery = (asyncio.create_task(self._recover_periodically(), context=contextvars.Context())
                          if r.recovery_interval_s is not None else None)
        r.client._accepting = True
        if r.process_guard is not None:
            self._guard_monitor = asyncio.create_task(self._watch_guard(), context=contextvars.Context())

    async def _watch_guard(self):
        while True:
            await asyncio.sleep(5)
            try:
                await self.runtime.process_guard.check()
            except Exception as exc:
                self.last_error = "store_lock_lost: " + str(exc)[:500]
                await self.shutdown(timeout_s=0)
                return

    def _activate(self, sid):
        if self._closing or sid in self._active:
            return
        if len(self._pending) >= 1024 and sid not in self._pending:
            self._overflow = True
            return
        self._pending.setdefault(sid, now_ms())
        self._wake.set()

    async def recover(self):
        page = await self.inputs.list_unfinished(self._recovery_cursor, self.runtime.claim_limit)
        self._recovery_cursor = page.next_cursor
        if page.next_cursor is None:
            self._overflow = False
        for ref in page.items:
            self._activate(ref.session_id)
        jobs = await self.inputs.db.fetchall(f"SELECT job_id,session_id FROM {self.inputs.jobs} "
            "WHERE kind IN ('subagent','background_tool','background_command','workflow','workflow_node') AND session_id IS NOT NULL "
            "AND status='running' AND outcome IS NULL AND job_id>? ORDER BY job_id LIMIT ?",
            (self._job_recovery_cursor, self.runtime.claim_limit))
        self._job_recovery_cursor = jobs[-1]["job_id"] if len(jobs) == self.runtime.claim_limit else ""
        for job in jobs:
            self._activate(job["session_id"])

    async def _recover_periodically(self):
        while True:
            await asyncio.sleep(self.runtime.recovery_interval_s)
            await self.recover()

    async def _listen(self, subscription):
        index = self._subscriptions.index(subscription)
        binding = self._bindings[index]
        while not self._closed:
            try:
                async for delivery in subscription:
                    packet = delivery.packet
                    ctx = self._active.get(packet.target.session_id)
                    if packet.channel == "control" and (packet.payload or {}).get("cancelled"):
                        deadline = self._wait_deadlines.pop(JobRef(packet.target.store_id,
                            packet.target.job_id, packet.payload["generation"]), None)
                        if deadline is not None:
                            deadline.cancel()
                    if packet.channel == "control" and ctx is not None:
                        await self._control(ctx, packet)
                    elif packet.channel == "work" and ctx is not None and not ctx.closing:
                        if packet.input_ref and len(ctx.append_refs) < 100:
                            ctx.append_refs[packet.input_ref] = None
                        elif packet.input_ref:
                            ctx.append_overflow = True
                    elif packet.channel in {"work", "activation"}:
                        self._activate(packet.target.session_id)
                    await delivery.ack()
            except Exception:
                log.exception("runtime %s subscription disconnected", binding.channel)
            if self._closing:
                return
            await subscription.aclose()
            while not self._closing:
                await asyncio.sleep(0.2)
                try:
                    subscription = await self.runtime.transport.subscribe(binding)
                    self._subscriptions[index] = subscription
                    await subscription.ready()
                    await self.recover()
                    break
                except Exception:
                    await subscription.aclose()
                    log.exception("runtime %s resubscribe failed", binding.channel)
            if self._closing:
                return
            if binding.channel == "control":
                # Each active context has at most one running job. Reconnect
                # checks those bounded slots, independent of execution capacity.
                for ctx in tuple(self._active.values()):
                    if ctx and ctx.job_ref:
                        await self._control(ctx, TransportPacket(42, binding.namespace,
                            "reconnect", TargetRef(ctx.store_id, ctx.session_id, ctx.job_ref.job_id), "control"))

    async def _control(self, ctx, packet):
        # Never enters the normal execution semaphore or the loop's send lock.
        if packet.input_ref is not None:
            # The serial control listener has one in-flight read. Keep its
            # consumption fact across awaits, even after the claim batch ends.
            read = _ControlInputRead(packet.input_ref, packet.input_ref in ctx.claiming_refs)
            ctx.control_read = read
            try:
                record = await self.inputs.input(self.inputs.db, packet.input_ref)
            finally:
                ctx.control_read = None
            if (record and record.row["status"] == "pending"
                    and record.row["mode"] == "steer" and ctx.job_ref is not None
                    and (record.row["target_job_id"] is None
                         or (record.row["target_job_id"], record.row["expected_job_generation"])
                         == (ctx.job_ref.job_id, ctx.job_ref.generation))):
                if read.hint_consumed:
                    return
                if len(ctx.append_refs) < 100:
                    ctx.append_refs[packet.input_ref] = None
                else:
                    ctx.append_overflow = True
                if ctx.loop:
                    ctx.loop.steer_event(ctx.session_id).set()
                return
        if ctx.job_ref is None or packet.target.job_id != ctx.job_ref.job_id:
            return
        generation = (packet.payload or {}).get("generation")
        if generation is not None and generation != ctx.job_ref.generation:
            return
        job = await self.inputs.db.fetchone(f"SELECT generation,cancel_requested_at,cancel_reason FROM {self.inputs.jobs} WHERE job_id=?",
                                            (ctx.job_ref.job_id,))
        if job["generation"] == ctx.job_ref.generation and job["cancel_requested_at"] is not None:
            ctx.stop.cancel(job["cancel_reason"])

    async def _dispatch(self):
        while not self._closing:
            await self._wake.wait()
            self._wake.clear()
            if (self._recovery_cursor is not None or self._overflow) and len(self._pending) < 924:
                await self.recover()
            while (self._pending and len(self._tasks) < 1024
                   and len(self._tasks) - len(self._waiting) < self.runtime.capacity and not self._closing):
                sid = next(iter(self._pending))
                self._pending.pop(sid)
                self._active[sid] = None
                task = asyncio.create_task(self._run_session(sid), context=contextvars.Context())
                self._tasks.add(task)
                task.add_done_callback(self._task_finished)
            if (self._recovery_cursor is not None or self._overflow) and len(self._pending) < 924:
                self._wake.set()

    def _task_finished(self, task):
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("runtime execution failed", exc_info=task.exception())
        self._wake.set()

    async def _run_session(self, sid):
        from .stateful_loop import _ExecutionLoop, _session_sync
        entry = _session_sync(sid)
        entry.users += 1
        acquired = False
        task = asyncio.current_task()
        try:
            await self._capacity.acquire()
            acquired = True
            async with entry.lock:
                await self._run_session_owned(sid)
        finally:
            if acquired and task not in self._waiting:
                self._capacity.release()
            self._waiting.discard(task)
            entry.users -= 1
            _ExecutionLoop._release_idle_sync(sid)

    @asynccontextmanager
    async def idle(self, ctx):
        """Keep the session owner while yielding its execution slot to children."""
        ctx.idle_users += 1
        if ctx.idle_users == 1:
            self._waiting.add(ctx.task)
            self._capacity.release()
            self._wake.set()
        try:
            yield
        finally:
            ctx.idle_users -= 1
            if ctx.idle_users == 0 and not ctx.stop.is_cancelled() and not ctx.shutdown_interrupted:
                await self._capacity.acquire()
                self._waiting.discard(ctx.task)

    async def _run_session_owned(self, sid):
        from contextlib import AsyncExitStack, ExitStack
        from types import SimpleNamespace

        from power_loop.agent.stateful_loop import _ExecutionLoop
        from power_loop.contracts.errors import CancellationRequested, SessionBusy, SessionLeaseLost
        from power_loop.contracts.hooks import HookPoint
        from power_loop.core.execution_context import _bind_execution_context
        from power_loop.core.execution_notifications import notify
        from power_loop.runtime.session_ownership import hold_session_ownership
        scopes = ExitStack()
        leases = AsyncExitStack()
        busy = False
        r, env, ctx, deadline_task, output_task = self.runtime, None, None, None, None
        bus, on_delta = None, None
        try:
            rows = await self.inputs.db.fetchall(f"SELECT q.*,s.metadata_json AS session_metadata,s.parent_session_id AS event_parent,s.spawn_depth AS event_depth FROM {self.inputs.q} q "
                f"JOIN {r.store.t.sessions} s ON s.session_id=q.session_id "
                "WHERE q.session_id=? AND q.status='pending' ORDER BY q.id LIMIT 2", (sid,))
            row = rows[0] if rows else None
            recovery_only = row is None
            if recovery_only:
                row = await self.inputs.db.fetchone(f"SELECT q.*,s.metadata_json AS session_metadata,s.parent_session_id AS event_parent,s.spawn_depth AS event_depth FROM {self.inputs.jobs} j "
                    f"JOIN {self.inputs.q} q ON q.delivered_job_id=j.job_id "
                    f"JOIN {r.store.t.sessions} s ON s.session_id=j.session_id WHERE j.session_id=? "
                    "AND j.kind IN ('subagent','background_tool','background_command','workflow','workflow_node') "
                    "AND j.status='running' AND j.outcome IS NULL ORDER BY q.id DESC LIMIT 1", (sid,))
                if row is None:
                    return
                record = self.inputs.record(row)
            else:
                record = self.inputs.record(row)
            identity = json.loads(row["session_metadata"])
            if record.meta["operation"] == "result_ready":
                record = await self.inputs.result_input(record)
            driver = record.meta["descriptor"]["kind"] == "workflow_driver"
            if driver and record.meta["operation"] == "result_ready":
                row = await self.inputs.db.fetchone(f"SELECT q.* FROM {self.inputs.jobs} j "
                    f"JOIN {self.inputs.q} q ON q.delivered_job_id=j.job_id WHERE j.session_id=? "
                    "AND j.kind='workflow' ORDER BY q.id DESC LIMIT 1", (sid,))
                record = self.inputs.record(row)
                recovery_only = True
            ctx = _SessionExecution(r.store.store_id, sid, r.node_id, r.boot_id, record.ref)
            # Inputs accepted before this execution registered left no notification for it.
            # The first round checkpoint picks that bounded backlog up from the store.
            ctx.append_overflow = len(rows) > 1
            ctx.options = RequestOptions(**record.meta["options"])
            scopes.enter_context(_bind_execution_context(ctx.snapshot, r.host_context_fields))
            ctx.task = asyncio.current_task()
            self._active[sid] = ctx
            if row["target_job_id"] and record.meta["operation"] != "resume":
                job = await self.inputs.db.fetchone(f"SELECT generation,status FROM {self.inputs.jobs} WHERE job_id=?", (row["target_job_id"],))
                reason = ("target_terminal" if job is None or job["status"] != "running" else
                          "stale_generation" if job["generation"] != row["expected_job_generation"] else None)
                if reason:
                    async with self.inputs.db.transaction() as tx:
                        await self.inputs.void_unbound(tx, record.ref, reason=reason)
                    return
            from .runtime_types import TrustedActor
            ctx.actor = TrustedActor(identity["tenant_id"], identity["principal_id"], sid)
            ctx.expires_at_ms = row["expires_at"]
            holder = SimpleNamespace(store=r.store, session_coordinator=r.coordinator,
                config=SimpleNamespace(session_lease_ttl_s=r.session_lease_ttl_s),
                _owner_id=r.node_id + "/" + r.boot_id,
                task_runtime=SimpleNamespace(executor_id=r.node_id, boot_id=r.boot_id),
                _cache_invalidate=lambda session: ctx.loop._cache_invalidate(session) if ctx.loop else None)
            ctx.lease = await leases.enter_async_context(hold_session_ownership(holder, sid))
            ctx.epoch = ctx.lease.epoch
            if row["expires_at"] is not None and row["expires_at"] <= now_ms():
                await self.inputs.fail_start(record.ref, "expired", ctx)
                return
            if recovery_only and not driver:
                from power_loop.runtime.session_ownership import recover_acquired_session
                recovered = SimpleNamespace(store=r.store, task_runtime=SimpleNamespace(manager=JobManager(r.store),
                    tenant_id=identity["tenant_id"], principal_id=identity["principal_id"]))
                await recover_acquired_session(recovered, sid, cache_valid=False)
                return
            if record.meta["operation"] == "result_ready" and not driver and identity.get("execution_role", "main") != "main":
                # An idle child/node has finished its execution. Keep a late
                # descendant result in its history for explicit resume, without
                # inventing a main send or claiming model consumption.
                await self._retain_late_result(ctx, record)
                return
            await notify(r.hooks, HookPoint.EXECUTION_ACQUIRED, {"execution": ctx.snapshot()}, timeout_s=r.hook_timeout_s)
            descriptor = ctx.descriptor = RuntimeDescriptor(**record.meta["descriptor"])
            try:
                env = await r.factory.build(descriptor, ctx.snapshot())
                ctx.resources = env.resources
                if env.runtime_env is not None:
                    from power_loop.runtime.env import runtime_env_context
                    scopes.enter_context(runtime_env_context(replace(env.runtime_env,
                        process_control=env.process_control or env.runtime_env.process_control)))
                    if r.shared_execution and env.runtime_env.workspace_dir is not None:
                        if env.resource_scope != "shared" or env.resource_profile_ref != descriptor.resource_profile_ref:
                            raise RuntimeRequestError("resource_unavailable", "shared workspace requires its stable resource profile")
                available = set(env.tool_registry.names() if env.tool_registry else ()) | {"pass_turn"}
                missing = set(ctx.options.tools or ()) - available
                if missing:
                    raise RuntimeRequestError("capability_unavailable", "missing tools: " + ", ".join(sorted(missing)))
                task_runtime = env.task_runtime or TaskRuntime(JobManager(r.store), tenant_id=identity["tenant_id"],
                    principal_id=identity["principal_id"], executor_id=r.node_id, boot_id=r.boot_id)
                model = env.llm
                if "child_spec" in descriptor.config:
                    from power_loop.runtime.spec import AgentSpec, _model_service
                    model = _model_service(SimpleNamespace(llm=model, task_runtime=task_runtime),
                                           AgentSpec.from_dict(plain(descriptor.config["child_spec"])))
            except (SessionLeaseLost, SessionBusy, CancellationRequested):
                raise
            except Exception as exc:
                code = exc.code if isinstance(exc, RuntimeRequestError) else "resource_unavailable"
                self.last_error = {"code": code, "message": str(exc)[:500],
                    "definition_ref": descriptor.definition_ref, "resource_profile_ref": descriptor.resource_profile_ref}
                if code in {"capability_unavailable", "resource_unavailable"}:
                    # This homogeneous node cannot satisfy its declared ports.
                    # Finish this visible failure, drain active work, and require
                    # a corrected node composition before taking more inputs.
                    r.client._accepting = False
                    self._closing = True
                    self._wake.set()
                    if r.timer_driver is not None:
                        await r.timer_driver.stop()
                parent_ref = await self.inputs.fail_start(record.ref, str(exc), ctx, error_code=code)
                if parent_ref:
                    await r.client._notify(TargetRef(parent_ref.store_id, parent_ref.session_id),
                                           parent_ref.message_id, parent_ref, ["work", "activation"])
                return
            options = record.meta["options"]
            if "child_spec" in descriptor.config:
                from power_loop.runtime.spec import AgentSpec, _child_config
                child_config = _child_config(AgentSpec.from_dict(plain(descriptor.config["child_spec"])),
                                             SimpleNamespace(config=env.config))
            else:
                child_config = env.config
            config = replace(child_config, delegation_policy="single_level", session_lease_ttl_s=r.session_lease_ttl_s)
            for field_name in ("model", "temperature", "max_tokens", "max_rounds", "max_tokens_per_run", "system_prompt", "response_format"):
                if options[field_name] is not None:
                    setattr(config, field_name, options[field_name])
            if record.row["expires_at"] is not None:
                deadline_task = asyncio.create_task(self._expire(ctx, record.row["expires_at"], identity))
            if descriptor.kind == "workflow_driver":
                from power_loop.workflow.runner import run
                await run(self, ctx, record, env, task_runtime, recovery=recovery_only)
                return
            if descriptor.kind in {"background_tool", "background_command"}:
                from power_loop.runtime.background_execution import run
                await run(self, ctx, record, env, task_runtime)
                return
            bus = env.event_bus if env.event_bus is not None else AgentEventBus()
            def on_delta(event):
                if event.session_id != sid:
                    return
                if ctx.output.full():
                    while not ctx.output.empty():
                        ctx.output.get_nowait()
                    ctx.output.put_nowait({"kind": "gap"})
                ctx.output.put_nowait({"kind": "delta", **event.payload, "job_ref": plain(ctx.job_ref)})
            bus.subscribe(AgentEventType.STREAM_DELTA, on_delta)
            bus.subscribe(AgentEventType.STREAM_THINK_DELTA, on_delta)
            output_task = asyncio.create_task(self._pump_output(ctx))
            loop = _ExecutionLoop(llm=model, store=r.store, config=config, tool_registry=env.tool_registry,
                                     hooks=env.hooks, event_bus=bus, task_runtime=task_runtime, session_coordinator=r.coordinator,
                                     session_cache_size=0, timer_backend=r.timer_backend)  # This worker builds a fresh loop for each input.
            loop._runtime_worker_context = ctx
            loop._runtime_worker = self
            loop._owner_id = r.node_id + "/" + r.boot_id
            ctx.loop = loop
            child_spec = descriptor.config.get("child_spec")
            child_result = None
            if child_spec is not None:
                from power_loop.contracts.event_payloads import SubagentTaskStartPayload
                body = json.loads(record.row["content"])
                bus.publish(AgentEvent(type=AgentEventType.SUBAGENT_TASK_START,
                    data=SubagentTaskStartPayload(task=body if isinstance(body, str) else str(body),
                        preset=child_spec.get("preset", "core"), sub_session_id=sid, depth=row["event_depth"]),
                    session_id=row["event_parent"], source="subagent"))
            try:
                send_index = int(await r.store.get_runtime_state(sid, "send_index") or 0) + 1
                start = _InputStart(self, ctx, record, send_index)
                await loop._interrupt_pending(sid)
                if env.command is not None:
                    from power_loop.runtime.host_commands import run
                    await run(loop, ctx, start, env.command, send_index)
                else:
                    child_result = await loop._run_loop(sid, stop_event=ctx.stop, tools=options["tools"],
                        system_prompt=options["system_prompt"], max_rounds=options["max_rounds"],
                        response_format=options["response_format"], with_history=options["history"],
                        _entry_send_index=send_index, _runtime_start=start)
            finally:
                if child_spec is not None:
                    from power_loop.contracts.event_payloads import (
                        SubagentCompletedPayload,
                        SubagentLimitPayload,
                        SubagentTextPayload,
                    )
                    status = child_result.status if child_result is not None else (
                        "cancelled" if ctx.stop.is_cancelled() else "failed")
                    fields = dict(sub_session_id=sid, status=status,
                        rounds=child_result.rounds if child_result is not None else 0,
                        final_text=child_result.final_text if child_result is not None else "")
                    if status == "completed":
                        bus.publish(AgentEvent(type=AgentEventType.SUBAGENT_TEXT,
                            data=SubagentTextPayload(**fields), session_id=row["event_parent"], source="subagent"))
                    elif status == "hit_round_limit":
                        bus.publish(AgentEvent(type=AgentEventType.SUBAGENT_LIMIT,
                            data=SubagentLimitPayload(sub_session_id=sid, max_rounds=config.max_rounds),
                            session_id=row["event_parent"], source="subagent"))
                    bus.publish(AgentEvent(type=AgentEventType.SUBAGENT_COMPLETED,
                        data=SubagentCompletedPayload(**fields), session_id=row["event_parent"], source="subagent"))
                ctx.closing = True

        except SessionBusy:
            busy = True

        except JobError as exc:
            if ctx.job_ref is not None:
                raise
            if exc.code == 'root_context_cancelled':
                async with self.inputs.db.transaction() as tx:
                    await self.inputs.db.dialect.lock_state(tx, self.inputs.state, sid)
                    await self.inputs.void_unbound(tx, record.ref, reason=exc.code, status='cancelled')
            else:
                # A normal admission refusal (for example a required budget
                # provider) must complete the accepted input with its code.
                parent_ref = await self.inputs.fail_start(record.ref, str(exc), ctx, error_code=exc.code)
                if parent_ref:
                    await r.client._notify(TargetRef(parent_ref.store_id, parent_ref.session_id),
                                           parent_ref.message_id, parent_ref, ["work", "activation"])

        except RuntimeRequestError as exc:
            if exc.code == "input_not_startable" and ctx.job_ref is None:
                return  # Cancel/expiry won the atomic start condition.
            if ctx.job_ref is None and record.meta["operation"] == "resume" and exc.code in {
                "target_cancelled", "target_not_resumable", "stale_generation",
            }:
                # The claim transaction rejected this continuation. Terminalize
                # only its unbound input; the original job/result stays intact.
                async with self.inputs.db.transaction() as tx:
                    await self.inputs.db.dialect.lock_state(tx, self.inputs.state, sid)
                    await self.inputs.void_unbound(tx, record.ref, reason=exc.code,
                        status="cancelled" if exc.code == "target_cancelled" else "failed")
                return
            if exc.code != "origin_cancelled":
                raise
            async with loop.task_runtime.manager.joined_write() as tx:
                await self.inputs.db.dialect.lock_state(tx, self.inputs.state, sid)
                await self.inputs.discard_cancelled_result(tx, record)
            from power_loop.runtime.finalization import try_finalize
            manager = loop.task_runtime.manager
            job = await manager._row(manager.db, record.meta["result_ref"]["job_id"])
            actor = loop.task_runtime.actor(sid)
            await try_finalize(manager, actor, job["job_id"], job["generation"])
            root = await manager._row(manager.db, job["root_job_id"])
            await try_finalize(manager, actor, root["job_id"], root["generation"])

        finally:
            try:
                if deadline_task:
                    deadline_task.cancel()
                    await asyncio.gather(deadline_task, return_exceptions=True)
                if output_task:
                    if ctx.shutdown_interrupted:
                        output_task.cancel()
                        await asyncio.gather(output_task, return_exceptions=True)
                    else:
                        await ctx.output.put(None)
                        await output_task
                if bus is not None:
                    bus.unsubscribe(on_delta)
                    await bus.drain(timeout=r.hook_timeout_s)
                if ctx:
                    ctx.closing = True
                    if ctx.loop:
                        from .stateful_loop import _release_session_sync
                        for owned_sid in ctx.loop._sync_sids:
                            _release_session_sync(owned_sid, id(ctx.loop))
                        ctx.loop._sync_sids.clear()
                        ctx.loop._session_cache.clear()
                        # A cancelled timer/callback may retain the execution
                        # context until its deadline; release model/history now.
                        ctx.loop = None
                        ctx.append_refs.clear()
                if ctx and ctx.lease:
                    if ctx.lease.lost:
                        await notify(r.hooks, HookPoint.EXECUTION_LOST, {"execution": ctx.snapshot()}, timeout_s=r.hook_timeout_s)
                    await notify(r.hooks, HookPoint.EXECUTION_FINISHED, {"execution": ctx.snapshot()}, timeout_s=r.hook_timeout_s)
            finally:
                try:
                    await leases.aclose()
                    if ctx and ctx.lease:
                        await notify(r.hooks, HookPoint.EXECUTION_RELEASED, {"execution": ctx.snapshot()}, timeout_s=r.hook_timeout_s)
                finally:
                    scopes.close()
                    self._active.pop(sid, None)
            if env:
                await env.aclose()
            # This check is AFTER the lease has been released. New submit always
            # sends activation as well, covering the other side of the handoff.
            if not busy and not (ctx and ctx.lease and ctx.lease.lost) and await self.inputs.has_pending(sid):
                await r.transport.publish(TransportPacket(42, r.namespace, "idle", TargetRef(r.store.store_id, sid), "activation"))
            if ctx and not busy:
                await r.transport.publish(TransportPacket(42, r.namespace, ctx.input_ref.message_id,
                    TargetRef(ctx.store_id, sid), "output", ctx.input_ref, {"kind": "completed"}))

    async def _expire(self, ctx, expires_at, identity):
        from .runtime_client import RuntimeClient
        from .runtime_types import TrustedActor
        await asyncio.sleep(max(0, (expires_at - now_ms()) / 1000))
        client = RuntimeClient(store=self.runtime.store, transport=self.runtime.transport, namespace=self.runtime.namespace)
        try:
            await client.cancel(ctx.input_ref,
                actor=TrustedActor(identity["tenant_id"], identity["principal_id"], None), reason="deadline_expired")
        finally:
            await client.aclose()

    async def _pump_output(self, ctx):
        while True:
            payload = await ctx.output.get()
            if payload is None:
                return
            try:
                await self.runtime.transport.publish(TransportPacket(42, self.runtime.namespace,
                    ctx.input_ref.message_id, TargetRef(ctx.store_id, ctx.session_id), "output", ctx.input_ref, payload))
            except Exception:
                log.warning("output notification lost for %s", ctx.input_ref.message_id, exc_info=True)

    async def _retain_late_result(self, ctx, rendered):
        manager = JobManager(self.runtime.store)
        async with manager.joined_write() as tx:
            records = await self.inputs.claim_inputs(tx, TargetRef(ctx.store_id, ctx.session_id),
                                                    ctx, 1, [rendered.ref])
            if not records or await self.inputs.discard_cancelled_result(tx, records[0]):
                return
            record = records[0]
        await self.runtime.store.append_message(ctx.session_id, role="user", name="follow_up",
            content=json.loads(rendered.row["content"]),
            meta={"inbox": {"item_ids": [record.ref.message_id], "kinds": [record.row["kind"]]}},
            inbox={"claim_token": record.row["claim_token"], "ids": [record.row["id"]]})

    async def drain_append(self, ctx, *, only=None):
        if only is None and ctx.append_overflow and len(ctx.append_refs) < self.runtime.claim_limit:
            rows = await self.inputs.db.fetchall(f"SELECT id,item_id FROM {self.inputs.q} "
                "WHERE session_id=? AND status='pending' AND id>? ORDER BY id LIMIT ?",
                (ctx.session_id, ctx.append_cursor, self.runtime.claim_limit - len(ctx.append_refs)))
            for row in rows:
                ctx.append_refs[InputRef(ctx.store_id, ctx.session_id, row["item_id"])] = None
                ctx.append_cursor = row["id"]
            if not rows:
                ctx.append_overflow, ctx.append_cursor = False, 0
        # Consume this checkpoint's hint even if its inputs were retracted.
        # Clear before yielding to claim, so newer control notifications survive.
        ctx.loop.steer_event(ctx.session_id).clear()
        if not (ctx.append_refs if only is None else only) or ctx.job_ref is None:
            return []
        refs = list(ctx.append_refs if only is None else only)[:self.runtime.claim_limit]
        for ref in refs:
            # Explicit durable references need not have reached the notification cache.
            ctx.append_refs.pop(ref, None)
        ctx.claiming_refs = frozenset(refs)
        if ctx.control_read and ctx.control_read.input_ref in ctx.claiming_refs:
            # A pending snapshot already being read must not re-issue the hint
            # consumed here. Claim/void/delivery still use the original transaction.
            ctx.control_read.hint_consumed = True
        try:
            async with ctx.loop.task_runtime.manager.joined_write() as tx:
                # A resume or an input already bound to its own job start stays pending.
                eligible = []
                for ref in refs:
                    record = await self.inputs.input(tx, ref)
                    if record and record.meta["operation"] == "result_ready":
                        eligible.append(ref)
                    elif record and record.meta["operation"] == "send" and record.row["delivered_job_id"] is None:
                        row = record.row
                        reason = ("expired" if row["expires_at"] is not None and row["expires_at"] <= now_ms() else
                                  "stale_generation" if row["target_job_id"] and
                                  (row["target_job_id"], row["expected_job_generation"]) !=
                                  (ctx.job_ref.job_id, ctx.job_ref.generation) else None)
                        if reason:
                            await self.inputs.void_unbound(tx, ref, reason=reason)
                        else:
                            eligible.append(ref)
                records = await self.inputs.claim_inputs(tx, TargetRef(ctx.store_id, ctx.session_id), ctx,
                                                        self.runtime.claim_limit, eligible)
                rendered = []
                for record in records:
                    if record.meta["operation"] == "result_ready":
                        if await self.inputs.discard_cancelled_result(tx, record):
                            continue
                        record = await self.inputs.result_input(record, tx)
                    rendered.append(record)
                records = rendered
        finally:
            ctx.claiming_refs = frozenset()
        rows = []
        for record in records:
            body = json.loads(record.row["content"])
            content, structured = (body, False) if isinstance(body, str) else _encode_content(body.get("content"))
            rows.append(dict(record.row, content=content, meta={"structured": structured}))
        messages = render_inbox_rows(rows, wrap=True) if rows else []
        for message in messages:
            message["_inbox"]["job_ref"] = plain(ctx.job_ref)
        return messages

    async def shutdown(self, *, timeout_s=30.0, cancel_running=False):
        async with self._lifecycle_lock:
            return await self._shutdown(timeout_s=timeout_s, cancel_running=cancel_running)

    async def _shutdown(self, *, timeout_s, cancel_running):
        if self._closed:
            return self.shutdown_result
        self._closing = True
        self.runtime.client._accepting = False
        self._pending.clear()
        deadlines = tuple(self._wait_deadlines.values())
        for task in deadlines:
            task.cancel()
        await asyncio.gather(*deadlines, return_exceptions=True)
        self._wait_deadlines.clear()
        if self.runtime.timer_driver is not None:
            await self.runtime.timer_driver.stop()
        if self._started:
            self._dispatcher.cancel()
            if self._recovery:
                self._recovery.cancel()
            if cancel_running:
                from .runtime_client import RuntimeClient
                from .runtime_types import TrustedActor
                client = RuntimeClient(store=self.runtime.store, transport=self.runtime.transport,
                                       namespace=self.runtime.namespace)
                for ctx in tuple(self._active.values()):
                    if ctx and ctx.job_ref and ctx.lease and not ctx.lease.lost:
                        session = await self.inputs.db.fetchone(
                            f"SELECT metadata_json FROM {self.runtime.store.t.sessions} WHERE session_id=? "
                            f"AND EXISTS (SELECT 1 FROM {self.inputs.jobs} WHERE job_id=? AND status='running')",
                            (ctx.session_id, ctx.job_ref.job_id))
                        if session is None:
                            continue
                        identity = json.loads(session["metadata_json"])
                        await client.cancel(ctx.input_ref, actor=TrustedActor(identity["tenant_id"], identity["principal_id"], None),
                                            reason="worker shutdown")
                await client.aclose()
            pending = set()
            if self._tasks:
                _, pending = await asyncio.wait(self._tasks, timeout=timeout_s)
            interrupted = tuple(ctx.job_ref for ctx in self._active.values() if ctx and ctx.job_ref)
            for ctx in self._active.values():
                if ctx and ctx.task in pending:
                    ctx.shutdown_interrupted = True
            for task in pending:
                task.cancel("worker shutdown timeout")
            await asyncio.gather(*pending, return_exceptions=True)
            for sub in self._subscriptions:
                await sub.aclose()
            await asyncio.gather(self._dispatcher, *self._listeners,
                                 *([self._recovery] if self._recovery else []), return_exceptions=True)
            self.shutdown_result = ShutdownResult(not pending, interrupted)
        if self._guard_monitor is not None and self._guard_monitor is not asyncio.current_task():
            self._guard_monitor.cancel()
            await asyncio.gather(self._guard_monitor, return_exceptions=True)
        if self.runtime.process_guard is not None:
            await self.runtime.process_guard.release()
        self._closed = True
        return self.shutdown_result
