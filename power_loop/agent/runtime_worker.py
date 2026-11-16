"""Session-native execution scheduling (design 215 P2).

The worker only discovers ``queued`` / ``recovering`` / running-expired sessions, claims
one batch on a fixed node, runs the ordinary agent loop on the session/send/owner identity,
and writes the result through the accept/claim/final storage boundary. There is no job
recovery, no work/control append dependency and no synthetic lease: the claim transaction
is the ownership acquisition, and every loop boundary reads the CURRENT inbox version.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
from dataclasses import replace

from power_loop.contracts.events import AgentEventType
from power_loop.core.events import AgentEventBus
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.session_ownership import SessionOwnership

from .runtime_store import RuntimeInputStore, now_ms
from .runtime_transport import PROTOCOL_VERSION as TRANSPORT_PROTOCOL
from .runtime_transport import SubscriptionSpec, TransportPacket
from .runtime_types import (
    InputRef,
    RequestOptions,
    RuntimeDescriptor,
    RuntimeRequestError,
    ShutdownResult,
    TargetRef,
    TrustedActor,
    plain,
)

log = logging.getLogger(__name__)


class _SessionExecution:
    """The local context of one running send (no job identity)."""

    def __init__(self, runtime, sid, descriptor, batch, actor, parent_session_id=None):
        self.runtime = runtime
        self.store_id = runtime.store.store_id
        self.session_id = sid
        self.parent_session_id = parent_session_id
        self.node_id = runtime.node_id
        self.boot_id = runtime.boot_id
        self.descriptor = descriptor
        self.batch = batch
        self.ownership = None
        self.input_ref = batch.input_refs[0]
        self.actor = actor
        self.stop = CancellationToken()
        self.task = None
        self.loop = None
        self.closing = False
        self.shutdown_interrupted = False
        # The ordinary LLM boundary extends this same run-local list. None means
        # claim/factory or a non-LLM path has not established that consumption fact.
        self.consumed: list[InputRef] | None = None
        self.wake_pending = False
        self.output: asyncio.Queue = asyncio.Queue(maxsize=256)
        option_values = descriptor.config.get("child_options", descriptor.config)
        self.options = RequestOptions(**{
            k: v for k, v in option_values.items() if k in RequestOptions.__dataclass_fields__
        })

    @property
    def send_index(self):
        return self.batch.send_index

    def snapshot(self):
        from power_loop.core.execution_context import ExecutionContext
        return ExecutionContext(
            self.store_id, self.session_id, self.node_id, self.boot_id,
            input_ref=self.input_ref, owner_epoch=self.batch.owner_epoch,
            owner_token=self.batch.owner_token, actor=self.actor, options=self.options,
            cancel_token=self.stop, host_context=self.descriptor.host_context,
            send_index=self.batch.send_index,
            parent_session_id=self.parent_session_id,
        )


class RuntimeWorker:
    def __init__(self, runtime):
        self.runtime = runtime
        self.inputs = RuntimeInputStore(runtime.store)
        self._subscriptions, self._listeners = [], []
        self._bindings = []
        self._pending = {}
        self._active = {}
        self._drivers = set()
        self._tasks = set()
        self._capacity = asyncio.Semaphore(runtime.capacity)
        self._wake = asyncio.Event()
        self._started, self._closing, self._closed = False, False, False
        self._lifecycle_lock = asyncio.Lock()
        self._overflow = False
        self._guard_monitor = None
        self.last_error = None
        # Local health: definitions this node cannot build. Never persisted as a task
        # registry; an incapable node simply leaves the session for a capable one.
        self.capability_missing = set()
        self.shutdown_result = ShutdownResult(True)

    # ── lifecycle ─────────────────────────────────────────────────────────────

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
            for channel in ("activation", "work"):
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

    def _activate(self, sid, input_ref=None, *, cancelled=False):
        if self._closing:
            return
        if sid in self._active:
            ctx = self._active[sid]
            if ctx is not None and ctx.consumed is not None \
                    and input_ref is not None and not cancelled \
                    and input_ref not in ctx.consumed:
                ctx.wake_pending = True
            return
        if len(self._pending) >= 1024 and sid not in self._pending:
            self._overflow = True
            return
        self._pending.setdefault(sid, now_ms())
        self._wake.set()

    async def recover(self):
        # The ONLY discovery path: bounded scan of unfinished sessions. No job scan, no
        # pending-inbox-only discovery — a running-expired session with no pending input
        # is still found here so its old wait can be terminated (design 215 §9.3).
        page = await self.inputs.discover_unfinished(limit=self.runtime.claim_limit)
        for ref in page:
            self._activate(ref.session_id)
        if len(page) < self.runtime.claim_limit:
            self._overflow = False

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
                    # Hints carry no durable input. Lost hints rely on a live boundary
                    # or unfinished discovery (explicit recover if the scan is disabled).
                    packet = delivery.packet
                    self._activate(
                        packet.target.session_id, packet.input_ref,
                        cancelled=bool((packet.payload or {}).get("cancelled")),
                    )
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

    async def _dispatch(self):
        while not self._closing:
            await self._wake.wait()
            self._wake.clear()
            if self._overflow and len(self._pending) < 924:
                await self.recover()
            while self._pending and len(self._tasks) < 1024 and not self._closing:
                used = sum(
                    1 for ctx in self._active.values()
                    if ctx is None or ctx.descriptor.kind != "workflow_driver"
                )
                if used < self.runtime.capacity:
                    sid = next(iter(self._pending))
                else:
                    sid = next((value for value in self._pending if value in self._drivers), None)
                    if sid is None:
                        break
                self._pending.pop(sid)
                self._active[sid] = None
                task = asyncio.create_task(self._run_session(sid), context=contextvars.Context())
                self._tasks.add(task)
                task.add_done_callback(self._task_finished)
            if self._overflow and len(self._pending) < 924:
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
        try:
            # _dispatch enforces capacity for LLM/background sessions. Non-LLM
            # workflow drivers never take a model slot, including during recovery
            # while all their healthy leaves are still running.
            async with entry.lock:
                await self._run_session_owned(sid)
        finally:
            entry.users -= 1
            _ExecutionLoop._release_idle_sync(sid)

    # ── one send on a fixed node ──────────────────────────────────────────────

    async def _run_session_owned(self, sid):
        from power_loop.contracts.errors import CancellationRequested, SessionLeaseLost
        from power_loop.contracts.hooks import HookPoint
        from power_loop.core.execution_context import _bind_execution_context
        from power_loop.core.execution_notifications import notify

        from .stateful_loop import _ExecutionLoop

        r = self.runtime
        store = r.store
        target = TargetRef(store.store_id, sid)
        ctx = None
        env = None
        env_ctx = None
        bus = None
        on_delta = None
        output_task = None
        renewal = None
        ownership = None
        external_ownership = None
        batch = None
        normal_final = None
        try:
            if getattr(r.coordinator, "installs_session_fence", False):
                # The claim transaction is the whole entry: under the state lock it reads
                # the descriptor/identity and the real pending batch, checks capability
                # before writing the owner, and returns the merged user row it wrote.
                batch = await r.coordinator.claim(
                    self.inputs, target, node_id=r.node_id, boot_id=r.boot_id,
                    ttl_s=r.session_lease_ttl_s, can_execute=self._capable)
                if batch is None:
                    if not await self._recover_stale(target):
                        return
                    batch = await r.coordinator.claim(
                        self.inputs, target, node_id=r.node_id, boot_id=r.boot_id,
                        ttl_s=r.session_lease_ttl_s, can_execute=self._capable)
                    if batch is None:
                        return
            else:
                # A foreign coordinator's acquire decision needs the session facts before
                # claim. Keep this single merged entry read on that topology only; the
                # database-fence topology gets the same facts inside claim.
                entry = await self.inputs.session_entry(target)
                if entry is None or entry["lifecycle"] != "open":
                    return
                descriptor = RuntimeDescriptor(**json.loads(entry["descriptor_json"]))
                if descriptor.kind == "workflow_driver":
                    self._drivers.add(sid)
                if not self._capable(descriptor):
                    return
                if self._needs_recovery(entry):
                    recovered = await self.inputs.recover(
                        target, node_id=r.node_id, boot_id=r.boot_id,
                        ttl_s=r.session_lease_ttl_s)
                    if recovered is None:
                        return
                    await self._notify_results(recovered)
                    if recovered.execution_state != "queued":
                        return
                elif entry["execution_state"] != "queued":
                    return
                external_ownership = await r.coordinator.acquire(
                    (store.store_id, sid), owner_id=r.node_id + "/" + r.boot_id,
                    ttl_s=r.session_lease_ttl_s, on_lost=lambda: None)
                if external_ownership is None:
                    return
                batch = await self.inputs.claim(
                    target, node_id=r.node_id, boot_id=r.boot_id,
                    ttl_s=r.session_lease_ttl_s,
                    _owner_token=external_ownership.token, can_execute=self._capable)
                if batch is None:
                    return

            descriptor = batch.descriptor
            if descriptor.kind == "workflow_driver":
                self._drivers.add(sid)
            actor = TrustedActor(batch.tenant_id, batch.principal_id, sid)

            # The claim installed the fence; build the local handle from it. Per-ownership
            # process caches are cleared so no seq/index survives a change of owner.
            ownership = SessionOwnership(
                (store.store_id, sid), r.node_id + "/" + r.boot_id,
                time.monotonic() + r.session_lease_ttl_s, lambda: None,
                token=batch.owner_token, epoch=batch.owner_epoch,
                node_id=r.node_id, boot_id=r.boot_id, installs_fence=True,
            )
            if external_ownership is not None:
                ownership = external_ownership
                ownership.epoch = batch.owner_epoch
                ownership.node_id, ownership.boot_id = r.node_id, r.boot_id
                ownership.installs_fence = True
            store._forget_appended(sid)
            for cache in ("_next_seqs", "_cached_sessions", "_send_indexes", "_change_versions"):
                getattr(store, cache, {}).pop(sid, None)

            ctx = _SessionExecution(
                r, sid, descriptor, batch, actor, batch.parent_session_id
            )
            ctx.task = asyncio.current_task()
            ctx.ownership = ownership
            self._active[sid] = ctx
            ownership.on_lost = lambda: ctx.stop.cancel("session ownership lost")
            renewal = asyncio.create_task(self._renew(sid, batch, ownership, ctx), context=contextvars.Context())

            with _bind_execution_context(ctx.snapshot, r.host_context_fields):
                await notify(r.hooks, HookPoint.EXECUTION_ACQUIRED, {"execution": ctx.snapshot()}, timeout_s=r.hook_timeout_s)
                try:
                    env = await r.factory.build(descriptor, ctx.snapshot())
                except Exception as exc:
                    code = exc.code if isinstance(exc, RuntimeRequestError) else "resource_unavailable"
                    self.last_error = {"code": code, "message": str(exc)[:500],
                                       "definition_ref": descriptor.definition_ref}
                    self.capability_missing.add(descriptor.definition_ref)
                    await self.inputs.release_claim(batch)
                    batch = None
                    return
                # The worker runs each session in a fresh context; honor the factory's
                # runtime_env (workspace/home/sandbox) so file-backed tools such as
                # workflow logs/export resolve paths inside this session's execution.
                if env.runtime_env is not None:
                    from power_loop.runtime.env import runtime_env_context
                    env_ctx = runtime_env_context(env.runtime_env)
                    env_ctx.__enter__()
                if ctx.options.tools is not None and env.tool_registry is not None:
                    env.tool_registry = env.tool_registry.subset(ctx.options.tools)
                config = self._config(env, descriptor)
                if descriptor.kind == "workflow_driver":
                    from power_loop.workflow.runner import run_driver
                    final_batch = await run_driver(self, ctx, env)
                    await self._notify_results(final_batch)
                    if final_batch.execution_state == "queued":
                        self._activate(sid)
                    return
                if descriptor.kind == "background":
                    if "managed_effect" in descriptor.config:
                        from power_loop.runtime.managed_effects import run_session
                    else:
                        from power_loop.runtime.background_execution import run_session
                    result = await run_session(self, ctx, env, config)
                    final_batch = await self.inputs.final(
                        batch, status=result.status, final_text=result.final_text,
                        usage=dict(result.usage or {}), structured_output=result.structured_output,
                        error=result.error, error_code=result.error_code, result_seq=None,
                    )
                    await self._notify_results(final_batch)
                    return
                bus = env.event_bus if env.event_bus is not None else AgentEventBus()

                def on_delta(event):
                    if event.session_id != sid:
                        return
                    if ctx.output.full():
                        while not ctx.output.empty():
                            ctx.output.get_nowait()
                        ctx.output.put_nowait({"kind": "gap"})
                    ctx.output.put_nowait({
                        "kind": "delta", **event.payload,
                        "session_id": sid, "send_index": batch.send_index,
                        "owner_epoch": batch.owner_epoch,
                    })

                bus.subscribe(AgentEventType.STREAM_DELTA, on_delta)
                bus.subscribe(AgentEventType.STREAM_THINK_DELTA, on_delta)
                output_task = asyncio.create_task(self._pump_output(ctx), context=contextvars.Context())

                loop = _ExecutionLoop(
                    llm=env.llm, store=store, config=config, tool_registry=env.tool_registry,
                    hooks=env.hooks, event_bus=bus,
                    session_coordinator=r.coordinator, session_cache_size=0,
                    timer_backend=r.timer_backend,
                )
                loop._runtime_worker = self
                loop._runtime_worker_context = ctx
                loop._owner_id = r.node_id + "/" + r.boot_id
                ctx.loop = loop

                if env.command is not None:
                    from power_loop.runtime.host_commands import run_session
                    result = await run_session(loop, ctx, env.command)
                    consumed = None
                else:
                    consumed = ctx.consumed = list(batch.input_refs)
                    result = await loop._run_session_send(
                        sid, batch=batch, ownership=ownership, config=config,
                        tool_registry=env.tool_registry, hooks=env.hooks, event_bus=bus,
                        stop_event=ctx.stop, system_prompt=config.system_prompt,
                        max_rounds=ctx.options.max_rounds, response_format=ctx.options.response_format,
                        with_history=ctx.options.history,
                        boundary=self._boundary(batch, ownership, ctx, consumed),
                    )
                status, error, error_code = self._map_result(result, ctx)
                structured = None
                if ctx.options.response_format and status == "completed":
                    from power_loop.runtime.structured import StructuredOutputError, parse_structured
                    schema = ctx.options.response_format.get("json_schema", {}).get("schema")
                    try:
                        structured = parse_structured(result.final_text, schema=schema)
                    except StructuredOutputError as exc:
                        status, error, error_code = "failed", f"structured_output: {exc}", "structured_output"
                final_batch = await self.inputs.final(
                    batch, consumed=consumed, status=status, final_text=result.final_text,
                    usage=dict(result.usage or {}), structured_output=structured,
                    error=error, error_code=error_code, result_seq=result.result_seq,
                    assistant_meta=result.result_meta,
                )
                if env.command is None:
                    normal_final = final_batch
                await self._notify_results(final_batch)
        except SessionLeaseLost:
            # Another owner took over; stop quietly. No compensation writes.
            if ctx is not None:
                ctx.stop.cancel("session ownership lost")
        except CancellationRequested:
            if batch is not None:
                await self._final_cancelled(batch, ctx)
        except RuntimeRequestError as exc:
            # Domain refusal before any model/tool call (e.g. projection_unavailable) keeps its
            # code and diagnostic detail instead of becoming a generic internal_error.
            log.warning("session send stopped for %s: %s", sid, exc)
            if batch is not None:
                await self._final_failed(batch, exc.code, str(exc))
        except Exception:
            log.exception("session send failed for %s", sid)
            if batch is not None:
                await self._final_failed(batch, "internal_error", "session send raised")
        finally:
            if env_ctx is not None:
                env_ctx.__exit__(None, None, None)
            if renewal is not None:
                renewal.cancel()
                await asyncio.gather(renewal, return_exceptions=True)
            if external_ownership is not None:
                await r.coordinator.release(external_ownership)
            if output_task is not None:
                await ctx.output.put(None)
                await asyncio.gather(output_task, return_exceptions=True)
            if bus is not None and on_delta is not None:
                bus.unsubscribe(on_delta)
                await bus.drain(timeout=r.hook_timeout_s)
            if ctx is not None:
                ctx.closing = True
                if ownership is not None and ownership.lost:
                    await notify(r.hooks, HookPoint.EXECUTION_LOST, {"execution": ctx.snapshot()}, timeout_s=r.hook_timeout_s)
                await notify(r.hooks, HookPoint.EXECUTION_FINISHED, {"execution": ctx.snapshot()}, timeout_s=r.hook_timeout_s)
                ctx.loop = None
            self._active.pop(sid, None)
            if env is not None:
                await env.aclose()
            # A normal final reports its pending snapshot; real new notifications
            # bridge the interval after that snapshot and before active.pop. After
            # pop the ordinary dispatcher waits on the same session lock for aclose.
            if normal_final is not None and ctx is not None and ctx.consumed is not None:
                if normal_final.execution_state == "queued" or ctx.wake_pending:
                    self._activate(sid)
            elif batch is not None and await self._has_pending(sid):
                self._activate(sid)

    def _capable(self, descriptor):
        """Cheap, synchronous capability decision used before any owner is written.

        The claim passes this as ``can_execute``; it must not build resources, call the
        model/tools or await external work. A descriptor this node cannot execute leaves
        the input pending for a capable node instead of acquiring and then failing it.
        """
        if descriptor.kind not in {"agent", "background", "workflow_driver"}:
            self.capability_missing.add(descriptor.definition_ref)
            return False
        try:
            capable = self._can_build(descriptor)
        except Exception as exc:
            code = exc.code if isinstance(exc, RuntimeRequestError) else "resource_unavailable"
            self.last_error = {"code": code, "message": str(exc)[:500],
                               "definition_ref": descriptor.definition_ref}
            self.capability_missing.add(descriptor.definition_ref)
            return False
        if not capable:
            self.last_error = {"code": "capability_unavailable",
                               "message": "factory rejected descriptor",
                               "definition_ref": descriptor.definition_ref}
            self.capability_missing.add(descriptor.definition_ref)
        return capable

    @staticmethod
    def _needs_recovery(entry):
        """True for a recovering row or a running row whose committed lease expired."""
        state = entry["execution_state"]
        expired = (entry["owner_lease_until_ms"] is not None
                   and int(entry["owner_lease_until_ms"]) <= int(entry["db_now"]))
        return state == "recovering" or (state == "running" and expired)

    async def _recover_stale(self, target):
        """Terminate an interrupted run before a fresh claim.

        Runs only after the queued claim attempt returned None, so the healthy queued
        path never pays for this read. Capability is checked before recovery, and
        ``recover`` re-reads expiry under the state lock.
        """
        entry = await self.inputs.session_entry(target)
        if entry is None or entry["lifecycle"] != "open":
            return False
        descriptor = RuntimeDescriptor(**json.loads(entry["descriptor_json"]))
        if descriptor.kind == "workflow_driver":
            self._drivers.add(target.session_id)
        if not self._capable(descriptor) or not self._needs_recovery(entry):
            return False
        r = self.runtime
        recovered = await r.coordinator.recover(
            self.inputs, target, node_id=r.node_id, boot_id=r.boot_id,
            ttl_s=r.session_lease_ttl_s)
        if recovered is None:
            return False
        await self._notify_results(recovered)
        return recovered.execution_state == "queued"

    def _can_build(self, descriptor):
        factory = self.runtime.factory
        can_execute = getattr(factory, "can_execute", None)
        if can_execute is not None:
            return can_execute(descriptor)
        supports = getattr(factory, "supports", None)
        if supports is not None:
            return supports(descriptor)
        builders = getattr(factory, "_builders", None)
        if builders is not None:
            return descriptor.definition_ref in builders
        return True

    def _config(self, env, descriptor):
        config = env.config
        overrides = {}
        option_values = descriptor.config.get("child_options", descriptor.config)
        for field_name in ("model", "temperature", "max_tokens", "max_rounds", "max_tokens_per_run",
                           "system_prompt", "response_format"):
            value = option_values.get(field_name)
            if value is not None:
                overrides[field_name] = value
        overrides["delegation_policy"] = "single_level"
        overrides["session_lease_ttl_s"] = self.runtime.session_lease_ttl_s
        return replace(config, **overrides)

    async def _renew(self, sid, batch, ownership, ctx):
        r = self.runtime
        ttl = r.session_lease_ttl_s
        try:
            while not ctx.closing:
                await asyncio.sleep(ttl / 3)
                async with asyncio.timeout(max(0, ownership.expires_at - time.monotonic())):
                    renewed = await r.coordinator.renew(ownership, ttl_s=ttl)
                    if renewed and not getattr(r.coordinator, "installs_session_fence", False):
                        from power_loop.runtime.store_session_ownership import StoreSessionOwnershipCoordinator
                        renewed = await StoreSessionOwnershipCoordinator(r.store).renew(ownership, ttl_s=ttl)
                if not renewed:
                    ownership.lose()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("session renewal failed for %s", sid)
            ownership.lose()

    def _boundary(self, batch, ownership, ctx, consumed):
        target = batch.target
        # Seed with the inbox version at claim time: the claim already consumed that
        # snapshot, so the first boundary only claims input committed AFTER the claim.
        last_version = [batch.inbox_version]

        async def boundary(*, final=False):
            row = await self.inputs.boundary(target)
            if row is None or (
                row["execution_state"] != "running"
                or row["active_send_index"] != batch.send_index
                or row["owner_epoch"] != batch.owner_epoch
                or row["owner_token"] != batch.owner_token
            ):
                from power_loop.contracts.errors import SessionLeaseLost
                raise SessionLeaseLost(target.session_id)
            if row["cancel_send_index"] == batch.send_index:
                ctx.stop.cancel(row["cancel_reason"] or "cancelled")
            from power_loop.runtime.store.change_signals import COLUMNS, reader
            changes = reader(self.runtime.store, target.session_id)
            if changes is not None:
                changes.versions = tuple(row[column] for column in COLUMNS)
            if final:
                # Final boundary only checks; final() turns remaining pending into queued.
                return []
            if row["inbox_version"] == last_version[0]:
                return []
            claimed = await self.inputs.claim_follow_up(batch)
            if claimed is None:
                last_version[0] = row["inbox_version"]
                return []
            refs, seq, merged, version = claimed
            last_version[0] = version
            consumed.extend(refs)
            from .follow_up import render_batch_content
            return [{
                "role": "user", "content": render_batch_content(merged),
                "_persisted_seq": seq,
                "meta": {"input_refs": [plain(ref) for ref in refs]},
            }]

        return boundary

    @staticmethod
    def _map_result(result, ctx):
        status = result.status
        if ctx.stop.is_cancelled():
            return "cancelled", ctx.stop.reason or "cancelled", "cancelled"
        if status == "completed":
            return "completed", None, None
        if status == "cancelled":
            return "cancelled", result.final_text or "cancelled", "cancelled"
        if status == "pending_tools":
            return "interrupted", "pending tool calls left unresolved", "interrupted"
        # degraded / timed_out / hit_round_limit / context_checkpoint / capacity_blocked / failed
        return "failed", result.final_text or status, status

    async def _final_failed(self, batch, code, message):
        try:
            final_batch = await self.inputs.final(
                batch, status="failed", final_text="", error=message, error_code=code)
            await self._notify_results(final_batch)
        except Exception:
            log.exception("failed to terminalize send for %s", batch.target.session_id)

    async def _final_cancelled(self, batch, ctx):
        try:
            final_batch = await self.inputs.final(
                batch, status="cancelled", final_text="",
                error=(ctx.stop.reason if ctx else None) or "cancelled", error_code="cancelled")
            await self._notify_results(final_batch)
        except Exception:
            log.exception("failed to cancel send for %s", batch.target.session_id)

    async def _notify_results(self, final_batch):
        for result in final_batch.results:
            await self.runtime.client._notify(
                result.target, result.input_ref.message_id, result.input_ref, ["output"],
                {"kind": "completed", "session_id": result.target.session_id,
                 "send_index": result.send_index},
            )
        if final_batch.result_activation is not None:
            target = final_batch.result_activation
            ref = final_batch.result_ref
            await self.runtime.client._notify(
                target, ref.message_id, ref, ["work", "activation"]
            )

    async def _has_pending(self, sid):
        row = await self.runtime.store._db.fetchone(
            f"SELECT 1 AS present FROM {self.inputs.q} "
            "WHERE session_id=? AND status='pending' LIMIT 1", (sid,))
        return row is not None

    async def _pump_output(self, ctx):
        while True:
            payload = await ctx.output.get()
            if payload is None:
                return
            try:
                await self.runtime.transport.publish(TransportPacket(
                    TRANSPORT_PROTOCOL, self.runtime.namespace, ctx.input_ref.message_id,
                    TargetRef(ctx.store_id, ctx.session_id), "output", ctx.input_ref, payload))
            except Exception:
                log.warning("output notification lost for %s", ctx.input_ref.message_id, exc_info=True)

    # ── shutdown ──────────────────────────────────────────────────────────────

    async def shutdown(self, *, timeout_s=30.0, cancel_running=False):
        async with self._lifecycle_lock:
            return await self._shutdown(timeout_s=timeout_s, cancel_running=cancel_running)

    async def _shutdown(self, *, timeout_s, cancel_running):
        if self._closed:
            return self.shutdown_result
        self._closing = True
        self.runtime.client._accepting = False
        self._pending.clear()
        if self.runtime.timer_driver is not None:
            await self.runtime.timer_driver.stop()
        if self._started:
            self._dispatcher.cancel()
            if self._recovery:
                self._recovery.cancel()
            if cancel_running:
                for ctx in tuple(self._active.values()):
                    if ctx is not None:
                        ctx.stop.cancel("worker shutdown")
            pending = set()
            if self._tasks:
                _, pending = await asyncio.wait(self._tasks, timeout=timeout_s)
            interrupted = tuple(
                InputRef(ctx.store_id, ctx.session_id, ctx.input_ref.message_id)
                for ctx in self._active.values() if ctx is not None
            )
            for ctx in self._active.values():
                if ctx is not None and ctx.task in pending:
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
