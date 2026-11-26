"""Session-Core Plan B resident lightweight workflow_driver (design 217 §4.5).

The driver is a non-LLM session that occupies 0 model concurrency slots and piggybacks
its lease onto node-level ``batch_renew``. Same-wave leaves are created in one atomic
``create_many`` transaction; original-leaf continuations (``ContinuationPolicy`` and
``LoopNode.continue_nodes``) use standard ``accept(kind='workflow_continue')``.
Same-process leaf results arrive via an in-memory waiter queue (0 DB reads); cross-process
leaf results arrive via ``WakeTransport`` / transport wake followed by a single indexed
``follow_up_queue`` read. The driver executes 1 Claim + 1 Final per run, writing back
``start_input_message_id``'s ``terminal_status``/``result_json``/``completed_at`` and
delivering ``workflow_result`` to the parent in the same ``Final`` transaction.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from power_loop.agent.runtime_store import ChildSpec, InputRecord, now_ms
from power_loop.agent.runtime_types import (
    CapacityError,
    InputRef,
    RequestOptions,
    Result,
    RuntimeDescriptor,
    RuntimeRequestError,
    TargetRef,
    TrustedActor,
    canonical,
    plain,
    sha256_canonical,
)
from power_loop.contracts.errors import SessionLeaseLost

from . import journal
from .engine import WorkflowEngine, _strip_continuation_todos
from .spec import WorkflowSpec


@dataclass
class _PendingLeafCreate:
    key: str
    request: Any
    target: TargetRef
    start: InputRef
    request_hash: str
    descriptor: RuntimeDescriptor
    spec: ChildSpec
    future: asyncio.Future


def _normalize_node_status(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    if out.get("error_code") in {"hit_round_limit", "context_checkpoint"}:
        out["status"] = out["error_code"]
    elif out.get("error_code") in {"unknown", "store_commit_unknown"}:
        out["status"] = "unknown"
    return out


async def submit(client, target, spec, *, message_id, actor, descriptor=None, options=None, parent=None):
    """Create only the independent driver and its frozen start input."""
    del parent
    client._require_open()
    spec = WorkflowSpec.from_json(spec.to_dict() if isinstance(spec, WorkflowSpec) else spec)
    port = client.inputs
    supplied_descriptor, supplied_options = descriptor, options
    if descriptor is None:
        caller = await port._session(port.db, target.session_id)
        descriptor = RuntimeDescriptor(**json.loads(caller["descriptor_json"]))
    options = options or RequestOptions()
    identity = [
        "workflow_driver", target.store_id, target.session_id, message_id,
        actor.call_ref.to_dict() if actor.call_ref is not None else None, actor.tool_call_id,
    ]
    sid = "ses_" + sha256_canonical(identity)[:56]
    driver = TargetRef(target.store_id, sid)
    start = InputRef(
        target.store_id, sid,
        "inp_" + sha256_canonical(["workflow_start", target.store_id, sid])[:56],
    )
    if supplied_descriptor is None and supplied_options is None:
        existing = await port.db.fetchone(
            f"SELECT descriptor_json FROM {port.store.t.sessions} WHERE session_id=?", (sid,),
        )
        if existing is not None:
            descriptor = RuntimeDescriptor(**json.loads(existing["descriptor_json"]))
            frozen_spec = WorkflowSpec.from_json(plain(descriptor.config["workflow"]["spec"]))
            if canonical(frozen_spec.to_dict()) != canonical(spec.to_dict()):
                raise RuntimeRequestError("creation_conflict")
            options = RequestOptions(**{
                key: value for key, value in descriptor.config["workflow"].get("options", {}).items()
                if key in RequestOptions.__dataclass_fields__
            })
    descriptor = replace(
        descriptor,
        kind="workflow_driver",
        config={
            **plain(descriptor.config),
            "workflow": {
                "spec": spec.to_dict(),
                "allowed_tools": plain(options.tools),
                "options": plain(options),
            },
        },
    )
    saved = {
        "version": 1,
        "run_id": start.message_id,
        "driver_target": plain(driver),
        "start_input_ref": plain(start),
        "caller_target": plain(target),
        "definition_ref": plain(start),
        "created_at_ms": now_ms(),
        "status": "running",
        "cancel_by": None,
        "result": None,
        "result_seq": None,
        "aggregate_delivered": False,
        "continuation": {},
        "execution_revision": 0,
    }
    await port.create_workflow_session(
        descriptor,
        parent=target,
        actor=actor,
        target=driver,
        start=start,
        body=spec.input,
        result_kind="workflow_result",
        cancel_propagation="independent",
        run_journal=saved,
    )
    await client._notify(driver, start.message_id, start, ["work", "activation"])
    if client.wake_transport is not None:
        try:
            await client.wake_transport.publish_wake(driver.session_id)
        except Exception:
            pass
    return WorkflowRunHandle(client, actor, driver, start)


class WorkflowRunHandle:
    """The complete durable handle: driver TargetRef, start InputRef and run alias."""

    def __init__(self, client, actor, target: TargetRef, input_ref: InputRef):
        if input_ref.message_id is None or (input_ref.store_id, input_ref.session_id) != (
            target.store_id, target.session_id
        ):
            raise RuntimeRequestError("invalid_options", "workflow handle target/start mismatch")
        self.client, self.actor = client, actor
        self.target, self.input_ref = target, input_ref
        self.run_id = input_ref.message_id

    async def wait(self, *, deadline_ms=None):
        result = await self.client.wait(self.input_ref, actor=self.actor, deadline_ms=deadline_ms)
        from .result import AgentResult, WorkflowResult

        data = plain(result.structured_output)
        if data is None:
            return WorkflowResult(
                self.run_id,
                result.status,
                errors=[result.error] if result.error else [],
                cancel_by=result.cancel_by,
            )
        data["results"] = {
            key: AgentResult(**value) for key, value in data.get("results", {}).items()
        }
        data["final"] = AgentResult(**data["final"]) if data.get("final") else None
        if data.get("cancel_by") is None and result.cancel_by is not None:
            data["cancel_by"] = result.cancel_by
        return WorkflowResult(**data)

    async def resume(self, *, node_id, message, message_id):
        return await resume_run(
            self.client, self.target, node_id=node_id, message=message,
            message_id=message_id, actor=self.actor,
        )

    async def cancel(self, reason="cancelled"):
        return await self.client.cancel(self.input_ref, actor=self.actor, reason=reason)


class DurableExecutor:
    """Session-Core Executor adapter over ``create_many``, ``workflow_continue``, and local/remote River."""

    def __init__(self, worker, ctx, env, run_id, saved, *, resident_waiter=None, is_recovery: bool = False):
        self.worker, self.ctx, self.env = worker, ctx, env
        self.client = worker.runtime.client
        self.port = worker.inputs
        self.run_id, self.saved = run_id, saved
        self.plan = saved.get("continuation") or {}
        self.driver = TargetRef(ctx.store_id, ctx.session_id)
        self.actor = TrustedActor(ctx.actor.tenant_id, ctx.actor.principal_id, ctx.session_id)
        # Turn 1 of a fresh resident run has 0 prior node-journal rows: initialize in memory
        # so parallel branches coalesce in one tick and skip an empty KV scan.
        self._nodes: dict[str, dict[str, Any]] | None = (
            {} if (resident_waiter is not None and ctx.send_index == 1 and not self.plan) else None
        )
        self._by_input: dict[InputRef, str] = {}
        self._journal_lock = asyncio.Lock()
        self._pending_creates: list[_PendingLeafCreate] = []
        self._flush_scheduled = False
        self._flush_tasks: set[asyncio.Task[None]] = set()
        if resident_waiter is not None:
            self._waiter_q, self._waiter_sig, self._subtree_stopped_flag = resident_waiter
        else:
            self._waiter_q = None
            self._waiter_sig = None
            self._subtree_stopped_flag = [False]
        self._in_memory_results: dict[InputRef, tuple[str, dict[str, Any]]] = {}
        self._remote_wake_pending = False
        self._is_recovery = bool(is_recovery)
        self._last_polled_id = 0

    async def _load_nodes(self, reader=None):
        """Called under the local journal lock; facts belong only to this executor."""
        if self._nodes is not None:
            return
        entries = await journal.nodes(
            self.port.store, self.port.db if reader is None else reader, self.ctx.session_id,
        )
        self._nodes = {}
        for entry in entries:
            key = journal.node_key(
                self.driver, self.run_id, entry["path"], entry["iteration"],
                entry["attempt"], entry["continuation_index"],
            )
            self._publish_node(key, entry)

    def _publish_node(self, key: str, entry: dict[str, Any]) -> None:
        assert self._nodes is not None
        self._nodes[key] = entry
        self._by_input[InputRef(**entry["input_ref"])] = key

    def _check_capacity(self, key: str, extra: int = 0) -> None:
        assert self._nodes is not None
        if key not in self._nodes and len(self._nodes) + extra + 1 > journal.NODE_LIMIT:
            raise CapacityError()

    def _coordinates(self, request):
        raw_attempt = request.attempt
        offset = self.plan.get("attempt_offsets", {}).get(request.node_path, 0)
        if offset:
            request = replace(request, attempt=request.attempt + offset)
        body = request.body
        if request.continuation == 0:
            body = self.plan.get("body_overrides", {}).get(request.node_path, request.body)
        if body != request.body:
            request = replace(request, body=body)
        hash_payload = plain(request)
        if request.continuation > 0:
            hash_payload["body"] = _strip_continuation_todos(hash_payload.get("body"))
        request_hash = sha256_canonical(hash_payload)
        resume_sid: str | None = None
        if request.session_id is None and request.continuation == 0 and raw_attempt == 1 and self.plan:
            route_key = canonical([request.node_path, request.iteration_path])
            resume_sid = (self.plan.get("resume_sessions") or {}).get(route_key)
        is_continuation = (
            request.session_id is not None
            or resume_sid is not None
            or request.continuation > 0
        )
        sid = request.session_id or resume_sid or "ses_" + sha256_canonical([
            "workflow_node", self.driver.store_id, self.driver.session_id, self.run_id,
            request.node_path, request.iteration_path, request.attempt,
        ])[:56]
        target = TargetRef(self.driver.store_id, sid)
        start = InputRef(
            self.driver.store_id, sid,
            "inp_" + sha256_canonical([
                "workflow_node_start", self.driver.store_id, self.driver.session_id, self.run_id,
                request.node_path, request.iteration_path, request.attempt,
                request.continuation, request_hash,
            ])[:56],
        )
        key = journal.node_key(
            self.driver, self.run_id, request.node_path, request.iteration_path,
            request.attempt, request.continuation,
        )
        return request, target, start, key, request_hash, is_continuation

    def _descriptor(self, request):
        from power_loop.runtime.execution_roles import role_tools
        from power_loop.runtime.spec import AgentSpec, _child_config

        spec = AgentSpec.from_dict(request.spec)
        config = _child_config(spec, SimpleNamespace(config=self.env.config))
        available = self.env.tool_registry.names() if self.env.tool_registry else []
        tools = role_tools(
            spec.tools if spec.tools is not None else available,
            "workflow_node", registry=self.env.tool_registry,
        )
        options = RequestOptions(
            tools=tuple(tools),
            **{
                name: getattr(config, name)
                for name in (
                    "model", "temperature", "max_tokens", "max_rounds", "max_tokens_per_run",
                    "system_prompt", "response_format",
                )
            },
        )
        return replace(
            self.ctx.descriptor,
            kind="agent",
            config={
                **{
                    key: value for key, value in plain(self.ctx.descriptor.config).items()
                    if key != "workflow"
                },
                "child_spec": request.spec,
                "child_options": plain(options),
                "workflow_node": {
                    "workflow_run_id": self.run_id,
                    "node_id": request.node_path,
                    "route": request.iteration_path,
                    "attempt": request.attempt,
                },
            },
        )

    async def _check_owner(self, tx):
        state = await self.port.store._db.dialect.lock_state(tx, self.port.s, self.ctx.session_id)
        batch = self.ctx.batch
        if (
            state["execution_state"] != "running"
            or int(state["send_index"]) != int(batch.send_index)
            or int(state["owner_epoch"]) != int(batch.owner_epoch)
        ):
            raise SessionLeaseLost(self.ctx.session_id)
        return state

    def _build_node_entry(self, request, target: TargetRef, start: InputRef, request_hash: str) -> dict[str, Any]:
        return {
            "version": 1,
            "driver_target": plain(self.driver),
            "run_id": self.run_id,
            "path": request.node_path,
            "iteration": request.iteration_path,
            "attempt": request.attempt,
            "continuation_index": request.continuation,
            "target_ref": plain(target),
            "session_id": target.session_id,
            "input_ref": plain(start),
            "request_hash": request_hash,
            "revision": int(self.plan.get("revision", 0)),
            "superseded": False,
            "result": None,
            "result_seq": None,
            "result_input_ref": None,
            "late_results": [],
        }

    async def _cancel_leaf_if_needed(self, target: TargetRef) -> None:
        if (
            not self.ctx.shutdown_interrupted
            and not (self.ctx.ownership is not None and self.ctx.ownership.lost)
            and self.ctx.stop.reason != "worker shutdown"
            and not self._subtree_stopped_flag[0]
            and self.ctx.cancel_by != "subtree"
        ):
            await self.client.cancel(
                target,
                actor=self.actor,
                scope="subtree",
                reason="workflow branch stopped",
                cancel_by="subtree",
            )

    async def _cancel_aborted_leaf_creates(
        self,
        items: list[_PendingLeafCreate],
        preclaimed_by_sid: dict[str, Any],
    ) -> None:
        """Cancel leaves whose caller future was cancelled during ``create_many`` / journal."""
        can_cancel = (
            not self.ctx.shutdown_interrupted
            and not (self.ctx.ownership is not None and self.ctx.ownership.lost)
            and self.ctx.stop.reason != "worker shutdown"
            and not getattr(self.client, "_closed", False)
        )
        for item in items:
            preclaimed_batch = preclaimed_by_sid.get(item.target.session_id)
            try:
                if can_cancel:
                    try:
                        await self.client.cancel(
                            item.target,
                            actor=self.actor,
                            scope="subtree",
                            reason="workflow branch stopped",
                            cancel_by="subtree",
                        )
                    except RuntimeRequestError:
                        pass
                if preclaimed_batch is not None:
                    try:
                        await self.port.final(
                            replace(preclaimed_batch, cancel_by="subtree"),
                            consumed=preclaimed_batch.input_refs,
                            status="cancelled",
                            final_text="",
                            error="workflow branch stopped",
                            error_code="cancelled",
                        )
                    except SessionLeaseLost:
                        pass
            finally:
                if preclaimed_batch is not None:
                    self.worker._unnote_local_claim(
                        preclaimed_batch.owner_node_id,
                        preclaimed_batch.owner_boot_id,
                        [(preclaimed_batch.target.session_id, preclaimed_batch.owner_epoch)],
                    )

    async def aclose(self) -> None:
        """Cancel any unstarted pending creates and drain all active flush tasks."""
        pending = list(self._pending_creates)
        self._pending_creates.clear()
        self._flush_scheduled = False
        for item in pending:
            if not item.future.done():
                item.future.cancel()
        while self._flush_tasks:
            tasks = list(self._flush_tasks)
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    async def _enqueue_leaf_create(
        self, key: str, request, target: TargetRef, start: InputRef,
        request_hash: str, descriptor: RuntimeDescriptor,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        identity = sha256_canonical([
            "workflow_session", self.driver.store_id, self.driver.session_id,
            target.session_id, descriptor, start,
        ])
        child_spec = ChildSpec(
            session_id=target.session_id,
            message_id=start.message_id,
            body=request.body,
            descriptor=descriptor,
            result_kind="workflow_node_result",
            reply_to_session_id=self.driver.session_id,
            cancel_propagation="linked",
            creation_identity_hash=identity,
        )
        self._pending_creates.append(_PendingLeafCreate(
            key=key, request=request, target=target, start=start,
            request_hash=request_hash, descriptor=descriptor, spec=child_spec, future=fut,
        ))
        if not self._flush_scheduled:
            self._flush_scheduled = True
            task = asyncio.create_task(self._flush_pending_creates())
            self._flush_tasks.add(task)
            task.add_done_callback(self._flush_tasks.discard)
        try:
            return await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                await self._cancel_leaf_if_needed(target)
            raise

    async def _flush_pending_creates(self) -> None:
        # Yield once so all parallel/foreach branches scheduled in this event-loop tick coalesce
        # into ONE single atomic create_many transaction + ONE batched node-journal transaction!
        await asyncio.sleep(0)
        while self._journal_lock.locked():
            await asyncio.sleep(0)
        batch_items = [item for item in self._pending_creates if not item.future.cancelled()]
        self._pending_creates.clear()
        self._flush_scheduled = False
        if not batch_items:
            return

        r = self.worker.runtime
        can_preclaim = (
            self.worker._started
            and not self.worker._closing
            and getattr(r, "fuse_local_send", True)
            and getattr(r.coordinator, "installs_session_fence", False)
            and all(self.worker._capable(item.descriptor) for item in batch_items)
        )
        free_slots = self.worker.available_llm_slots() if can_preclaim else 0

        try:
            cm = await self.port.create_many(
                parent=self.driver,
                actor=self.actor,
                specs=[item.spec for item in batch_items],
                node_id=r.node_id,
                boot_id=r.boot_id,
                ttl_s=r.session_lease_ttl_s,
                can_execute=can_preclaim,
                free_slots=free_slots,
                parent_owner_epoch=self.ctx.batch.owner_epoch,
                parent_cancel_seq=self.ctx.batch.cancel_seq,
            )
            preclaimed_by_sid = {b.target.session_id: b for b in cm.preclaimed}
            live_items = list(batch_items)
            cancelled_sids: set[str] = set()

            async def _cancel_newly_cancelled() -> None:
                nonlocal live_items
                while True:
                    newly_cancelled = [
                        item for item in live_items
                        if item.future.cancelled() and item.target.session_id not in cancelled_sids
                    ]
                    if not newly_cancelled:
                        break
                    for item in newly_cancelled:
                        cancelled_sids.add(item.target.session_id)
                    live_items = [
                        item for item in live_items if item.target.session_id not in cancelled_sids
                    ]
                    await self._cancel_aborted_leaf_creates(newly_cancelled, preclaimed_by_sid)

            # Cancel any leaves whose futures were cancelled during create_many FIRST —
            # before dispatching or awaiting notifications for surviving siblings in the
            # same batch, so a concurrent worker scan cannot claim a cancelled queued leaf.
            await _cancel_newly_cancelled()

            live_sids = {item.target.session_id for item in live_items}
            handled_sids: set[str] = set()
            for b in cm.preclaimed:
                if b.target.session_id in live_sids:
                    handled_sids.add(b.target.session_id)
                    self.worker.dispatch_claimed(b)
            spec_by_sid = {item.target.session_id: item for item in live_items}
            for qt in cm.queued_targets:
                await _cancel_newly_cancelled()
                it = spec_by_sid.get(qt.session_id)
                if it is None or it.target.session_id in cancelled_sids:
                    continue
                handled_sids.add(qt.session_id)
                await self.client._notify(qt, it.start.message_id, it.start, ["work", "activation"])
                if self.client.wake_transport is not None:
                    try:
                        await self.client.wake_transport.publish_wake(qt.session_id)
                    except Exception:
                        pass
            if cm.already_created:
                for item in list(live_items):
                    await _cancel_newly_cancelled()
                    if item.target.session_id in cancelled_sids or item.target.session_id in handled_sids:
                        continue
                    await self.client._notify(
                        item.target, item.start.message_id, item.start, ["work", "activation"],
                    )
                    if self.client.wake_transport is not None:
                        try:
                            await self.client.wake_transport.publish_wake(item.target.session_id)
                        except Exception:
                            pass
            await _cancel_newly_cancelled()
        except BaseException as exc:
            for item in batch_items:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        if not live_items:
            return

        # Fixed order: session+start committed above, journal committed below.
        accepted_items: list[tuple[_PendingLeafCreate, dict[str, Any]]] = []
        late_cancelled: list[_PendingLeafCreate] = []
        async with self._journal_lock:
            assert self._nodes is not None
            extra = 0
            for item in live_items:
                existing = self._nodes.get(item.key)
                if existing is not None:
                    if existing["request_hash"] != item.request_hash:
                        if not item.future.done():
                            item.future.set_exception(
                                RuntimeRequestError("request_conflict", "workflow dispatch input changed")
                            )
                    else:
                        if item.future.cancelled():
                            late_cancelled.append(item)
                        elif not item.future.done():
                            item.future.set_result(existing)
                    continue
                try:
                    self._check_capacity(item.key, extra=extra)
                except BaseException as cap_exc:
                    if not item.future.done():
                        item.future.set_exception(cap_exc)
                    continue
                entry = self._build_node_entry(item.request, item.target, item.start, item.request_hash)
                accepted_items.append((item, entry))
                extra += 1

            if accepted_items:
                try:
                    async with self.port.db.transaction() as tx:
                        await self._check_owner(tx)
                        await journal.put_many(
                            self.port.store,
                            tx,
                            self.ctx.session_id,
                            [(it.key, ent) for it, ent in accepted_items],
                        )
                except BaseException as j_exc:
                    for it, _ in accepted_items:
                        if not it.future.done():
                            it.future.set_exception(j_exc)
                    return
                for it, ent in accepted_items:
                    self._publish_node(it.key, ent)
                    if it.future.cancelled():
                        late_cancelled.append(it)
                    elif not it.future.done():
                        it.future.set_result(ent)

        for item in late_cancelled:
            await self._cancel_leaf_if_needed(item.target)

    async def _send_leaf_continuation(
        self, target: TargetRef, start: InputRef, body: Any, *, max_rounds: int | None = None,
    ) -> None:
        """Original-leaf continuation via standard accept/accept_and_claim(kind='workflow_continue')."""
        r = self.worker.runtime
        reply_route = {"target": self.driver, "result_kind": "workflow_node_result"}
        can_fuse = (
            self.worker._started
            and not self.worker._closing
            and getattr(r, "fuse_local_send", True)
            and getattr(r.coordinator, "installs_session_fence", False)
            and self.worker.can_fuse_locally(target.session_id)
        )
        batch = None
        if can_fuse:
            receipt, batch, activate = await self.port.accept_and_claim(
                target, body, message_id=start.message_id, actor=self.actor,
                node_id=r.node_id, boot_id=r.boot_id, ttl_s=r.session_lease_ttl_s,
                kind="workflow_continue", reply_route=reply_route, public=False,
                can_execute=self.worker._capable, max_rounds=max_rounds,
            )
        else:
            receipt, activate = await self.port.accept(
                target, body, message_id=start.message_id, actor=self.actor,
                kind="workflow_continue", reply_route=reply_route, public=False,
                max_rounds=max_rounds, _with_activation=True,
            )
        if batch is not None:
            self.worker.dispatch_claimed(batch)
            return
        if receipt.state == "accepted" and activate:
            await self.client._notify(target, start.message_id, start, ["work", "activation"])
            if self.client.wake_transport is not None:
                try:
                    await self.client.wake_transport.publish_wake(target.session_id)
                except Exception:
                    pass

    def _drain_local_waiter(self) -> None:
        if self._waiter_q is None:
            return
        while True:
            try:
                ev = self._waiter_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            ev_type = ev.get("type")
            if ev_type == "leaf_result":
                payload = ev["payload"]
                src = InputRef(**payload["input_ref"])
                self._in_memory_results[src] = (str(ev["item_id"]), _normalize_node_status(payload))
            elif ev_type == "remote_wake":
                self._remote_wake_pending = True
            elif ev_type == "cancel":
                if ev.get("cancel_by") is not None:
                    self.ctx.cancel_by = ev["cancel_by"]

    async def _pull_remote_pending_results(self, *, include_claimed_send: bool = False) -> int:
        if include_claimed_send:
            rows = await self.port.poll_driver_results_for_send(
                self.ctx.session_id, self.ctx.send_index, after_id=self._last_polled_id,
            )
        else:
            rows = await self.port.poll_pending_driver_results(
                self.ctx.session_id, after_id=self._last_polled_id,
            )
        for row in rows:
            self._last_polled_id = max(self._last_polled_id, int(row["id"]))
            payload = json.loads(row["content_json"])
            src = InputRef(**payload["input_ref"])
            self._in_memory_results[src] = (str(row["item_id"]), _normalize_node_status(payload))
        if rows and self._waiter_sig is not None:
            # Wake sibling coroutines so they pick up newly cached results without another DB read.
            self._waiter_sig.set()
        return len(rows)

    async def _redeliver_missing_node_result(self, leaf_target: TargetRef, start: InputRef, res: Result) -> None:
        driver_events: list[tuple[str, str, dict[str, Any]]] = []
        async with self.port.db.transaction() as tx:
            leaf_row = await self.port._session(tx, leaf_target.session_id)
            await self.port._river_deliver_tx(
                tx,
                child_target=leaf_target,
                child_tenant_id=leaf_row["tenant_id"],
                child_principal_id=leaf_row["principal_id"],
                parent_session_id=self.driver.session_id,
                start_ref=start,
                result=res,
                result_kind="workflow_node_result",
                child_wake=1,
                _driver_events_out=driver_events,
            )
        for drv_sid, item_id, payload in driver_events:
            self.port._push_driver_leaf_result(drv_sid, item_id, payload)

    def _is_leaf_local_pending_or_active(self, leaf_sid: str) -> bool:
        """True when this Worker owns or is actively draining the local queue for ``leaf_sid``."""
        if (
            leaf_sid in self.worker._active
            or leaf_sid in self.worker._pending
            or leaf_sid in self.worker._claimed_pending
        ):
            return True
        if not self.worker.runtime.shared_execution and (
            self.worker._active_llm_count() > 0
            or bool(self.worker._pending)
            or bool(self.worker._claimed_pending)
        ):
            return True
        return False

    async def _wait_resident_node_result(self, key: str, start: InputRef, target: TargetRef) -> dict[str, Any]:
        """Wait for a leaf result: 0 DB reads in-process; 1 indexed SELECT on cross-process wake."""
        assert self._waiter_sig is not None and self._nodes is not None
        polled_initial = False
        idle_polls = 0
        while True:
            self._drain_local_waiter()
            existing_res = self._nodes[key].get("result")
            if existing_res is not None:
                return existing_res
            hit = self._in_memory_results.get(start)
            if hit is not None:
                item_id, raw = hit
                entry = self._nodes[key]
                entry["result"] = raw
                entry["result_seq"] = raw.get("result_seq")
                entry["result_input_ref"] = plain(InputRef(self.ctx.store_id, self.ctx.session_id, item_id))
                return raw

            # Only query DB on real crash-takeover entry (origin kind == 'workflow_continue') OR
            # an external remote wake when this target leaf is NOT running/queued locally!
            # Normal local explicit handle.resume() (send_index > 1, kind == 'user') does 0 DB reads.
            is_recovery_init = self._is_recovery and not polled_initial
            is_local_leaf = self._is_leaf_local_pending_or_active(target.session_id)
            if is_recovery_init or (self._remote_wake_pending and not is_local_leaf):
                self._remote_wake_pending = False
                polled_initial = True
                await self._pull_remote_pending_results(include_claimed_send=is_recovery_init)
                if start in self._in_memory_results:
                    continue
            elif self._remote_wake_pending and not self.worker.runtime.shared_execution:
                self._remote_wake_pending = False

            self._waiter_sig.clear()
            self._drain_local_waiter()
            if start in self._in_memory_results or self._nodes[key].get("result") is not None:
                continue
            if self._remote_wake_pending and not self._is_leaf_local_pending_or_active(target.session_id):
                continue

            try:
                await asyncio.wait_for(self._waiter_sig.wait(), timeout=0.25)
            except TimeoutError:
                # Same-process leaf still executing or queued locally: 0 extra DB reads!
                # When it commits Final, RuntimeInputStore pushes directly to _waiter_q & sets _waiter_sig.
                if self._is_leaf_local_pending_or_active(target.session_id):
                    continue
                idle_polls += 1
                # Fallback for cross-process DatabasePollingTransport / lost hints or full inbox
                found = await self._pull_remote_pending_results(
                    include_claimed_send=self._is_recovery,
                )
                if found or start in self._in_memory_results:
                    continue
                if idle_polls >= 2:
                    res = await self.port.get_result(start, actor=self.actor)
                    if res is not None:
                        await self._consume_results(for_key=key)
                        if self._nodes[key].get("result") is not None:
                            return self._nodes[key]["result"]
                        await self._redeliver_missing_node_result(target, start, res)

    async def run_agent(self, request, *, allow_start=True):
        request, target, start, key, request_hash, is_continuation = self._coordinates(request)
        async with self._journal_lock:
            await self._load_nodes()
            assert self._nodes is not None
            saved = self._nodes.get(key)
            if saved is not None and saved["request_hash"] != request_hash:
                raise RuntimeRequestError("request_conflict", "workflow dispatch input changed")
            if saved is None:
                if not allow_start:
                    return {"status": "budget_exceeded", "usage": {}, "session_id": target.session_id}
                self._check_capacity(key)

        if saved is None:
            descriptor = self._descriptor(request)
            if not is_continuation:
                saved = await self._enqueue_leaf_create(
                    key, request, target, start, request_hash, descriptor,
                )
            else:
                cont_max_rounds = request.spec.get("max_rounds") if request.continuation > 0 else None
                await self._send_leaf_continuation(target, start, request.body, max_rounds=cont_max_rounds)
                async with self._journal_lock:
                    saved = self._nodes.get(key)
                    if saved is not None and saved["request_hash"] != request_hash:
                        raise RuntimeRequestError("request_conflict", "workflow dispatch input changed")
                    if saved is None:
                        self._check_capacity(key)
                        saved = self._build_node_entry(request, target, start, request_hash)
                        async with self.port.db.transaction() as tx:
                            await self._check_owner(tx)
                            await journal.put(self.port.store, tx, self.ctx.session_id, key, saved)
                        self._publish_node(key, saved)

        if saved.get("result") is None:
            use_resident_waiter = self._waiter_q is not None and "wait" not in vars(self.client)
            try:
                if use_resident_waiter:
                    raw_res = await self._wait_resident_node_result(key, start, target)
                    saved = self._nodes[key]
                    _ = raw_res
                else:
                    result = await self.client.wait(start, actor=self.actor, deadline_ms=None)
                    consumed = await self._consume_results(for_key=key)
                    saved = consumed.get(key, self._nodes[key])
                    if saved.get("result") is None:
                        await self._redeliver_missing_node_result(target, start, result)
                        consumed = await self._consume_results(for_key=key)
                        saved = consumed.get(key, self._nodes[key])
                    if saved.get("result") is None:
                        raise RuntimeRequestError("result_delivery_failed", start.message_id)
            except asyncio.CancelledError:
                # Graph halt, workflow timeout, or single-InputRef cancel stops sibling/active leaves
                # with cancel_by='subtree' (child_wake=0 -> NEVER ghost-rewakes finished Driver!).
                # If the Driver itself was already cancelled with scope='subtree', RuntimeInputStore.cancel
                # already stopped all active leaves in 1 atomic DB transaction (0 redundant SQL!).
                await self._cancel_leaf_if_needed(target)
                raise

        assert self._nodes is not None
        raw = dict(saved["result"])
        next_key = journal.node_key(
            self.driver, self.run_id, request.node_path, request.iteration_path,
            request.attempt, request.continuation + 1,
        )
        following = self._nodes.get(next_key)
        replayed = bool(self.plan) and int(saved.get("revision", 0)) < int(self.plan.get("revision", 0))
        return {
            **raw,
            "session_id": target.session_id,
            "has_next_continuation": following is not None,
            "accounting_usage": {} if replayed else raw.get("usage") or {},
        }

    async def _consume_results(self, *, for_key=None):
        async with self._journal_lock:
            if (
                for_key is not None
                and self._nodes is not None
                and for_key in self._nodes
                and self._nodes[for_key].get("result") is not None
            ):
                return {for_key: self._nodes[for_key]}
            return await self._consume_results_locked()

    async def _consume_results_locked(self):
        """Consume ready node results and journal them in one conditionally-fenced transaction."""
        async with self.port.db.transaction() as tx:
            rows = await tx.fetchall(
                f"SELECT * FROM {self.port.q} WHERE session_id=? "
                "AND kind='workflow_node_result' AND terminal_status IS NULL AND completed_at IS NULL "
                "AND (status='pending' OR (status='delivered' AND delivered_send_index=?)) "
                "ORDER BY id LIMIT 1000",
                (self.ctx.session_id, self.ctx.send_index),
            )
            if not rows:
                # Verify ownership even when empty if called explicitly
                state = await tx.fetchone(
                    f"SELECT execution_state,send_index,owner_epoch FROM {self.port.s} WHERE session_id=?",
                    (self.ctx.session_id,),
                )
                if (
                    state is None
                    or state["execution_state"] != "running"
                    or int(state["send_index"]) != int(self.ctx.send_index)
                    or int(state["owner_epoch"]) != int(self.ctx.batch.owner_epoch)
                ):
                    raise SessionLeaseLost(self.ctx.session_id)
                return {}

            await self._load_nodes(tx)
            assert self._nodes is not None
            updates: dict[str, dict[str, Any]] = {}
            consumed_row_ids: list[int] = []
            w_sub = p_sub = b_sub = 0
            database_now = now_ms()

            for row in rows:
                body = json.loads(row["content_json"])
                source = InputRef(**body["input_ref"])
                key = self._by_input.get(source)
                if key is None:
                    continue
                if key not in updates:
                    prior = self._nodes[key]
                    updates[key] = {**prior, "late_results": list(prior["late_results"])}
                entry = updates[key]
                raw = _normalize_node_status(body)
                if "send_index" not in body:
                    source_record = await self.port.input(tx, source)
                    if source_record is None:
                        raise RuntimeRequestError("workflow_result_source_not_terminal")
                    source_sess = await self.port._session(tx, source.session_id)
                    derived_src = await self.port._derive_input_terminal(
                        tx, dict(source_sess), source_record.row,
                    )
                    if derived_src["terminal_status"] is None:
                        raise RuntimeRequestError("workflow_result_source_not_terminal")
                    raw["send_index"] = derived_src["result_send_index"]
                    if raw.get("cancel_by") is None and derived_src.get("cancel_by") is not None:
                        raw["cancel_by"] = derived_src["cancel_by"]

                is_same_primary = (
                    entry.get("result") is not None
                    and entry["result"].get("input_ref") == body["input_ref"]
                    and not self.saved.get("aggregate_delivered")
                    and not entry.get("superseded")
                )
                if is_same_primary:
                    entry["result"] = raw
                    entry["result_seq"] = body.get("result_seq")
                    entry["result_input_ref"] = plain(
                        InputRef(self.ctx.store_id, self.ctx.session_id, row["item_id"])
                    )
                elif self.saved.get("aggregate_delivered") or entry.get("superseded") or entry.get("result") is not None:
                    if not any(item.get("input_ref") == body["input_ref"] for item in entry["late_results"]):
                        entry["late_results"].append(raw)
                else:
                    entry["result"] = raw
                    entry["result_seq"] = body.get("result_seq")
                    entry["result_input_ref"] = plain(
                        InputRef(self.ctx.store_id, self.ctx.session_id, row["item_id"])
                    )

                consumed_row_ids.append(int(row["id"]))
                if row["status"] == "pending":
                    w_sub += int(row["wake"])
                    p_sub += 1
                    b_sub += int(row["content_bytes"])

            # Conditional owner fence + pending counter drain in 1 UPDATE sessions statement!
            fenced = await tx.execute(
                f"UPDATE {self.port.s} SET "
                "wake_pending_count=CASE WHEN wake_pending_count>=? THEN wake_pending_count-? ELSE 0 END,"
                "pending_count=CASE WHEN pending_count>=? THEN pending_count-? ELSE 0 END,"
                "pending_bytes=CASE WHEN pending_bytes>=? THEN pending_bytes-? ELSE 0 END "
                "WHERE session_id=? AND execution_state='running' AND send_index=? AND owner_epoch=?",
                (
                    w_sub, w_sub, p_sub, p_sub, b_sub, b_sub,
                    self.ctx.session_id, self.ctx.send_index, self.ctx.batch.owner_epoch,
                ),
            )
            if fenced != 1:
                raise SessionLeaseLost(self.ctx.session_id)

            if updates:
                await journal.put_many(
                    self.port.store, tx, self.ctx.session_id, list(updates.items()),
                )
            if consumed_row_ids:
                ph = ",".join("?" for _ in consumed_row_ids)
                changed = await tx.execute(
                    f"UPDATE {self.port.q} SET status='delivered',delivered_at=?,"
                    "delivered_seq=NULL,delivered_send_index=?,terminal_status='completed',"
                    "result_send_index=?,completed_at=? "
                    f"WHERE session_id=? AND id IN ({ph}) "
                    "AND (status='pending' OR (status='delivered' AND delivered_send_index=? "
                    "AND terminal_status IS NULL))",
                    (
                        database_now, self.ctx.send_index, self.ctx.send_index, database_now,
                        self.ctx.session_id, *consumed_row_ids, self.ctx.send_index,
                    ),
                )
                if changed != len(consumed_row_ids):
                    raise SessionLeaseLost(self.ctx.session_id)

        for key, entry in updates.items():
            self._publish_node(key, entry)
        return updates

    async def todo_remaining(self, sid):
        state = await self.port.store.get_runtime_state(sid, "todo") or {}
        return [item["text"] for item in state.get("items", []) if item["status"] != "completed"]


async def _requeue_deferred_inputs_tx(port, tx, sid: str, send_index: int, requeue_ids: tuple[str, ...]) -> None:
    """Restore co-claimed secondary formal resumes to ``pending`` before starting the first graph."""
    if not requeue_ids:
        return
    ph = ",".join("?" for _ in requeue_ids)
    urows = await tx.fetchall(
        f"SELECT item_id,wake,content_bytes FROM {port.q} WHERE session_id=? "
        f"AND status='delivered' AND delivered_send_index=? AND terminal_status IS NULL "
        f"AND item_id IN ({ph})",
        (sid, send_index, *requeue_ids),
    )
    if not urows:
        return
    live_ids = [str(r["item_id"]) for r in urows]
    lph = ",".join("?" for _ in live_ids)
    await tx.execute(
        f"UPDATE {port.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
        "delivered_send_index=NULL WHERE session_id=? AND status='delivered' "
        f"AND delivered_send_index=? AND terminal_status IS NULL AND item_id IN ({lph})",
        (sid, send_index, *live_ids),
    )
    w_add = sum(int(r["wake"]) for r in urows)
    b_add = sum(int(r["content_bytes"]) for r in urows)
    await tx.execute(
        f"UPDATE {port.s} SET wake_pending_count=wake_pending_count+?,"
        "pending_count=pending_count+?,pending_bytes=pending_bytes+? WHERE session_id=?",
        (w_add, len(urows), b_add, sid),
    )


async def _finalize_cancelled_driver_batch_tx(port, tx, state, ctx):
    """Settle an already-cancelled driver batch inside an existing locked-state transaction."""
    undeliverables: list[tuple[Any, str, Any, str, str]] = []
    driver_events_out: list[tuple[str, str, dict[str, Any]]] = []
    eff_state = state
    if eff_state["cancel_by"] is None and eff_state["lifecycle"] not in ("closing", "closed"):
        eff_state = dict(eff_state)
        eff_state["cancel_by"] = ctx.cancel_by or ctx.batch.cancel_by or "user"
        if eff_state.get("cancel_reason") is None:
            eff_state["cancel_reason"] = ctx.stop.reason or "cancelled"
    final_batch = await port._recover_workflow_tx(
        tx,
        eff_state,
        ctx.batch.target,
        int(ctx.send_index),
        int(ctx.batch.owner_epoch),
        await port._db_now(tx),
        undeliverables,
        driver_events_out,
    )
    return final_batch, undeliverables, driver_events_out


async def _flush_cancelled_driver_post_commit(port, undeliverables, driver_events_out) -> None:
    for drv_sid, item_id, payload in driver_events_out:
        port._push_driver_leaf_result(drv_sid, item_id, payload)
    for c_target, p_sid, r_res, r_kind, r_reason in undeliverables:
        await port._fire_result_undeliverable(
            child_target=c_target,
            parent_session_id=p_sid,
            result=r_res,
            result_kind=r_kind,
            reason=r_reason,
        )


def _driver_batch_cancelled_in_state(state, ctx) -> bool:
    return (
        int(state["cancel_seq"]) > int(ctx.batch.cancel_seq)
        or state["lifecycle"] in ("closing", "closed")
        or bool(ctx.batch.reclaimed_cancelled)
        or ctx.batch.cancel_by is not None
        or ctx.stop.is_cancelled()
    )


async def _prepare_explicit(worker, ctx, saved, origin, options, *, requeue_ids: tuple[str, ...] = ()):
    from .continuation import plan_continuation

    if getattr(ctx, "ownership", None) is not None and ctx.ownership.lost:
        raise SessionLeaseLost(ctx.session_id)
    port = worker.inputs
    spec = WorkflowSpec.from_json(plain(ctx.descriptor.config["workflow"]["spec"]))
    plan = plan_continuation(spec, options["node_id"])
    entries = await journal.nodes(port.store, port.db, ctx.session_id)
    offsets: dict[str, int] = {}
    resume_sessions: dict[str, tuple[int, str]] = {}
    for entry in entries:
        if entry["path"] in plan.invalidated:
            attempt = int(entry["attempt"])
            offsets[entry["path"]] = max(offsets.get(entry["path"], 0), attempt)
            route_key = canonical([entry["path"], entry["iteration"]])
            prev = resume_sessions.get(route_key)
            if prev is None or attempt >= prev[0]:
                resume_sessions[route_key] = (attempt, entry["session_id"])
    revision = int(saved.get("execution_revision", 0)) + 1
    continuation = {
        "attempt_offsets": offsets,
        "resume_sessions": {k: v[1] for k, v in resume_sessions.items()},
        "body_overrides": {options["node_id"]: options["message"]},
        "prior_usage": (saved.get("result") or {}).get("usage") or {},
        "invalidated": sorted(plan.invalidated),
        "revision": revision,
    }
    cancelled_batch = None
    undeliverables: list[tuple[Any, str, Any, str, str]] = []
    driver_events_out: list[tuple[str, str, dict[str, Any]]] = []
    async with port.db.transaction() as tx:
        state = await port.store._db.dialect.lock_state(tx, port.s, ctx.session_id)
        if (
            state["execution_state"] != "running"
            or int(state["send_index"]) != int(ctx.send_index)
            or int(state["owner_epoch"]) != int(ctx.batch.owner_epoch)
        ):
            raise SessionLeaseLost(ctx.session_id)
        if _driver_batch_cancelled_in_state(state, ctx):
            cancelled_batch, undeliverables, driver_events_out = (
                await _finalize_cancelled_driver_batch_tx(port, tx, state, ctx)
            )
        else:
            if requeue_ids:
                await _requeue_deferred_inputs_tx(port, tx, ctx.session_id, ctx.send_index, requeue_ids)
            to_write: list[tuple[str, dict[str, Any]]] = []
            for entry in entries:
                if entry["path"] in plan.invalidated:
                    entry["superseded"] = True
                    key = journal.node_key(
                        TargetRef(ctx.store_id, ctx.session_id), saved["run_id"],
                        entry["path"], entry["iteration"], entry["attempt"], entry["continuation_index"],
                    )
                    to_write.append((key, entry))
            saved.update(
                continuation=continuation,
                execution_revision=revision,
                definition_ref=plain(origin.ref),
                status="running",
                cancel_by=None,
                aggregate_delivered=False,
            )
            to_write.append((journal.run_key(saved["run_id"]), saved))
            await journal.put_many(port.store, tx, ctx.session_id, to_write)
    if cancelled_batch is not None:
        await _flush_cancelled_driver_post_commit(port, undeliverables, driver_events_out)
        return saved, cancelled_batch
    return saved, None


async def _resolve_input_chain_ids(port, store_id: str, session_id: str, origin: InputRecord) -> list[str]:
    """Walk ``continuation_of`` from ``origin`` back to its business root."""
    chain_ids: list[str] = [origin.ref.message_id]
    current = origin
    while current.row.get("continuation_of") is not None:
        parent_rec = await port.input(
            port.db, InputRef(store_id, session_id, current.row["continuation_of"]),
        )
        if parent_rec is None:
            break
        chain_ids.append(parent_rec.ref.message_id)
        current = parent_rec
    return chain_ids


async def _complete_driver(
    worker, ctx, saved, origin, aggregate, *,
    drain_node_results: bool = False,
    consumed: tuple[InputRef, ...] | None = None,
    extra_late_chains: tuple[tuple[str, tuple[str, ...]], ...] = (),
):
    """Finalize the resident driver in 1 Final transaction (0 final_lock_asc!)."""
    if getattr(ctx, "ownership", None) is not None and ctx.ownership.lost:
        raise SessionLeaseLost(ctx.session_id)
    port = worker.inputs
    status = aggregate["status"]
    if status == "budget_exceeded":
        status = "failed"
    if ctx.stop.is_cancelled():
        status = "cancelled"
    final_text = (aggregate.get("final") or {}).get("text") or ""
    error_list = list(aggregate.get("errors") or [])
    error = (
        None if status == "completed"
        else ((ctx.stop.reason or "cancelled") if status == "cancelled" else (error_list[0] if error_list else status))
    )
    error_code = None if status == "completed" else status

    chain_ids = await _resolve_input_chain_ids(port, ctx.store_id, ctx.session_id, origin)
    root_id = chain_ids[-1]
    chain_id_set = set(chain_ids)
    # Only settle verified non-formal plain late chains whose tip is still delivered in this turn —
    # never sweep unrelated formal workflow_resume inputs or restored-to-pending siblings!
    late_chains_to_check = [
        (
            tip_id,
            tuple(
                mid for mid in dict.fromkeys(cids)
                if mid not in chain_id_set and mid != ctx.batch.start_input_message_id
            ),
        )
        for tip_id, cids in extra_late_chains
        if tip_id not in chain_id_set and tip_id != ctx.batch.start_input_message_id
    ]
    late_chains_to_check = [(tip_id, cids) for tip_id, cids in late_chains_to_check if cids]

    batch_for_final = ctx.batch
    if ctx.cancel_by is not None and batch_for_final.cancel_by is None:
        batch_for_final = replace(batch_for_final, cancel_by=ctx.cancel_by)
    if batch_for_final.result_route is not None and root_id != batch_for_final.start_input_message_id:
        # Explicit resume_run on a new user input: route with that resume input's ref
        new_route = {
            **batch_for_final.result_route,
            "input_ref": plain(InputRef(ctx.store_id, ctx.session_id, root_id)),
        }
        batch_for_final = replace(batch_for_final, result_route=new_route)

    async def _driver_tx_hook(tx, info):
        eff_status = info["effective_status"]
        eff_cby = info["effective_cancel_by"] if eff_status == "cancelled" else None
        agg_final = dict(aggregate)
        if eff_status == "cancelled":
            agg_final["status"] = "cancelled"
            agg_final["cancel_by"] = eff_cby
        elif agg_final.get("cancel_by") != eff_cby:
            agg_final["cancel_by"] = eff_cby
        saved.update(
            status=agg_final["status"],
            cancel_by=eff_cby,
            result=agg_final,
            result_seq=info["result_seq"],
            aggregate_delivered=True,
            aggregate_delivered_at=info["database_now"] or now_ms(),
            continuation={},
            last_input_ref=plain(origin.ref),
        )
        await journal.put(port.store, tx, ctx.session_id, journal.run_key(saved["run_id"]), saved)
        if late_chains_to_check:
            tip_ids = [tip_id for tip_id, _ in late_chains_to_check]
            tph = ",".join("?" for _ in tip_ids)
            live_rows = await tx.fetchall(
                f"SELECT item_id FROM {port.q} WHERE session_id=? AND item_id IN ({tph}) "
                "AND status='delivered' AND delivered_send_index=? AND terminal_status IS NULL",
                (ctx.session_id, *tip_ids, ctx.send_index),
            )
            live_tips = {str(r["item_id"]) for r in live_rows}
            live_late_ids = list(dict.fromkeys(
                mid for tip_id, cids in late_chains_to_check if tip_id in live_tips for mid in cids
            ))
            if live_late_ids:
                ep = ",".join("?" for _ in live_late_ids)
                await tx.execute(
                    f"UPDATE {port.q} SET status='delivered',terminal_status='completed',"
                    "result_send_index=?,completed_at=? "
                    f"WHERE session_id=? AND item_id IN ({ep}) "
                    "AND status IN ('delivered','waiting_continuation') AND terminal_status IS NULL",
                    (ctx.send_index, info["database_now"] or now_ms(), ctx.session_id, *live_late_ids),
                )
        return {"structured_output": agg_final, "result_json": agg_final}

    return await port.final(
        batch_for_final,
        consumed=ctx.batch.input_refs if consumed is None else consumed,
        status=status,
        final_text=final_text,
        usage=aggregate.get("usage") or {},
        structured_output=aggregate,
        error=error,
        error_code=error_code,
        next_seq=ctx.batch.message_seq + 1,
        result_json=aggregate,
        settle_start_input=(root_id == ctx.batch.start_input_message_id),
        settle_chain_ids=tuple(dict.fromkeys(chain_ids)),
        drain_driver_node_results=drain_node_results,
        deliver_to_parent=True,
        write_driver_message=True,
        tx_hook=_driver_tx_hook,
    )


async def _execute_driver_graph(
    worker, ctx, env, run_id: str, saved, spec, resident_waiter, *, is_recovery: bool = False,
):
    """Run one workflow graph execution wave and return ``(result_dict, aborted_early)``."""
    executor = DurableExecutor(
        worker, ctx, env, run_id, saved, resident_waiter=resident_waiter, is_recovery=is_recovery,
    )
    engine = WorkflowEngine(
        executor,
        run_id=run_id,
        driver_sid=ctx.session_id,
        allowed_tools=ctx.descriptor.config["workflow"]["allowed_tools"],
        stop_event=ctx.stop,
        file_io=env.workflow_file_io,
    )
    if executor.plan:
        engine._usage_acc = dict(executor.plan.get("prior_usage") or {})

    if ctx.stop.is_cancelled():
        cby = ctx.cancel_by or "user"
        return engine._build_result(spec, "cancelled", cancel_by=cby).to_dict(), True

    graph = asyncio.create_task(engine.run(spec))
    stopped = asyncio.create_task(ctx.stop.wait())
    aborted_early = False
    try:
        remaining = (
            max(0, spec.timeout_s - (now_ms() - saved["created_at_ms"]) / 1000)
            if spec.timeout_s else None
        )
        done, _ = await asyncio.wait(
            (graph, stopped), timeout=remaining, return_when=asyncio.FIRST_COMPLETED,
        )
        if graph in done:
            result = graph.result()
            if result.status == "cancelled" and result.cancel_by is None:
                executor._drain_local_waiter()
                result.cancel_by = ctx.cancel_by or "user"
        else:
            if getattr(ctx, "ownership", None) is not None and ctx.ownership.lost:
                raise SessionLeaseLost(ctx.session_id)
            aborted_early = True
            timed_out = stopped not in done
            if timed_out:
                ctx.cancel_by = "timeout"
                await worker.inputs.cancel_active(
                    ctx.batch, reason="workflow timeout", cancel_by="timeout",
                )
            else:
                executor._drain_local_waiter()
            graph.cancel()
            await asyncio.gather(graph, return_exceptions=True)
            if timed_out:
                result = engine._build_result(spec, "failed")
                result.errors.append("workflow timeout")
            else:
                cby = ctx.cancel_by or "user"
                result = engine._build_result(spec, "cancelled", cancel_by=cby)
    finally:
        stopped.cancel()
        graph.cancel()
        await asyncio.gather(stopped, graph, return_exceptions=True)
        await executor.aclose()

    if not aborted_early:
        await executor._consume_results()
    return result.to_dict(), aborted_early


async def _finish_late(worker, ctx, refs):
    """Terminalize late result inputs (including any continuation ancestors) without another graph run."""
    port = worker.inputs
    settle_ids: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        cur = await port.input(port.db, ref)
        if cur is None:
            if ref.message_id not in seen:
                seen.add(ref.message_id)
                settle_ids.append(ref.message_id)
            continue
        for mid in await _resolve_input_chain_ids(port, ctx.store_id, ctx.session_id, cur):
            if mid not in seen:
                seen.add(mid)
                settle_ids.append(mid)
    return await port.final(
        ctx.batch,
        consumed=tuple(refs),
        status="completed",
        final_text="",
        next_seq=max(1, ctx.batch.message_seq),
        settle_chain_ids=tuple(settle_ids),
        drain_driver_node_results=True,
        deliver_to_parent=False,
        write_driver_message=False,
    )


def _fast_first_turn_origin(ctx, run_id: str) -> InputRecord | None:
    """Construct Turn-1 origin in memory with 0 DB reads when input_refs == (start_input_ref,)."""
    refs = ctx.batch.input_refs
    if ctx.send_index == 1 and len(refs) == 1 and refs[0].message_id == run_id:
        ref = refs[0]
        return InputRecord(
            ref,
            {
                "session_id": ctx.session_id,
                "item_id": run_id,
                "kind": "user",
                "status": "delivered",
                "content_json": "null",
                "continuation_of": None,
                "continuation_index": None,
            },
            {},
        )
    return None


async def run_driver(worker, ctx, env):
    """Execute the Plan B resident workflow_driver coroutine (0 LLM slots, 1 Claim + 1 Final)."""
    port = worker.inputs
    run_id = ctx.batch.start_input_message_id
    if not run_id:
        session = await port._session(port.db, ctx.session_id)
        run_id = session["start_input_message_id"]
    saved = await journal.get(port.store, port.db, ctx.session_id, journal.run_key(run_id))
    if saved is None:
        raise RuntimeRequestError("workflow_journal_missing")

    fast_origin = _fast_first_turn_origin(ctx, run_id)
    formal_items: list[tuple[InputRecord, InputRecord, dict[str, Any] | None]] = []
    plain_late_chains: list[tuple[str, tuple[str, ...]]] = []

    if fast_origin is not None:
        formal_items.append((fast_origin, fast_origin, None))
    else:
        origins = []
        live_input_refs: list[InputRef] = []
        for ref in ctx.batch.input_refs:
            record = await port.input(port.db, ref)
            if (
                record is not None
                and record.row["status"] == "delivered"
                and record.row["delivered_send_index"] == ctx.send_index
            ):
                live_input_refs.append(ref)
                if record.row["kind"] in {"user", "workflow_continue"}:
                    origins.append(record)
        if len(live_input_refs) != len(ctx.batch.input_refs) and live_input_refs:
            ctx.batch = replace(ctx.batch, input_refs=tuple(live_input_refs))
        def_ref = saved.get("definition_ref") if isinstance(saved, dict) else None
        def_msg_id = def_ref.get("message_id") if isinstance(def_ref, dict) else None

        # Classify every candidate by its business root so:
        # 1) Plain late inputs (and their wfc_late continuations) never shadow formal runs/resumes.
        # 2) Multiple distinct accepted explicit handle.resume() inputs in the same batch each
        #    execute in order with independent results & Main receipts (never collapsed as late!).
        by_root: dict[str, tuple[InputRecord, InputRecord, dict[str, Any] | None]] = {}
        root_order: list[str] = []
        plain_late_records: list[InputRecord] = []
        for o in origins:
            parsed = json.loads(o.row["content_json"])
            root_rec = o
            res_opts = parsed["workflow_resume"] if isinstance(parsed, dict) and "workflow_resume" in parsed else None
            if (
                o.row["kind"] == "workflow_continue"
                and isinstance(parsed, dict)
                and isinstance(parsed.get("workflow_continue"), dict)
                and isinstance(parsed["workflow_continue"].get("origin_input_ref"), dict)
            ):
                root_ref = InputRef(**parsed["workflow_continue"]["origin_input_ref"])
                loaded_root = await port.input(port.db, root_ref)
                if loaded_root is not None:
                    root_rec = loaded_root
                    root_parsed = json.loads(loaded_root.row["content_json"])
                    if isinstance(root_parsed, dict) and "workflow_resume" in root_parsed:
                        res_opts = root_parsed["workflow_resume"]
            is_run_or_resume = (
                res_opts is not None
                or root_rec.ref.message_id == run_id
                or (def_msg_id is not None and root_rec.ref.message_id == def_msg_id)
            )
            if is_run_or_resume:
                rk = root_rec.ref.message_id
                if rk not in by_root:
                    root_order.append(rk)
                    by_root[rk] = (o, root_rec, res_opts)
                elif o.row["kind"] == "workflow_continue":
                    # Prefer latest continuation record for the same business root
                    by_root[rk] = (o, root_rec, res_opts)
            else:
                plain_late_records.append(o)

        for rk in root_order:
            item = by_root[rk]
            # Skip already-terminalized formal roots on crash-replay unless no runnable item remains
            if item[1].row.get("terminal_status") is not None and saved.get("aggregate_delivered"):
                continue
            formal_items.append(item)

        def _formal_priority(item):
            orig_rec, r_rec, _ = item
            if orig_rec.row["kind"] == "workflow_continue":
                return 0
            if r_rec.ref.message_id == run_id or (def_msg_id is not None and r_rec.ref.message_id == def_msg_id):
                return 1
            return 2

        formal_items.sort(key=_formal_priority)

        formal_id_set = {
            mid for item in formal_items for mid in (item[0].ref.message_id, item[1].ref.message_id)
        }
        seen_late: set[str] = set(formal_id_set)
        if run_id:
            seen_late.add(run_id)
        for rec in plain_late_records:
            rec_chain: list[str] = []
            for mid in await _resolve_input_chain_ids(port, ctx.store_id, ctx.session_id, rec):
                if mid not in seen_late:
                    seen_late.add(mid)
                    rec_chain.append(mid)
            if rec_chain:
                plain_late_chains.append((rec.ref.message_id, tuple(rec_chain)))

    # Filter out non-resume run roots when aggregate was already delivered (pure late batch)
    runnable_formal = [
        item for item in formal_items
        if not (saved.get("aggregate_delivered") and item[2] is None)
    ]
    if not runnable_formal:
        if saved.get("aggregate_delivered"):
            executor = DurableExecutor(worker, ctx, env, run_id, saved)
            await executor._consume_results()
            return await _finish_late(worker, ctx, list(ctx.batch.input_refs))
        if not formal_items and not plain_late_chains:
            raise RuntimeRequestError("workflow_origin_missing")
        # Non-delivered fallback (e.g. unexpected non-start input before first delivery)
        fallback_rec = await port.input(port.db, ctx.batch.input_refs[-1])
        if fallback_rec is None:
            raise RuntimeRequestError("workflow_origin_missing")
        runnable_formal = [(fallback_rec, fallback_rec, None)]

    spec = WorkflowSpec.from_json(plain(ctx.descriptor.config["workflow"]["spec"]))
    origin_0, resume_origin_0, resume_opts_0 = runnable_formal[0]
    # When multiple distinct formal runs/resumes are claimed in one batch, immediately restore
    # deferred secondary resumes back to status='pending' BEFORE starting the first graph so:
    # 1) cancel(second_resume) voids ONLY the second resume without cancelling the running first!
    # 2) If not cancelled, live wake_pending_count > 0 cleanly re-queues Driver for send_index+1!
    deferred_formal_ids = tuple(dict.fromkeys(
        mid for item in runnable_formal[1:]
        for mid in (item[0].ref.message_id, item[1].ref.message_id)
    ))
    deferred_set = set(deferred_formal_ids)
    if deferred_set:
        consumed_refs = tuple(
            ref for ref in ctx.batch.input_refs if ref.message_id not in deferred_set
        )
    else:
        consumed_refs = ctx.batch.input_refs

    need_prepare = (
        resume_opts_0 is not None and saved.get("definition_ref") != plain(resume_origin_0.ref)
    )
    if need_prepare:
        saved, cancelled_batch = await _prepare_explicit(
            worker, ctx, saved, resume_origin_0, resume_opts_0, requeue_ids=deferred_formal_ids,
        )
        if cancelled_batch is not None:
            return cancelled_batch
    elif deferred_formal_ids:
        if getattr(ctx, "ownership", None) is not None and ctx.ownership.lost:
            raise SessionLeaseLost(ctx.session_id)
        cancelled_batch = None
        undeliverables: list[tuple[Any, str, Any, str, str]] = []
        driver_events_out: list[tuple[str, str, dict[str, Any]]] = []
        async with port.db.transaction() as tx:
            state = await port.store._db.dialect.lock_state(tx, port.s, ctx.session_id)
            if (
                state["execution_state"] != "running"
                or int(state["send_index"]) != int(ctx.send_index)
                or int(state["owner_epoch"]) != int(ctx.batch.owner_epoch)
            ):
                raise SessionLeaseLost(ctx.session_id)
            if _driver_batch_cancelled_in_state(state, ctx):
                cancelled_batch, undeliverables, driver_events_out = (
                    await _finalize_cancelled_driver_batch_tx(port, tx, state, ctx)
                )
            else:
                await _requeue_deferred_inputs_tx(
                    port, tx, ctx.session_id, ctx.send_index, deferred_formal_ids,
                )
        if cancelled_batch is not None:
            await _flush_cancelled_driver_post_commit(port, undeliverables, driver_events_out)
            return cancelled_batch
    if deferred_formal_ids:
        # Deferred resumes are already back in status='pending' (tracked by wake_pending_count),
        # so drop them from ctx.batch.input_refs: if cancelled/voided mid-flight, port.final sees
        # wake_pending_count==0 and cleanly transitions Driver to 'idle'.
        ctx.batch = replace(ctx.batch, input_refs=consumed_refs)

    resident_waiter = port.register_driver_waiter(ctx.session_id)
    try:
        if ctx.batch.reclaimed_cancelled or ctx.batch.cancel_by is not None:
            ctx.cancel_by = ctx.batch.cancel_by or "user"
            ctx.stop.cancel("cancelled")

        result_dict, aborted_early = await _execute_driver_graph(
            worker, ctx, env, run_id, saved, spec, resident_waiter,
            is_recovery=(origin_0.row["kind"] == "workflow_continue"),
        )
        return await _complete_driver(
            worker, ctx, saved, origin_0, result_dict,
            drain_node_results=aborted_early,
            consumed=consumed_refs,
            extra_late_chains=tuple(plain_late_chains),
        )
    finally:
        port.unregister_driver_waiter(ctx.session_id)


async def resume_run(client, target, *, node_id, message, message_id, actor):
    """Start a new driver input and recompute only the selected dependency tail."""
    port = client.inputs
    session = await port._session(port.db, target.session_id)
    if session["session_kind"] != "workflow_driver":
        raise RuntimeRequestError("not_found")
    run_id = session["start_input_message_id"]
    saved = await journal.get(port.store, port.db, target.session_id, journal.run_key(run_id))
    if saved is None or not saved.get("aggregate_delivered"):
        raise RuntimeRequestError("target_not_resumable", "Use send for a running leaf.")
    descriptor = RuntimeDescriptor(**json.loads(session["descriptor_json"]))
    spec = WorkflowSpec.from_json(plain(descriptor.config["workflow"]["spec"]))
    from .continuation import plan_continuation

    plan_continuation(spec, node_id)
    if spec.budget:
        from .result import SharedBudget

        budget = SharedBudget(
            spec.budget.max_tokens, stop_at_remaining_pct=spec.budget.stop_at_remaining_pct,
        )
        budget.commit((saved.get("result") or {}).get("usage") or {})
        if not budget.can_spawn():
            raise RuntimeRequestError("budget_exceeded", "The cumulative workflow budget is exhausted.")
    if spec.timeout_s and now_ms() - saved["created_at_ms"] >= spec.timeout_s * 1000:
        raise RuntimeRequestError("workflow_timeout")
    receipt = await client.send(
        target,
        {"workflow_resume": {"node_id": node_id, "message": message}},
        message_id=message_id,
        actor=actor,
    )
    return WorkflowRunHandle(client, actor, target, receipt.input_ref)
