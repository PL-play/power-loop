"""Authenticated, ID-only submission, cancellation and result collection."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import replace

from power_loop.runtime.jobs import JobManager

from .runtime_store import RuntimeInputStore, now_ms
from .runtime_transport import SubscriptionSpec, TransportPacket
from .runtime_types import (
    CapacityError,
    Command,
    InputRef,
    JobPage,
    JobRef,
    OutputItem,
    Receipt,
    RequestOptions,
    Result,
    RuntimeRequestError,
    StoreError,
    TargetRef,
    WaitTimeout,
    plain,
)

log = logging.getLogger(__name__)


class RuntimeClient:
    def __init__(self, *, store, transport, namespace="power-loop", hooks=None, hook_timeout_s=1.0):
        if not namespace:
            raise ValueError("namespace must be nonempty")
        self.store, self.transport, self.namespace = store, transport, namespace
        self.hooks, self.hook_timeout_s = hooks, hook_timeout_s
        self.inputs = RuntimeInputStore(store)
        self._waiters, self._outputs = set(), set()
        self._closed = False
        self._slots = 0
        self._accepting = True

    def _require_open(self, *, reading=False):
        if self._closed or (not reading and not self._accepting):
            raise RuntimeRequestError("session_closed")

    async def send(self, target, body, *, message_id, actor, options=None, descriptor=None, deadline_ms=None):
        if target.job_id is not None:
            raise RuntimeRequestError("invalid_options", "send targets a session")
        return await self._submit(Command(message_id, target, "send", {"body": body}, "new_turn",
            options=plain(options or RequestOptions()), expires_at_ms=deadline_ms), actor=actor, descriptor=descriptor)

    async def append(self, target, body, *, message_id, actor, deadline_ms=None):
        return await self._submit_internal(Command(message_id, target, "append", {"body": body}, "append",
                                         expires_at_ms=deadline_ms), actor=actor)

    async def submit(self, target, body, *, message_id, actor, options=None, deadline_ms=None):
        """Smart dispatch: automatically chooses send (no active job) or append (active job).

        Host does not need to judge operation type manually.
        send and append semantics remain unchanged, submit just encapsulates the judgment logic.

        Note: target must be TargetRef (session level), not JobRef.
        """
        if target.job_id is not None:
            raise RuntimeRequestError("invalid_options", "submit targets a session")

        state = await self.inputs.db.fetchone(
            f"SELECT owner_token FROM {self.inputs.state} WHERE session_id=?",
            (target.session_id,))
        has_active = state is not None and state["owner_token"] is not None

        if has_active:
            return await self.append(target, body, message_id=message_id, actor=actor,
                                     deadline_ms=deadline_ms)
        else:
            return await self.send(target, body, message_id=message_id, actor=actor,
                                   options=options, deadline_ms=deadline_ms)

    async def _job_input(self, ref, operation, body, *, message_id, actor, deadline_ms=None):
        job = await self.inputs.resolve_job(self.inputs.db, ref, actor)
        if ref.generation is None:
            raise RuntimeRequestError("stale_generation")
        return await self._submit_internal(Command(message_id, TargetRef(ref.store_id, job["session_id"], ref.job_id),
            operation, {"body": body}, "control" if operation == "steer" else "append",
            expires_at_ms=deadline_ms, expected_job_generation=ref.generation), actor=actor)

    async def nudge(self, ref, body, *, message_id, actor, deadline_ms=None):
        return await self._job_input(ref, "nudge", body, message_id=message_id, actor=actor, deadline_ms=deadline_ms)

    async def steer(self, ref, body, *, message_id, actor, deadline_ms=None):
        """Deprecated: use priority() instead.

        High-priority message delivery. This is a message primitive without control side effects.
        """
        return await self._job_input(ref, "steer", body, message_id=message_id, actor=actor, deadline_ms=deadline_ms)

    async def answer(self, ref, *, interaction_id, wait_round, value, message_id, actor):
        """Internal API for answering wait_review interactions. Not exposed in STABLE_API."""
        self._require_open()
        from power_loop.runtime.interactions import answer
        return await answer(self, ref, interaction_id=interaction_id, wait_round=wait_round,
                            value=value, message_id=message_id, actor=actor)

    async def priority(self, target, body, *, message_id, actor, deadline_ms=None):
        """High-priority message: arrives faster than append, can interrupt rounds waiting for tool results.

        Pure message primitive, no control side effects. Does not stop any tasks or clear queues.
        If you need to stop or cancel, use control primitives stop() or cancel().

        Note: target can be TargetRef (session level) or JobRef (job level).
        """
        if target.job_id is not None:
            # JobRef: delegate to steer (job-level steering)
            return await self.steer(target, body, message_id=message_id, actor=actor, deadline_ms=deadline_ms)
        else:
            # TargetRef: need to get active job first
            state = await self.inputs.db.fetchone(
                f"SELECT owner_token FROM {self.inputs.state} WHERE session_id=?",
                (target.session_id,))
            if state is None or state["owner_token"] is None:
                raise RuntimeRequestError("no_active_job", "priority requires an active job")

            # Get the active job to create JobRef for steer
            job = await self.inputs.db.fetchone(
                f"SELECT job_id, generation FROM {self.inputs.jobs} "
                "WHERE session_id=? AND status IN ('running','settling','model_closed') "
                "ORDER BY created_at DESC LIMIT 1",
                (target.session_id,))
            if job is None:
                raise RuntimeRequestError("no_active_job", "priority requires an active job")

            from power_loop.agent.runtime_types import JobRef
            job_ref = JobRef(target.store_id, job["job_id"], job["generation"])
            return await self.steer(job_ref, body, message_id=message_id, actor=actor, deadline_ms=deadline_ms)

    async def resume(self, ref, *, message_id, actor, body=None, deadline_ms=None, options=None):
        if ref.generation is None or ref.generation == 0:
            raise RuntimeRequestError("stale_generation")
        job = await self.inputs.resolve_job(self.inputs.db, ref, actor)
        if job["kind"] not in {"subagent", "workflow_node"}:
            raise RuntimeRequestError("invalid_options", "resume requires a child task")
        import json
        creation = await self.inputs.db.fetchone(f"SELECT payload_json FROM {self.inputs.events} "
            "WHERE job_id=? AND action='create'", (ref.job_id,))
        start_ref = InputRef(**json.loads(creation["payload_json"])["request"]["input_ref"])
        start = await self.inputs.input(self.inputs.db, start_ref)
        original = start.meta["options"]
        narrowed = dict(original)
        if options is not None:
            if not isinstance(options, Mapping) or set(options) - {"tools", "max_rounds", "max_tokens", "max_tokens_per_run"}:
                raise RuntimeRequestError("invalid_options", "resume only narrows tools and execution limits")
            for key, value in options.items():
                if key == "tools":
                    if not isinstance(value, (list, tuple)) or any(not isinstance(name, str) for name in value):
                        raise RuntimeRequestError("invalid_options", "resume tools")
                    if original[key] is not None and set(value) - set(original[key]):
                        raise RuntimeRequestError("forbidden", "resume tools exceed the original grant")
                elif type(value) is not int or value < 1 or (original[key] is not None and value > original[key]):
                    raise RuntimeRequestError("invalid_options", "resume cannot increase " + key)
                narrowed[key] = value
        from .runtime_types import RuntimeDescriptor
        return await self._submit(Command(message_id, TargetRef(ref.store_id, job["session_id"], ref.job_id),
            "resume", {"body": body or "Continue the original task using saved results. Do not replay unknown external actions."},
            "new_turn", options=narrowed, expires_at_ms=deadline_ms,
            expected_job_generation=ref.generation), actor=actor,
            descriptor=RuntimeDescriptor(**start.meta["descriptor"]))

    async def stop(self, job_id, *, actor, reason="Stopped by host", expected_generation=None):
        """Stop task: request persistent cancellation of job and its supervised subtree.

        Note: stop is a request, not immediate physical stop. acceptance != physical stop.
        Can target main agent's job or subagent's job.
        """
        from power_loop.runtime import task_control
        manager = self.inputs.store._task_manager
        return await task_control.request_cancel(
            manager, actor, job_id,
            reason=reason, expected_generation=expected_generation,
        )

    async def cancel(self, job_id, *, actor, reason="Cancelled by host", clear_queue=False):
        """Cancel task, optionally clearing pending messages in input_queue.

        clear_queue=True: set status='void' for status='pending' AND mode='queue' messages
        in the session, preventing stopped task from processing queued messages.

        Use case: user clicks "stop" button and wants to completely stop, not process backlog.
        """
        result = await self.stop(job_id, actor=actor, reason=reason)
        if clear_queue:
            job = await self.inputs.db.fetchone(
                f"SELECT session_id FROM {self.inputs.jobs} WHERE job_id=?",
                (job_id,))
            if job:
                await self.inputs.db.execute(
                    f"UPDATE {self.inputs.q} SET status='void' "
                    "WHERE session_id=? AND status='pending' AND mode='queue'",
                    (job["session_id"],))
        return result

    async def _query_session_core(self, session_id):
        """Internal status query, shared by jobs tool and host interface."""
        state = await self.inputs.db.fetchone(
            f"SELECT lifecycle, owner_token, next_seq FROM {self.inputs.state} "
            "WHERE session_id=?", (session_id,))
        if state is None:
            raise RuntimeRequestError("not_found")
        active_job = await self.inputs.db.fetchone(
            f"SELECT job_id, kind, status FROM {self.inputs.jobs} "
            "WHERE session_id=? AND status IN ('running','settling','model_closed') "
            "ORDER BY created_at DESC LIMIT 1", (session_id,))
        pending_count = await self.inputs.db.fetchone(
            f"SELECT COUNT(*) AS n FROM {self.inputs.q} "
            "WHERE session_id=? AND status='pending'", (session_id,))
        return {
            "lifecycle": state["lifecycle"],
            "has_active_job": active_job is not None,
            "active_job_id": active_job["job_id"] if active_job else None,
            "active_job_kind": active_job["kind"] if active_job else None,
            "pending_count": int(pending_count["n"]) if pending_count else 0,
        }

    async def get_session_status(self, session_id, *, actor):
        """Get session status. Host no longer needs raw SQL queries.

        Returns:
          lifecycle: open / closing / closed
          has_active_job: bool
          active_job_id: str | None
          active_job_kind: send / subagent / workflow_node / ...
          pending_count: int  # status='pending' in input_queue
        """
        await self.inputs.authorize(self.inputs.db, TargetRef(self.store.store_id, session_id), actor)
        return await self._query_session_core(session_id)

    async def get_job_status(self, job_id, *, actor):
        """Get job status.

        Returns:
          job_id, kind, status, outcome, generation
          cancel_requested: bool
        """
        from power_loop.agent.runtime_types import JobRef
        job = await self.inputs.resolve_job(self.inputs.db, JobRef(self.store.store_id, job_id, None), actor)
        return {
            "job_id": job["job_id"],
            "kind": job["kind"],
            "status": job["status"],
            "outcome": job["outcome"],
            "generation": job["generation"],
            "cancel_requested": job["cancel_requested_at"] is not None,
        }

    async def list_jobs(self, session_id, *, actor, status="active", kinds=None, limit=20):
        """List jobs under the session.

        status: active (running/settling/model_closed) / terminal (completed/cancelled) / all
        kinds: filter by job types (send/subagent/workflow_node/...)
        """
        from power_loop.runtime import task_query
        manager = self.inputs.store._task_manager
        return await task_query.list_jobs(
            manager, self.inputs._actor(actor, session_id),
            status=status, kinds=kinds, limit=limit,
        )

    async def workflow(self, target, spec, *, message_id, actor, descriptor=None, options=None):
        from power_loop.workflow.runner import submit
        return await submit(self, target, spec, message_id=message_id, actor=actor,
                            descriptor=descriptor, options=options)

    async def _submit_internal(self, command, *, actor):
        if command.operation in {"answer", "resume"}:
            raise RuntimeRequestError("invalid_options", command.operation + " requires its dedicated control port")
        if command.operation == "cancel":
            if (set(command.payload) - {"input_ref", "reason"} or "reason" not in command.payload
                    or not isinstance(command.payload["reason"], str) or command.options
                    or command.descriptor_ref is not None or command.expires_at_ms is not None):
                raise RuntimeRequestError("invalid_options", "cancel payload")
            if "input_ref" in command.payload:
                value = command.payload["input_ref"]
                if (not isinstance(value, Mapping) or set(value) != {"store_id", "session_id", "message_id"}
                        or command.target.job_id is not None or command.expected_job_generation is not None):
                    raise RuntimeRequestError("invalid_options", "cancel input reference")
                ref = InputRef(**value)
                if (ref.store_id, ref.session_id) != (command.target.store_id, command.target.session_id):
                    raise RuntimeRequestError("invalid_options", "cancel target does not match input")
            else:
                if command.target.job_id is None or command.expected_job_generation is None:
                    raise RuntimeRequestError("invalid_options", "cancel requires an exact job generation")
                ref = JobRef(command.target.store_id, command.target.job_id, command.expected_job_generation)
            return await self._cancel(ref, actor=actor, scope=command.scope,
                                      reason=command.payload["reason"], target=command.target)
        return await self._submit(command, actor=actor)

    async def _submit(self, command, *, actor, descriptor=None):
        self._require_open()
        if command.operation not in {"send", "append", "nudge", "steer", "resume"}:
            raise RuntimeRequestError("invalid_options", "operation requires its dedicated control port")
        if set(command.payload) != {"body"} or command.descriptor_ref is not None:
            raise RuntimeRequestError("invalid_options")
        body = command.payload["body"]
        if not isinstance(body, str) and (not isinstance(body, Mapping) or "content" not in body):
            raise RuntimeRequestError("invalid_options", "body must be text or a message")
        if command.scope != "self":
            raise RuntimeRequestError("invalid_options", "ordinary inputs have self scope")
        if command.operation not in {"send", "resume"} and command.options:
            raise RuntimeRequestError("invalid_options", "append does not accept execution overrides")
        try:
            async with self.inputs.db.transaction() as tx:
                receipt = await self.inputs.accept_input(tx, command, actor, descriptor=descriptor)
        except RuntimeRequestError:
            raise
        except Exception as exc:
            raise StoreError(InputRef(command.target.store_id, command.target.session_id,
                                      command.message_id)) from exc
        if receipt.newly_accepted:
            from power_loop.contracts.hooks import HookPoint
            from power_loop.core.execution_notifications import notify
            await notify(self.hooks, HookPoint.MESSAGE_ACCEPTED, {"receipt": receipt}, timeout_s=self.hook_timeout_s)
        if receipt.state not in {"accepted"}:
            return receipt
        channels = ["work", "activation"]
        if command.operation == "steer":
            channels.insert(0, "control")
        published = await self._notify(receipt.target, receipt.message_id, receipt.input_ref, channels)
        return replace(receipt, notification="published" if published else "failed",
                       reason=None if published else "wake_pending")

    async def _notify(self, target, message_id, input_ref, channels, payload=None):
        success = True
        for channel in channels:
            try:
                sent = await self.transport.publish(TransportPacket(42, self.namespace, message_id,
                    target, channel, input_ref, payload))
                success = sent.accepted and success
            except Exception:
                log.warning("input saved; %s notification failed for %s", channel, message_id, exc_info=True)
                success = False
        return success

    async def cancel(self, ref, *, actor, scope="self", reason="cancelled"):
        return await self._cancel(ref, actor=actor, scope=scope, reason=reason)

    async def _cancel(self, ref, *, actor, scope, reason, target=None):
        self._require_open()
        if scope not in {"self", "subtree"}:
            raise RuntimeRequestError("invalid_options")
        manager = JobManager(self.store)
        input_ref = ref if isinstance(ref, InputRef) else None
        job_ref = ref if isinstance(ref, JobRef) else None
        if target is None:
            if input_ref:
                target = TargetRef(ref.store_id, ref.session_id)
            else:
                row = await self.inputs.resolve_job(self.inputs.db, ref, actor)
                target = TargetRef(ref.store_id, row["session_id"], ref.job_id)
        parent_ref = None
        async with manager.joined_write() as tx:
            await self.inputs.authorize(tx, target, actor)
            result_route = None
            if input_ref:
                original = await self.inputs.input(tx, input_ref)
                if original is None:
                    raise RuntimeRequestError("not_found")
                if original.row["delivered_job_id"] is not None:
                    job_ref = JobRef(ref.store_id, original.row["delivered_job_id"],
                                     original.row["delivered_job_generation"])
            if job_ref is not None:
                result_route = await self.inputs._lock_result_sessions(tx, job_ref)
            else:
                await self.inputs.db.dialect.lock_state(tx, self.inputs.state, target.session_id)
            if input_ref:
                record = await self.inputs.input(tx, input_ref)
                if record is None:
                    raise RuntimeRequestError("not_found")
                if record.row["delivered_job_id"] is None:
                    receipt = await self.inputs.void_unbound(tx, input_ref, reason=reason, status="cancelled")
                else:
                    job_ref = JobRef(ref.store_id, record.row["delivered_job_id"], record.row["delivered_job_generation"])
                if record.meta["operation"] in {"append", "nudge", "steer"} and record.row["status"] == "delivered":
                    return self.inputs.receipt(record, reason="already_applied")
            if job_ref is not None:
                row = await manager._row(tx, job_ref.job_id)
                if job_ref.generation is None or row["generation"] != job_ref.generation:
                    raise RuntimeRequestError("stale_generation")
                reason = row["cancel_reason"] or reason
                if row["cancel_requested_at"] is None and row["supervision_status"] != "closed":
                    await tx.write_job(job_ref.job_id,
                        f"UPDATE {manager.jobs} SET cancel_requested_at=?,cancel_reason=?,intake='closed',version=version+1 "
                        "WHERE job_id=? AND generation=? AND version=?",
                        (now_ms(), reason, job_ref.job_id, job_ref.generation, row["version"]))
                    if job_ref.generation == 0:
                        await tx.write_job(job_ref.job_id,
                            f"UPDATE {manager.jobs} SET status='cancelled',outcome='cancelled',resource_state='stopped',"
                            "supervision_status=?,result_status='available',version=version+1 WHERE job_id=? AND generation=0",
                            ("open" if row["kind"] in {"subagent", "background_tool", "background_command"} and result_route["parent_session_id"] else "closed", job_ref.job_id))
                        await tx.execute(f"UPDATE {self.inputs.q} SET status='void',claim_token=NULL,claimed_at=NULL,claim_owner_epoch=NULL "
                            "WHERE session_id=? AND delivered_job_id=? AND delivered_job_generation=0 AND status IN ('pending','claimed')",
                            (target.session_id, job_ref.job_id))
                        parent_ref = await self.inputs._commit_result(tx, job_ref,
                            Result(job_ref, input_ref, "cancelled", error=reason), None, result_route)
                if row["status"] == "waiting" and row["outcome"] is None:
                    import json
                    state = await tx.fetchone(f"SELECT pending_json,next_seq FROM {self.inputs.state} WHERE session_id=?", (target.session_id,))
                    pending = json.loads(state["pending_json"] or "{}")
                    seq = state["next_seq"]
                    for call in pending.get("tool_calls", []):
                        await tx.execute(f"INSERT INTO {self.store.t.messages} "
                            "(session_id,seq,role,tool_call_id,name,content,state,round_index,send_index,created_at) "
                            "VALUES (?,?,'tool',?,?,?,'active',?,?,?)", (target.session_id, seq, call["id"],
                            call["function"]["name"], "<cancelled>", pending["round_index"],
                            pending["runtime_wait"]["send_index"], now_ms()))
                        seq += 1
                    await tx.execute(f"UPDATE {self.inputs.state} SET pending_json=NULL,next_seq=?,runtime_version=runtime_version+1 WHERE session_id=?",
                                     (seq, target.session_id))
                    await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET status='cancelled',outcome='cancelled',"
                        "intake='closed',resource_state='stopped',version=version+1 WHERE job_id=? AND generation=? AND outcome IS NULL",
                        (row["job_id"], job_ref.generation), obligations=True)
                    parent_ref = await self.inputs._commit_result(tx, job_ref, Result(job_ref, input_ref, "cancelled", error=reason), None, result_route)
                receipt = Receipt(input_ref, target, input_ref.message_id if input_ref else
                    f"cancel:{job_ref.job_id}:{job_ref.generation}", False, job_ref, "applied", "not_needed", reason)
        if job_ref is None:
            success = await self._notify(target, receipt.message_id, input_ref, ['output'], {'kind': 'completed'})
            return replace(receipt, notification='published' if success else 'failed')
        success = await self._notify(TargetRef(target.store_id, target.session_id, job_ref.job_id),
            receipt.message_id, input_ref, ["control"], {"generation": job_ref.generation, "cancelled": True})
        if parent_ref:
            success = await self._notify(TargetRef(parent_ref.store_id, parent_ref.session_id),
                parent_ref.message_id, parent_ref, ["work", "activation"]) and success
        if scope == "subtree":
            cursor = ""
            while True:
                rows = await self.inputs.db.fetchall(
                    f"WITH RECURSIVE descendants(job_id) AS (SELECT job_id FROM {self.inputs.jobs} WHERE parent_job_id=? "
                    f"UNION ALL SELECT j.job_id FROM {self.inputs.jobs} j JOIN descendants d ON j.parent_job_id=d.job_id) "
                    f"SELECT j.job_id,j.generation FROM {self.inputs.jobs} j JOIN descendants d ON j.job_id=d.job_id "
                    "WHERE j.job_id>? AND j.supervision_status<>'closed' ORDER BY j.job_id LIMIT 100", (job_ref.job_id, cursor))
                for row in rows:
                    child = await self.cancel(JobRef(job_ref.store_id, row["job_id"], row["generation"]),
                                              actor=actor, reason=reason)
                    success = child.notification != "failed" and success
                if len(rows) < 100:
                    break
                cursor = rows[-1]["job_id"]
        return replace(receipt, notification="published" if success else "failed")

    async def inspect(self, ref, *, actor):
        self._require_open(reading=True)
        return (await self.inputs.read(ref, actor))[0]

    async def list_jobs(self, target, *, actor, scope="self", unfinished=True, cursor=None, limit=100):
        self._require_open(reading=True)
        if scope not in {"self", "subtree"} or not 1 <= limit <= 1000:
            raise RuntimeRequestError("invalid_options")
        await self.inputs.authorize(self.inputs.db, target, actor)
        params = [target.session_id]
        prefix = ""
        condition = "session_id=?"
        if scope == "subtree":
            prefix = (f"WITH RECURSIVE descendants(session_id) AS (SELECT session_id FROM {self.store.t.sessions} "
                      f"WHERE session_id=? UNION ALL SELECT s.session_id FROM {self.store.t.sessions} s "
                      "JOIN descendants d ON s.parent_session_id=d.session_id) ")
            condition = "session_id IN (SELECT session_id FROM descendants)"
        if target.job_id:
            if scope == "subtree":
                prefix = (f"WITH RECURSIVE descendants(job_id) AS (SELECT job_id FROM {self.inputs.jobs} WHERE job_id=? "
                          f"UNION ALL SELECT j.job_id FROM {self.inputs.jobs} j JOIN descendants d ON j.parent_job_id=d.job_id) ")
                condition = "job_id IN (SELECT job_id FROM descendants)"
                params = []
            else:
                condition += " AND job_id=?"
            params.append(target.job_id)
        if unfinished:
            condition += " AND supervision_status<>'closed'"
        rows = await self.inputs.db.fetchall(prefix + f"SELECT job_id FROM {self.inputs.jobs} WHERE {condition} "
            "AND job_id>? ORDER BY job_id LIMIT ?", (*params, cursor or "", limit))
        items = tuple([await self.inspect(JobRef(target.store_id, row["job_id"]), actor=actor) for row in rows])
        return JobPage(items, rows[-1]["job_id"] if len(rows) == limit else None)

    async def wait(self, ref, *, actor, deadline_ms):
        from power_loop.core.agent_context import get_current_loop
        loop = get_current_loop()
        worker = getattr(loop, "_runtime_worker", None)
        if worker is not None and worker.runtime.client is self:
            async with worker.idle(loop._runtime_worker_context):
                return await self._wait(ref, actor=actor, deadline_ms=deadline_ms)
        return await self._wait(ref, actor=actor, deadline_ms=deadline_ms)

    async def _wait(self, ref, *, actor, deadline_ms):
        self._require_open(reading=True)
        if self._slots >= 1024:
            raise CapacityError()
        self._slots += 1
        task = asyncio.current_task()
        self._waiters.add(task)
        original = ref
        subscription = None
        try:
            # Subscribe first, then inspect. A completion in either window is visible.
            if isinstance(ref, InputRef):
                target = TargetRef(ref.store_id, ref.session_id)
                await self.inputs.authorize(self.inputs.db, target, actor)
                subscription = await self.transport.subscribe(SubscriptionSpec(
                    self.namespace, ref.store_id, "output", "client", ref.session_id))
                await subscription.ready()
            while True:
                _, result, selected = await self.inputs.read(ref, actor)
                if result:
                    return replace(result, input_ref=original if isinstance(original, InputRef) else None)
                ref = selected
                remaining = 30.0 if deadline_ms is None else (deadline_ms - now_ms()) / 1000
                if remaining <= 0:
                    raise WaitTimeout(str(original))
                if subscription is None:
                    await asyncio.sleep(min(0.05, remaining))
                else:
                    try:
                        delivery = await asyncio.wait_for(anext(subscription), min(30.0, remaining))
                        await delivery.ack()
                    except TimeoutError:
                        # A lost notification is not a lost durable result.
                        # Recheck once before the deadline decision above.
                        continue
        finally:
            self._slots -= 1
            self._waiters.discard(task)
            if subscription:
                await subscription.aclose()

    async def subscribe_output(self, ref, *, actor, before_send=False):
        self._require_open(reading=True)
        if self._slots >= 1024:
            raise CapacityError()
        self._slots += 1
        subscription = None
        try:
            if before_send:
                # The trusted ingress may subscribe to its imminent request after
                # authorizing the session. A guessed ID alone grants nothing.
                if not isinstance(ref, InputRef) or actor.session_id is not None:
                    raise RuntimeRequestError("forbidden")
                await self.inputs.authorize(self.inputs.db, TargetRef(ref.store_id, ref.session_id), actor)
                if await self.inputs.input(self.inputs.db, ref) is not None:
                    await self.inputs.read(ref, actor)
            else:
                await self.inputs.read(ref, actor)
            self._require_open(reading=True)
            subscription = await self.transport.subscribe(SubscriptionSpec(
                self.namespace, ref.store_id, "output", "client", ref.session_id))
            await subscription.ready()
            self._require_open(reading=True)
            output = OutputSubscription(self, ref, actor, subscription)
            self._outputs.add(output)
            return output
        except BaseException:
            self._slots -= 1
            if subscription:
                await subscription.aclose()
            raise

    async def aclose(self):
        self._closed = True
        for task in tuple(self._waiters):
            task.cancel("client closed")
        if self._waiters:
            await asyncio.gather(*tuple(self._waiters), return_exceptions=True)
        for output in tuple(self._outputs):
            await output.aclose()


class OutputSubscription:
    def __init__(self, client, ref, actor, subscription):
        self.client, self.ref, self.actor, self.subscription = client, ref, actor, subscription
        self.sequence = 0
        self.closed = False
        self.initial = True
        self.job_ref = None

    async def ready(self):
        await self.subscription.ready()

    def __aiter__(self):
        return self

    async def __anext__(self):
        while not self.closed:
            payload = None
            if not self.initial:
                try:
                    delivery = await anext(self.subscription)
                except BaseException:
                    await self.aclose()
                    raise
                await delivery.ack()
                if (isinstance(self.ref, InputRef) and delivery.packet.input_ref is not None
                        and delivery.packet.input_ref != self.ref):
                    continue
                payload = delivery.packet.payload or {}
                if payload.get("kind") == "delta":
                    if self.job_ref is None:
                        status = await self.client.inspect(self.ref, actor=self.actor)
                        self.job_ref = status.job_ref
                    if plain(self.job_ref) != plain(payload.get("job_ref")):
                        continue
                    self.sequence += 1
                    return OutputItem(self.ref, self.job_ref, self.sequence, "delta", plain(payload))
            status, result, _ = await self.client.inputs.read(self.ref, self.actor)
            self.job_ref = status.job_ref
            if result:
                self.sequence += 1
                await self.aclose()
                return OutputItem(self.ref, result.job_ref, self.sequence, "completed", plain(result))
            if self.initial or payload.get("kind") in {"gap", "error"}:
                self.initial = False
                self.sequence += 1
                return OutputItem(self.ref, status.job_ref, self.sequence, payload.get("kind", "gap") if payload else "gap", plain(payload or {}))
        raise StopAsyncIteration

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        self.client._slots -= 1
        self.client._outputs.discard(self)
        await self.subscription.aclose()
