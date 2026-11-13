"""Session-native workflow admission, execution and recovery.

The driver is a non-LLM session. Leaves are ordinary agent sessions created with
one atomic session+start transaction; their journal entries commit afterwards.
A replacement driver replays the frozen graph from those entries and ordinary
node result inputs, never from a Python call stack or a job generation.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

from power_loop.agent.runtime_store import FinalBatch, now_ms
from power_loop.agent.runtime_types import (
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
from .engine import WorkflowEngine
from .spec import WorkflowSpec


async def submit(client, target, spec, *, message_id, actor, descriptor=None, options=None, parent=None):
    """Create only the independent driver and its frozen start input."""
    del parent  # Parent authority is the direct caller session, never a job lease.
    client._require_open()
    spec = WorkflowSpec.from_json(spec.to_dict() if isinstance(spec, WorkflowSpec) else spec)
    port = client.inputs
    caller = await port._session(port.db, target.session_id)
    supplied_descriptor, supplied_options = descriptor, options
    if descriptor is None:
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
    existing = await port.db.fetchone(
        f"SELECT descriptor_json FROM {port.store.t.sessions} WHERE session_id=?", (sid,),
    )
    if existing is not None and supplied_descriptor is None and supplied_options is None:
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
        run_journal=saved if existing is None else None,
    )
    await client._notify(driver, start.message_id, start, ["work", "activation"])
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
                self.run_id, result.status, errors=[result.error] if result.error else [],
            )
        data["results"] = {
            key: AgentResult(**value) for key, value in data.get("results", {}).items()
        }
        data["final"] = AgentResult(**data["final"]) if data.get("final") else None
        return WorkflowResult(**data)

    async def resume(self, *, node_id, message, message_id):
        return await resume_run(
            self.client, self.target, node_id=node_id, message=message,
            message_id=message_id, actor=self.actor,
        )

    async def cancel(self, reason="cancelled"):
        return await self.client.cancel(self.input_ref, actor=self.actor, reason=reason)


class DurableExecutor:
    """Narrow Executor adapter over the unified session Client."""

    def __init__(self, worker, ctx, env, run_id, saved):
        self.worker, self.ctx, self.env = worker, ctx, env
        self.client = worker.runtime.client
        self.port = worker.inputs
        self.run_id, self.saved = run_id, saved
        self.plan = saved.get("continuation") or {}
        self.driver = TargetRef(ctx.store_id, ctx.session_id)
        self.actor = TrustedActor(ctx.actor.tenant_id, ctx.actor.principal_id, ctx.session_id)

    def _coordinates(self, request):
        offset = self.plan.get("attempt_offsets", {}).get(request.node_path, 0)
        if offset:
            request = replace(request, attempt=request.attempt + offset)
        body = request.body
        if request.continuation == 0:
            body = self.plan.get("body_overrides", {}).get(request.node_path, request.body)
        if body != request.body:
            request = replace(request, body=body)
        request_hash = sha256_canonical(plain(request))
        sid = request.session_id or "ses_" + sha256_canonical([
            "workflow_node", self.driver.store_id, self.driver.session_id, self.run_id,
            request.node_path, request.iteration_path,
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
        return request, target, start, key, request_hash

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
        state = await self.port.store._db.dialect.lock_state(tx, self.port.state, self.ctx.session_id)
        batch = self.ctx.batch
        if (
            state["execution_state"] != "running"
            or state["active_send_index"] != batch.send_index
            or state["owner_epoch"] != batch.owner_epoch
            or state["owner_token"] != batch.owner_token
        ):
            raise SessionLeaseLost(self.ctx.session_id)
        return state

    async def run_agent(self, request, *, allow_start=True):
        request, target, start, key, request_hash = self._coordinates(request)
        saved = await journal.get(self.port.store, self.port.db, self.ctx.session_id, key)
        if saved is None:
            if not allow_start:
                return {"status": "budget_exceeded", "usage": {}, "session_id": target.session_id}
            descriptor = self._descriptor(request)
            if request.session_id is None and request.attempt == 1 and request.continuation == 0:
                await self.port.create_workflow_session(
                    descriptor,
                    parent=self.driver,
                    actor=self.actor,
                    target=target,
                    start=start,
                    body=request.body,
                    result_kind="workflow_node_result",
                    cancel_propagation="linked",
                )
                await self.client._notify(target, start.message_id, start, ["work", "activation"])
            else:
                await self.client.send(
                    target, request.body, message_id=start.message_id, actor=self.actor,
                )
            # Fixed order: session+start committed above, journal committed below.
            async with self.port.db.transaction() as tx:
                await self._check_owner(tx)
                saved = await journal.get(self.port.store, tx, self.ctx.session_id, key)
                if saved is None:
                    saved = {
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
                    await journal.put(self.port.store, tx, self.ctx.session_id, key, saved)
        elif saved["request_hash"] != request_hash:
            raise RuntimeRequestError("request_conflict", "workflow dispatch input changed")

        if saved.get("result") is None:
            try:
                result = await self.client.wait(start, actor=self.actor, deadline_ms=None)
            except asyncio.CancelledError:
                # Graph halt and explicit cancellation stop outstanding leaves. Driver
                # shutdown/ownership replacement leaves healthy independent sessions alone.
                if (
                    not self.ctx.shutdown_interrupted
                    and not self.ctx.ownership.lost
                    and self.ctx.stop.reason != "worker shutdown"
                ):
                    await self.client.cancel(
                        target, actor=self.actor, scope="subtree", reason="workflow branch stopped",
                    )
                raise
            await self._consume_results()
            saved = await journal.get(self.port.store, self.port.db, self.ctx.session_id, key)
            if saved.get("result") is None:
                # A full driver inbox can be drained and the ordinary result re-reported
                # without re-executing the node. Unknown external actions are not replayed.
                await self.port.accept_result(target, result, result_kind="workflow_node_result")
                await self._consume_results()
                saved = await journal.get(self.port.store, self.port.db, self.ctx.session_id, key)
            if saved.get("result") is None:
                raise RuntimeRequestError("result_delivery_failed", start.message_id)

        raw = dict(saved["result"])
        next_key = journal.node_key(
            self.driver, self.run_id, request.node_path, request.iteration_path,
            request.attempt, request.continuation + 1,
        )
        following = await journal.get(self.port.store, self.port.db, self.ctx.session_id, next_key)
        replayed = bool(self.plan) and int(saved.get("revision", 0)) < int(self.plan.get("revision", 0))
        return {
            **raw,
            "session_id": target.session_id,
            "has_next_continuation": following is not None,
            "accounting_usage": {} if replayed else raw.get("usage") or {},
        }

    async def _consume_results(self):
        """Consume every ready node result and journal it in one fenced transaction."""
        async with self.port.db.transaction() as tx:
            await self._check_owner(tx)
            rows = await tx.fetchall(
                f"SELECT * FROM {self.port.q} WHERE session_id=? "
                "AND kind='workflow_node_result' AND (status='pending' "
                "OR (status='delivered' AND delivered_send_index=? AND terminal_status IS NULL)) "
                "ORDER BY id LIMIT 1000",
                (self.ctx.session_id, self.ctx.send_index),
            )
            if not rows:
                return 0
            entries = await journal.nodes(self.port.store, tx, self.ctx.session_id)
            by_input = {canonical(entry["input_ref"]): (key, entry) for entry, key in zip(
                entries,
                [journal.node_key(
                    self.driver, self.run_id, entry["path"], entry["iteration"],
                    entry["attempt"], entry["continuation_index"],
                ) for entry in entries], strict=True,
            )}
            run = await journal.get(
                self.port.store, tx, self.ctx.session_id, journal.run_key(self.run_id),
            )
            consumed = 0
            database_now = now_ms()
            for row in rows:
                body = json.loads(row["content_json"])
                source = InputRef(**body["input_ref"])
                found = by_input.get(canonical(source))
                if found is None:
                    # The node may have committed before its journal. Recovery aligns
                    # that journal first; only then can this result advance the graph.
                    continue
                key, entry = found
                raw = dict(body)
                source_record = await self.port.input(tx, source)
                if source_record is None or source_record.row["terminal_status"] is None:
                    raise RuntimeRequestError("workflow_result_source_not_terminal")
                raw["send_index"] = source_record.row["result_send_index"]
                if raw.get("error_code") in {"hit_round_limit", "context_checkpoint"}:
                    raw["status"] = raw["error_code"]
                elif raw.get("error_code") in {"unknown", "store_commit_unknown"}:
                    raw["status"] = "unknown"
                if run.get("aggregate_delivered") or entry.get("superseded") or entry.get("result") is not None:
                    if not any(item.get("input_ref") == body["input_ref"] for item in entry["late_results"]):
                        entry["late_results"].append(raw)
                else:
                    entry["result"] = raw
                    entry["result_seq"] = body.get("result_seq")
                    entry["result_input_ref"] = plain(
                        InputRef(self.ctx.store_id, self.ctx.session_id, row["item_id"])
                    )
                await journal.put(self.port.store, tx, self.ctx.session_id, key, entry)
                changed = await tx.execute(
                    f"UPDATE {self.port.q} SET status='delivered',delivered_at=?,"
                    "delivered_seq=NULL,delivered_send_index=?,terminal_status='completed',"
                    "result_send_index=?,completed_at=? WHERE session_id=? AND item_id=? "
                    "AND (status='pending' OR (status='delivered' AND delivered_send_index=? "
                    "AND terminal_status IS NULL))",
                    (
                        database_now, self.ctx.send_index, self.ctx.send_index, database_now,
                        self.ctx.session_id, row["item_id"], self.ctx.send_index,
                    ),
                )
                if changed != 1:
                    raise SessionLeaseLost(self.ctx.session_id)
                consumed += 1
            return consumed

    async def todo_remaining(self, sid):
        state = await self.port.store.get_runtime_state(sid, "todo") or {}
        return [item["text"] for item in state.get("items", []) if item["status"] != "completed"]


async def _prepare_explicit(worker, ctx, saved, origin, options):
    from .continuation import plan_continuation

    port = worker.inputs
    spec = WorkflowSpec.from_json(plain(ctx.descriptor.config["workflow"]["spec"]))
    plan = plan_continuation(spec, options["node_id"])
    entries = await journal.nodes(port.store, port.db, ctx.session_id)
    offsets = {}
    for entry in entries:
        if entry["path"] in plan.invalidated:
            offsets[entry["path"]] = max(offsets.get(entry["path"], 0), int(entry["attempt"]))
    revision = int(saved.get("execution_revision", 0)) + 1
    continuation = {
        "attempt_offsets": offsets,
        "body_overrides": {options["node_id"]: options["message"]},
        "prior_usage": (saved.get("result") or {}).get("usage") or {},
        "invalidated": sorted(plan.invalidated),
        "revision": revision,
    }
    async with port.db.transaction() as tx:
        state = await port.store._db.dialect.lock_state(tx, port.state, ctx.session_id)
        if (
            state["execution_state"] != "running"
            or state["active_send_index"] != ctx.send_index
            or state["owner_epoch"] != ctx.batch.owner_epoch
            or state["owner_token"] != ctx.batch.owner_token
        ):
            raise SessionLeaseLost(ctx.session_id)
        for entry in entries:
            if entry["path"] in plan.invalidated:
                entry["superseded"] = True
                key = journal.node_key(
                    TargetRef(ctx.store_id, ctx.session_id), saved["run_id"],
                    entry["path"], entry["iteration"], entry["attempt"], entry["continuation_index"],
                )
                await journal.put(port.store, tx, ctx.session_id, key, entry)
        saved.update(
            continuation=continuation,
            execution_revision=revision,
            definition_ref=plain(origin.ref),
            status="running",
            aggregate_delivered=False,
        )
        await journal.put(port.store, tx, ctx.session_id, journal.run_key(saved["run_id"]), saved)
    return saved


async def _complete_driver(worker, ctx, saved, origin, aggregate):
    """Bind/finalize the driver chain and route one aggregate in the same transaction."""
    port = worker.inputs
    target = TargetRef(ctx.store_id, ctx.session_id)
    async with port.db.transaction() as tx:
        session = await port._session(tx, ctx.session_id)
        route = json.loads(session["result_route_json"]) if session["result_route_json"] else None
        lock_ids = {ctx.session_id}
        if route is not None:
            lock_ids.add(route["target"]["session_id"])
        for sid in sorted(lock_ids):
            state = await port.store._db.dialect.lock_state(tx, port.state, sid)
            if sid == ctx.session_id:
                own_state = state
        if (
            own_state["execution_state"] != "running"
            or own_state["active_send_index"] != ctx.send_index
            or own_state["owner_epoch"] != ctx.batch.owner_epoch
            or own_state["owner_token"] != ctx.batch.owner_token
        ):
            raise SessionLeaseLost(ctx.session_id)
        root = await port._continuation_root(tx, origin)
        status = aggregate["status"]
        if status == "budget_exceeded":
            status = "failed"
        if own_state["cancel_send_index"] == ctx.send_index:
            status = "cancelled"
            aggregate = {**aggregate, "status": "cancelled"}
        error = None if status == "completed" else (
            own_state["cancel_reason"] if status == "cancelled" else status
        )
        database_now = now_ms()
        result_seq = int(own_state["next_seq"])
        final_text = (aggregate.get("final") or {}).get("text") or ""
        result_meta = {
            "usage": aggregate.get("usage") or {},
            "structured_output": aggregate,
            "error": error,
            "error_code": None if status == "completed" else status,
        }
        await port._insert_message(
            tx, ctx.session_id, result_seq, "assistant", final_text,
            result_meta, ctx.send_index, database_now,
        )
        chain = [origin]
        current = origin
        while current.row["continuation_of"] is not None:
            current = await port.input(tx, InputRef(
                current.ref.store_id, current.ref.session_id, current.row["continuation_of"],
            ))
            chain.append(current)
        result_ref = delivery_error = None
        activations = []
        if route is not None:
            result_ref, delivery_error = await port._route_result_tx(
                tx, session, route,
                Result(
                    root.ref, target, status, ctx.send_index, result_seq, final_text,
                    aggregate.get("usage") or {}, aggregate, error,
                    None if status == "completed" else status,
                ),
                route["result_kind"], database_now, activations,
            )
        ids = [record.ref.message_id for record in chain]
        await tx.execute(
            f"UPDATE {port.q} SET status='delivered',terminal_status=?,result_seq=?,"
            "result_send_index=?,error_code=?,error_detail=?,result_delivery_error=?,completed_at=? "
            f"WHERE session_id=? AND item_id IN ({','.join('?' for _ in ids)}) "
            "AND terminal_status IS NULL",
            (
                status, result_seq, ctx.send_index, None if status == "completed" else status,
                error, delivery_error, database_now, ctx.session_id, *ids,
            ),
        )
        other = [ref for ref in ctx.batch.input_refs if ref not in {record.ref for record in chain}]
        if other:
            await tx.execute(
                f"UPDATE {port.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
                f"delivered_send_index=NULL WHERE session_id=? AND status='delivered' "
                "AND delivered_send_index=? AND terminal_status IS NULL "
                f"AND item_id IN ({','.join('?' for _ in other)})",
                (ctx.session_id, ctx.send_index, *[ref.message_id for ref in other]),
            )
        pending = await tx.fetchone(
            f"SELECT COUNT(*) AS n FROM {port.q} WHERE session_id=? AND status='pending'",
            (ctx.session_id,),
        )
        next_state = "queued" if int(pending["n"]) else "idle"
        changed = await tx.execute(
            f"UPDATE {port.state} SET execution_state=?,active_send_index=NULL,"
            "owner_token=NULL,owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
            "cancel_send_index=NULL,cancel_reason=NULL,next_seq=? WHERE session_id=? "
            "AND execution_state='running' AND active_send_index=? AND owner_epoch=? AND owner_token=?",
            (
                next_state, result_seq + 1, ctx.session_id, ctx.send_index,
                ctx.batch.owner_epoch, ctx.batch.owner_token,
            ),
        )
        if changed != 1:
            raise SessionLeaseLost(ctx.session_id)
        saved.update(
            status=aggregate["status"], result=aggregate, result_seq=result_seq,
            aggregate_delivered=True, aggregate_delivered_at=database_now,
            continuation={}, last_input_ref=plain(origin.ref),
        )
        await journal.put(port.store, tx, ctx.session_id, journal.run_key(saved["run_id"]), saved)
        results = tuple(
            Result(
                record.ref, target, status, ctx.send_index, result_seq, final_text,
                aggregate.get("usage") or {}, aggregate, error,
                None if status == "completed" else status, delivery_error,
            )
            for record in chain
        )
        return FinalBatch(
            next_state, results, result_ref,
            activations[0] if activations else None,
        )


async def _finish_late(worker, ctx, refs):
    """Terminalize late result inputs without another graph run or parent result."""
    port = worker.inputs
    async with port.db.transaction() as tx:
        state = await port.store._db.dialect.lock_state(tx, port.state, ctx.session_id)
        if (
            state["execution_state"] != "running"
            or state["active_send_index"] != ctx.send_index
            or state["owner_epoch"] != ctx.batch.owner_epoch
            or state["owner_token"] != ctx.batch.owner_token
        ):
            raise SessionLeaseLost(ctx.session_id)
        database_now = now_ms()
        ids = [ref.message_id for ref in refs]
        if ids:
            await tx.execute(
                f"UPDATE {port.q} SET terminal_status='completed',result_send_index=?,completed_at=? "
                f"WHERE session_id=? AND item_id IN ({','.join('?' for _ in ids)}) "
                "AND status='delivered' AND delivered_send_index=? AND terminal_status IS NULL",
                (ctx.send_index, database_now, ctx.session_id, *ids, ctx.send_index),
            )
        pending = await tx.fetchone(
            f"SELECT COUNT(*) AS n FROM {port.q} WHERE session_id=? AND status='pending'",
            (ctx.session_id,),
        )
        next_state = "queued" if int(pending["n"]) else "idle"
        await tx.execute(
            f"UPDATE {port.state} SET execution_state=?,active_send_index=NULL,"
            "owner_token=NULL,owner_node_id=NULL,owner_boot_id=NULL,owner_lease_until_ms=NULL,"
            "cancel_send_index=NULL,cancel_reason=NULL WHERE session_id=? "
            "AND execution_state='running' AND active_send_index=? AND owner_epoch=? AND owner_token=?",
            (
                next_state, ctx.session_id, ctx.send_index,
                ctx.batch.owner_epoch, ctx.batch.owner_token,
            ),
        )
        target = TargetRef(ctx.store_id, ctx.session_id)
        return FinalBatch(next_state, tuple(
            Result(ref, target, "completed", ctx.send_index, None) for ref in refs
        ))


async def run_driver(worker, ctx, env):
    """Rebuild the graph at each send; healthy nodes are never adopted or cancelled.

    RuntimeWorker's dispatcher excludes non-LLM drivers from model capacity, so a
    driver can run (and recover) while all its healthy leaves still own the slots.
    """
    port = worker.inputs
    session = await port._session(port.db, ctx.session_id)
    run_id = session["start_input_message_id"]
    saved = await journal.get(port.store, port.db, ctx.session_id, journal.run_key(run_id))
    if saved is None:
        raise RuntimeRequestError("workflow_journal_missing")
    origins = []
    for ref in ctx.batch.input_refs:
        record = await port.input(port.db, ref)
        if record is not None and record.row["kind"] in {"user", "workflow_continue"}:
            origins.append(record)
    origin = origins[-1] if origins else None
    resume_options = None
    if origin is not None:
        content = json.loads(origin.row["content_json"])
        if isinstance(content, dict) and "workflow_resume" in content:
            resume_options = content["workflow_resume"]
    executor = DurableExecutor(worker, ctx, env, run_id, saved)
    if saved.get("aggregate_delivered") and resume_options is None:
        await executor._consume_results()
        return await _finish_late(worker, ctx, list(ctx.batch.input_refs))
    if origin is None:
        raise RuntimeRequestError("workflow_origin_missing")
    if resume_options is not None and saved.get("definition_ref") != plain(origin.ref):
        saved = await _prepare_explicit(worker, ctx, saved, origin, resume_options)
        executor = DurableExecutor(worker, ctx, env, run_id, saved)
    spec = WorkflowSpec.from_json(plain(ctx.descriptor.config["workflow"]["spec"]))
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
    graph = asyncio.create_task(engine.run(spec))
    stopped = asyncio.create_task(ctx.stop.wait())
    timed_out = False
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
        else:
            timed_out = stopped not in done
            if timed_out:
                await worker.inputs.cancel_active(ctx.batch, reason="workflow timeout")
            graph.cancel()
            await asyncio.gather(graph, return_exceptions=True)
            result = engine._build_result(spec, "failed" if timed_out else "cancelled")
            if timed_out:
                result.errors.append("workflow timeout")
    finally:
        stopped.cancel()
        graph.cancel()
        await asyncio.gather(stopped, graph, return_exceptions=True)
    await executor._consume_results()
    return await _complete_driver(worker, ctx, saved, origin, result.to_dict())


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
