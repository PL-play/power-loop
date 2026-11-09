"""Workflow admission and driver execution through the common message runtime."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, replace
from types import SimpleNamespace

from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import (
    Command,
    InputRef,
    JobRef,
    RequestOptions,
    Result,
    RuntimeDescriptor,
    RuntimeRequestError,
    TargetRef,
    TrustedActor,
    canonical,
    plain,
)
from power_loop.runtime.jobs import JobActor, JobLease, JobManager, JobRegistration, _hash, _record
from power_loop.runtime.store.types import SubagentLifecycle

from . import journal
from .engine import WorkflowEngine
from .spec import WorkflowSpec


@dataclass(frozen=True)
class DriverActor(TrustedActor):
    job_ref: JobRef | None = None


async def authorize_driver(port, reader, actor):
    from power_loop.runtime.session_ownership import current_ownership

    ownership = current_ownership()
    if ownership is None or ownership.key != (port.store.store_id, actor.session_id):
        raise RuntimeRequestError("forbidden", "driver ownership required")
    ownership.require_owned()
    row = await reader.fetchone(
        f"SELECT kind,session_id,generation FROM {port.jobs} WHERE job_id=?", (actor.job_ref.job_id,)
    )
    if row is None or (row["kind"], row["session_id"], row["generation"]) != (
        "workflow",
        actor.session_id,
        actor.job_ref.generation,
    ):
        raise RuntimeRequestError("forbidden", "driver generation changed")


async def submit(client, target, spec, *, message_id, actor, descriptor=None, options=None, parent=None):
    """Freeze a graph and atomically create its independent driver and start input."""
    client._require_open()
    spec = WorkflowSpec.from_json(spec.to_dict() if isinstance(spec, WorkflowSpec) else spec)
    port, manager = client.inputs, JobManager(client.store)
    defaults = await port.authorize(port.db, target, actor)
    origin = JobActor(actor.tenant_id, actor.principal_id, target.session_id)
    key = manager._origin_key(
        origin, "create", actor.call_ref, actor.tool_call_id, None if actor.call_ref else "workflow:" + message_id
    )
    sid = "driver_" + key[:32]
    start = InputRef(target.store_id, sid, "start_" + key[:32])
    options = options or RequestOptions()
    async with manager.joined_write() as tx:
        await port.db.dialect.lock_state(tx, port.state, target.session_id)
        if parent:
            await manager._fence(tx, origin, parent)
        old = await tx.fetchone(f"SELECT job_id FROM {manager.events} WHERE operation_key=?", (key,))
        accepted = await port.input(tx, start) if old else None
        descriptor = descriptor or RuntimeDescriptor(
            **(accepted.meta["descriptor"] if accepted else defaults["runtime_descriptor"])
        )
        allowed_tools = options.tools
        if allowed_tools is None and accepted:
            allowed_tools = accepted.meta["descriptor"]["config"]["workflow"]["allowed_tools"]
        descriptor = replace(
            descriptor,
            kind="workflow_driver",
            config={
                **plain(descriptor.config),
                "workflow": {"spec": spec.to_dict(), "allowed_tools": plain(allowed_tools)},
            },
        )
        if old is None:
            await port.store._create_session_tx(
                tx,
                sid,
                now_ms(),
                parent_session_id=target.session_id,
                lifecycle=SubagentLifecycle.LINKED,
                metadata={
                    "tenant_id": actor.tenant_id,
                    "principal_id": actor.principal_id,
                    "execution_role": "workflow",
                    "runtime_descriptor": plain(descriptor),
                },
            )
        registration = await manager.register_job(
            origin,
            kind="workflow",
            title=spec.name,
            session_id=sid,
            parent=parent,
            independent=True,
            call_ref=actor.call_ref,
            tool_call_id=actor.tool_call_id,
            system_key=None if actor.call_ref else "workflow:" + message_id,
            request={"input_ref": plain(start)},
        )
        receipt = await port.accept_input(
            tx,
            Command(
                start.message_id,
                TargetRef(start.store_id, sid),
                "send",
                {"body": spec.input},
                options=plain(options),
            ),
            actor,
            descriptor=descriptor, child_start=True,
        )
        run_id = registration.operation.operation_id[3:]
        if receipt.newly_accepted:
            await tx.execute(
                f"UPDATE {port.q} SET delivered_job_id=?,delivered_job_generation=0 WHERE session_id=? AND item_id=?",
                (registration.job.job_id, sid, start.message_id),
            )
            await journal.put(
                port.store,
                tx,
                sid,
                journal.run_key(run_id),
                {
                    "run_id": run_id,
                    "job_id": registration.job.job_id,
                    "driver_sid": sid,
                    "definition_ref": plain(start),
                    "created_at_ms": now_ms(),
                },
            )
    await client._notify(TargetRef(start.store_id, sid), start.message_id, start, ["work", "activation"])
    return WorkflowRunHandle(client, actor, run_id, JobRef(start.store_id, registration.job.job_id, 0), start)


class WorkflowRunHandle:
    """A durable reference, with no local task or callback ownership."""

    def __init__(self, client, actor, run_id, job_ref, input_ref):
        self.client, self.actor, self.run_id = client, actor, run_id
        self.job_ref, self.input_ref = job_ref, input_ref
        self.job_id, self.operation_id = job_ref.job_id, "op_" + run_id

    async def wait(self, *, deadline_ms=None):
        result = await self.client.wait(self.input_ref, actor=self.actor, deadline_ms=deadline_ms)
        from .result import AgentResult, WorkflowResult

        data = plain(result.structured_output)
        if data is None:
            return WorkflowResult(self.run_id, result.status, errors=[result.error] if result.error else [])
        data["results"] = {key: AgentResult(**value) for key, value in data["results"].items()}
        data["final"] = AgentResult(**data["final"]) if data["final"] else None
        return WorkflowResult(**data)

    async def resume(self, *, generation, node_id, message, message_id):
        from .resume import resume_run

        return await resume_run(
            self.client,
            replace(self.job_ref, generation=generation),
            node_id=node_id,
            message=message,
            message_id=message_id,
            actor=self.actor,
        )

    async def cancel(self, reason="cancelled"):
        return await self.client.cancel(self.input_ref, actor=self.actor, scope="subtree", reason=reason)


class DurableExecutor:
    def __init__(self, worker, ctx, runtime, binding, run_id, env):
        self.worker, self.ctx, self.runtime, self.binding = worker, ctx, runtime, binding
        self.port, self.manager = worker.inputs, runtime.manager
        self.run_id, self.env = run_id, env
        self.plan = {}
        self.actor = DriverActor(ctx.actor.tenant_id, ctx.actor.principal_id, ctx.session_id, job_ref=ctx.job_ref)

    async def run_agent(self, request, *, allow_start=True):
        """G: stable dispatch plus node record; consume results under driver ownership."""
        from power_loop.runtime.execution_roles import role_tools
        from power_loop.runtime.spec import AgentSpec, _child_config

        spec = AgentSpec.from_dict(request.spec)
        config = _child_config(spec, SimpleNamespace(config=self.env.config))
        available = self.env.tool_registry.names() if self.env.tool_registry else []
        tools = role_tools(
            spec.tools if spec.tools is not None else available, "workflow_node", registry=self.env.tool_registry
        )
        options = RequestOptions(
            tools=tools,
            **{
                name: getattr(config, name)
                for name in (
                    "model",
                    "temperature",
                    "max_tokens",
                    "max_rounds",
                    "max_tokens_per_run",
                    "system_prompt",
                    "response_format",
                )
            },
        )
        descriptor = replace(
            self.ctx.descriptor,
            kind="workflow_node",
            config={
                **{key: value for key, value in plain(self.ctx.descriptor.config).items() if key != "workflow"},
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
        # This exact system key is the G0 contract. Only the creation event is unique.
        system_key = (
            "wf:"
            + hashlib.sha256(
                canonical([self.run_id, request.node_path, request.iteration_path, request.attempt]).encode()
            ).hexdigest()
        )
        origin = self.runtime.actor(self.ctx.session_id)
        offset = self.plan.get("attempt_offsets", {}).get(request.node_path, 0)
        if offset:
            request = replace(request, attempt=request.attempt + offset)
            system_key = (
                "wf:"
                + hashlib.sha256(
                    canonical([self.run_id, request.node_path, request.iteration_path, request.attempt]).encode()
                ).hexdigest()
            )
        key = self.manager._origin_key(origin, "create", None, None, system_key)
        sid = request.session_id or "node_" + key[:32]
        start = InputRef(self.ctx.store_id, sid, "start_" + key[:32])
        entry_key = journal.node_key(key)
        async with self.manager.joined_write() as tx:
            await self.port.lock_owned(tx, TargetRef(self.ctx.store_id, self.ctx.session_id), self.ctx)
            await self.manager._fence(tx, origin, self.binding.lease)
            saved = await journal.get(self.port.store, tx, self.ctx.session_id, entry_key)
            if saved is None:
                if not allow_start:
                    return {"status": "budget_exceeded", "usage": {}}
                if request.session_id is None:
                    await self.port.store._create_session_tx(
                        tx,
                        sid,
                        now_ms(),
                        parent_session_id=self.ctx.session_id,
                        lifecycle=SubagentLifecycle.LINKED,
                        metadata={
                            **plain(spec.metadata),
                            "tenant_id": self.actor.tenant_id,
                            "principal_id": self.actor.principal_id,
                            "execution_role": "workflow_node",
                            "workflow_run_id": self.run_id,
                            "runtime_descriptor": plain(descriptor),
                        },
                    )
                registration = await self.manager.register_job(
                    origin,
                    kind="workflow_node",
                    title=request.node_path,
                    session_id=sid,
                    parent=self.binding.lease,
                    system_key=system_key,
                    request={"input_ref": plain(start)},
                )
                receipt = await self.port.accept_input(
                    tx,
                    Command(
                        start.message_id,
                        TargetRef(start.store_id, sid),
                        "send",
                        {"body": request.body},
                        options=plain(options),
                    ),
                    self.actor,
                    descriptor=descriptor, child_start=True,
                )
                await tx.execute(
                    f"UPDATE {self.port.q} SET delivered_job_id=?,delivered_job_generation=0 WHERE session_id=? AND item_id=?",
                    (registration.job.job_id, sid, start.message_id),
                )
                saved = {
                    "path": request.node_path,
                    "iteration": request.iteration_path,
                    "attempt": request.attempt,
                    "session_id": sid,
                    "job_id": registration.job.job_id,
                    "input_ref": plain(receipt.input_ref),
                    "request_hash": _hash(plain(request)),
                    "continuations": {},
                    "result_ref": None,
                }
                await journal.put(self.port.store, tx, self.ctx.session_id, entry_key, saved)
            elif request.continuation == 0 and saved["request_hash"] != _hash(plain(request)):
                raise RuntimeRequestError("request_conflict", "workflow dispatch input changed")
            wait_ref = InputRef(**self.plan.get("overrides", {}).get(key, saved["input_ref"]))
            if request.continuation:
                continuation_key = str(request.continuation)
                existing = saved["continuations"].get(continuation_key)
                if existing is None:
                    if not allow_start:
                        return {"status": "budget_exceeded", "session_id": saved["session_id"], "usage": {}}
                    row = await self.manager._row(tx, saved["job_id"])
                    wait_ref = InputRef(
                        self.ctx.store_id,
                        saved["session_id"],
                        "continue_" + _hash([key, plain(wait_ref), request.continuation])[:32],
                    )
                    receipt = await self.port.accept_input(
                        tx,
                        Command(
                            wait_ref.message_id,
                            TargetRef(wait_ref.store_id, wait_ref.session_id, row["job_id"]),
                            "resume",
                            {"body": request.body},
                            expected_job_generation=row["generation"],
                            options=plain(options),
                        ),
                        self.actor,
                        descriptor=descriptor,
                    )
                    saved["continuations"][continuation_key] = plain(receipt.input_ref)
                    await journal.put(self.port.store, tx, self.ctx.session_id, entry_key, saved)
                else:
                    wait_ref = InputRef(**existing)
        await self.worker.runtime.client._notify(
            TargetRef(wait_ref.store_id, wait_ref.session_id), wait_ref.message_id, wait_ref, ["work", "activation"]
        )
        try:
            result = await self.worker.runtime.client.wait(wait_ref, actor=self.actor, deadline_ms=None)
        except asyncio.CancelledError:
            # A graph halt stops its outstanding leaves. Losing the driver
            # process leaves healthy nodes alone for the replacement driver.
            if not self.ctx.shutdown_interrupted and not self.ctx.lease.lost:
                await self.worker.runtime.client.cancel(
                    wait_ref, actor=self.actor, scope="subtree", reason="workflow branch stopped"
                )
            raise
        from power_loop.runtime.result_revisions import result_id

        reference = {
            "result_id": result_id(result.job_ref.job_id, result.job_ref.generation),
            "job_id": result.job_ref.job_id,
            "generation": result.job_ref.generation,
        }
        async with self.manager.joined_write() as tx:
            await self.port.lock_owned(tx, TargetRef(self.ctx.store_id, self.ctx.session_id), self.ctx)
            await self.manager._fence(tx, origin, self.binding.lease)
            saved = await journal.get(self.port.store, tx, self.ctx.session_id, entry_key)
            # Re-reading earlier generations rebuilds usage, not the latest
            # node projection. Keep the already consumed continuation result.
            if saved["result_ref"] is None or saved["result_ref"]["generation"] < reference["generation"]:
                saved["result_ref"] = reference
                await journal.put(self.port.store, tx, self.ctx.session_id, entry_key, saved)
            item = hashlib.sha256(("result:" + reference["result_id"]).encode()).hexdigest()
            await tx.execute(
                f"UPDATE {self.port.q} SET status='delivered',delivered_at=?,claim_token=NULL,claimed_at=NULL,"
                "claim_owner_epoch=NULL WHERE session_id=? AND item_id=?",
                (now_ms(), self.ctx.session_id, item),
            )
            await tx.update_event(
                reference["job_id"],
                f"UPDATE {self.port.events} SET delivery_status='handled' WHERE event_id=?",
                (reference["result_id"],),
            )
            await tx.write_job(
                reference["job_id"],
                f"UPDATE {self.port.jobs} SET result_status='handled',version=version+1 "
                "WHERE job_id=? AND generation=?",
                (reference["job_id"], reference["generation"]),
                obligations=True,
            )
        from power_loop.runtime.finalization import try_finalize

        await try_finalize(self.manager, origin, result.job_ref.job_id, result.job_ref.generation)
        raw = plain(result)
        # Public results classify incomplete model turns as interrupted. The
        # graph retains the explicit round-limit reason for its declared policy.
        if result.status == "interrupted" and result.error in {"hit_round_limit", "context_checkpoint"}:
            raw["status"] = result.error
        return {
            **raw,
            "session_id": saved["session_id"],
            "has_next_continuation": str(request.continuation + 1) in saved["continuations"],
            "accounting_usage": (
                {}
                if result.job_ref.generation <= self.plan.get("already_counted", {}).get(result.job_ref.job_id, 0)
                else raw["usage"]
            ),
        }

    async def todo_remaining(self, sid):
        state = await self.port.store.get_runtime_state(sid, "todo") or {}
        return [item["text"] for item in state.get("items", []) if item["status"] != "completed"]


async def run(worker, ctx, record, env, runtime, *, recovery=False):
    """Reconstruct a driver from the frozen definition and existing dispatches."""
    from power_loop.runtime.task_runtime import Execution

    port, manager = worker.inputs, runtime.manager
    await runtime.ensure_started()
    job_id = record.row["delivered_job_id"] or record.row["target_job_id"]
    async with manager.joined_write() as tx:
        await port.lock_owned(tx, TargetRef(ctx.store_id, ctx.session_id), ctx)
        row = await manager._row(tx, job_id)
        if row["outcome"] is not None and (record.meta["operation"] != "resume" or recovery):
            await consume_late_results(worker, ctx, row, tx)
            from power_loop.runtime.finalization import try_finalize

            await try_finalize(manager, runtime.actor(ctx.session_id), job_id, row["generation"])
            return
        explicit = record.meta["operation"] == "resume" and not recovery
        if explicit:
            claimed = await port.claim_inputs(tx, TargetRef(ctx.store_id, ctx.session_id), ctx, 1, [record.ref])
            if not claimed:
                return
            from power_loop.runtime.subagent_resume import resume_claim

            binding = await resume_claim(runtime, tx, record, row)
            ctx.job_ref = JobRef(ctx.store_id, job_id, binding.lease.generation)
            await port.bind_input_job(tx, record.ref, ctx.job_ref, ctx, claim_token=claimed[0].row["claim_token"])
            await tx.execute(
                f"UPDATE {port.q} SET status='delivered',delivered_at=?,claim_token=NULL,claimed_at=NULL,"
                "claim_owner_epoch=NULL WHERE session_id=? AND item_id=?",
                (now_ms(), ctx.session_id, record.ref.message_id),
            )
        elif row["generation"] == 0:
            claimed = await port.claim_inputs(tx, TargetRef(ctx.store_id, ctx.session_id), ctx, 1, [record.ref])
            if not claimed:
                return
            binding = await runtime.claim(runtime.actor(ctx.session_id), JobRegistration(_record(row), None, False))
            ctx.job_ref = JobRef(ctx.store_id, job_id, binding.lease.generation)
            await port.bind_input_job(tx, record.ref, ctx.job_ref, ctx, claim_token=claimed[0].row["claim_token"])
            await tx.execute(
                f"UPDATE {port.q} SET status='delivered',delivered_at=?,claim_token=NULL,claimed_at=NULL,"
                "claim_owner_epoch=NULL WHERE session_id=? AND item_id=?",
                (now_ms(), ctx.session_id, record.ref.message_id),
            )
        else:
            # Only the new driver owner enters a new generation. No leaf is
            # re-claimed or stopped, and its original result/cost owner remains.
            old_ref = JobRef(ctx.store_id, job_id, row["generation"])
            await port._commit_result(
                tx,
                old_ref,
                Result(old_ref, record.ref, "interrupted", error="workflow driver replaced"),
                ctx,
                {"parent_session_id": None},
            )
            row = await manager._row(tx, job_id)
            lease = JobLease(job_id, runtime.executor_id, runtime.boot_id, row["generation"] + 1)
            from power_loop.runtime.executor_registry import ExecutorIdentity, require_accepting

            executor = await require_accepting(
                manager,
                tx,
                ExecutorIdentity(row["tenant_id"], row["principal_id"], runtime.executor_id, runtime.boot_id),
                required=row["executor_key"] is not None,
            )
            changed = await tx.write_job(
                job_id,
                f"UPDATE {manager.jobs} SET generation=?,executor_id=?,boot_id=?,executor_key=?,version=version+1,"
                "result_json=NULL,result_status='none',updated_at=? WHERE job_id=? AND generation=? AND version=? AND outcome IS NULL",
                (
                    lease.generation,
                    lease.executor_id,
                    lease.boot_id,
                    executor["executor_key"] if executor else None,
                    now_ms(),
                    job_id,
                    row["generation"],
                    row["version"],
                ),
            )
            if not changed:
                raise RuntimeRequestError("stale_generation")
            binding = Execution(runtime, runtime.actor(ctx.session_id), lease)
            from power_loop.runtime.execution_membership import start

            await start(manager, tx, await manager._row(tx, job_id))
            ctx.job_ref = JobRef(ctx.store_id, job_id, lease.generation)
        ctx.parent_job_id, ctx.root_job_id = row["parent_job_id"], row["root_job_id"]
        creation = await tx.fetchone(
            f"SELECT event_id FROM {manager.events} WHERE job_id=? AND action='create'", (job_id,)
        )
        run_id = creation["event_id"][3:]
        saved = await journal.get(port.store, tx, ctx.session_id, journal.run_key(run_id))
        if explicit:
            saved = await prepare_continuation(worker, ctx, runtime, binding, run_id, saved, tx)
            await journal.put(port.store, tx, ctx.session_id, journal.run_key(run_id), saved)
    executor = DurableExecutor(worker, ctx, runtime, binding, run_id, env)
    executor.plan = saved.get("continuation", {})
    spec = WorkflowSpec.from_json(plain(ctx.descriptor.config["workflow"]["spec"]))
    engine = WorkflowEngine(
        executor,
        run_id=run_id,
        driver_sid=ctx.session_id,
        allowed_tools=ctx.descriptor.config["workflow"]["allowed_tools"],
        stop_event=ctx.stop,
        file_io=env.workflow_file_io,
    )
    engine._usage_acc = dict(executor.plan.get("prior_usage", {}))
    # The scheduler holds ownership but no model slot while awaiting leaves.
    async with runtime.execution(binding, cancel_token=ctx.stop), worker.idle(ctx):
        graph = asyncio.create_task(engine.run(spec))
        stopped = asyncio.create_task(ctx.stop.wait())
        timed_out = False
        try:
            remaining = max(0, spec.timeout_s - (now_ms() - saved["created_at_ms"]) / 1000) if spec.timeout_s else None
            done, _ = await asyncio.wait((graph, stopped), timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            if graph in done:
                result = graph.result()
            else:
                timed_out = stopped not in done
                if timed_out:
                    await worker.runtime.client.cancel(
                        ctx.job_ref, actor=executor.actor, scope="subtree", reason="workflow timeout"
                    )
                graph.cancel()
                await asyncio.gather(graph, return_exceptions=True)
                result = engine._build_result(spec, "failed" if timed_out else "cancelled")
                if timed_out:
                    result.errors.append("workflow timeout")
        finally:
            stopped.cancel()
            graph.cancel()
            await asyncio.gather(stopped, graph, return_exceptions=True)
        async with manager.joined_write() as tx:
            route = await port._lock_result_sessions(tx, ctx.job_ref)
            await port.lock_owned(tx, TargetRef(ctx.store_id, ctx.session_id), ctx)
            row = await manager._authorize(tx, binding.actor, job_id)
            manager._check_proof(row, binding.lease)
            status = "failed" if timed_out else "cancelled" if row["cancel_requested_at"] is not None else result.status
            await tx.write_job(
                job_id,
                f"UPDATE {manager.jobs} SET status='settling',outcome=?,resource_state='stopped',intake='closed',"
                "version=version+1 WHERE job_id=? AND generation=? AND version=?",
                (
                    "succeeded" if status == "completed" else "failed" if status == "budget_exceeded" else status,
                    job_id,
                    ctx.job_ref.generation,
                    row["version"],
                ),
                obligations=True,
            )
            parent = await port._commit_result(
                tx,
                ctx.job_ref,
                Result(
                    ctx.job_ref,
                    record.ref,
                    status,
                    result.final.text if result.final else "",
                    result.usage,
                    result.to_dict(),
                ),
                ctx,
                route,
            )
    from power_loop.runtime.finalization import try_finalize

    await try_finalize(manager, binding.actor, job_id, ctx.job_ref.generation)
    if parent:
        await worker.runtime.client._notify(
            TargetRef(parent.store_id, parent.session_id), parent.message_id, parent, ["work", "activation"]
        )


async def prepare_continuation(worker, ctx, runtime, binding, run_id, saved, tx):
    """The explicit new input selects one node and a new dependent tail."""
    from .continuation import plan_continuation

    port, manager = worker.inputs, runtime.manager
    options = plain(ctx.descriptor.config["workflow_resume"])
    plan = plan_continuation(
        WorkflowSpec.from_json(plain(ctx.descriptor.config["workflow"]["spec"])), options["node_id"]
    )
    rows = await tx.fetchall(
        f"SELECT state_key,value_json FROM {port.store.t.session_runtime_state} "
        "WHERE session_id=? AND state_key LIKE 'workflow:node:%' LIMIT 10001",
        (ctx.session_id,),
    )
    import json

    nodes = [(r["state_key"].removeprefix("workflow:node:"), json.loads(r["value_json"])) for r in rows]
    selected = [(key, n) for key, n in nodes if n["path"] == options["node_id"]]
    if not selected or len({canonical(n["iteration"]) for _, n in selected}) != 1:
        raise RuntimeRequestError("workflow_node_ambiguous_choose_job")
    key, node = max(selected, key=lambda pair: pair[1]["attempt"])
    child = await manager._row(tx, node["job_id"])
    original = await port.input(tx, InputRef(**node["input_ref"]))
    actor = DriverActor(ctx.actor.tenant_id, ctx.actor.principal_id, ctx.session_id, job_ref=ctx.job_ref)
    start = InputRef(ctx.store_id, node["session_id"], "resume_" + _hash([ctx.input_ref.message_id, key])[:32])
    receipt = await port.accept_input(
        tx,
        Command(
            start.message_id,
            TargetRef(ctx.store_id, node["session_id"], node["job_id"]),
            "resume",
            {"body": options["message"]},
            expected_job_generation=child["generation"],
            options=original.meta["options"],
        ),
        actor,
        descriptor=RuntimeDescriptor(**original.meta["descriptor"]),
    )
    # This explicit input starts a new continuation sequence. Driver recovery
    # skips this preparation and retains that sequence's accepted references.
    node["continuations"] = {}
    await journal.put(port.store, tx, ctx.session_id, journal.node_key(key), node)
    previous = saved.get("continuation", {})
    offsets = dict(previous.get("attempt_offsets", {}))
    for _, n in nodes:
        if n["path"] in plan.invalidated and n["path"] != options["node_id"]:
            offsets[n["path"]] = max(offsets.get(n["path"], 0), n["attempt"])
    overrides = dict(previous.get("overrides", {}))
    overrides[key] = plain(receipt.input_ref)
    import json

    from power_loop.runtime.result_revisions import result_id

    old_result = await tx.fetchone(
        f"SELECT payload_json FROM {manager.events} WHERE event_id=?",
        (result_id(ctx.job_ref.job_id, ctx.job_ref.generation - 1),),
    )
    usage = json.loads(old_result["payload_json"])["result"]["usage"]
    counted = {}
    for _, n in nodes:
        job = await manager._row(tx, n["job_id"])
        counted[job["job_id"]] = job["generation"]
    return {
        **saved,
        "definition_ref": plain(ctx.input_ref),
        "continuation": {
            "attempt_offsets": offsets,
            "overrides": overrides,
            "prior_usage": usage,
            "already_counted": counted,
        },
    }


async def consume_late_results(worker, ctx, workflow, tx):
    """A finished graph records late node results without starting another graph."""
    port = worker.inputs
    records = await tx.fetchall(
        f"SELECT * FROM {port.q} WHERE session_id=? AND status='pending' ORDER BY id LIMIT 100", (ctx.session_id,)
    )
    for row in records:
        record = port.record(row)
        if record.meta["operation"] != "result_ready":
            await port.void_unbound(tx, record.ref, reason="workflow_finished")
            continue
        ref = record.meta["result_ref"]
        source = await tx.fetchone(
            f"SELECT j.parent_job_id,e.operation_key FROM {port.jobs} j JOIN {port.events} e "
            "ON e.job_id=j.job_id AND e.action='create' WHERE j.job_id=?",
            (ref["job_id"],),
        )
        if source["parent_job_id"] != workflow["job_id"]:
            raise RuntimeRequestError("forbidden")
        key = journal.node_key(source["operation_key"])
        node = await journal.get(port.store, tx, ctx.session_id, key)
        if node is not None:
            node["result_ref"] = ref
            await journal.put(port.store, tx, ctx.session_id, key, node)
        await tx.execute(
            f"UPDATE {port.q} SET status='delivered',delivered_at=? WHERE session_id=? AND item_id=?",
            (now_ms(), ctx.session_id, record.ref.message_id),
        )
        await tx.update_event(
            ref["job_id"], f"UPDATE {port.events} SET delivery_status='handled' WHERE event_id=?", (ref["result_id"],)
        )
        await tx.write_job(
            ref["job_id"],
            f"UPDATE {port.jobs} SET result_status='handled',version=version+1 WHERE job_id=? AND generation=?",
            (ref["job_id"], ref["generation"]),
            obligations=True,
        )
