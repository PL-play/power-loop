"""Independent, single-dispatch tool execution on a session owner."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import (
    JobRef,
    RequestOptions,
    Result,
    RuntimeRequestError,
    TargetRef,
    canonical,
    plain,
)
from power_loop.contracts.tools import ToolResult
from power_loop.core import agent_context

from . import tool_attempts
from .child_tasks import admit, tool_actor
from .control_runtime import current, flush_binding, forget_completed, source
from .jobs import JobError, _hash
from .pricing import BillingReceipt, PricedCall, PricingError
from .store.types import SubagentLifecycle
from .tool_contracts import background_allowed, validate_registration


async def create(loop, name, arguments, *, command=False, automatic=False):
    from power_loop.tools.registry import async_capable_for

    from .tool_call_plans import registration_value
    binding, actor = await tool_actor(loop)
    registry = loop.tool_registry
    registered = registry.get(name)
    allowed = agent_context.get_effective_tools()
    if registered is None or allowed is not None and name not in allowed:
        raise RuntimeRequestError("forbidden", "background target is outside the tool grant")
    if not background_allowed(registered) or (not command and not automatic and not async_capable_for(registered.definition, arguments)):
        raise RuntimeRequestError("invalid_options", "tool does not allow background execution")
    registry._raise_if_invalid(name, arguments)
    validate_registration(registry, name, registered)
    descriptor = loop._runtime_worker_context.descriptor
    kind = "background_command" if command else "background_tool"
    descriptor = replace(descriptor, kind=kind, config={**plain(descriptor.config), "background": {
        "tool": name, "arguments": arguments, "registration": registration_value(registered),
        "call_ref": actor.call_ref.to_dict(), "call_id": actor.tool_call_id}})
    return await admit(loop, binding, actor, kind=kind, title=name, descriptor=descriptor,
        options=RequestOptions(tools=[name]), body="", role=source(binding)["execution_role"],
        lifecycle=SubagentLifecycle.LINKED)


def automatic(registered):
    return registered.definition.interrupt == "background" and background_allowed(registered)


async def invoke_and_wait(pipeline, registered, arguments, detach):
    loop = agent_context.get_current_loop()
    receipt = await create(loop, registered.definition.name, arguments, automatic=True)
    _, actor = await tool_actor(loop)
    wait = asyncio.create_task(loop._runtime_worker.runtime.client.wait(
        JobRef(**receipt["job_ref"]), actor=actor, deadline_ms=None))
    detached = asyncio.create_task(detach.wait())
    try:
        done, _ = await asyncio.wait({wait, detached}, return_when=asyncio.FIRST_COMPLETED)
        if wait in done:
            result = wait.result()
            return result.final_text
        return canonical({**receipt, "task_id": receipt["job_id"],
            "message": "The original call continues in the background. Its result will be delivered automatically; do not repeat it."})
    finally:
        wait.cancel()
        detached.cancel()
        await asyncio.gather(wait, detached, return_exceptions=True)


async def action(loop, action, *, command=None, task_id=None, tool=None, args=None, task_ids=None, timeout_s=None):
    from .background import encode_results, load_record
    _, actor = await tool_actor(loop)
    worker, port = loop._runtime_worker, loop._runtime_worker.inputs
    client = worker.runtime.client
    if action in {"tool", "run"}:
        receipt = await create(loop, "bash" if action == "run" else tool,
            {"command": command, "description": "Run the requested background command"} if action == "run" else dict(args or {}), command=action == "run")
        return canonical({**receipt, "task_id": receipt["job_id"],
            "message": "The result will be delivered automatically. Do not poll with check; use check only if the task appears stuck."})
    if action not in {"list", "check", "wait", "cancel"}:
        raise RuntimeRequestError("invalid_options", "unknown background action")
    if action in {"check", "cancel"} and not task_id:
        raise RuntimeRequestError("invalid_options", "task_id is required")
    ids = list(task_ids or ([task_id] if task_id else []))
    if not ids:
        from .task_query import list_jobs
        listing = await list_jobs(loop.task_runtime.manager, loop.task_runtime.actor(actor.session_id),
            kinds=["background_tool", "background_command"], status="active", limit=100)
        ids = [row["job_id"] for row in listing["nodes"] if row["kind"] in {"background_tool", "background_command"}]
    deadline = now_ms() + int(min(max(timeout_s or 600, 0.01), 3600) * 1000)
    records = []
    for job_id in ids:
        ref = JobRef(port.store.store_id, job_id)
        row = await port.resolve_job(port.db, ref, actor)
        if row["kind"] not in {"background_tool", "background_command"}:
            raise RuntimeRequestError("invalid_options", "not a background task")
        ref = replace(ref, generation=row["generation"])
        if action == "cancel":
            return canonical(plain(await client.cancel(ref, actor=actor, scope="subtree")))
        if action == "wait":
            from power_loop.agent.runtime_types import WaitTimeout
            try:
                await client.wait(ref, actor=actor, deadline_ms=deadline)
            except WaitTimeout:
                pass  # The caller stopped waiting; the original execution continues.
        status, result, _ = await port.read(ref, actor)
        record = await load_record(port.store, actor.session_id, job_id) if result else None
        records.append(record or {"task_id": job_id, "job_ref": plain(status.job_ref), "status": status.state,
            "result": result.final_text if result else None, "result_ref": plain(status.job_ref)})
    return encode_results(records)


@dataclass
class BackgroundReceipt:
    budget_attempt: object
    billing_receipt: BillingReceipt | None = None

    def record(self, receipt):
        tool_attempts._original_owner(self.budget_attempt)
        if not isinstance(receipt, BillingReceipt):
            raise PricingError("billing_receipt_required")
        if self.billing_receipt is not None and self.billing_receipt != receipt:
            raise PricingError("tool_billing_receipt_changed")
        self.billing_receipt = receipt


@contextmanager
def _context(variable, value, reset):
    token = variable(value)
    try:
        yield
    finally:
        reset(token)


async def run(worker, ctx, record, env, runtime):
    from .jobs import JobRegistration, _record
    from .model_attempts import enable
    from .tool_billing import receipt_context
    from .tool_call_plans import registration_value
    port, manager = worker.inputs, runtime.manager
    frozen = plain(ctx.descriptor.config["background"])
    name, arguments = frozen["tool"], frozen["arguments"]
    registered = env.tool_registry.get(name)
    if registration_value(registered) != frozen["registration"]:
        parent_ref = await port.fail_start(record.ref, "definition_unavailable: background tool changed", ctx)
        if parent_ref:
            await worker.runtime.client._notify(TargetRef(parent_ref.store_id, parent_ref.session_id),
                parent_ref.message_id, parent_ref, ["work", "activation"])
        return
    await runtime.ensure_started()
    async with manager.joined_write() as tx:
        claimed = await port.claim_inputs(tx, TargetRef(ctx.store_id, ctx.session_id), ctx, 1, [record.ref])
        if not claimed:
            return
        row = await manager._row(tx, record.row["delivered_job_id"])
        binding = await runtime.claim(runtime.actor(ctx.session_id), JobRegistration(_record(row), None, False))
        ctx.job_ref = JobRef(ctx.store_id, row["job_id"], binding.lease.generation)
        ctx.parent_job_id, ctx.root_job_id = row["parent_job_id"], row["root_job_id"]
        await port.bind_input_job(tx, record.ref, ctx.job_ref, ctx, claim_token=claimed[0].row["claim_token"])
        await tx.execute(f"UPDATE {port.q} SET status='delivered',delivered_at=?,claim_token=NULL,claimed_at=NULL,"
            "claim_owner_epoch=NULL WHERE session_id=? AND item_id=?", (now_ms(), ctx.session_id, record.ref.message_id))
    try:
        await enable(binding)
        loop = SimpleNamespace(config=env.config, tool_registry=env.tool_registry, task_runtime=runtime,
                               _runtime_worker=worker, _runtime_worker_context=ctx, store=worker.runtime.store)
        with ExitStack() as stack:
            for setter, value, reset in (
                (agent_context.set_current_loop, loop, agent_context.reset_current_loop),
                (agent_context.set_session_id, ctx.session_id, agent_context.reset_session_id),
                (agent_context.set_current_cancel_token, ctx.stop, agent_context.reset_current_cancel_token)):
                stack.enter_context(_context(setter, value, reset))
            async with runtime.execution(binding, cancel_token=ctx.stop):
                from power_loop.tools.registry import validate_pricing
                pricing = validate_pricing(registered)
                key = "toolattempt_" + _hash([ctx.store_id, asdict(binding.lease), "background"])[:48]
                prepared = dict(version=1, attempt_id=key, scope=asdict(binding.lease), session_id=ctx.session_id,
                    call_ref=frozen["call_ref"], call_id=frozen["call_id"], request_hash=_hash(arguments), tool_name=name,
                    pricing_rule={"rule_id": pricing.rule_id, "rule_version": registered.pricing_version})
                control = current(binding)
                value = control.begin_attempt(key, "tool", prepared)
                value["budget_required"] = (runtime._original_budget_config is not None
                    and callable(getattr(runtime._original_budget_config.provider, "prepare_tool", None)))
                control.record("attempt", key, value)
                attempt = tool_attempts.ToolAttempt(binding, key, prepared,
                    PricedCall(name, arguments, source(binding)["execution_role"]), pricing, asyncio.current_task())
                receipt = BackgroundReceipt(attempt)
                status, text, error = "completed", "", None
                async def stop_original():
                    await ctx.stop.wait()
                    policy = env.config.stop_policy
                    grace = (policy.abort_tool_s if registered.definition.interrupt == "abort" else
                             policy.subtask_s if registered.definition.interrupt == "background" else policy.finish_tool_s)
                    await asyncio.sleep(grace)
                    ctx.task.cancel("background cancellation grace elapsed")
                stopper = asyncio.create_task(stop_original())
                try:
                    await tool_attempts.reserve(attempt)
                    await tool_attempts.authorize_dispatch(attempt, None, source(binding))
                    with receipt_context(receipt):
                        output = await env.tool_registry.invoke_async(name, arguments)
                    if isinstance(output, ToolResult):
                        if output.billing_receipt:
                            receipt.record(output.billing_receipt)
                        output = output.content
                    text = output if isinstance(output, str) else canonical(output)
                    if text.startswith("Error:"):
                        status = "failed"
                except BaseException as exc:
                    error = exc
                    status = "interrupted" if ctx.shutdown_interrupted else "cancelled" if ctx.stop.is_cancelled() else "failed"
                    text = f"{type(exc).__name__}: {exc}"
                finally:
                    stopper.cancel()
                    await asyncio.gather(stopper, return_exceptions=True)
                    if control.get("attempt", key)["phase"] == "prepared":
                        from .model_budgets import close_unstarted
                        await close_unstarted(attempt)
                        control.abort_prepared(key)
                    else:
                        tool_attempts.capture(attempt, receipt.billing_receipt, error)
                        await tool_attempts.finish(attempt)
                    async with manager.joined_write() as tx:
                        route = await port._lock_result_sessions(tx, ctx.job_ref)
                        await flush_binding(binding, close=True, transaction=tx)
                        row = await manager._row(tx, ctx.job_ref.job_id)
                        if row["cancel_requested_at"] is not None:
                            status = "cancelled"
                        await tx.write_job(row["job_id"], f"UPDATE {manager.jobs} SET status='settling',outcome=?,"
                            "resource_state=?,intake='closed',version=version+1 WHERE job_id=? AND generation=? AND version=?",
                            ("succeeded" if status == "completed" else status,
                             "unknown" if ctx.shutdown_interrupted else binding.resource_evidence.get("resource_state", "stopped"), row["job_id"], row["generation"], row["version"]), obligations=True)
                        parent_ref = await port._commit_result(tx, ctx.job_ref,
                            Result(ctx.job_ref, record.ref, status, text, error=type(error).__name__ if error else None), ctx, route)
                        manager.after_outcome(lambda: forget_completed(binding), lambda: None)
    except JobError as exc:
        if exc.code != 'cancel_requested':
            raise
        # Cancellation may arrive after the claim commits but before callback
        # admission. No external action started, so finish that same generation.
        async with manager.joined_write() as tx:
            route = await port._lock_result_sessions(tx, ctx.job_ref)
            await port.lock_owned(tx, TargetRef(ctx.store_id, ctx.session_id), ctx)
            if binding.resource_evidence.get('_control_state') is not None:
                await flush_binding(binding, close=True, transaction=tx)
                manager.after_outcome(lambda: forget_completed(binding), lambda: None)
            row = await manager._row(tx, ctx.job_ref.job_id)
            changed = await tx.write_job(row['job_id'], f"UPDATE {manager.jobs} SET status='settling',outcome='cancelled',"
                "resource_state='stopped',intake='closed',version=version+1 WHERE job_id=? AND generation=? "
                "AND version=? AND outcome IS NULL AND cancel_requested_at IS NOT NULL",
                (row['job_id'], ctx.job_ref.generation, row['version']), obligations=True)
            if not changed:
                raise RuntimeRequestError('stale_generation') from exc
            parent_ref = await port._commit_result(tx, ctx.job_ref,
                Result(ctx.job_ref, record.ref, 'cancelled', error='cancel_requested', error_code='cancel_requested'), ctx, route)

    if parent_ref:
        await worker.runtime.client._notify(TargetRef(parent_ref.store_id, parent_ref.session_id),
            parent_ref.message_id, parent_ref, ["work", "activation"])
