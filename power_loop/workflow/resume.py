"""Explicit graph continuation, starting new executions from saved results."""

from dataclasses import replace

from power_loop.agent.runtime_types import (
    Command,
    InputRef,
    RequestOptions,
    RuntimeDescriptor,
    RuntimeRequestError,
    TargetRef,
    plain,
)

from .runner import WorkflowRunHandle


async def resume_run(client, ref, *, node_id, message, message_id, actor):
    """Continue one settled node and recompute its dependent tail.

    Live leaves use client.nudge/steer; this entry never resumes a Python stack.
    """
    port = client.inputs
    job = await port.resolve_job(port.db, ref, actor)
    if job["kind"] != "workflow" or ref.generation != job["generation"]:
        raise RuntimeRequestError("stale_generation")
    if job["outcome"] is None:
        raise RuntimeRequestError("target_not_resumable", "Use nudge or steer for a running leaf.")
    import json

    creation = await port.db.fetchone(
        f"SELECT event_id,payload_json FROM {port.events} WHERE job_id=? AND action='create'", (job["job_id"],)
    )
    first = await port.input(port.db, InputRef(**json.loads(creation["payload_json"])["request"]["input_ref"]))
    descriptor = RuntimeDescriptor(**first.meta["descriptor"])
    from .continuation import plan_continuation
    from .spec import WorkflowSpec

    spec = WorkflowSpec.from_json(plain(descriptor.config["workflow"]["spec"]))
    plan_continuation(spec, node_id)
    if spec.budget:
        from .result import SharedBudget

        budget = SharedBudget(spec.budget.max_tokens, stop_at_remaining_pct=spec.budget.stop_at_remaining_pct)
        budget.commit(json.loads(job["result_json"])["usage"])
        if not budget.can_spawn():
            raise RuntimeRequestError("budget_exceeded", "The cumulative workflow budget is exhausted.")
    if spec.timeout_s:
        from power_loop.agent.runtime_store import now_ms

        from . import journal

        saved = await journal.get(port.store, port.db, job["session_id"], journal.run_key(creation["event_id"][3:]))
        if now_ms() - saved["created_at_ms"] >= spec.timeout_s * 1000:
            raise RuntimeRequestError("workflow_timeout")
    descriptor = replace(
        descriptor, config={**plain(descriptor.config), "workflow_resume": {"node_id": node_id, "message": message}}
    )
    receipt = await client._submit(
        Command(
            message_id,
            TargetRef(ref.store_id, job["session_id"], job["job_id"]),
            "resume",
            {"body": message},
            "new_turn",
            expected_job_generation=ref.generation,
            options=plain(RequestOptions()),
        ),
        actor=actor,
        descriptor=descriptor,
    )
    return WorkflowRunHandle(client, actor, creation["event_id"][3:], ref, receipt.input_ref)
