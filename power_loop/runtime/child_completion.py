"""Finish a child only after accepted nudges and its bounded output repair."""
import asyncio
import json
from dataclasses import replace

from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import JobRef, TargetRef
from power_loop.core.agent_context import get_current_loop


async def continue_before_finish(pipeline, text, round_index):
    loop = get_current_loop()
    ctx = getattr(loop, "_runtime_worker_context", None)
    if ctx is None or "child_spec" not in ctx.descriptor.config:
        return False
    port, manager = loop._runtime_worker.inputs, loop.task_runtime.manager
    pending = await port.db.fetchall(f"SELECT j.job_id,j.generation,e.call_ref_json,e.tool_call_id FROM {manager.jobs} j "
        f"JOIN {manager.jobs} p ON p.job_id=j.parent_job_id "
        f"JOIN {manager.events} e ON e.job_id=j.job_id AND e.action='create' WHERE p.session_id=? "
        "AND j.kind IN ('background_tool','background_command','subagent') AND j.outcome IS NULL LIMIT 1001", (ctx.session_id,))
    if len(pending) > 1000:
        from power_loop.agent.runtime_types import CapacityError
        raise CapacityError()
    for job in pending:
        call = await loop.store.message_ref(ctx.session_id, int(json.loads(job["call_ref_json"])["seq"]))
        actor = replace(ctx.actor, call_ref=call, tool_call_id=job["tool_call_id"])
        wait = asyncio.create_task(loop._runtime_worker.runtime.client.wait(
            JobRef(ctx.store_id, job["job_id"], job["generation"]), actor=actor, deadline_ms=now_ms()+1800000))
        _, why = await pipeline._race(wait)
        if why == "stop":
            wait.cancel()
            await asyncio.gather(wait, return_exceptions=True)
            return True
    ctx.append_overflow = True
    drained = await loop._runtime_worker.drain_append(ctx)
    if await pipeline._append_drained(drained, round_index=round_index):
        return True
    prompt = None
    response_format = ctx.options.response_format
    if not text.strip() and not getattr(pipeline, "_child_answer_attempted", False):
        pipeline._child_answer_attempted = True
        prompt = ("Use the work already completed and return the requested final answer as assistant text. "
                  "Do not repeat completed actions or call pass_turn.")
    elif text.strip() and response_format and not getattr(pipeline, "_child_repair_attempted", False):
        from .structured import StructuredOutputError, parse_structured
        schema = response_format.get("json_schema", {}).get("schema")
        try:
            parse_structured(text, schema=schema)
        except StructuredOutputError:
            pipeline._child_repair_attempted = True
            prompt = "Rewrite the previous answer as the requested JSON object. Do not gather new facts."
    if prompt and round_index + 1 < pipeline._round_limit:
        await pipeline._append_message({"role": "user", "content": prompt}, round_index=round_index)
        pipeline.runtime_tools = []
        pipeline.tool_registry = pipeline.tool_registry.subset([]) if pipeline.tool_registry else None
        return True
    # Intake and finalization serialize with nudge acceptance on the same
    # session state row. A nudge either joins this execution or is rejected.
    async with manager.joined_write() as tx:
        await port.lock_owned(tx, TargetRef(ctx.store_id, ctx.session_id), ctx)
        changed = await tx.write_job(ctx.job_ref.job_id, f"UPDATE {manager.jobs} SET status='settling',version=version+1 "
            "WHERE job_id=? AND generation=? AND status='running' AND NOT EXISTS "
            f"(SELECT 1 FROM {port.q} WHERE session_id=? AND status='pending' AND target_job_id=? "
            "AND expected_job_generation=?)",
            (ctx.job_ref.job_id, ctx.job_ref.generation, ctx.session_id, ctx.job_ref.job_id, ctx.job_ref.generation))
    if changed:
        return False
    ctx.append_overflow = True
    drained = await loop._runtime_worker.drain_append(ctx)
    return bool(await pipeline._append_drained(drained, round_index=round_index))
