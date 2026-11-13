"""Optional final-answer repair for child agents; no parent/job supervision."""
from __future__ import annotations


async def continue_before_finish(pipeline, text, round_index):
    loop = pipeline._loop if hasattr(pipeline, "_loop") else None
    if loop is None:
        from power_loop.core.agent_context import get_current_loop
        loop = get_current_loop()
    ctx = getattr(loop, "_runtime_worker_context", None)
    if ctx is None or "child_spec" not in ctx.descriptor.config:
        return False
    prompt = None
    response_format = ctx.options.response_format
    if not text.strip() and not getattr(pipeline, "_child_answer_attempted", False):
        pipeline._child_answer_attempted = True
        prompt = (
            "Use the work already completed and return the requested final answer as assistant text. "
            "Do not repeat completed actions or call pass_turn."
        )
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
    return False
