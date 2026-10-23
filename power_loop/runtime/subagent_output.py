"""Bounded original-session completion of empty managed child output."""
from __future__ import annotations

import asyncio
import contextlib

from .spec import workflow_continuation_loop


async def ensure_answer(loop, running, sid, raw, deadline):
    binding = running.task_binding
    if ((binding is not None and binding.lease.generation != 1) or running.answer_attempted or
            raw.get("status") != "completed" or (raw.get("final_text") or "").strip() or
            running.token.is_cancelled()):
        return raw
    running.answer_attempted = True
    child = await workflow_continuation_loop(loop, sid)
    remaining = int(child.config.max_rounds or 3) - int(raw.get("rounds") or 0)
    if remaining <= 0:
        return {**raw, "status": "failed", "error": "subagent_output_missing"}
    with contextlib.ExitStack() as guards:
        for _, guard in getattr(loop, "_child_run_guards", ()):
            guards.enter_context(guard())
        extra = await asyncio.wait_for(child.follow_up(
            "Your previous turn did not contain a final answer. There is no outstanding delegated task. "
            "Use the work already completed in this session and return the requested final answer as assistant text. "
            "Do not repeat completed actions or call pass_turn.", session_id=sid,
            stop_event=running.token, max_rounds=remaining, tools=[]),
            timeout=max(0.0, deadline-asyncio.get_running_loop().time()))
    usage = dict(raw.get("usage") or {})
    for name, value in (extra.usage or {}).items():
        usage[name] = usage.get(name, 0) + value
    result = {**raw, "status": extra.status, "final_text": extra.final_text,
              "rounds": int(raw.get("rounds") or 0)+int(extra.rounds), "usage": usage,
              "new_input_after_pass": bool(getattr(extra, "new_input_after_pass", False))}
    if extra.status == "completed" and not (extra.final_text or "").strip():
        result.update(status="failed", error="subagent_output_missing")
    return result
