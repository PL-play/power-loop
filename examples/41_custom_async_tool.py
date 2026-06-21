"""41 · 自定义异步唤醒工具 / Custom async-wake tool (the extension recipe)

How to build YOUR OWN long-running tool that kicks off async work, returns immediately, and brings
the result back into the agent — using only public, callable APIs (no magic). This is the pattern
behind ``background_run`` / timers, applied to your own system.

What you learn / 你将学到
--------------------------
- ``ToolDefinition`` + an **async def** handler + ``ToolRegistry.register`` — the tool seam
- ``get_tool_runtime_context()`` inside a handler → ``store`` / ``session_id`` / ``loop`` / ``config``
- waking the agent when async work finishes: ``loop.schedule_timer(sid, delay_s=…, note=…)`` →
  a ``TimerRunner`` delivers the note via ``follow_up`` → the agent takes another round
- the agent reads the result back with a second custom tool — the result re-enters the loop

设计要点 / Design notes
-----------------------
- power-loop runs **no daemon**: the host drives every wake. Here the wake is a durable timer; it
  could equally be a ``follow_up`` from your own "job done" callback.
- the handler MUST be ``async def`` (a sync lambda returning a coroutine is run as sync and never
  awaited — see docs/user-guide/async-orchestration.md).

Run / 运行
----------
    python examples/41_custom_async_tool.py        # needs .env with POWER_LOOP_*
"""

from __future__ import annotations

import asyncio
from tempfile import TemporaryDirectory
from typing import Any

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    StatefulAgentLoop,
    TimerRunner,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
)

# A stand-in for "your async system" (a DB export, an HTTP job, a render farm…). Keyed by job id.
_JOBS: dict[str, dict[str, Any]] = {}
# Phrasing-independent signal that the agent re-checked after the wake (used by the real test).
_CHECKED: list[str] = []


async def _start_job(label: str) -> str:
    """Kick off async work, return immediately, and schedule a durable wake-up to check it."""
    ctx = get_tool_runtime_context(required=True)  # the callable seam: store / session_id / loop
    job_id = f"job-{len(_JOBS) + 1}"
    _JOBS[job_id] = {"label": label, "status": "running", "result": None}

    async def _work() -> None:  # simulate slow async work finishing later
        await asyncio.sleep(4)
        _JOBS[job_id].update(status="done", result=f"processed {label!r} ✓")

    asyncio.create_task(_work())
    # Durable wake: when the work should be done, a timer fires its note via follow_up, which
    # gives the agent another round to call check_job. (Alternatively, from _work's completion:
    # await ctx.loop.follow_up(f"{job_id} is done", ctx.session_id).)
    await ctx.loop.schedule_timer(
        ctx.session_id, delay_s=5, note=f"{job_id} should be finished — call check_job('{job_id}') and report."
    )
    return f"Started {job_id} ({label}); I'll be reminded to check it shortly."


async def _check_job(job_id: str) -> str:
    _CHECKED.append(job_id)
    job = _JOBS.get(job_id)
    if job is None:
        return f"No such job {job_id}."
    return f"{job_id}: status={job['status']} result={job['result']!r}"


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="start_job",
            description="Begin a long background job; you'll be reminded to check its result.",
            input_schema={"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]},
            required_params=("label",),
        ),
        _start_job,
    )
    reg.register(
        ToolDefinition(
            name="check_job",
            description="Check a job's status/result by its id.",
            input_schema={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
            required_params=("job_id",),
        ),
        _check_job,
    )
    return reg


async def main() -> None:
    with TemporaryDirectory() as tmp:
        loop = StatefulAgentLoop(
            llm=make_llm(),
            db_path=f"{tmp}/sessions.db",
            config=AgentLoopConfig(
                system_prompt=(
                    "You can run long jobs. Call start_job to begin one and tell the user. When you "
                    "are later reminded that a job should be done, call check_job and report the result."
                ),
                max_rounds=6,
            ),
            tool_registry=_build_registry(),
        )
        sid = await loop.new_session()

        # 1) The agent kicks off the job (returns immediately) — no result yet.
        r1 = await loop.send("Please run a long job labelled 'nightly-report'.", session_id=sid)
        print("after start:", r1.final_text[:100])
        print("armed timers:", [(t.timer_id, t.note[:40]) for t in await loop.list_timers(sid)])

        # 2) Drive the wake: the timer fires (~5s) → follow_up → the agent takes another round,
        #    calls check_job, and reports. No daemon — the host runs the TimerRunner.
        runner = TimerRunner(loop, scan_interval_s=1.0)
        await runner.start()
        try:
            await asyncio.sleep(9)
        finally:
            await runner.stop()

        msgs = await loop.get_messages(sid)
        last = next((m for m in reversed(msgs) if m.get("role") == "assistant" and m.get("content")), None)
        print("after wake-up:", (str(last.get("content"))[:140] if last else "(none)"))
        await loop.aclose()
        return {
            "job_started": bool(_JOBS),                               # the agent kicked off the job
            "job_done": any(j["status"] == "done" for j in _JOBS.values()),  # async work finished
            "checked_after_wake": bool(_CHECKED),                     # woke → re-checked (no daemon)
        }


if __name__ == "__main__":
    asyncio.run(main())
