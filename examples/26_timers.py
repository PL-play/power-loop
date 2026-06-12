"""26 · 持久定时唤醒 / Durable timers: schedule_wakeup + TimerRunner

What you learn / 你将学到
--------------------------
- ``schedule_wakeup`` / ``list_wakeups`` / ``cancel_wakeup`` / ``current_time``
  默认工具：模型给**自己**排闹钟，到点收到自己的 note
  / the agent schedules its own durable wake-ups
- ``loop.schedule_timer(...)``：外部（编排层）也能给 session 排定时
  / the external path writes the same rows
- ``TimerRunner``：扫描 store、到点把 note 经 ``follow_up`` 注入会话——
  空闲 = 正常 send，运行中 = 轮边界插话，**进会话只有一条路**
- ``HookPoint.TIMER_FIRE``：投递前的否决点（busy 判断 / 去重 / 改期 / 取消），
  无 hook 默认直接投递

设计要点 / Design notes
-----------------------
- **timer 是数据不是任务**：存 SQLite，跨重启存活；进程内只是扫描加速器
- **at-least-once**：崩在投递中途的 timer 会被重新 arm，可能投两次——
  需要严格一次就在 TIMER_FIRE hook 里按 timer_id 去重
- 没有 TimerRunner（或外部调度器轮询 ``store.due_timers()``）时 timer 永远
  不会触发——宿主必须明确选择由谁来"醒"

Run / 运行
----------
    python examples/26_timers.py        # 需要 .env 配置 POWER_LOOP_*
"""

from __future__ import annotations

import asyncio
from tempfile import TemporaryDirectory

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    HookPoint,
    StatefulAgentLoop,
    TimerFireCtx,
    TimerRunner,
    create_default_tool_registry,
    get_tool_definitions,
)
from power_loop.core.hooks import AgentHooks
from power_loop.tools import DEFAULT_TOOL_HANDLERS


async def main() -> None:
    llm = make_llm()

    registry = create_default_tool_registry(preset="core", bind=False)
    for d in get_tool_definitions(
        include=["schedule_wakeup", "list_wakeups", "cancel_wakeup", "current_time"]
    ):
        registry.register(d, DEFAULT_TOOL_HANDLERS[d.name], overwrite=True)

    # 可选：投递前的编排否决点。这里只做审计日志（CONTINUE = 照常投递）。
    hooks = AgentHooks()

    def audit(ctx: TimerFireCtx) -> None:
        print(f"  [TIMER_FIRE hook] #{ctx.timer_id} note={ctx.note!r} → deliver")

    hooks.register(HookPoint.TIMER_FIRE, audit)

    with TemporaryDirectory() as tmp:
        loop = StatefulAgentLoop(
            llm=llm,
            db_path=f"{tmp}/sessions.db",
            config=AgentLoopConfig(
                system_prompt=(
                    "你是一个会自我管理时间的助手。需要稍后做某事时，用 "
                    "schedule_wakeup 给自己排一个唤醒，note 写清楚届时要做什么。"
                ),
                max_rounds=6,
            ),
            tool_registry=registry,
            hooks=hooks,
        )
        sid = loop.new_session()

        # 1) 模型自己排闹钟（也可以用 loop.schedule_timer 从外部排）。
        res = await loop.send(
            "请在 6 秒后提醒自己向我汇报『定时演示完成』，排好后告诉我闹钟编号。",
            session_id=sid,
        )
        print(f"reply: {res.final_text[:80]}")
        print("armed timers:", [(t.timer_id, t.note) for t in loop.list_timers(sid)])

        # 2) 启动 TimerRunner，等闹钟响。
        runner = TimerRunner(loop, scan_interval_s=1.0)
        await runner.start()
        try:
            await asyncio.sleep(9)  # 闹钟 6s 后到期 → 扫描捕获 → follow_up 进会话
        finally:
            await runner.stop()

        # 3) 看看醒来后它说了什么。
        msgs = loop.get_messages(sid)
        last_assistant = next(
            (m for m in reversed(msgs) if m.get("role") == "assistant" and m.get("content")), None
        )
        print(f"after wake-up: {str(last_assistant.get('content'))[:100] if last_assistant else '(none)'}")
        print("timer status:", loop.store.get_timer(sid, 1).status)

        loop.close()


if __name__ == "__main__":
    asyncio.run(main())
