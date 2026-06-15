"""35 · 扩展与读连接池 / Scaling & the read pool (0.16.0 — SCALE-1/2/3)

What you learn / 你将学到
--------------------------
- power-loop 是单进程内核:一个写者(单连接+RLock 串行,保证 next_seq 与写原子),一个事件循环
  可驱动任意多并发会话。读路径已卸载出事件循环,读可用**只读连接池**与写者并发。
  / Single-process kernel: one writer (single connection + RLock), one event loop driving
  many concurrent sessions; reads are offloaded off the loop and can run on a read pool.
- `SessionStore.open(read_pool_size=N)`(SCALE-2,文件库,opt-in)让读不再排在写锁后面;
  每次 send 的历史加载走 `to_thread`(SCALE-3),不阻塞其它会话。
  / ``read_pool_size=N`` keeps reads off the writer's lock; the per-send history load runs
  in a thread so it doesn't stall other sessions.
- 用自带压测台把"推理出的"上限变成"测出来的"(SCALE-1):``python -m bench [--smoke]``。
  / Turn the reasoned ceiling into measured numbers with the bundled harness.

Run / 运行
----------
    python examples/35_scaling_and_read_pool.py
    python -m bench --smoke        # the benchmark/soak harness (no provider needed)
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from _helpers import make_llm

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop


async def main() -> str:
    tmp = Path(tempfile.mkdtemp(prefix="pl_scaling_"))
    db = str(tmp / "agent.db")

    # SCALE-2: a read pool (file DB only; :memory: would decline it). Default is 0 (off);
    # enable it under read-heavy fan-out so reads run concurrently with the writer.
    store = SessionStore.open(db, read_pool_size=4)
    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=128),
        store=store,
        config=AgentLoopConfig(
            system_prompt="Answer in one short sentence.", max_rounds=1, max_tokens=128,
        ),
    )

    # One event loop, several concurrent sessions. Each send's history read is offloaded
    # (SCALE-3) and may use a pooled connection (SCALE-2) — they don't serialize on reads.
    questions = [
        "What is the capital of France?",
        "Name a primary color.",
        "What is 2 + 2?",
    ]
    async with loop:  # aclose() on exit: drain in-flight sends, checkpoint, close store
        sids = [loop.new_session() for _ in questions]
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(loop.send(q, session_id=sid) for q, sid in zip(questions, sids))
        )
        wall = time.perf_counter() - t0

    ok = sum(1 for r in results if r.status == "completed")
    summary = (
        f"{ok}/{len(questions)} concurrent sessions completed in {wall*1000:.0f} ms "
        f"(read_pool_size=4). For measured ceilings run: python -m bench --smoke"
    )
    print(summary)
    return summary


if __name__ == "__main__":
    asyncio.run(main())
