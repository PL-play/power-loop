"""35 · 扩展与并发会话 / Scaling & concurrent sessions (0.16.0 — SCALE-1/3)

What you learn / 你将学到
--------------------------
- power-loop 是单进程内核:一个写者(单连接 + 一把 asyncio.Lock 串行写,保证 next_seq 与写
  原子),一个事件循环可驱动任意多并发会话。store 的每条语句都卸载到工作线程执行,因此读不
  会阻塞事件循环。
  / Single-process kernel: one writer (a single connection serialized behind one
  ``asyncio.Lock``, so ``next_seq`` and the write stay atomic) and one event loop driving
  many concurrent sessions. Every store statement is offloaded to a worker thread, so reads
  never block the event loop.
- 异步 store 没有单独的「读连接池」可配:并发性来自语句级 ``to_thread`` 卸载 + 写者锁的细
  粒度持有,而不是预开一组连接。多个 ``loop.send`` 并发跑时,各自的历史加载互不串行化。
  / The async store has no separate read pool to configure: concurrency comes from per-statement
  ``to_thread`` offload plus a finely-held writer lock, not from a pre-opened pool of connections.
  Concurrent ``loop.send`` calls don't serialize on each other's history loads.
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

    # The async store offloads each statement to a worker thread and serializes writers
    # behind one asyncio.Lock — there is no read-pool knob; concurrency is built in.
    store = await SessionStore.open(db)
    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=128),
        store=store,
        config=AgentLoopConfig(
            system_prompt="Answer in one short sentence.", max_rounds=1, max_tokens=128,
        ),
    )

    # One event loop, several concurrent sessions. Each send's history read is offloaded
    # to a worker thread (SCALE-3), so the sessions don't serialize on reads.
    questions = [
        "What is the capital of France?",
        "Name a primary color.",
        "What is 2 + 2?",
    ]
    async with loop:  # aclose() on exit: drain in-flight sends, checkpoint, close store
        sids = [await loop.new_session() for _ in questions]
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(loop.send(q, session_id=sid) for q, sid in zip(questions, sids))
        )
        wall = time.perf_counter() - t0

    ok = sum(1 for r in results if r.status == "completed")
    summary = (
        f"{ok}/{len(questions)} concurrent sessions completed in {wall*1000:.0f} ms "
        f"on the async store. For measured ceilings run: python -m bench --smoke"
    )
    print(summary)
    return summary


if __name__ == "__main__":
    asyncio.run(main())
