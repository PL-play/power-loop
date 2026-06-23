"""31 · 记忆 + 压缩共存 / Memory recall + compaction together (H1.1 regression)

What you learn / 你将学到
--------------------------
- 召回由内置的 ``MemoryRecallHook``（``LLM_BEFORE`` hook）完成：把召回的 ``memory_*``
  消息**临时拼到请求末尾**（不进 ``self.history``、不落库），**天然躲过压缩折叠**；
  同一会话里长历史照常触发 ``DefaultCompactor``，两者互不干扰。
  / recall is done by the built-in ``MemoryRecallHook`` (an ``LLM_BEFORE`` hook): it
  appends the recalled ``memory_*`` messages EPHEMERALLY at the request TAIL (never
  into ``self.history`` or the store), so they are NOT folded; a long history still
  triggers ``DefaultCompactor`` in the same session — the two coexist cleanly.
- 召回的记忆**永不落库**（MemoryProvider 自己拥有它），所以 store 里不会出现
  ``name=memory_*`` 的行；而压缩会精确地把**真实对话行**标 ``compacted_out``。
  / recalled memory is NEVER persisted (the provider owns it), so the store has no
  ``name=memory_*`` rows; compaction marks exactly the REAL conversation rows
  ``compacted_out``.
- 因为记忆从不进入 history，压缩只看真实持久行——无需任何 index↔seq 对齐补偿
  （3.1 起，旧的前端插入 + ``on_messages_inserted`` 路径已移除）。
  / because memory never enters history, compaction sees only real persisted rows —
  no index↔seq realignment needed (3.1 removed the old front-splice +
  ``on_messages_inserted`` path).

Run / 运行
----------
    python examples/31_memory_with_compaction.py
"""

from __future__ import annotations

import asyncio
import os

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    MessageState,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.runtime.compact import DefaultCompactor


class StaticMemory:
    """A trivial MemoryProvider that always recalls the same two facts and never
    persists. ``recall`` returns plain messages; the framework auto-tags them as
    ``role=system, name=memory_*``."""

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        return [
            {"content": "Known fact: the user's name is Alan."},
            {"content": "Known fact: the user's favorite number is 37."},
        ]

    async def remember(self, *, snapshot, session_id):
        return None  # this demo's memory is static


async def _seed_fat_history(store: SessionStore, sid: str, turns: int = 4) -> None:
    for i in range(turns):
        await store.append_message(sid, role="user", content="filler " + ("u" * 400), round_index=i)
        await store.append_message(
            sid, role="assistant", content="filler ack " + ("a" * 400), round_index=i
        )


async def main() -> str:
    os.environ["CONTEXT_COMPACT_THRESHOLD"] = "500"  # force compaction every round

    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session(system_prompt="S")
        await _seed_fat_history(store, sid, turns=4)

        loop = StatefulAgentLoop(
            llm=make_llm(),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "Earlier turns may be summarized into a compact_note. Some facts "
                    "about the user are provided in the system region. Answer concisely."
                ),
                max_rounds=1,
                compactor=DefaultCompactor(trigger_ratio=0.5, keep_last_n=1),
                memory=StaticMemory(),
            ),
        )
        r = await loop.send(
            "Using what you know about me, what is my name and favorite number?",
            session_id=sid,
        )
        print(f"status   : {r.status}, rounds: {r.rounds}")
        print(f"reply    : {r.final_text}")

        # ── invariants the C1 fix guarantees ──
        all_rows = await store.load_all_messages(sid)
        memory_rows = [m for m in all_rows if (m.name or "").startswith("memory_")]
        folded = {m.seq for m in all_rows if m.state is MessageState.COMPACTED_OUT}
        active = {m.seq for m in all_rows if m.state is MessageState.ACTIVE}
        comps = await store.list_compactions(sid)

        print(f"compactions recorded : {len(comps)}")
        print(f"recalled memory rows persisted : {len(memory_rows)} (expect 0)")
        print(f"rows compacted_out   : {sorted(folded)}")

        assert not memory_rows, "recalled memory must never be persisted as session rows"
        assert comps, "the fat history should have compacted"
        assert folded and folded.isdisjoint(active), "compacted rows must be a clean, disjoint set"
        return r.final_text
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
