"""33 · 联动记忆的压缩器 / Memory-coordinating compactor (H7 Phase 2)

What you learn / 你将学到
--------------------------
- ``Compactor.maybe_compact`` 现在可选接收 ``context: CompactionContext``,暴露
  **注入的** ``MemoryProvider`` + session_id + 只读读取器。自定义压缩器可以在折叠
  **前**把要点 ``remember`` 进记忆,跨 session 留存。
  / ``maybe_compact`` optionally receives ``context`` (the injected MemoryProvider,
  session_id, a read accessor). A custom compactor can ``remember`` must-keep detail
  BEFORE folding, so it survives across sessions.
- ``DefaultCompactor`` 行为不变;旧签名的压缩器照常工作(pipeline 按签名判断是否传
  context)。本例只是**在库的现有 seam 上**做联动,记忆后端仍由调用方注入。
  / ``DefaultCompactor`` is unchanged; old-signature compactors still work (the
  pipeline only passes context to compactors that accept it). Coordination is built
  on the existing seam; the memory backend stays injected.

Run / 运行
----------
    python examples/33_coordinating_compactor.py
"""

from __future__ import annotations

import asyncio
import os
import re

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    MemorySnapshot,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.runtime.compact import DefaultCompactor

CODENAME = "BLUEHERON"


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class CapturingMemory:
    """A tiny cross-session memory. ``remember`` with status='compaction' captures
    the about-to-be-folded slice; ``recall`` injects captured snippets back."""

    def __init__(self) -> None:
        self.captured: list[str] = []

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        if not self.captured:
            return []
        return [{"content": "Context captured from earlier (now-compacted) turns:\n" + "\n".join(self.captured)}]

    async def remember(self, *, snapshot: MemorySnapshot, session_id):
        if snapshot.status == "compaction":
            text = " ".join((m.get("content") or "") for m in snapshot.messages)
            self.captured.append(text[:2000])


class CoordinatingCompactor(DefaultCompactor):
    """On every fold, persist the folded slice into the injected memory so its
    detail outlives the active window — then fold as usual."""

    async def maybe_compact(self, messages, *, llm, max_tokens, round_index, context=None):
        plan = await super().maybe_compact(
            messages, llm=llm, max_tokens=max_tokens, round_index=round_index, context=context)
        if plan and context and context.memory:
            slice_ = messages[plan.fold_start_idx : plan.fold_end_idx + 1]
            await context.memory.remember(
                snapshot=MemorySnapshot(
                    session_id=context.session_id or "", messages=slice_, status="compaction"),
                session_id=context.session_id,
            )
        return plan


def _seed_with_codename(store: SessionStore, sid: str) -> None:
    store.append_message(
        sid, role="user",
        content=f"For the record, the project codename is {CODENAME}. " + ("context " * 80),
        round_index=0,
    )
    store.append_message(sid, role="assistant", content="Noted. " + ("ok " * 80), round_index=0)
    for i in range(1, 5):
        store.append_message(sid, role="user", content="(work) " + ("filler " * 120), round_index=i)
        store.append_message(sid, role="assistant", content="done " + ("ok " * 120), round_index=i)


async def main() -> dict:
    os.environ["CONTEXT_COMPACT_THRESHOLD"] = "500"
    memory = CapturingMemory()
    store = SessionStore.open(":memory:")
    try:
        # ── Session A: long convo → compaction → compactor captures the slice ──
        sid_a = store.create_session(system_prompt="S")
        _seed_with_codename(store, sid_a)
        loop_a = StatefulAgentLoop(
            llm=make_llm(max_tokens=200), store=store,
            config=AgentLoopConfig(
                system_prompt="Answer concisely.", max_rounds=1,
                compactor=CoordinatingCompactor(trigger_ratio=0.5, keep_last_n=1, summary_max_tokens=40),
                memory=memory,
            ),
        )
        await loop_a.send("Continue the project work.", session_id=sid_a)
        captured_has_codename = any(CODENAME in c for c in memory.captured)
        print(f"[A] captured snippets: {len(memory.captured)}; has codename: {captured_has_codename}")

        # ── Session B: brand-new session — the codename comes back via recall ──
        sid_b = store.create_session(system_prompt="S")
        loop_b = StatefulAgentLoop(
            llm=make_llm(max_tokens=200), store=store,
            config=AgentLoopConfig(
                system_prompt="Answer concisely using what you know.", max_rounds=1,
                compactor=None, memory=memory,
            ),
        )
        r = await loop_b.send("What is the project codename?", session_id=sid_b)
        answer_has_codename = _norm(CODENAME) in _norm(r.final_text)
        print(f"[B] reply: {r.final_text}")
        print(f"[B] answer has codename: {answer_has_codename}")
        assert sid_b != sid_a
        return {
            "captured_has_codename": captured_has_codename,
            "final_text": r.final_text,
            "answer_has_codename": answer_has_codename,
        }
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
