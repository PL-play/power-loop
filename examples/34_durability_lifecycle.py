"""34 · 持久化生命周期 / Durability lifecycle (0.15.0 — OPS-2..5)

What you learn / 你将学到
--------------------------
- 长期存活的磁盘 `SessionStore` 是生产可运维的:**保留/裁剪**折叠出的原文、**回收**磁盘、
  **导出/导入**整个会话、**优雅停机**。全部 opt-in、调用方驱动。
  / The on-disk SessionStore is operable for the long haul: opt-in **retention** of
  folded-out originals, disk **reclamation**, full-session **export/import**, and
  **graceful shutdown** — all caller-driven.
- `prune_compacted_messages` 只删 `compacted_out` 原文,绝不动 `compact_note`/active,
  所以活动窗口 byte 级不变;`vacuum()`/`checkpoint()` 真正缩小文件。
  / Pruning removes only the folded originals; the active window is byte-identical.
- `export_session` → `import_session` 无损搬运一个会话(含 compact_note 排序);
  `async with loop:` 触发 `aclose()`(排空在飞 send、drain 事件、checkpoint 后关库)。
  / export→import is lossless; ``async with`` runs aclose() (drain in-flight sends +
  pending events, checkpoint, close).

Run / 运行
----------
    python examples/34_durability_lifecycle.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from _helpers import make_llm

from power_loop import AgentLoopConfig, MessageState, SessionStore, StatefulAgentLoop
from power_loop.runtime.compact import DefaultCompactor


async def main() -> str:
    # Demo: a tiny compaction threshold so the seeded turns fold (real apps tune this
    # via max_tokens / CONTEXT_COMPACT_THRESHOLD). Folding is what creates the
    # compacted_out originals that retention then reclaims.
    os.environ["CONTEXT_COMPACT_THRESHOLD"] = "400"
    tmp = Path(tempfile.mkdtemp(prefix="pl_durability_"))
    db = str(tmp / "agent.db")

    # A read pool (SCALE-2) is opt-in and harmless here — shows the knob.
    store = SessionStore.open(db, read_pool_size=2)
    cfg = AgentLoopConfig(
        system_prompt="Answer the user's latest question concisely.",
        max_rounds=1, max_tokens=400,
        compactor=DefaultCompactor(trigger_ratio=0.5, keep_last_n=1),
    )

    pruned = 0
    async with StatefulAgentLoop(llm=make_llm(max_tokens=200), store=store, config=cfg) as loop:
        sid = loop.new_session()
        # Seed several fat earlier turns the compactor must fold on the next send
        # (big enough that reclaiming them visibly shrinks the file).
        for i in range(12):
            store.append_message(sid, role="user", content="context " + "u" * 3000, round_index=i)
            store.append_message(sid, role="assistant", content="noted " + "a" * 3000, round_index=i)

        result = await loop.send("Name the largest planet in our solar system.", session_id=sid)
        assert result.status == "completed"

        compactions = store.list_compactions(sid)
        active_before = [(m.role, m.name) for m in store.load_active_messages(sid)]

        # OPS-4: ARCHIVE the full session first (incl. the folded-out originals) — so the
        # export is complete; then reclaim locally. This is the "archive then prune" order.
        blob = store.export_session(sid)

        # OPS-2/3: reclaim the folded-out originals + shrink the file on disk.
        store.checkpoint(mode="TRUNCATE")
        size_before = Path(db).stat().st_size
        pruned = store.prune_compacted_messages(sid)
        store.vacuum(incremental=False)
        store.checkpoint(mode="TRUNCATE")  # flush the VACUUM's rewrite out of the WAL
        size_after = Path(db).stat().st_size

        # The active window (compact_note + kept tail) is untouched by pruning.
        assert [(m.role, m.name) for m in store.load_active_messages(sid)] == active_before
    # ← async-with exit ran aclose(): drained the send, checkpointed, closed the store.

    # OPS-4: import the (complete) archive into a brand-new store under a new id.
    other = SessionStore.open(str(tmp / "restored.db"))
    try:
        new_sid = other.import_session(blob, new_session_id="restored")
        restored_first = other.load_active_messages(new_sid)[0].name  # the compact_note, in order
        restored_compacted = sum(
            1 for m in other.load_all_messages(new_sid) if m.state is MessageState.COMPACTED_OUT
        )
    finally:
        other.close()

    summary = (
        f"answer={result.final_text!r} | compactions={len(compactions)} | "
        f"pruned_originals={pruned} | file {size_before}->{size_after} bytes | "
        f"reimported first-active={restored_first!r} compacted_kept={restored_compacted}"
    )
    print(summary)
    return summary


if __name__ == "__main__":
    asyncio.run(main())
