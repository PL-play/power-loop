"""34 · 持久化生命周期 / Durability lifecycle (design 216 projection-only)

What you learn / 你将学到
--------------------------
- 长期存活的磁盘 `SessionStore` 是生产可运维的: **导出/导入**整个会话、**优雅停机**。
  / The on-disk SessionStore supports full-session **export/import** and graceful shutdown.
- 已结束的 send 只通过投影 (`pl_project_messages`) 进入模型; 原始终审 `pl_messages` 不变。
  折叠只产生一个 `compact` 投影行, 不删原文。
  / Finished sends reach the model only through their projection; the raw audit stays immutable.
  A fold writes one `compact` projection row and deletes nothing from `pl_messages`.
- `export_session` → `import_session` 无损搬运一个会话(含 compact 投影);
  `async with loop:` 触发 `aclose()`(排空在飞 send、drain 事件、checkpoint 后关库)。
  / export→import is lossless; ``async with`` runs aclose().

Run / 运行
----------
    python examples/34_durability_lifecycle.py
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from _runtime import demo_runtime

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    LLMSummaryFold,
    ProjectedRepresentation,
    SessionStore,
)


class PlanetModel(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text='Jupiter')

    async def close(self):
        pass


async def main() -> str:
    tmp = Path(tempfile.mkdtemp(prefix="pl_durability_"))
    db = str(tmp / "agent.db")
    store = await SessionStore.open(db)
    cfg = AgentLoopConfig(
        system_prompt="Answer the user's latest question concisely.",
        max_rounds=1, max_tokens=400,
        representation=ProjectedRepresentation(),
        # Tiny fold budget → the fold fires after the first foldable sends.
        fold_strategy=LLMSummaryFold(keep_last_sends=1, trigger_ratio=1.0),
        context_budget_tokens=1,
    )

    async with demo_runtime(PlanetModel(), store=store, config=cfg) as (runtime, loop, target, actor):
        sid = target.session_id
        for index in range(3):
            result = await loop.send(
                target, "context " + "u" * 300,
                message_id=f"history-{index}", actor=actor,
                deadline_ms=int(time.time() * 1000) + 10000,
            )
            assert result.status == "completed"
        result = await loop.send(
            target, "Name the largest planet in our solar system.",
            message_id="planet", actor=actor,
            deadline_ms=int(time.time() * 1000) + 10000,
        )
        assert result.status == "completed"

        compact = await store.latest_project_compact(sid)
        assert compact is not None, "the configured fold must produce a compact projection"
        projection_rows = await store.load_project_messages(sid)
        raw_rows = await store.load_all_messages(sid)
        blob = await store.export_session(sid)
    # ← async-with exit ran aclose(): drained the send, checkpointed, closed the store.

    other = await SessionStore.open(str(tmp / "restored.db"))
    try:
        new_sid = await other.import_session(blob, new_session_id="restored")
        restored_compact = await other.latest_project_compact(new_sid)
        restored_projection = await other.load_project_messages(new_sid)
        restored_raw = await other.load_all_messages(new_sid)
    finally:
        await other.close()

    assert restored_compact is not None and restored_compact.send_index == compact.send_index
    assert len(restored_projection) == len(projection_rows)
    assert len(restored_raw) == len(raw_rows)
    summary = (
        f"answer={result.final_text!r} | projection_rows={len(projection_rows)} | "
        f"compact_to_send={compact.compact_to_send} | raw_rows={len(raw_rows)} | "
        f"reimported_projection={len(restored_projection)}"
    )
    print(summary)
    return summary


if __name__ == "__main__":
    asyncio.run(main())