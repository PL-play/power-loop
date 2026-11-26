"""34 · 持久化生命周期 / Durability lifecycle (close, reopen, read original session)

What you learn / 你将学到
--------------------------
- 磁盘 `SessionStore` 支持优雅关闭并在进程/实例重启后重新打开同一数据库文件，按原会话 ID 读取完整状态。
  / The on-disk SessionStore supports graceful shutdown and reopening the same DB file
  to read the original session by its session ID.
- 已结束的 send 只通过投影 (`pl_project_messages`) 进入模型；原始审计表 `pl_messages` 保持完整。
  折叠只追加 `compact` 投影行，不删除历史原文。
  / Finished sends reach the model through projections; the raw audit stays intact.
  A fold writes a ``compact`` projection row without deleting rows from ``pl_messages``.
- 退出 `async with demo_runtime(...)` 会触发运行时与存储的正常关闭 (`aclose()` 排空在飞 send 并关闭数据库)；
  随后用新的 `SessionStore.open(db)` 重开同一磁盘文件，核对摘要覆盖范围、原始消息与投影内容完全一致。
  / Exiting ``async with demo_runtime(...)`` gracefully closes the runtime and store;
  reopening the same disk file with a new ``SessionStore`` verifies the compact span,
  raw messages, and projections match what was written before shutdown.

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
        return LLMResponse(raw_text="Jupiter")

    async def close(self):
        pass


async def main() -> str:
    with tempfile.TemporaryDirectory(prefix="pl_durability_") as tmpdir:
        db = str(Path(tmpdir) / "agent.db")
        store = await SessionStore.open(db)
        cfg = AgentLoopConfig(
            system_prompt="Answer the user's latest question concisely.",
            max_rounds=1,
            max_tokens=400,
            representation=ProjectedRepresentation(),
            # Tiny fold budget → the fold fires after the first foldable sends.
            fold_strategy=LLMSummaryFold(keep_last_sends=1, trigger_ratio=1.0),
            context_budget_tokens=1,
        )

        async with demo_runtime(PlanetModel(), store=store, config=cfg) as (runtime, loop, target, actor):
            sid = target.session_id
            for index in range(3):
                result = await loop.send(
                    target,
                    "context " + "u" * 300,
                    message_id=f"history-{index}",
                    actor=actor,
                    deadline_ms=int(time.time() * 1000) + 10000,
                )
                assert result.status == "completed"
            result = await loop.send(
                target,
                "Name the largest planet in our solar system.",
                message_id="planet",
                actor=actor,
                deadline_ms=int(time.time() * 1000) + 10000,
            )
            assert result.status == "completed" and result.final_text == "Jupiter"

            compact = await store.latest_project_compact(sid)
            assert compact is not None, "the configured fold must produce a compact projection"
            projection_rows = await store.load_project_messages(sid)
            raw_rows = await store.load_all_messages(sid)

            compact_snapshot = (
                compact.send_index,
                compact.compact_from_send,
                compact.compact_to_send,
                compact.content,
                compact.rendered_text,
            )
            projection_snapshot = [
                (r.send_index, r.kind, r.compact_from_send, r.compact_to_send, r.content, r.rendered_text)
                for r in projection_rows
            ]
            raw_snapshot = [(r.seq, r.send_index, r.role, r.content) for r in raw_rows]
        # ← Exiting demo_runtime ran runtime.aclose(), draining in-flight work and closing store.

        # Reopen the same on-disk database file with a fresh SessionStore and read by original sid.
        reopened = await SessionStore.open(db)
        try:
            reopened_session = await reopened.get_session(sid)
            reopened_compact = await reopened.latest_project_compact(sid)
            reopened_projection = await reopened.load_project_messages(sid)
            reopened_raw = await reopened.load_all_messages(sid)
        finally:
            await reopened.close()

        assert reopened_session is not None and reopened_session.session_id == sid
        assert reopened_compact is not None
        assert (
            reopened_compact.send_index,
            reopened_compact.compact_from_send,
            reopened_compact.compact_to_send,
            reopened_compact.content,
            reopened_compact.rendered_text,
        ) == compact_snapshot
        assert [
            (r.send_index, r.kind, r.compact_from_send, r.compact_to_send, r.content, r.rendered_text)
            for r in reopened_projection
        ] == projection_snapshot
        assert [(r.seq, r.send_index, r.role, r.content) for r in reopened_raw] == raw_snapshot
        assert reopened_raw[-1].role == "assistant" and reopened_raw[-1].content == "Jupiter"
        assert reopened_projection[-1].kind == "project"
        assert reopened_projection[-1].content.get("final_text") == "Jupiter"

        summary = (
            f"session_id={sid} | answer={result.final_text!r} | "
            f"compact_span={reopened_compact.compact_from_send}..{reopened_compact.compact_to_send} | "
            f"reopened_raw={len(reopened_raw)} | reopened_projection={len(reopened_projection)}"
        )
        print(summary)
        return summary


if __name__ == "__main__":
    asyncio.run(main())
