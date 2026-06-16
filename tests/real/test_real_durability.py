"""Real-LLM end-to-end check of the 0.15.0 durability phase (OPS-1..OPS-5).

One lifecycle against a real model and a real on-disk file:

* a send that triggers compaction (the loop still answers correctly),
* OPS-2/OPS-3: prune the folded originals + checkpoint + full VACUUM — the active
  window is byte-identical and the file shrinks,
* OPS-4: export the session,
* OPS-5: graceful ``aclose()`` of the owned file store,
* OPS-1: REOPEN the file (the migration gate stamps/keeps the schema version) and the
  session survives — a real follow-up still answers correctly,
* OPS-4: import the export into a brand-new store and a real send works there too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from power_loop import (
    AgentLoopConfig,
    MessageState,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.runtime.compact import DefaultCompactor
from power_loop.runtime.session_store import CURRENT_SCHEMA_VERSION

from ._llm import make_llm

_SP = (
    "Earlier turns may have been summarized into a compact_note. "
    "Answer the user's latest question concisely in English."
)


def _loop(db_path: str) -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=make_llm(max_tokens=256, temperature=0),
        db_path=db_path,
        config=AgentLoopConfig(
            system_prompt=_SP, max_rounds=1, max_tokens=256, temperature=0,
            compactor=DefaultCompactor(trigger_ratio=0.5, keep_last_n=1),
        ),
    )


async def _seed_fat_history(store: SessionStore, sid: str) -> None:
    for i in range(4):
        await store.append_message(sid, role="user", content="filler " + "u" * 400, round_index=i)
        await store.append_message(
            sid, role="assistant", content="filler ack " + "a" * 400, round_index=i
        )


@pytest.mark.asyncio
async def test_durability_lifecycle_with_real_llm(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "500")  # force compaction on the fat seed
    db = str(tmp_path / "dur.db")

    # ── phase 1: real send → compaction → prune + vacuum (file store) ──────────
    loop = _loop(db)
    sid = await loop.new_session()
    await _seed_fat_history(loop.store, sid)

    r1 = await loop.send("Name the largest planet in our solar system.", session_id=sid)
    assert r1.status == "completed"
    assert "jupiter" in (r1.final_text or "").lower(), r1.final_text
    assert await loop.store.list_compactions(sid), "compaction should have fired on the fat seed"
    assert any(
        m.state is MessageState.COMPACTED_OUT for m in await loop.store.load_all_messages(sid)
    )

    active_before = [
        (m.role, m.name, m.content) for m in await loop.store.load_active_messages(sid)
    ]
    await loop.store.checkpoint(mode="TRUNCATE")
    size_before = Path(db).stat().st_size
    pruned = await loop.store.prune_compacted_messages(sid)
    assert pruned > 0
    await loop.store.vacuum(incremental=False)
    # active window (compact_note + kept tail) is untouched by pruning the originals
    assert [
        (m.role, m.name, m.content) for m in await loop.store.load_active_messages(sid)
    ] == active_before
    assert Path(db).stat().st_size <= size_before  # reclaimed (or at worst unchanged)

    blob = await loop.store.export_session(sid)  # OPS-4 export before shutdown
    await loop.aclose()                          # OPS-5 graceful shutdown of the owned store

    # ── phase 2: REOPEN the file (migration gate) + a real follow-up ──────────
    loop2 = _loop(db)
    store2 = await loop2._ensure_store()
    assert store2.schema_version == CURRENT_SCHEMA_VERSION
    assert await store2.get_session(sid) is not None  # survived reopen
    r2 = await loop2.send("Which planet is known as the Red Planet?", session_id=sid)
    assert r2.status == "completed"
    assert "mars" in (r2.final_text or "").lower(), r2.final_text
    await loop2.aclose()

    # ── phase 3: import the export into a fresh store + a real send ───────────
    other = _loop(str(tmp_path / "restored.db"))
    other_store = await other._ensure_store()
    new_sid = await other_store.import_session(blob, new_session_id="restored")
    assert (await other_store.load_active_messages(new_sid))[0].name == "compact_note"  # ordering
    r3 = await other.send("In one word, what is the largest planet in the solar system?", session_id=new_sid)
    assert r3.status == "completed"
    assert "jupiter" in (r3.final_text or "").lower(), r3.final_text
    await other.aclose()
