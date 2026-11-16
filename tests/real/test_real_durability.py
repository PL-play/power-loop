"""Real-LLM end-to-end durability check: projection fold, export, reopen and import.

One lifecycle against a real model and a real on-disk file:

* a few sends trigger a projection fold (the loop still answers correctly),
* export the session,
* graceful ``aclose()`` of the owned file store,
* REOPEN the file (the schema gate stamps/keeps the version) and the session survives,
* import the export into a brand-new store and a real send works there too.

In-place compaction / ``prune_compacted_messages`` no longer exists: finished history is a
derived projection and the raw audit remains immutable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, SessionStore, TargetRef
from power_loop.runtime.fold import LLMSummaryFold
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.schema import CURRENT_SCHEMA_VERSION

from ._llm import make_llm

_SP = "Answer the user's latest question concisely in English."


def _config() -> AgentLoopConfig:
    return AgentLoopConfig(
        system_prompt=_SP, max_rounds=1, max_tokens=256, temperature=0,
        representation=ProjectedRepresentation(),
        fold_strategy=LLMSummaryFold(keep_last_sends=1, trigger_ratio=1.0),
        context_budget_tokens=1,  # threshold 0 → the fold fires as soon as a send is foldable
    )


async def _loop(runtime_agent, db_path, *, target=None):
    store = await SessionStore.open(db_path)
    return await runtime_agent(
        model=make_llm(max_tokens=256, temperature=0), store=store, target=target,
        config=_config(),
    )


@pytest.mark.asyncio
async def test_durability_lifecycle_with_real_llm(tmp_path: Path, runtime_agent) -> None:
    db = str(tmp_path / "dur.db")

    loop = await _loop(runtime_agent, db)
    sid = loop.target.session_id
    assert (await loop.send("Say OK.")).status == "completed"
    assert (await loop.send("Say OK again.")).status == "completed"
    r1 = await loop.send("Name the largest planet in our solar system.")
    assert r1.status == "completed"
    assert "jupiter" in (r1.final_text or "").lower(), r1.final_text
    assert await loop.runtime.store.latest_project_compact(sid) is not None  # fold fired

    blob = await loop.runtime.store.export_session(sid)
    await loop.aclose()
    await loop.runtime.store.close()

    # ── REOPEN the file + a real follow-up ────────────────────────────────────
    loop2 = await _loop(runtime_agent, db, target=loop.target)
    store2 = loop2.runtime.store
    assert store2.schema_version == CURRENT_SCHEMA_VERSION
    assert await store2.get_session(sid) is not None
    r2 = await loop2.send("Which planet is known as the Red Planet?", message_id="after-reopen")
    assert r2.status == "completed"
    assert "mars" in (r2.final_text or "").lower(), r2.final_text
    await loop2.aclose()
    await store2.close()

    # ── import the export into a fresh store + a real send ────────────────────
    other_store = await SessionStore.open(str(tmp_path / "restored.db"))
    new_sid = await other_store.import_session(blob, new_session_id="restored")
    other = await runtime_agent(
        model=make_llm(max_tokens=256, temperature=0), store=other_store,
        target=TargetRef(other_store.store_id, new_sid),
        config=AgentLoopConfig(system_prompt=_SP, max_tokens=256,
                               representation=ProjectedRepresentation(),
                               fold_strategy=LLMSummaryFold(keep_last_sends=1, trigger_ratio=1.0),
                               context_budget_tokens=1),
    )
    assert await other_store.load_project_messages(new_sid)
    r3 = await other.send("In one word, what is the largest planet in the solar system?")
    assert r3.status == "completed"
    assert "jupiter" in (r3.final_text or "").lower(), r3.final_text
    await other.aclose()
    await other_store.close()