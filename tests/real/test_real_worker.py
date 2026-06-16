"""Real-LLM Phase 0 gate: the isolated worker core on the *serializable* path.

The factory path is covered by unit tests; this proves the path a real subprocess
would take — ``WorkerBootstrap(llm_from_env=True)`` rebuilds the provider from
environment config alone (no parent objects), runs an agent to completion in its
own db, and that db is inspectable afterward.

Skipped when no provider env is set (or with --no-real). See tests/conftest.py.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from power_loop import SessionStore
from power_loop.workflow import WorkerBootstrap, run_spec_isolated

pytestmark = pytest.mark.real_llm


async def test_isolated_worker_real_llm_from_env() -> None:
    db = tempfile.mktemp(suffix=".db")
    boot = WorkerBootstrap(llm_from_env=True)  # serializable: rebuilds provider from env

    result = await run_spec_isolated(
        {"name": "leaf", "system_prompt": "Answer in one short sentence."},
        "Name the capital of France.",
        bootstrap=boot,
        db_path=db,
    )

    assert result["status"] == "completed", result
    assert "paris" in result["final_text"].lower()
    assert result["usage"].get("total_tokens", 0) > 0  # real provider was used

    # The private ledger is inspectable after the fact.
    store = await SessionStore.open(db)
    try:
        msgs = await store.load_all_messages(result["session_id"])
        assert any(m.role == "assistant" for m in msgs)
    finally:
        await store.close()
    os.remove(db)
