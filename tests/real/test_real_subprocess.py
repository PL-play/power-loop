"""Real-LLM Phase 1b gate: a leaf actually runs a real model in a subprocess.

Proves the production path end-to-end: the orchestrator drives the unchanged
WorkflowEngine with a SubprocessExecutor; each leaf is a spawned
`python -m power_loop.workflow.worker` that rebuilds the real provider from env
(WorkerBootstrap(llm_from_env=True)) and runs in its own db.

Skipped when no provider env is set (or with --no-real). See tests/conftest.py.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from power_loop import (
    AgentLoopConfig,
    SessionStore,
    StatefulAgentLoop,
    create_llm_service_from_env,
)
from power_loop.workflow import SubprocessExecutor, WorkerBootstrap, WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine

pytestmark = pytest.mark.real_llm


async def test_leaf_runs_real_model_in_subprocess() -> None:
    runs = tempfile.mkdtemp(prefix="wf_real_")
    ex = SubprocessExecutor(
        bootstrap=WorkerBootstrap(llm_from_env=True),  # child rebuilds the real provider from env
        runs_dir=runs,
        timeout_s=120,
    )
    loop = StatefulAgentLoop(
        llm=create_llm_service_from_env(), db_path=":memory:",
        config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=3),
    )
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "France",
        "root": {"type": "agent", "id": "q",
                 "spec": {"name": "q", "system_prompt": "Answer in one short sentence."},
                 "input": "What is the capital of {{input}}?"},
    })

    res = await WorkflowEngine(loop, executor=ex, run_id="real1").run(spec)

    assert res.status == "completed", res.errors
    leaf = res.results["q"]
    assert "paris" in leaf.text.lower()
    # The leaf ran in its own db file, inspectable afterward.
    db_path = leaf.session_id and next(iter(os.listdir(os.path.join(runs, "real1"))), None)
    assert db_path is not None
    store = SessionStore.open(os.path.join(runs, "real1", db_path))
    try:
        assert any(m.role == "assistant" for m in store.load_all_messages(leaf.session_id))
    finally:
        store.close()
