"""G1 regression: the workflow journal's read-modify-write must be atomic.

The async store's ``get_runtime_state`` / ``set_runtime_state`` yield the event loop, so the
pre-fix ``journal.record_step`` (read → … → write) lost updates when the engine recorded
steps from parallel branches of one run concurrently, and ``_append_index`` lost run_ids
when ``seed`` ran concurrently for several runs. Both now funnel through the store's atomic
``mutate_runtime_state``. These tests FAIL against the pre-fix get+set implementation.
"""

from __future__ import annotations

import asyncio

import pytest

from power_loop.runtime.store.store import SessionStore
from power_loop.workflow import journal
from power_loop.workflow.result import WorkflowResult

pytestmark = pytest.mark.unit


async def test_concurrent_record_step_keeps_every_step() -> None:
    async with await SessionStore.open(":memory:") as store:
        parent = await store.create_session(system_prompt="parent")
        run_id = "run-A"
        await journal.seed(store, parent, run_id, "wf", spec={})

        n = 8  # mirrors a parallel node with 8 concurrent successful branches
        await asyncio.gather(*[
            journal.record_step(
                store, parent, run_id, node_id=f"leaf{i}", status="completed", text=f"out{i}"
            )
            for i in range(n)
        ])

        rec = await journal.read(store, parent, run_id)
        node_ids = sorted(s["node_id"] for s in rec["steps"])
        assert node_ids == [f"leaf{i}" for i in range(n)], (
            f"lost-update race: only {len(node_ids)}/{n} steps survived ({node_ids})"
        )


async def test_concurrent_seed_keeps_every_run_in_index() -> None:
    async with await SessionStore.open(":memory:") as store:
        parent = await store.create_session(system_prompt="parent")
        run_ids = [f"run-{i}" for i in range(12)]
        await asyncio.gather(*[
            journal.seed(store, parent, r, "wf", spec={}) for r in run_ids
        ])
        idx = await journal.list_run_ids(store, parent)
        assert sorted(idx) == sorted(run_ids), (
            f"lost-update race: only {len(idx)}/{len(run_ids)} run_ids in index"
        )


async def test_record_step_still_frozen_after_finalize() -> None:
    """The atomic path must preserve the terminal-freeze contract: a step settling after
    the run finalized (an orphaned halt sibling) must not append or revert status."""
    async with await SessionStore.open(":memory:") as store:
        parent = await store.create_session()
        run_id = "r"
        await journal.seed(store, parent, run_id, "wf", spec={})
        await journal.record_step(store, parent, run_id, node_id="a", status="completed")
        await journal.finalize(store, parent, run_id, WorkflowResult(name="wf", status="completed"))

        await journal.record_step(store, parent, run_id, node_id="late", status="completed")

        rec = await journal.read(store, parent, run_id)
        assert rec["status"] == "completed"
        assert sorted(s["node_id"] for s in rec["steps"]) == ["a"]


async def test_update_returns_none_when_absent_and_frozen_blob_when_terminal() -> None:
    async with await SessionStore.open(":memory:") as store:
        parent = await store.create_session()
        # absent run → None
        assert await journal.update(store, parent, "missing", woke=True) is None
        # terminal run → frozen blob unchanged (unless allow_terminal)
        run_id = "r"
        await journal.seed(store, parent, run_id, "wf", spec={})
        await journal.finalize(store, parent, run_id, WorkflowResult(name="wf", status="failed"))
        frozen = await journal.update(store, parent, run_id, status="running")
        assert frozen is not None and frozen["status"] == "failed"
        # allow_terminal lets a legitimate writer through (e.g. resume / wake guard)
        woke = await journal.update(store, parent, run_id, allow_terminal=True, woke=True)
        assert woke is not None and woke["woke"] is True
