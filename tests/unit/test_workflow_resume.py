"""Orchestration-level resume across a (simulated) process restart.

Drives a first attempt with a controllable executor that crashes partway, lets
the exception abort the engine (the journal keeps what completed), then calls
``resume_run`` with a healthy executor and asserts:

* completed steps are REPLAYED, not re-run (the executor never sees them again);
* the unfinished tail re-runs and the run completes;
* replayed structured output still feeds downstream foreach / inputs_from refs;
* a foreach is atomic — replayed whole via its aggregate, or re-run whole.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import dataclass, field

import pytest

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.workflow import (
    WorkflowSpec,
    rehydrate,
    replayable_node_ids,
    resume_run,
)
from power_loop.workflow import journal as journal_mod
from power_loop.workflow.engine import WorkflowEngine
from power_loop.workflow.runner import make_on_step

pytestmark = pytest.mark.unit


# ── a tiny LLM (unused by the stub executor, required by the loop ctor) ───────


@dataclass
class _LLM(LLMService):
    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        return LLMResponse(raw_text="x", content_text="x")

    def stream(self, request: LLMRequest):
        async def _e():
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


@pytest.fixture
def loop() -> Generator[StatefulAgentLoop, None, None]:
    lp = StatefulAgentLoop(
        llm=_LLM(), store=SessionStore.open(":memory:"),
        config=AgentLoopConfig(system_prompt="o", max_rounds=2, compactor=None),
    )
    yield lp
    lp.close()


# ── controllable executor ─────────────────────────────────────────────────────


@dataclass
class _CtlExec:
    """Records every leaf it runs (by workflow_node_id) and can crash on demand."""

    runs: list[tuple[str | None, str]] = field(default_factory=list)
    fail_nodes_once: set[str] = field(default_factory=set)         # raise once per node id
    fail_input_substr_once: set[str] = field(default_factory=set)  # raise once per matching input
    structured: dict[str, dict] = field(default_factory=dict)      # node id -> payload (JSON output)
    _spent: set[str] = field(default_factory=set)
    seen_meta: list[dict] = field(default_factory=list)

    async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
        nid = spec.metadata.get("workflow_node_id")
        self.runs.append((nid, user_input))
        self.seen_meta.append(dict(spec.metadata))
        for sub in list(self.fail_input_substr_once):
            if sub in user_input and f"in:{sub}" not in self._spent:
                self._spent.add(f"in:{sub}")
                raise RuntimeError(f"boom-input:{sub}")
        if nid in self.fail_nodes_once and f"node:{nid}" not in self._spent:
            self._spent.add(f"node:{nid}")
            raise RuntimeError(f"boom-node:{nid}")
        txt = json.dumps(self.structured[nid]) if nid in self.structured else f"out:{nid}"
        return {"status": "completed", "final_text": txt, "session_id": None,
                "usage": {"total_tokens": 7}}

    def ran(self, nid: str) -> int:
        return sum(1 for n, _ in self.runs if n == nid)


# ── tests ──────────────────────────────────────────────────────────────────


def test_replayable_ids_excludes_foreach_body() -> None:
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "plan", "spec": {"name": "p", "system_prompt": "p"},
             "output_schema": {"name": "P", "schema": {"type": "object"}}},
            {"type": "foreach", "id": "research", "items_from": "plan.subs", "as": "s",
             "body": {"type": "agent", "id": "body", "spec": {"name": "b", "system_prompt": "{{s}}"}}},
            {"type": "agent", "id": "write", "spec": {"name": "w", "system_prompt": "w"}},
        ]},
    })
    ids = replayable_node_ids(spec)
    assert ids == {"plan", "research", "write"}  # 'body' (foreach body) excluded


def test_rehydrate_refuses_journal_without_spec(loop: StatefulAgentLoop) -> None:
    sid = loop.new_session()
    journal_mod.seed(loop.store, sid, "r1", "w", spec=None)  # legacy-style: no spec
    with pytest.raises(Exception, match="no spec persisted"):
        rehydrate(loop.store, sid, "r1")


async def test_resume_replays_completed_and_reruns_tail(loop: StatefulAgentLoop) -> None:
    sid = loop.new_session()
    run_id = "run-linear"
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "topic",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "a"}},
            {"type": "agent", "id": "b", "spec": {"name": "b", "system_prompt": "b"}},
            {"type": "agent", "id": "c", "spec": {"name": "c", "system_prompt": "c"},
             "inputs_from": ["a"]},
        ]},
    })

    # Attempt 1: crash on 'b'. 'a' completes and is journaled.
    ex1 = _CtlExec(fail_nodes_once={"b"})
    journal_mod.seed(loop.store, sid, run_id, spec.name, spec=spec.to_dict())
    eng1 = WorkflowEngine(loop, executor=ex1, on_step=make_on_step(loop.store, sid, run_id),
                          run_id=run_id)
    with pytest.raises(RuntimeError, match="boom-node:b"):
        await eng1.run(spec)
    j = journal_mod.read(loop.store, sid, run_id)
    assert [s["node_id"] for s in j["steps"] if s["status"] == "completed"] == ["a"]

    # Resume with a healthy executor.
    ex2 = _CtlExec()
    result = await resume_run(loop, sid, run_id, executor=ex2)

    assert result.status == "completed"
    assert ex2.ran("a") == 0          # replayed, NOT re-run
    assert ex2.ran("b") == 1 and ex2.ran("c") == 1  # tail re-ran
    # replayed 'a' fed 'c' via inputs_from (its text appears in c's input)
    c_input = next(inp for nid, inp in ex2.runs if nid == "c")
    assert "out:a" in c_input
    # journal finalized as completed, attempts bumped
    j2 = journal_mod.read(loop.store, sid, run_id)
    assert j2["status"] == "completed" and j2["attempts"] == 2


async def test_resume_foreach_replayed_atomically(loop: StatefulAgentLoop) -> None:
    sid = loop.new_session()
    run_id = "run-foreach-done"
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "plan", "spec": {"name": "plan", "system_prompt": "p"},
             "output_schema": {"name": "P", "schema": {"type": "object", "required": ["subs"],
                "properties": {"subs": {"type": "array", "items": {"type": "string"}}}}}},
            {"type": "foreach", "id": "research", "items_from": "plan.subs", "as": "s",
             "parallel": False,
             "body": {"type": "agent", "id": "body", "spec": {"name": "body", "system_prompt": "{{s}}"},
                      "input": "do {{s}}"}},
            {"type": "agent", "id": "write", "spec": {"name": "write", "system_prompt": "w"},
             "inputs_from": ["research"]},
        ]},
    })

    # Attempt 1: plan + the whole foreach complete; crash on 'write'.
    ex1 = _CtlExec(fail_nodes_once={"write"}, structured={"plan": {"subs": ["x", "y"]}})
    journal_mod.seed(loop.store, sid, run_id, spec.name, spec=spec.to_dict())
    eng1 = WorkflowEngine(loop, executor=ex1, on_step=make_on_step(loop.store, sid, run_id),
                          run_id=run_id)
    with pytest.raises(RuntimeError, match="boom-node:write"):
        await eng1.run(spec)
    done = {s["node_id"] for s in journal_mod.read(loop.store, sid, run_id)["steps"]
            if s["status"] == "completed"}
    assert {"plan", "research"} <= done  # foreach aggregate journaled

    # Resume: plan + research replayed atomically; only 'write' re-runs.
    ex2 = _CtlExec()
    result = await resume_run(loop, sid, run_id, executor=ex2)
    assert result.status == "completed"
    assert ex2.ran("plan") == 0
    assert ex2.ran("body") == 0   # foreach bodies NOT re-run (aggregate replayed)
    assert ex2.ran("write") == 1
    assert len(result.results["research"].payload["items"]) == 2  # replayed aggregate intact


async def test_resume_reruns_foreach_when_it_died_midway(loop: StatefulAgentLoop) -> None:
    sid = loop.new_session()
    run_id = "run-foreach-partial"
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "plan", "spec": {"name": "plan", "system_prompt": "p"},
             "output_schema": {"name": "P", "schema": {"type": "object", "required": ["subs"],
                "properties": {"subs": {"type": "array", "items": {"type": "string"}}}}}},
            {"type": "foreach", "id": "research", "items_from": "plan.subs", "as": "s",
             "parallel": False, "on_error": "halt",
             "body": {"type": "agent", "id": "body", "spec": {"name": "body", "system_prompt": "{{s}}"},
                      "input": "do {{s}}"}},
        ]},
    })

    # Attempt 1: plan completes; foreach crashes on the 2nd item → aggregate NOT journaled.
    ex1 = _CtlExec(fail_input_substr_once={"yy"}, structured={"plan": {"subs": ["xx", "yy"]}})
    journal_mod.seed(loop.store, sid, run_id, spec.name, spec=spec.to_dict())
    eng1 = WorkflowEngine(loop, executor=ex1, on_step=make_on_step(loop.store, sid, run_id),
                          run_id=run_id)
    with pytest.raises(RuntimeError, match="boom-input:yy"):
        await eng1.run(spec)
    done = {s["node_id"] for s in journal_mod.read(loop.store, sid, run_id)["steps"]
            if s["status"] == "completed"}
    assert "plan" in done and "research" not in done  # no aggregate → foreach re-runs

    # Resume: plan replayed; the WHOLE foreach re-runs (both items).
    ex2 = _CtlExec()
    result = await resume_run(loop, sid, run_id, executor=ex2)
    assert result.status == "completed"
    assert ex2.ran("plan") == 0
    assert ex2.ran("body") == 2  # both items re-run, including the one that had succeeded


async def test_idempotency_key_threaded_into_leaf_metadata(loop: StatefulAgentLoop) -> None:
    sid = loop.new_session()
    run_id = "run-idem"
    spec = WorkflowSpec.from_json({
        "name": "w", "root": {"type": "agent", "id": "only",
                              "spec": {"name": "only", "system_prompt": "p"}}})
    ex = _CtlExec()
    journal_mod.seed(loop.store, sid, run_id, spec.name, spec=spec.to_dict())
    eng = WorkflowEngine(loop, executor=ex, on_step=make_on_step(loop.store, sid, run_id),
                         run_id=run_id)
    await eng.run(spec)
    meta = ex.seen_meta[0]
    assert meta["workflow_run_id"] == run_id
    assert meta["workflow_node_id"] == "only"
    assert meta["idempotency_key"] == f"{run_id}:only"


async def test_foreach_usage_counted_and_body_not_journaled_individually(
    loop: StatefulAgentLoop,
) -> None:
    """The fan-out's tokens roll into the aggregate + run total, and the journal
    records the foreach as ONE aggregate step (not one-overwriting-record-per-item)."""
    sid = loop.new_session()
    run_id = "run-usage"
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "plan", "spec": {"name": "plan", "system_prompt": "p"},
             "output_schema": {"name": "P", "schema": {"type": "object", "required": ["subs"],
                "properties": {"subs": {"type": "array", "items": {"type": "string"}}}}}},
            {"type": "foreach", "id": "research", "items_from": "plan.subs", "as": "s",
             "parallel": False,
             "body": {"type": "agent", "id": "body", "spec": {"name": "body", "system_prompt": "{{s}}"},
                      "input": "do {{s}}"}},
        ]},
    })
    ex = _CtlExec(structured={"plan": {"subs": ["a", "b", "c"]}})  # 7 tokens per leaf
    journal_mod.seed(loop.store, sid, run_id, spec.name, spec=spec.to_dict())
    eng = WorkflowEngine(loop, executor=ex, on_step=make_on_step(loop.store, sid, run_id),
                         run_id=run_id)
    result = await eng.run(spec)

    # plan(7) + 3 body leaves(7 each) = 28 — foreach work is NOT lost.
    assert result.usage["total_tokens"] == 28
    assert result.results["research"].usage["total_tokens"] == 21  # aggregate carries the fan-out
    # Journal records the foreach as ONE aggregate step, body leaves not journaled.
    steps = {s["node_id"]: s for s in journal_mod.read(loop.store, sid, run_id)["steps"]}
    assert set(steps) == {"plan", "research"}  # 'body' not individually journaled
    assert steps["research"]["usage"]["total_tokens"] == 21


async def test_resume_budget_precharge_includes_replayed_foreach(loop: StatefulAgentLoop) -> None:
    """Resuming pre-charges the fresh budget with replayed usage (incl. a foreach
    aggregate), so the shared ceiling accounts for the whole run."""
    from power_loop.workflow import SharedBudget

    sid = loop.new_session()
    run_id = "run-budget"
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "plan", "spec": {"name": "plan", "system_prompt": "p"},
             "output_schema": {"name": "P", "schema": {"type": "object", "required": ["subs"],
                "properties": {"subs": {"type": "array", "items": {"type": "string"}}}}}},
            {"type": "foreach", "id": "research", "items_from": "plan.subs", "as": "s",
             "parallel": False,
             "body": {"type": "agent", "id": "body", "spec": {"name": "body", "system_prompt": "{{s}}"},
                      "input": "do {{s}}"}},
            {"type": "agent", "id": "write", "spec": {"name": "write", "system_prompt": "w"}},
        ]},
    })
    # Attempt 1: plan + foreach(2 items) complete (3 leaves * 7 = 21 tokens), crash on write.
    ex1 = _CtlExec(fail_nodes_once={"write"}, structured={"plan": {"subs": ["a", "b"]}})
    journal_mod.seed(loop.store, sid, run_id, spec.name, spec=spec.to_dict())
    eng1 = WorkflowEngine(loop, executor=ex1, on_step=make_on_step(loop.store, sid, run_id),
                          run_id=run_id)
    with pytest.raises(RuntimeError):
        await eng1.run(spec)

    # Resume with a fresh budget whose ceiling is below the already-spent total:
    # pre-charge of replayed usage (plan 7 + foreach 14 = 21) exhausts it, so the
    # remaining 'write' step is refused with budget_exceeded.
    ex2 = _CtlExec()
    result = await resume_run(loop, sid, run_id, executor=ex2, budget=SharedBudget(21))
    assert result.status == "budget_exceeded"
    assert ex2.ran("write") == 0  # never spawned — budget already consumed by replay
