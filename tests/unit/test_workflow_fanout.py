"""H1.2 / C2: parallel & foreach `on_error="halt"` cancel in-flight siblings.

asyncio.gather(return_exceptions=False) re-raises on the first failure but leaves
the other branches running detached — they keep burning real LLM calls and a late
record_step can clobber the finalized journal. The engine must instead cancel the
siblings when a branch fails under halt.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from power_loop.workflow import WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine

pytestmark = pytest.mark.unit






@dataclass
class _Tracker:
    """An executor where 'boom' fails immediately and slow leaves block until
    cancelled — records which leaves started, finished, or were cancelled."""

    started: list = None  # type: ignore[assignment]
    finished: list = None  # type: ignore[assignment]
    cancelled: list = None  # type: ignore[assignment]

    def __post_init__(self):
        self.started, self.finished, self.cancelled = [], [], []

    async def run_agent(self, request, *, allow_start=True):
        # Key off the rendered input so a foreach (whose leaves share one spec name)
        # is distinguishable per item; fall back to the spec name for parallel.
        key = request.body.strip() or request.node_path
        self.started.append(key)
        try:
            if key == "boom":
                raise RuntimeError("branch failed")
            await asyncio.sleep(1)  # slow sibling; gets cancelled before this returns
            self.finished.append(key)
            return {"status": "completed", "final_text": "ok", "session_id": None, "usage": {}}
        except asyncio.CancelledError:
            self.cancelled.append(key)
            raise


def _leftover_tasks() -> list:
    return [t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()]


async def test_parallel_halt_cancels_inflight_siblings() -> None:
    ex = _Tracker()
    spec = WorkflowSpec.from_json({
        "name": "w", "root": {"type": "parallel", "on_error": "halt", "branches": [
            {"type": "agent", "id": "slow1", "input": "slow1", "spec": {"name": "slow1", "system_prompt": "p"}},
            {"type": "agent", "id": "boom", "input": "boom", "spec": {"name": "boom", "system_prompt": "p"}},
            {"type": "agent", "id": "slow2", "input": "slow2", "spec": {"name": "slow2", "system_prompt": "p"}},
        ]}})
    eng = WorkflowEngine(ex, run_id="test", driver_sid="driver")

    with pytest.raises(RuntimeError, match="branch failed"):
        await eng.run(spec)

    assert "boom" in ex.started
    assert ex.finished == []                       # no slow sibling ran to completion
    assert set(ex.cancelled) == {"slow1", "slow2"}  # both in-flight siblings cancelled
    assert _leftover_tasks() == []                  # nothing left running detached


async def test_foreach_parallel_halt_cancels_inflight_siblings() -> None:
    ex = _Tracker()
    spec = WorkflowSpec.from_json({
        "name": "w", "root": {
            "type": "foreach", "id": "f", "parallel": True, "on_error": "halt",
            "items": ["slow1", "boom", "slow2"], "as": "name",
            "body": {"type": "agent", "id": "leaf", "input": "{{name}}",
                     "spec": {"name": "leaf", "system_prompt": "p"}},
        }})
    eng = WorkflowEngine(ex, run_id="test", driver_sid="driver")

    with pytest.raises(RuntimeError, match="branch failed"):
        await eng.run(spec)

    assert "boom" in ex.started
    assert ex.finished == []
    assert set(ex.cancelled) == {"slow1", "slow2"}
    assert _leftover_tasks() == []


async def test_parallel_continue_still_collects_all_errors() -> None:
    """on_error='continue' is unchanged: every branch runs, errors are collected."""
    ex = _Tracker()
    spec = WorkflowSpec.from_json({
        "name": "w", "root": {"type": "parallel", "on_error": "continue", "branches": [
            {"type": "agent", "id": "ok1", "input": "ok1", "spec": {"name": "ok1", "system_prompt": "p"}},
            {"type": "agent", "id": "boom", "input": "boom", "spec": {"name": "boom", "system_prompt": "p"}},
        ]}})
    # 'ok1' must finish (not cancelled) under continue; shorten its sleep via name
    eng = WorkflowEngine(ex, run_id="test", driver_sid="driver")
    res = await eng.run(spec)  # does not raise under continue
    assert res.status in {"completed", "failed"}
    assert any("branch error" in e or "failed" in e for e in res.errors)
    assert ex.cancelled == []  # continue never cancels siblings


# ── H3 (BUG_REVIEW_3.4): fanout is bounded. create_workflow is LLM-facing, so a spec may be
# hallucinated/adversarial; without caps a foreach/parallel explodes into millions of sessions.

def test_foreach_max_concurrency_capped() -> None:
    from power_loop.workflow.spec import MAX_FANOUT_CONCURRENCY, WorkflowSpecError
    with pytest.raises(WorkflowSpecError, match=f"<= {MAX_FANOUT_CONCURRENCY}"):
        WorkflowSpec.from_json({"name": "w", "root": {
            "type": "foreach", "items": ["a"], "as": "x", "max_concurrency": 1_000_000,
            "body": {"type": "agent", "input": "{{x}}", "spec": {"name": "l", "system_prompt": "p"}},
        }})


def test_foreach_items_literal_capped() -> None:
    from power_loop.workflow.spec import MAX_FOREACH_ITEMS, WorkflowSpecError
    with pytest.raises(WorkflowSpecError, match=f"max {MAX_FOREACH_ITEMS}"):
        WorkflowSpec.from_json({"name": "w", "root": {
            "type": "foreach", "items": list(range(MAX_FOREACH_ITEMS + 1)), "as": "x",
            "body": {"type": "agent", "input": "{{x}}", "spec": {"name": "l", "system_prompt": "p"}},
        }})


# ── M-workflow-engine-4: references must target a node guaranteed completed BEFORE this one runs.

def _agent(nid, **extra):
    return {"type": "agent", "id": nid, "spec": {"name": nid, "system_prompt": "p"}, **extra}


def test_forward_reference_in_sequence_rejected() -> None:
    from power_loop.workflow.spec import WorkflowSpecError
    with pytest.raises(WorkflowSpecError, match="not guaranteed to have completed"):
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            _agent("a", inputs_from=["b"]),  # b runs AFTER a
            _agent("b"),
        ]}})


def test_parallel_sibling_reference_rejected() -> None:
    from power_loop.workflow.spec import WorkflowSpecError
    with pytest.raises(WorkflowSpecError, match="not guaranteed to have completed"):
        WorkflowSpec.from_json({"name": "w", "root": {"type": "parallel", "branches": [
            _agent("a"),
            _agent("b", inputs_from=["a"]),  # concurrent sibling — a may not be done
        ]}})


def test_cross_branch_case_reference_rejected() -> None:
    from power_loop.workflow.spec import WorkflowSpecError
    with pytest.raises(WorkflowSpecError, match="not guaranteed to have completed"):
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            {"type": "branch", "on": "src.k",
             "cases": {"x": _agent("ca")},
             "default": _agent("cd")},
            _agent("after", inputs_from=["ca"]),  # ca only runs if case 'x' was taken
            _agent("src"),
        ]}})


def test_output_schema_shape_validated() -> None:
    # workflow-engine-5: a malformed output_schema must be rejected at parse, not blow up at runtime.
    from power_loop.workflow.spec import WorkflowSpecError
    with pytest.raises(WorkflowSpecError, match="output_schema.name"):
        WorkflowSpec.from_json({"name": "w", "root": _agent("a", output_schema={"schema": {"type": "object"}})})
    with pytest.raises(WorkflowSpecError, match="output_schema.schema"):
        WorkflowSpec.from_json({"name": "w", "root": _agent("a", output_schema={"name": "S", "schema": "x"})})
    with pytest.raises(WorkflowSpecError, match="unknown key"):
        WorkflowSpec.from_json({"name": "w", "root": _agent("a", output_schema={"name": "S", "schema": {}, "x": 1})})
    # valid shape passes
    assert WorkflowSpec.from_json(
        {"name": "w", "root": _agent("a", output_schema={"name": "S", "schema": {"type": "object"}})}
    ).name == "w"




def test_valid_reference_to_completed_node_accepted() -> None:
    # s1 → parallel{A, B} → C references both: A and B both complete before C. Valid.
    spec = WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
        _agent("s1"),
        {"type": "parallel", "branches": [_agent("a", inputs_from=["s1"]), _agent("b", inputs_from=["s1"])]},
        _agent("c", inputs_from=["a", "b", "s1"]),
    ]}})
    assert spec.name == "w"


def test_foreach_as_must_be_a_valid_identifier() -> None:
    # M-workflow-engine-3: an 'as' that isn't a {{var}}-substitutable identifier would silently
    # never bind into the body → per-iteration input corruption. Reject it at validation.
    from power_loop.workflow.spec import WorkflowSpecError
    with pytest.raises(WorkflowSpecError, match="valid identifier"):
        WorkflowSpec.from_json({"name": "w", "root": {
            "type": "foreach", "items": ["a"], "as": "my var",
            "body": {"type": "agent", "id": "l", "input": "{{my var}}",
                     "spec": {"name": "l", "system_prompt": "p"}},
        }})


def test_parallel_max_concurrency_capped() -> None:
    from power_loop.workflow.spec import MAX_FANOUT_CONCURRENCY, WorkflowSpecError
    with pytest.raises(WorkflowSpecError, match=f"<= {MAX_FANOUT_CONCURRENCY}"):
        WorkflowSpec.from_json({"name": "w", "root": {
            "type": "parallel", "max_concurrency": 99_999, "branches": [
                {"type": "agent", "input": "a", "spec": {"name": "a", "system_prompt": "p"}},
            ]}})


def test_foreach_items_from_runtime_capped() -> None:
    """A DYNAMIC items_from list is capped BEFORE _exec_foreach eagerly creates one task per item."""
    import dataclasses

    from power_loop.workflow.engine import WorkflowRunError
    from power_loop.workflow.result import AgentResult
    from power_loop.workflow.spec import MAX_FOREACH_ITEMS

    eng = WorkflowEngine(_Tracker(), run_id="test", driver_sid="driver")
    eng._results["src"] = AgentResult(
        node_id="src", status="completed", text="",
        payload={"xs": list(range(MAX_FOREACH_ITEMS + 1))},
    )
    base = WorkflowSpec.from_json({"name": "w", "root": {
        "type": "foreach", "id": "f", "items": ["a"], "as": "x",
        "body": {"type": "agent", "id": "leaf", "input": "{{x}}", "spec": {"name": "l", "system_prompt": "p"}},
    }}).root
    node = dataclasses.replace(base, items=None, items_from="src.xs")
    with pytest.raises(WorkflowRunError, match=f"max {MAX_FOREACH_ITEMS}"):
        eng._resolve_items(node)


async def test_total_leaf_ceiling_enforced(monkeypatch) -> None:
    """The per-run leaf ceiling fail-closes nested/programmatic fanout independent of any budget."""
    import power_loop.workflow.engine as engmod

    monkeypatch.setattr(engmod, "MAX_TOTAL_LEAVES", 2)

    @dataclass
    class _Counter:
        runs: int = 0

        async def run_agent(self, request, *, allow_start=True):
            self.runs += 1
            return {"status": "completed", "final_text": "ok", "session_id": None, "usage": {}}

    ex = _Counter()
    spec = WorkflowSpec.from_json({"name": "w", "root": {
        "type": "foreach", "parallel": False, "on_error": "continue",
        "items": ["a", "b", "c", "d"], "as": "x",
        "body": {"type": "agent", "id": "leaf", "input": "{{x}}", "spec": {"name": "l", "system_prompt": "p"}},
    }})
    res = await engmod.WorkflowEngine(ex, run_id="test", driver_sid="driver").run(spec)
    assert ex.runs == 2                       # only 2 leaves actually executed
    assert res.status in {"completed", "failed"}
    assert any("leaf ceiling" in e for e in res.errors)
