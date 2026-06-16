"""Coverage: crash-recovery replay/journal/budget invariants the existing suite doesn't exercise (probed clean; these lock the higher-arity / multi-restart / branch-decider / spec-budget paths)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_replayable_ids_deeply_nested_parallel_sequence_foreach_branch() -> None:
    from power_loop.workflow import WorkflowSpec, replayable_node_ids

    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "parallel", "branches": [
            {"type": "sequence", "steps": [
                {"type": "agent", "id": "s1",
                 "spec": {"name": "s1", "system_prompt": "p"},
                 "output_schema": {"name": "S", "schema": {"type": "object"}}},
                {"type": "foreach", "id": "fe", "items_from": "s1.xs", "as": "v",
                 "body": {"type": "sequence", "steps": [
                     {"type": "agent", "id": "deep",
                      "spec": {"name": "d", "system_prompt": "{{v}}"}}]}},
            ]},
            {"type": "branch", "on": "s1.k", "cases": {
                "a": {"type": "agent", "id": "ca",
                      "spec": {"name": "ca", "system_prompt": "ca"}}},
             "default": {"type": "agent", "id": "cd",
                         "spec": {"name": "cd", "system_prompt": "cd"}}},
        ]},
    })
    # s1 + foreach aggregate 'fe' + branch case 'ca' + default 'cd' replayable;
    # 'deep' (inside the foreach body) is NOT individually replayable.
    assert replayable_node_ids(spec) == {"s1", "fe", "ca", "cd"}

async def test_resume_twice_replays_across_both_prior_attempts() -> None:
    from dataclasses import dataclass, field

    from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
    from power_loop._vendor.llm_client.interface import (
        LLMResponse,
        LLMService,
        LLMStreamChunk,
    )
    from power_loop.workflow import WorkflowSpec, resume_run
    from power_loop.workflow import journal as journal_mod
    from power_loop.workflow.engine import WorkflowEngine
    from power_loop.workflow.runner import make_on_step

    @dataclass
    class _LLM(LLMService):
        async def complete(self, request, *, on_chunk_delta_text=None,
                           on_chunk_think=None, on_stream_end=None) -> LLMResponse:
            return LLMResponse(raw_text="x", content_text="x")

        def stream(self, request):
            async def _e():
                if False:
                    yield LLMStreamChunk()
            return _e()

        async def close(self) -> None:
            return None

    @dataclass
    class _Exec:
        runs: list = field(default_factory=list)
        fail_once: set = field(default_factory=set)
        _spent: set = field(default_factory=set)

        async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
            nid = spec.metadata.get("workflow_node_id")
            self.runs.append((nid, user_input))
            if nid in self.fail_once and nid not in self._spent:
                self._spent.add(nid)
                raise RuntimeError(f"boom:{nid}")
            return {"status": "completed", "final_text": f"out:{nid}",
                    "session_id": None, "usage": {"total_tokens": 7}}

        def ran(self, nid):
            return sum(1 for n, _ in self.runs if n == nid)

    lp = StatefulAgentLoop(
        llm=_LLM(), store=SessionStore.open(":memory:"),
        config=AgentLoopConfig(system_prompt="o", max_rounds=2, compactor=None),
    )
    try:
        sid = lp.new_session()
        run_id = "run-multi"
        spec = WorkflowSpec.from_json({
            "name": "w", "input": "x",
            "root": {"type": "sequence", "steps": [
                {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "a"}},
                {"type": "agent", "id": "b", "spec": {"name": "b", "system_prompt": "b"}},
                {"type": "agent", "id": "c", "spec": {"name": "c", "system_prompt": "c"},
                 "inputs_from": ["a", "b"]},
            ]},
        })
        journal_mod.seed(lp.store, sid, run_id, spec.name, spec=spec.to_dict())

        ex1 = _Exec(fail_once={"b"})
        eng1 = WorkflowEngine(lp, executor=ex1,
                              on_step=make_on_step(lp.store, sid, run_id), run_id=run_id)
        with pytest.raises(RuntimeError):
            await eng1.run(spec)

        ex2 = _Exec(fail_once={"c"})
        with pytest.raises(RuntimeError):
            await resume_run(lp, sid, run_id, executor=ex2)
        assert ex2.ran("a") == 0 and ex2.ran("b") == 1
        done = {s["node_id"] for s in journal_mod.read(lp.store, sid, run_id)["steps"]
                if s["status"] == "completed"}
        assert done == {"a", "b"}

        ex3 = _Exec()
        result = await resume_run(lp, sid, run_id, executor=ex3)
        assert result.status == "completed"
        assert ex3.ran("a") == 0 and ex3.ran("b") == 0 and ex3.ran("c") == 1
        c_input = next(inp for nid, inp in ex3.runs if nid == "c")
        assert "out:a" in c_input and "out:b" in c_input
        assert result.usage["total_tokens"] == 21
        assert journal_mod.read(lp.store, sid, run_id)["attempts"] == 3
    finally:
        lp.close()

async def test_resume_branch_replayed_decider_picks_same_case() -> None:
    import json
    from dataclasses import dataclass, field

    from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
    from power_loop._vendor.llm_client.interface import (
        LLMResponse,
        LLMService,
        LLMStreamChunk,
    )
    from power_loop.workflow import WorkflowSpec, resume_run
    from power_loop.workflow import journal as journal_mod
    from power_loop.workflow.engine import WorkflowEngine
    from power_loop.workflow.runner import make_on_step

    @dataclass
    class _LLM(LLMService):
        async def complete(self, request, *, on_chunk_delta_text=None,
                           on_chunk_think=None, on_stream_end=None) -> LLMResponse:
            return LLMResponse(raw_text="x", content_text="x")

        def stream(self, request):
            async def _e():
                if False:
                    yield LLMStreamChunk()
            return _e()

        async def close(self) -> None:
            return None

    @dataclass
    class _Exec:
        runs: list = field(default_factory=list)
        fail_once: set = field(default_factory=set)
        structured: dict = field(default_factory=dict)
        _spent: set = field(default_factory=set)

        async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
            nid = spec.metadata.get("workflow_node_id")
            self.runs.append(nid)
            if nid in self.fail_once and nid not in self._spent:
                self._spent.add(nid)
                raise RuntimeError(f"boom:{nid}")
            txt = json.dumps(self.structured[nid]) if nid in self.structured else f"out:{nid}"
            return {"status": "completed", "final_text": txt,
                    "session_id": None, "usage": {"total_tokens": 7}}

        def ran(self, nid):
            return self.runs.count(nid)

    lp = StatefulAgentLoop(
        llm=_LLM(), store=SessionStore.open(":memory:"),
        config=AgentLoopConfig(system_prompt="o", max_rounds=2, compactor=None),
    )
    try:
        sid = lp.new_session()
        run_id = "run-branch"
        spec = WorkflowSpec.from_json({
            "name": "w", "input": "x",
            "root": {"type": "sequence", "steps": [
                {"type": "agent", "id": "pick",
                 "spec": {"name": "pick", "system_prompt": "p"},
                 "output_schema": {"name": "P", "schema": {"type": "object",
                    "required": ["k"], "properties": {"k": {"type": "string"}}}}},
                {"type": "branch", "on": "pick.k", "cases": {
                    "go": {"type": "agent", "id": "ca",
                           "spec": {"name": "ca", "system_prompt": "ca"}},
                    "stop": {"type": "agent", "id": "cb",
                             "spec": {"name": "cb", "system_prompt": "cb"}}}},
                {"type": "agent", "id": "tail",
                 "spec": {"name": "tail", "system_prompt": "t"}},
            ]},
        })
        journal_mod.seed(lp.store, sid, run_id, spec.name, spec=spec.to_dict())
        ex1 = _Exec(structured={"pick": {"k": "go"}}, fail_once={"tail"})
        eng1 = WorkflowEngine(lp, executor=ex1,
                              on_step=make_on_step(lp.store, sid, run_id), run_id=run_id)
        with pytest.raises(RuntimeError):
            await eng1.run(spec)
        done = {s["node_id"] for s in journal_mod.read(lp.store, sid, run_id)["steps"]
                if s["status"] == "completed"}
        assert {"pick", "ca"} <= done

        ex2 = _Exec()
        result = await resume_run(lp, sid, run_id, executor=ex2)
        assert result.status == "completed"
        assert ex2.ran("pick") == 0 and ex2.ran("ca") == 0  # replayed
        assert ex2.ran("cb") == 0  # non-chosen case never ran
        assert ex2.ran("tail") == 1  # only the tail re-runs
    finally:
        lp.close()

async def test_resume_replayed_payload_drives_rerunning_foreach() -> None:
    import json
    from dataclasses import dataclass, field

    from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
    from power_loop._vendor.llm_client.interface import (
        LLMResponse,
        LLMService,
        LLMStreamChunk,
    )
    from power_loop.workflow import WorkflowSpec, resume_run
    from power_loop.workflow import journal as journal_mod
    from power_loop.workflow.engine import WorkflowEngine
    from power_loop.workflow.runner import make_on_step

    @dataclass
    class _LLM(LLMService):
        async def complete(self, request, *, on_chunk_delta_text=None,
                           on_chunk_think=None, on_stream_end=None) -> LLMResponse:
            return LLMResponse(raw_text="x", content_text="x")

        def stream(self, request):
            async def _e():
                if False:
                    yield LLMStreamChunk()
            return _e()

        async def close(self) -> None:
            return None

    @dataclass
    class _Exec:
        runs: list = field(default_factory=list)
        fail_once: set = field(default_factory=set)
        structured: dict = field(default_factory=dict)
        _spent: set = field(default_factory=set)

        async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
            nid = spec.metadata.get("workflow_node_id")
            self.runs.append((nid, user_input))
            if nid in self.fail_once and nid not in self._spent:
                self._spent.add(nid)
                raise RuntimeError(f"boom:{nid}")
            txt = json.dumps(self.structured[nid]) if nid in self.structured else f"out:{nid}"
            return {"status": "completed", "final_text": txt,
                    "session_id": None, "usage": {"total_tokens": 7}}

        def ran(self, nid):
            return sum(1 for n, _ in self.runs if n == nid)

    lp = StatefulAgentLoop(
        llm=_LLM(), store=SessionStore.open(":memory:"),
        config=AgentLoopConfig(system_prompt="o", max_rounds=2, compactor=None),
    )
    try:
        sid = lp.new_session()
        run_id = "run-payload"
        spec = WorkflowSpec.from_json({
            "name": "w", "input": "x",
            "root": {"type": "sequence", "steps": [
                {"type": "agent", "id": "plan",
                 "spec": {"name": "plan", "system_prompt": "p"},
                 "output_schema": {"name": "P", "schema": {"type": "object",
                    "required": ["subs"], "properties": {
                        "subs": {"type": "array", "items": {"type": "string"}}}}}},
                {"type": "foreach", "id": "fe", "items_from": "plan.subs", "as": "s",
                 "parallel": False, "on_error": "halt",
                 "body": {"type": "agent", "id": "body",
                          "spec": {"name": "b", "system_prompt": "{{s}}"},
                          "input": "do {{s}}"}},
            ]},
        })
        journal_mod.seed(lp.store, sid, run_id, spec.name, spec=spec.to_dict())
        # plan completes with subs=[p,q,r]; foreach crashes on its 1st body call.
        ex1 = _Exec(structured={"plan": {"subs": ["p", "q", "r"]}}, fail_once={"body"})
        eng1 = WorkflowEngine(lp, executor=ex1,
                              on_step=make_on_step(lp.store, sid, run_id), run_id=run_id)
        with pytest.raises(RuntimeError):
            await eng1.run(spec)
        done = {s["node_id"] for s in journal_mod.read(lp.store, sid, run_id)["steps"]
                if s["status"] == "completed"}
        assert "plan" in done and "fe" not in done

        ex2 = _Exec()
        result = await resume_run(lp, sid, run_id, executor=ex2)
        assert result.status == "completed"
        assert ex2.ran("plan") == 0  # plan replayed from journal
        assert ex2.ran("body") == 3  # foreach re-ran fully off the replayed payload
        inputs = [inp for nid, inp in ex2.runs if nid == "body"]
        assert any("do p" in i for i in inputs) and any("do r" in i for i in inputs)
    finally:
        lp.close()

async def test_resume_precharges_spec_defined_budget() -> None:
    from dataclasses import dataclass, field

    from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
    from power_loop._vendor.llm_client.interface import (
        LLMResponse,
        LLMService,
        LLMStreamChunk,
    )
    from power_loop.workflow import WorkflowSpec, resume_run
    from power_loop.workflow import journal as journal_mod
    from power_loop.workflow.engine import WorkflowEngine
    from power_loop.workflow.runner import make_on_step

    @dataclass
    class _LLM(LLMService):
        async def complete(self, request, *, on_chunk_delta_text=None,
                           on_chunk_think=None, on_stream_end=None) -> LLMResponse:
            return LLMResponse(raw_text="x", content_text="x")

        def stream(self, request):
            async def _e():
                if False:
                    yield LLMStreamChunk()
            return _e()

        async def close(self) -> None:
            return None

    @dataclass
    class _Exec:
        runs: list = field(default_factory=list)

        async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
            nid = spec.metadata.get("workflow_node_id")
            self.runs.append(nid)
            return {"status": "completed", "final_text": f"out:{nid}",
                    "session_id": None, "usage": {"total_tokens": 7}}

        def ran(self, nid):
            return self.runs.count(nid)

    lp = StatefulAgentLoop(
        llm=_LLM(), store=SessionStore.open(":memory:"),
        config=AgentLoopConfig(system_prompt="o", max_rounds=2, compactor=None),
    )
    try:
        sid = lp.new_session()
        run_id = "run-specbudget"
        spec = WorkflowSpec.from_json({
            "name": "w", "input": "x",
            "budget": {"max_tokens": 7},
            "root": {"type": "sequence", "steps": [
                {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "a"}},
                {"type": "agent", "id": "b", "spec": {"name": "b", "system_prompt": "b"}},
            ]},
        })
        journal_mod.seed(lp.store, sid, run_id, spec.name, spec=spec.to_dict())
        # Attempt 1: spec budget 7 -> a completes (7), b refused (budget_exceeded).
        ex1 = _Exec()
        eng1 = WorkflowEngine(lp, executor=ex1,
                              on_step=make_on_step(lp.store, sid, run_id), run_id=run_id)
        r1 = await eng1.run(spec)
        assert r1.status == "budget_exceeded" and ex1.ran("a") == 1 and ex1.ran("b") == 0
        journal_mod.finalize(lp.store, sid, run_id, r1)
        done = {s["node_id"] for s in journal_mod.read(lp.store, sid, run_id)["steps"]
                if s["status"] == "completed"}
        assert "a" in done

        # Resume: spec budget 7 rebuilt + pre-charged with a's 7 -> b stays refused.
        ex2 = _Exec()
        result = await resume_run(lp, sid, run_id, executor=ex2)
        assert ex2.ran("a") == 0  # replayed
        assert result.status == "budget_exceeded"
        assert ex2.ran("b") == 0  # spec-budget pre-charge blocks it
    finally:
        lp.close()
