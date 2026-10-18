import asyncio

import pytest

from power_loop.workflow import WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine


class Parent:
    async def new_session(self, **kwargs):
        return "driver"


def leaf(n):
    return {"type": "agent", "id": n, "spec": {"name": n, "system_prompt": n}}


@pytest.mark.asyncio
@pytest.mark.parametrize("with_loop", [False, True])
async def test_failed_branch_resumes_while_sibling_runs_and_review_blocks_assembly(with_loop):
    sibling = asyncio.Event()
    repairing = asyncio.Event()
    finish_repair = asyncio.Event()
    runs = []

    class Executor:
        async def run_agent(self, spec, user_input, **kwargs):
            nid = spec.metadata['workflow_node_id']
            runs.append(nid)
            if nid == "sibling":
                await sibling.wait()
            return {"status": "context_checkpoint" if nid == "dev" else "completed",
                    "final_text": '{"pass": true}' if nid == "review" else "OK", "session_id": nid + "-sid"}

        async def continue_agent(self, sid, message, **kwargs):
            assert sid == "dev-sid" and message == "finish development"
            repairing.set()
            await finish_repair.wait()
            return {"status": "completed", "final_text": "repaired", "session_id": sid}

    review = {**leaf("review"), "output_schema": {"name": "Review", "schema": {
        "type": "object", "properties": {"pass": {"type": "boolean"}}, "required": ["pass"]}}}
    branch = {"type": "sequence", "steps": [leaf("dev"), review]}
    if with_loop:
        branch = {"type": "loop", "id": "dev-loop", "max_iterations": 2,
                  "until": {"ref": "review.pass"}, "body": branch}
    spec = WorkflowSpec.from_json({"name": "w", "input": "", "root": {
        "type": "sequence", "steps": [
            {"type": "parallel", "on_error": "continue", "branches": [
                branch, leaf("sibling")]},
            leaf("assembly")]}})
    engine = WorkflowEngine(Parent(), executor=Executor(), close_driver=False, run_id="test")
    task = asyncio.create_task(engine.run(spec))
    for _ in range(100):
        if "dev" in engine._results and "sibling" in runs:
            break
        await asyncio.sleep(0)
    repair = asyncio.create_task(engine.continue_node("dev", "finish development", 12))
    await asyncio.wait_for(repairing.wait(), 2)
    sibling.set()
    await asyncio.sleep(0)
    assert "review" not in runs and "assembly" not in runs
    finish_repair.set()
    await repair
    result = await task
    assert result.status == "completed"
    assert runs == ["dev", "sibling", "review", "assembly"]
    assert result.results["dev"].text == "repaired"


def test_recovery_replay_invalidates_review_but_preserves_parallel_siblings():
    from power_loop.workflow.result import AgentResult
    from power_loop.workflow.resume import continuation_replay
    spec = WorkflowSpec.from_json({"name": "r", "input": "", "root": {
        "type": "sequence", "steps": [{"type": "parallel", "branches": [
            {"type": "loop", "id": "l", "max_iterations": 2, "until": {"ref": "review.pass"},
             "body": {"type": "sequence", "steps": [leaf("dev"), leaf("review")]}},
            leaf("sibling")]}, leaf("assembly")]}})
    results = {n: AgentResult(node_id=n, status="completed", text="old")
               for n in ["dev", "review", "l", "sibling", "assembly"]}
    assert set(continuation_replay(spec, "dev", results)) == {"sibling"}


@pytest.mark.asyncio
async def test_accepted_terminal_window_steering_is_processed_before_review():
    started, finish, delivered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    seen = []

    class InboxParent(Parent):
        async def follow_up(self, message, **kwargs):
            assert kwargs["enqueue_only"] is True
            await delivered.wait()
            return "queued"

        async def inbox_pending(self, sid):
            return {"pending": 1}

    class Executor:
        async def run_agent(self, spec, user_input, **kwargs):
            if spec.name == "dev":
                started.set()
                await finish.wait()
            else:
                seen.append(user_input)
            return {"status": "completed", "final_text": "old", "session_id": "sid"}

        async def continue_agent(self, sid, message, **kwargs):
            seen.append("flushed")
            return {"status": "completed", "final_text": "new evidence", "session_id": sid}

    review = {**leaf("review"), "inputs_from": ["dev"]}
    spec = WorkflowSpec.from_json({"name": "s", "input": "", "root": {
        "type": "sequence", "steps": [leaf("dev"), review]}})
    engine = WorkflowEngine(InboxParent(), executor=Executor(), close_driver=False, run_id="steer")
    run = asyncio.create_task(engine.run(spec))
    await started.wait()
    steering = asyncio.create_task(engine.steer_node("dev", "sid", "fix"))
    await asyncio.sleep(0)
    finish.set()
    await asyncio.sleep(0)
    assert not seen
    delivered.set()
    await steering
    result = await run
    assert result.status == "completed"
    assert seen[0] == "flushed" and "new evidence" in seen[1]
