import asyncio

import pytest

from power_loop.workflow import WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine, WorkflowRunError


class Parent:
    async def new_session(self, **kwargs):
        return "driver"


def leaf(n):
    return {"type": "agent", "id": n, "spec": {"name": n, "system_prompt": n}}


@pytest.mark.asyncio
@pytest.mark.parametrize("with_loop", [False, True])
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
async def test_failed_branch_resumes_while_sibling_runs_and_review_blocks_assembly(with_loop, queued, outcome):
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
            return {"status": outcome, "final_text": "repaired", "session_id": sid}

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
    repair = (engine.start_node_continuation("dev", "finish development", 12) if queued else
              asyncio.create_task(engine.continue_node("dev", "finish development", 12)))
    await asyncio.wait_for(repairing.wait(), 2)
    with pytest.raises(WorkflowRunError, match="active|in progress"):
        engine.start_node_continuation("dev", "duplicate", 12)
    sibling.set()
    await asyncio.sleep(0)
    assert "review" not in runs and "assembly" not in runs
    finish_repair.set()
    await repair
    result = await task
    if outcome == "completed":
        assert result.status == "completed"
        assert runs == ["dev", "sibling", "review", "assembly"]
        assert result.results["dev"].text == "repaired"
    else:
        assert result.status != "completed" and "assembly" not in runs and "review" not in runs
    with pytest.raises(WorkflowRunError, match="settling"):
        engine.start_node_continuation("dev", "too late", 12)


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


@pytest.mark.asyncio
async def test_rework_exception_never_replays_side_effectful_leaf():
    calls = []
    class Executor:
        async def run_agent(self, spec, user_input, **kwargs):
            calls.append(spec.name)
            return {"status": "completed", "session_id": spec.name,
                    "final_text": '{"pass": false}' if spec.name == "review" else "artifact created"}
        async def continue_agent(self, *args, **kwargs):
            raise RuntimeError("transport interrupted after mutation")
    review = {**leaf("review"), "output_schema": {"name": "Review", "schema": {
        "type": "object", "properties": {"pass": {"type": "boolean"}}, "required": ["pass"]}}}
    spec = WorkflowSpec.from_json({"name": "r", "input": "", "root": {
        "type": "loop", "id": "repair", "max_iterations": 2, "continue_nodes": ["dev"],
        "until": {"ref": "review.pass"}, "body": {"type": "sequence", "steps": [leaf("dev"), review]}}})
    result = await WorkflowEngine(Parent(), executor=Executor(), close_driver=False).run(spec)
    assert result.status == "failed"
    assert calls.count("dev") == 1
    assert result.results["dev"].session_id == "dev"


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["template", "file"])
@pytest.mark.parametrize("consumer_state", ["active", "settled"])
async def test_cross_branch_dependency_refuses_local_repair_before_any_side_effect(reference, consumer_state):
    held = asyncio.Event()
    calls = []
    continued = []

    class Executor:
        async def run_agent(self, node_spec, user_input, **kwargs):
            calls.append(node_spec.name)
            if node_spec.name in ({"consumer", "blocker"} if consumer_state == "active" else {"blocker"}):
                await held.wait()
            return {"status": "completed", "final_text": '{"answer":42}', "session_id": node_spec.name+"-sid"}

        async def continue_agent(self, *args, **kwargs):
            continued.append(1)
            raise AssertionError("rejected repair must never dispatch")

    producer = {**leaf("dev"), "output_file": "outputs/dev.txt"}
    consumer = {**leaf("consumer"), "input": "{{dev.answer}}" if reference == "template" else "@@FILEREF:outputs/dev.txt@@"}
    specification = WorkflowSpec.from_json({"name": "cross-ref", "root": {"type": "parallel", "branches": [producer, consumer, leaf("blocker")]}})
    steps = []
    async def observed(step):
        steps.append((step.node_id, step.status))
    engine = WorkflowEngine(Parent(), executor=Executor(), close_driver=False, on_step=observed)
    run = asyncio.create_task(engine.run(specification))
    try:
        await wait_until(lambda: "blocker" in calls and "dev" in engine._results and
                         ("consumer" in engine._results if consumer_state == "settled" else "consumer" in calls))
        before = dict(engine._results), list(steps), dict(engine._replay)
        with pytest.raises(WorkflowRunError, match="active|outside this branch"):
            engine.start_node_continuation("dev", "correct", 3)
        assert (dict(engine._results), steps, engine._replay) == before
        assert not continued and not engine._repairs
    finally:
        held.set()
        await run


@pytest.mark.asyncio
async def test_parallel_join_closes_all_branch_intake_before_awaiting_repairs():
    original_release, repair_one, repair_two = asyncio.Event(), asyncio.Event(), asyncio.Event()
    repairing = set()
    calls = []

    class Executor:
        async def run_agent(self, node_spec, user_input, **kwargs):
            calls.append(node_spec.name)
            if node_spec.name == "blocker":
                await original_release.wait()
            return {"status": "completed", "final_text": "old", "session_id": node_spec.name+"-sid"}

        async def continue_agent(self, sid, *args, **kwargs):
            repairing.add(sid)
            await (repair_one if sid == "one-sid" else repair_two).wait()
            return {"status": "completed", "final_text": "new", "session_id": sid}

    specification = WorkflowSpec.from_json({"name": "join", "root": {"type": "sequence", "steps": [
        {"type": "parallel", "branches": [leaf("one"), leaf("two"), leaf("blocker")]}, leaf("assembly")]}})
    engine = WorkflowEngine(Parent(), executor=Executor(), close_driver=False)
    run = asyncio.create_task(engine.run(specification))
    try:
        await wait_until(lambda: "one" in engine._results and "two" in engine._results and "blocker" in calls)
        one = engine.start_node_continuation("one", "repair", 3)
        two = engine.start_node_continuation("two", "repair", 3)
        await wait_until(lambda: len(repairing) == 2)
        repair_one.set()
        await one
        original_release.set()
        await wait_until(lambda: bool(engine._continuation_barriers))
        assert not two.done() and "assembly" not in calls
        with pytest.raises(WorkflowRunError, match="intake is closed"):
            engine.start_node_continuation("one", "late new work", 3)
        repair_two.set()
        await two
        result = await run
        assert result.status == "completed" and calls.count("assembly") == 1
    finally:
        original_release.set()
        repair_one.set()
        repair_two.set()
        await asyncio.gather(run, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["task_cancel", "scope_error"])
async def test_run_exit_cancels_and_drains_repairs_before_closing_the_driver(ending):
    original_release, repairing = asyncio.Event(), asyncio.Event()
    stopped = asyncio.Event()
    calls, closed = [], []

    class OwnedParent(Parent):
        async def close_session(self, sid, **kwargs):
            assert sid == "driver" and kwargs == {"cascade": False}
            assert stopped.is_set()
            closed.append(sid)

    class Executor:
        async def run_agent(self, node_spec, user_input, **kwargs):
            calls.append(node_spec.name)
            if node_spec.name == "blocker":
                await original_release.wait()
            return {"status": "completed", "final_text": "old", "session_id": node_spec.name+"-sid"}

        async def continue_agent(self, sid, *args, **kwargs):
            repairing.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    specification = WorkflowSpec.from_json({"name": "stop", "root": {"type": "parallel", "branches": [leaf("dev"), leaf("blocker")]}})
    engine = WorkflowEngine(OwnedParent(), executor=Executor())
    original_exec = engine._exec_parallel
    async def execute(*args):
        if ending == "task_cancel":
            return await original_exec(*args)
        task = asyncio.create_task(original_exec(*args))
        try:
            await repairing.wait()
            raise WorkflowRunError("scope failed after accepting repair")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    engine._exec_parallel = execute
    run = asyncio.create_task(engine.run(specification))
    try:
        await wait_until(lambda: "dev" in engine._results and "blocker" in calls)
        repair = engine.start_node_continuation("dev", "repair", 3)
        await repairing.wait()
        if ending == "task_cancel":
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
        else:
            assert (await run).status == "failed"
        assert repair.cancelled() and stopped.is_set() and closed == ["driver"]
        assert all(task.done() for task in engine._repairs.values())
        assert not engine._accept_continuations
    finally:
        original_release.set()
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_continuation_caller_leaves_registered_work_for_original_engine_to_join():
    original_release, repair_release, repairing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class Executor:
        async def run_agent(self, node_spec, user_input, **kwargs):
            if node_spec.name == "blocker":
                await original_release.wait()
            return {"status": "completed", "final_text": "old", "session_id": node_spec.name+"-sid"}
        async def continue_agent(self, sid, *args, **kwargs):
            repairing.set()
            await repair_release.wait()
            return {"status": "completed", "final_text": "new", "session_id": sid}
    specification = WorkflowSpec.from_json({"name": "owner", "root": {"type": "parallel", "branches": [leaf("dev"), leaf("blocker")]}})
    engine = WorkflowEngine(Parent(), executor=Executor(), close_driver=False)
    run = asyncio.create_task(engine.run(specification))
    try:
        await wait_until(lambda: "dev" in engine._results)
        caller = asyncio.create_task(engine.continue_node("dev", "repair", 3))
        await repairing.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        [repair] = engine._repairs.values()
        assert not repair.done()
        original_release.set()
        await wait_until(lambda: bool(engine._continuation_barriers))
        assert not run.done()
        repair_release.set()
        assert (await run).status == "completed"
        assert engine._results["dev"].text == "new"
    finally:
        original_release.set()
        repair_release.set()
        await asyncio.gather(run, return_exceptions=True)


@pytest.mark.asyncio
async def test_live_replay_of_foreach_prefix_does_not_charge_its_aggregate_twice():
    release, sibling_started = asyncio.Event(), asyncio.Event()
    calls = []
    class Executor:
        async def run_agent(self, node_spec, user_input, **kwargs):
            calls.append(node_spec.name)
            if node_spec.name == "blocker":
                sibling_started.set()
                await release.wait()
            return {"status": "completed", "final_text": "old", "session_id": node_spec.name+"-sid", "usage": {"total_tokens": 10}}
        async def continue_agent(self, sid, *args, **kwargs):
            return {"status": "completed", "final_text": "new", "session_id": sid, "usage": {"total_tokens": 7}}
    specification = WorkflowSpec.from_json({"name": "usage", "root": {"type": "parallel", "branches": [
        {"type": "sequence", "steps": [
            {"type": "foreach", "id": "prefix", "items": [1, 2], "as": "item", "body": leaf("body")}, leaf("dev"), leaf("review")]},
        leaf("blocker")]}})
    engine = WorkflowEngine(Parent(), executor=Executor(), close_driver=False)
    run = asyncio.create_task(engine.run(specification))
    try:
        await sibling_started.wait()
        await wait_until(lambda: "review" in engine._results)
        repair = engine.start_node_continuation("dev", "correct", 3)
        await repair
        assert calls.count("body") == 2 and calls.count("dev") == 1 and calls.count("review") == 2
        release.set()
        result = await run
        assert result.status == "completed" and result.usage["total_tokens"] == 67
    finally:
        release.set()
        await asyncio.gather(run, return_exceptions=True)


@pytest.mark.asyncio
async def test_repair_task_creation_failure_closes_coroutine_without_mutating_results(monkeypatch):
    import inspect
    release = asyncio.Event()
    calls, coroutines = [], []
    class Executor:
        async def run_agent(self, node_spec, user_input, **kwargs):
            calls.append(node_spec.name)
            if node_spec.name == "blocker":
                await release.wait()
            return {"status": "completed", "final_text": "old", "session_id": node_spec.name+"-sid"}
        async def continue_agent(self, *args, **kwargs):
            raise AssertionError("task creation failed; no dispatch is permitted")
    specification = WorkflowSpec.from_json({"name": "creation", "root": {"type": "parallel", "branches": [leaf("dev"), leaf("blocker")]}})
    engine = WorkflowEngine(Parent(), executor=Executor(), close_driver=False)
    run = asyncio.create_task(engine.run(specification))
    try:
        await wait_until(lambda: "dev" in engine._results and "blocker" in calls)
        before = dict(engine._results)
        def fail_create(coroutine, **kwargs):
            coroutines.append(coroutine)
            raise RuntimeError("scheduler unavailable before dispatch")
        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "create_task", fail_create)
            with pytest.raises(RuntimeError, match="scheduler unavailable"):
                engine.start_node_continuation("dev", "repair", 3)
        assert len(coroutines) == 1 and inspect.getcoroutinestate(coroutines[0]) == inspect.CORO_CLOSED
        assert engine._results == before and not engine._repairs and not engine._resume_sessions
        release.set()
        assert (await run).status == "completed"
    finally:
        release.set()
        await asyncio.gather(run, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["template", "file"])
async def test_repair_cannot_rerun_earlier_work_before_its_changed_dependency(reference):
    release = asyncio.Event()
    calls = []
    class Executor:
        async def run_agent(self, node_spec, user_input, **kwargs):
            calls.append(node_spec.name)
            if node_spec.name == "blocker":
                await release.wait()
            return {"status": "completed", "final_text": "old", "session_id": node_spec.name+"-sid"}
        async def continue_agent(self, *args, **kwargs):
            raise AssertionError("backward dependency must be refused before dispatch")
    # Templates/files are not reachability-checked by from_json. Such an earlier
    # consumer cannot be repaired correctly by rerunning it before the target.
    prefix = {**leaf("prefix"), "input": "{{dev.answer}}" if reference == "template" else "@@FILEREF:outputs/dev.txt@@"}
    dev = {**leaf("dev"), "output_file": "outputs/dev.txt"}
    specification = WorkflowSpec.from_json({"name": "backward", "root": {"type": "parallel", "branches": [
        {"type": "sequence", "steps": [prefix, dev, leaf("review")]}, leaf("blocker")]}})
    engine = WorkflowEngine(Parent(), executor=Executor(), close_driver=False)
    run = asyncio.create_task(engine.run(specification))
    try:
        await wait_until(lambda: "review" in engine._results and "blocker" in calls)
        before = dict(engine._results), list(calls)
        with pytest.raises(WorkflowRunError, match="earlier work"):
            engine.start_node_continuation("dev", "correct", 3)
        assert (dict(engine._results), calls) == before and not engine._repairs
    finally:
        release.set()
        await run
