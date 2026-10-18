import asyncio

import pytest

from power_loop.workflow import WorkflowSpec
from power_loop.workflow.admission import idle
from power_loop.workflow.engine import WorkflowEngine


class Parent:
    async def new_session(self, **kwargs):
        return "driver"


def leaf(name):
    return {"type": "agent", "id": name, "spec": {"name": name, "system_prompt": name}}


def graph(scope):
    return WorkflowSpec.from_json(
        {
            "name": "fair",
            "input": "",
            "root": {
                "type": "parallel",
                "max_concurrency": 1,
                "concurrency_scope": scope,
                "branches": [{"type": "sequence", "steps": [leaf("dev"), leaf("review")]}, leaf("other")],
            },
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,order", [("branch", ["dev", "review", "other"]), ("leaf", ["dev", "other", "review"])])
async def test_leaf_fairness_and_legacy_branch_semantics(scope, order):
    calls = []

    class Executor:
        async def run_agent(self, spec, *args, **kwargs):
            calls.append(spec.name)
            await asyncio.sleep(0.01)
            return {"status": "completed", "final_text": "ok"}

    result = await WorkflowEngine(Parent(), executor=Executor(), close_driver=False).run(graph(scope))
    assert result.status == "completed"
    assert calls == order


@pytest.mark.asyncio
async def test_idle_settlement_releases_capacity():
    other = asyncio.Event()

    class Executor:
        async def run_agent(self, spec, *args, **kwargs):
            if spec.name == "dev":
                async with idle():
                    await other.wait()
            if spec.name == "other":
                other.set()
            return {"status": "completed", "final_text": "ok"}

    result = await asyncio.wait_for(
        WorkflowEngine(Parent(), executor=Executor(), close_driver=False).run(graph("leaf")), 2
    )
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_queued_leaf_can_be_cancelled_without_running():
    entered, finish = asyncio.Event(), asyncio.Event()
    calls = []

    class Executor:
        async def run_agent(self, spec, *args, **kwargs):
            calls.append(spec.name)
            if spec.name == "dev":
                entered.set()
                await finish.wait()
            return {"status": "completed", "final_text": "ok"}

    engine = WorkflowEngine(Parent(), executor=Executor(), close_driver=False)
    task = asyncio.create_task(engine.run(graph("leaf")))
    await entered.wait()
    assert engine.cancel_node("other") == 1
    await asyncio.sleep(0.15)
    assert engine._results["other"].status == "cancelled"
    finish.set()
    await asyncio.wait_for(task, 2)
    assert "other" not in calls


@pytest.mark.asyncio
async def test_nested_model_requests_share_capacity_without_parent_deadlock():
    from power_loop.workflow.admission import request_slot
    running = maximum = 0

    class Executor:
        async def run_agent(self, spec, *args, **kwargs):
            async def request():
                nonlocal running, maximum
                async with request_slot():
                    running += 1
                    maximum = max(maximum, running)
                    await asyncio.sleep(.01)
                    running -= 1
            # The parent leaf retains its admission slot while children need model slots.
            await asyncio.gather(*(request() for _ in range(4)))
            return {"status": "completed", "final_text": "ok"}

    result = await asyncio.wait_for(WorkflowEngine(Parent(), executor=Executor(), close_driver=False).run(graph("leaf")), 2)
    assert result.status == "completed" and maximum == 1


@pytest.mark.asyncio
async def test_cancelled_model_waiter_does_not_leak_slot():
    from power_loop.runtime.cancellation import CancellationToken
    from power_loop.workflow.admission import REQUEST_GATES, request_slot
    gate = asyncio.Semaphore(1)
    token = REQUEST_GATES.set((gate,))
    stop = CancellationToken()
    async def wait():
        async with request_slot(stop):
            pytest.fail('cancelled waiter entered')
    try:
        await gate.acquire()
        waiter = asyncio.create_task(wait())
        await asyncio.sleep(.01)
        stop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, 1)
        gate.release()
        async with asyncio.timeout(1):
            async with request_slot():
                pass
        assert gate._value == 1
    finally:
        REQUEST_GATES.reset(token)
