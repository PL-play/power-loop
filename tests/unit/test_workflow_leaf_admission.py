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
