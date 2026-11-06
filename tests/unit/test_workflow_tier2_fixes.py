"""Container failure policy over serializable node requests."""

import asyncio

import pytest

from power_loop.workflow import WorkflowEngine, WorkflowRunError, WorkflowSpec

from ._workflow_fixture_203 import leaf

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("policy", ["halt", "continue"])
async def test_branch_failure_stops_only_halt_siblings(policy):
    class Executor:
        cancelled = False
        finished = False
        started = asyncio.Event()

        async def run_agent(self, request, *, allow_start=True):
            if request.node_path == "bad":
                await self.started.wait()
                raise WorkflowRunError("failed leaf")
            self.started.set()
            try:
                await asyncio.sleep(0.05)
                self.finished = True
                return {"status": "completed", "final_text": "done", "usage": {}}
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    executor = Executor()
    spec = WorkflowSpec.from_json(
        {"name": "failure", "root": {"type": "parallel", "on_error": policy, "branches": [leaf("bad"), leaf("good")]}}
    )
    result = await WorkflowEngine(executor, run_id="graph", driver_sid="driver").run(spec)
    assert result.status == ("failed" if policy == "halt" else "completed")
    assert not result.ok and result.errors
    assert executor.cancelled == (policy == "halt")
    assert executor.finished == (policy == "continue")
