"""Host-facing entry points for building and running workflows.

``create_workflow`` validates a :class:`WorkflowSpec` JSON (raising
:class:`WorkflowSpecError` with every problem aggregated) and returns a
:class:`Workflow` handle whose :meth:`Workflow.run` interprets it in-process.

This tier runs synchronously (``await wf.run()``). Detached execution with a
completion callback that wakes the parent agent, and orchestration-level resume,
are the next tiers — :meth:`Workflow.start` is present as a forward-compatible
shape but raises ``NotImplementedError`` for ``detached=True`` for now.
"""

from __future__ import annotations

from typing import Any

from .engine import Executor, WorkflowEngine
from .result import SharedBudget, WorkflowResult
from .spec import WorkflowSpec

__all__ = ["Workflow", "create_workflow"]


class Workflow:
    """A validated workflow bound to a parent loop, ready to run."""

    def __init__(
        self,
        spec: WorkflowSpec,
        *,
        parent_loop: Any,
        executor: Executor | None = None,
        budget: SharedBudget | None = None,
    ) -> None:
        self.spec = spec
        self._loop = parent_loop
        self._executor = executor
        self._budget = budget

    async def run(self) -> WorkflowResult:
        """Interpret the spec to completion (in-process) and return the result."""
        engine = WorkflowEngine(self._loop, executor=self._executor, budget=self._budget)
        return await engine.run(self.spec)

    async def start(self, *, detached: bool = False) -> WorkflowResult:
        """Forward-compatible entry point.

        ``detached=True`` (run in the background, wake the parent agent via a
        completion hook) is a later tier and not yet implemented.
        """
        if detached:
            raise NotImplementedError(
                "detached execution + completion callback is a later tier; "
                "use `await workflow.run()` for synchronous in-process execution."
            )
        return await self.run()


def create_workflow(
    spec_json: str | dict[str, Any] | WorkflowSpec,
    *,
    parent_loop: Any,
    executor: Executor | None = None,
    budget: SharedBudget | None = None,
) -> Workflow:
    """Validate ``spec_json`` and return a runnable :class:`Workflow`.

    Accepts raw JSON / a dict (validated here) or an already-built
    :class:`WorkflowSpec`. Raises
    :class:`~power_loop.workflow.spec.WorkflowSpecError` (all problems
    aggregated) if an unvalidated payload is invalid.
    """
    spec = spec_json if isinstance(spec_json, WorkflowSpec) else WorkflowSpec.from_json(spec_json)
    return Workflow(spec, parent_loop=parent_loop, executor=executor, budget=budget)
