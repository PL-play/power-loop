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

import asyncio
from typing import Any

from power_loop.runtime.cancellation import CancellationToken

from .engine import Executor, WorkflowEngine
from .result import SharedBudget, WorkflowResult, WorkflowRunHandle
from .spec import WorkflowSpec

__all__ = ["Workflow", "create_workflow"]


class Workflow:
    """A validated workflow bound to a parent loop, ready to run.

    ``parent_session_id`` is the agent session that owns this run — required for
    detached execution (it is journaled under that session and woken on
    completion). Synchronous :meth:`run` does not need it.
    """

    def __init__(
        self,
        spec: WorkflowSpec,
        *,
        parent_loop: Any,
        executor: Executor | None = None,
        budget: SharedBudget | None = None,
        parent_session_id: str | None = None,
    ) -> None:
        self.spec = spec
        self._loop = parent_loop
        self._executor = executor
        self._budget = budget
        self._parent_sid = parent_session_id
        self._tasks: set[asyncio.Task[None]] = set()  # retain detached tasks (GC guard)
        # Owned token, flipped by cancel(). Forwarded into the engine → every
        # in-flight sub-agent, so cancelling tears down the whole run.
        self._cancel = CancellationToken()

    def cancel(self, reason: str = "cancelled") -> None:
        """Request cancellation of this workflow.

        In-flight sub-agents stop at their next round/checkpoint and the run
        settles with ``status="cancelled"``; no further steps are spawned. Safe
        to call from any thread or before the run starts. For a detached run,
        prefer :meth:`WorkflowRunHandle.cancel`.
        """
        self._cancel.cancel(reason)

    async def run(self) -> WorkflowResult:
        """Interpret the spec to completion (in-process, synchronous) and return it."""
        engine = WorkflowEngine(
            self._loop, executor=self._executor, budget=self._budget,
            stop_event=self._cancel,
        )
        return await engine.run(self.spec)

    async def start(
        self, *, detached: bool = False, eager_wake: bool = False
    ) -> WorkflowResult | WorkflowRunHandle:
        """Run the workflow.

        * ``detached=False`` (default): run synchronously, return a
          :class:`WorkflowResult`.
        * ``detached=True``: run in the background, return a
          :class:`WorkflowRunHandle` immediately; the parent agent (identified by
          ``parent_session_id``) is woken on completion via a durable timer.
          Requires a ``parent_session_id`` and a running ``TimerRunner`` on the
          host. ``eager_wake=True`` adds a non-durable instant wake on top.
        """
        if detached:
            from .runner import run_detached  # local import avoids import cycle

            return await run_detached(self, eager_wake=eager_wake)
        return await self.run()


def create_workflow(
    spec_json: str | dict[str, Any] | WorkflowSpec,
    *,
    parent_loop: Any,
    executor: Executor | None = None,
    budget: SharedBudget | None = None,
    parent_session_id: str | None = None,
) -> Workflow:
    """Validate ``spec_json`` and return a runnable :class:`Workflow`.

    Accepts raw JSON / a dict (validated here) or an already-built
    :class:`WorkflowSpec`. Raises
    :class:`~power_loop.workflow.spec.WorkflowSpecError` (all problems
    aggregated) if an unvalidated payload is invalid. Pass
    ``parent_session_id`` to enable detached execution.
    """
    spec = spec_json if isinstance(spec_json, WorkflowSpec) else WorkflowSpec.from_json(spec_json)
    return Workflow(
        spec, parent_loop=parent_loop, executor=executor, budget=budget,
        parent_session_id=parent_session_id,
    )
