"""power_loop.workflow — deterministic, declarative multi-agent workflows.

An OPTIONAL submodule (import it explicitly; it adds no dependencies to the
core). It layers a *code-driven* workflow capability on top of the existing
sub-agent primitives:

* a declarative JSON DSL (:class:`WorkflowSpec`) the parent LLM can author at
  runtime, validated strictly on creation;
* a deterministic in-process engine (:class:`WorkflowEngine`) that interprets
  ``sequence`` / ``parallel`` / ``foreach`` / ``branch`` whose leaves are
  ordinary sub-agents (``run_agent_spec``);
* typed results (:class:`AgentResult` / :class:`WorkflowResult`) and an optional
  shared token budget (:class:`SharedBudget`);
* host API (:func:`create_workflow`, :class:`Workflow`) and an LLM-facing
  meta-tool (:data:`CREATE_WORKFLOW_DEFINITION`, :func:`register_workflow_tools`).

Quick start (host code)::

    from power_loop import StatefulAgentLoop
    from power_loop.workflow import create_workflow

    loop = StatefulAgentLoop(llm=llm, db_path="./app.db")
    wf = create_workflow(spec_json, parent_loop=loop)
    result = await wf.run()
    print(result.summary())

Beyond the synchronous in-process core, this package also provides detached
execution with a completion callback that wakes the parent agent
(:func:`run_detached` + :func:`register_wake_guard`) and orchestration-level
resume across a process restart (:func:`resume_run` / :func:`resume_detached`,
which replay completed steps from the run journal and re-run only the unfinished
tail). An out-of-process executor remains deferred (see
``docs/dynamic-workflow-feasibility.md``).
"""

from __future__ import annotations

from .api import Workflow, create_workflow
from .engine import (
    Executor,
    InProcessExecutor,
    WorkflowEngine,
    WorkflowRunError,
    in_workflow,
)
from .introspect import get_workflow, list_workflows
from .result import AgentResult, SharedBudget, WorkflowResult, WorkflowRunHandle
from .resume import (
    rehydrate,
    replayable_node_ids,
    resume_detached,
    resume_run,
)
from .runner import register_wake_guard, run_detached
from .spec import (
    AgentNode,
    BranchNode,
    ForeachNode,
    ParallelNode,
    SequenceNode,
    WorkflowBudget,
    WorkflowNode,
    WorkflowSpec,
    WorkflowSpecError,
)
from .subprocess_executor import SubprocessExecutor
from .tool import (
    CREATE_WORKFLOW_DEFINITION,
    WORKFLOW_STATUS_DEFINITION,
    register_workflow_tools,
)
from .worker import WorkerBootstrap, WorkerBootstrapError, run_spec_isolated

__all__ = [
    # spec / DSL
    "WorkflowSpec",
    "WorkflowSpecError",
    "WorkflowBudget",
    "WorkflowNode",
    "AgentNode",
    "SequenceNode",
    "ParallelNode",
    "ForeachNode",
    "BranchNode",
    # engine
    "WorkflowEngine",
    "Executor",
    "InProcessExecutor",
    "WorkflowRunError",
    "in_workflow",
    # results
    "AgentResult",
    "WorkflowResult",
    "SharedBudget",
    "WorkflowRunHandle",
    # host api
    "Workflow",
    "create_workflow",
    # detached execution + wake (D3)
    "run_detached",
    "register_wake_guard",
    # orchestration-level resume across restart (full tier)
    "resume_run",
    "resume_detached",
    "rehydrate",
    "replayable_node_ids",
    # introspection (D4)
    "list_workflows",
    "get_workflow",
    # isolated worker core + out-of-process executor (Phase 0/1)
    "WorkerBootstrap",
    "WorkerBootstrapError",
    "run_spec_isolated",
    "SubprocessExecutor",
    # llm-facing tools
    "CREATE_WORKFLOW_DEFINITION",
    "WORKFLOW_STATUS_DEFINITION",
    "register_workflow_tools",
]
