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

This is the minimal tier: in-process, synchronous. Detached execution with a
completion callback that wakes the parent agent, orchestration-level resume, and
an out-of-process executor are deferred to later tiers (see
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
from .tool import (
    CREATE_WORKFLOW_DEFINITION,
    WORKFLOW_STATUS_DEFINITION,
    register_workflow_tools,
)

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
    # introspection (D4)
    "list_workflows",
    "get_workflow",
    # llm-facing tools
    "CREATE_WORKFLOW_DEFINITION",
    "WORKFLOW_STATUS_DEFINITION",
    "register_workflow_tools",
]
