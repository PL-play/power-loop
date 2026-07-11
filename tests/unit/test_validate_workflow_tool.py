"""3.16: ``validate_workflow`` — dry-run spec checking with aggregated errors."""

from __future__ import annotations

import pytest

from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import register_workflow_tools
from power_loop.workflow.spec import WorkflowSpec, WorkflowSpecError

pytestmark = pytest.mark.asyncio

GOOD = {
    "name": "wf",
    "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "plan", "spec": {"name": "p", "system_prompt": "x"},
         "output_schema": {"name": "Plan", "schema": {"type": "object"}}},
        {"type": "agent", "id": "sum", "spec": {"name": "s", "system_prompt": "y"},
         "inputs_from": ["plan"]},
    ]},
}


def _handler(**seams):
    reg = ToolRegistry()
    register_workflow_tools(reg, **seams)
    return reg.get("validate_workflow").handler


async def test_valid_spec_returns_summary():
    out = await _handler()(spec=GOOD)
    assert out.startswith("VALID.")
    assert "2 leaf agent(s): plan, sum" in out
    assert "structured output declared on: plan" in out
    assert "Ready for create_workflow" in out


async def test_invalid_spec_reports_all_problems_at_once():
    bad = {
        "name": "",  # problem 1: empty name
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "a", "spec": {"name": "x", "system_prompt": "p"},
             "inputs_from": ["missing"]},          # problem 2: forward/unknown-ish ref
            {"type": "agent", "id": "a",           # problem 3: duplicate id
             "spec": {"name": "y"}},               # problem 4: spec missing system_prompt
        ]},
    }
    out = await _handler()(spec=bad)
    assert out.startswith("INVALID —")
    # Aggregated: several distinct problems in one reply.
    assert out.count("- ") >= 3
    assert "duplicate node id" in out


async def test_host_policy_rejection_surfaces():
    def spec_transform(spec: WorkflowSpec) -> WorkflowSpec:
        raise WorkflowSpecError(["policy: foreach items exceed the platform limit"])

    out = await _handler(spec_transform=spec_transform)(spec=GOOD)
    assert out.startswith("INVALID (platform policy)")
    assert "platform limit" in out


async def test_validate_never_executes_anything():
    """No loop/session context is required — validation is pure."""
    out = await _handler()(spec=GOOD)  # note: no set_current_loop at all
    assert out.startswith("VALID.")


async def test_missing_spec_argument():
    assert "requires 'spec'" in await _handler()()
