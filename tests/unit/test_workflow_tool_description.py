"""3.15: the create_workflow default description is complete and drift-proof.

The description is the model's only manual for authoring a WorkflowSpec, so it
must cover every AgentSpec field and every node type — these tests fail when a
field/node is added without updating the model-facing text.
"""

from __future__ import annotations

from dataclasses import fields

from power_loop.runtime.spec import AgentSpec
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import register_workflow_tools
from power_loop.workflow.spec import _NODE_TYPES
from power_loop.workflow.tool import (
    _AGENT_SPEC_FIELD_DOCS,
    CREATE_WORKFLOW_DEFINITION,
    WORKFLOW_STATUS_DEFINITION,
)


def test_field_docs_cover_agent_spec_exactly() -> None:
    """Anti-drift guard: adding/renaming an AgentSpec field without updating
    _AGENT_SPEC_FIELD_DOCS fails here."""
    spec_fields = {f.name for f in fields(AgentSpec)}
    documented = set(_AGENT_SPEC_FIELD_DOCS)
    assert documented == spec_fields, (
        f"AgentSpec fields and the create_workflow description drifted: "
        f"missing docs {sorted(spec_fields - documented)}, "
        f"stale docs {sorted(documented - spec_fields)}"
    )


def test_description_mentions_every_field_and_node_type() -> None:
    desc = CREATE_WORKFLOW_DEFINITION.description
    for f in fields(AgentSpec):
        assert f'"{f.name}"' in desc, f"AgentSpec field {f.name!r} absent from description"
    for node_type in _NODE_TYPES:
        assert f'"{node_type}"' in desc, f"node type {node_type!r} absent from description"


def test_description_states_key_authoring_rules() -> None:
    desc = CREATE_WORKFLOW_DEFINITION.description
    # The rules a model most often violates; each costs a validation retry if unstated.
    for token in (
        "items_from",
        "inputs_from",
        "output_schema",
        "node_id.key",
        "{{input}}",
        "globally",          # id uniqueness
        "parallel sibling",  # reachability rule
        "detached",
        "ALL problems at once",
        "EXAMPLE",
    ):
        assert token in desc, f"authoring rule {token!r} absent from description"
    assert "woken" in WORKFLOW_STATUS_DEFINITION.description


def test_description_suffix_appended_only_when_given() -> None:
    reg = ToolRegistry()
    register_workflow_tools(reg)
    assert reg.get("create_workflow").definition.description == (
        CREATE_WORKFLOW_DEFINITION.description
    )
    reg2 = ToolRegistry()
    register_workflow_tools(reg2, description_suffix="HOST LIMITS: concurrency <= 3\n")
    desc2 = reg2.get("create_workflow").definition.description
    assert desc2.startswith(CREATE_WORKFLOW_DEFINITION.description)
    assert desc2.endswith("HOST LIMITS: concurrency <= 3")
    # Schema/handler untouched by the suffix.
    assert reg2.get("create_workflow").definition.required_params == ("spec",)
