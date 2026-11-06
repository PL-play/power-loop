"""Graph validation and soft budget arithmetic; runtime cases use test_workflow_task_runtime_192."""
import pytest

from power_loop.workflow import AgentNode, SequenceNode, SharedBudget, WorkflowSpec, WorkflowSpecError


def test_valid_spec_parses():
    spec = WorkflowSpec.from_json({
        "name": "wf", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "p"}},
        ]},
    })
    assert spec.name == "wf"
    assert isinstance(spec.root, SequenceNode)
    assert isinstance(spec.root.steps[0], AgentNode)


def test_invalid_spec_aggregates_all_problems():
    bad = {
        "name": "",
        "root": {
            "type": "foreach", "as": "x", "items_from": "ghost.k",
            "body": {"type": "agent", "spec": {"name": "a"}},  # missing id + system_prompt
            "max_concurrency": 0, "on_error": "boom", "retires": 1,  # typo'd key
        },
    }
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json(bad)
    problems = ei.value.problems
    # All problems reported at once (loud-fail), not just the first.
    assert len(problems) >= 6
    joined = "\n".join(problems)
    assert "name" in joined
    assert "retires" in joined  # unknown key surfaced
    assert "ghost" in joined  # dangling items_from reference
    assert "max_concurrency" in joined
    assert "on_error" in joined
    assert "system_prompt" in joined  # nested AgentSpec error folded in (no crash)


def test_unknown_top_level_key_rejected():
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "agent", "id": "a",
                                "spec": {"name": "a", "system_prompt": "p"}}, "oops": 1})
    assert any("oops" in p for p in ei.value.problems)


def test_foreach_requires_items_source():
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {
            "type": "foreach", "as": "i",
            "body": {"type": "agent", "id": "b", "spec": {"name": "b", "system_prompt": "p"}},
        }})
    assert any("items_from" in p or "items" in p for p in ei.value.problems)


def test_duplicate_agent_ids_rejected():
    """Agent ids key results/journal/replay — duplicates must fail validation,
    not silently mis-replay the wrong node on resume."""
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "dup", "spec": {"name": "a", "system_prompt": "p"}},
            {"type": "agent", "id": "dup", "spec": {"name": "b", "system_prompt": "p"}},
        ]}})
    assert any("duplicate" in p and "dup" in p for p in ei.value.problems)


def test_container_id_colliding_with_agent_id_rejected():
    """C5 regression: a container (foreach aggregate) id sharing an agent id
    collides in the _results/journal/replay namespace — must be rejected globally,
    not only checked across agent ids."""
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "x", "spec": {"name": "a", "system_prompt": "p"}},
            {"type": "foreach", "id": "x", "items": ["a"], "as": "i",
             "body": {"type": "agent", "id": "leaf", "spec": {"name": "b", "system_prompt": "p"}}},
        ]}})
    assert any("duplicate node id" in p and "x" in p for p in ei.value.problems)


def test_reference_to_foreach_body_id_rejected():
    """C4 regression: a foreach-body node id is not individually addressable on
    resume; inputs_from/items_from/branch.on must not reference it (would work on
    attempt 1 but raise on resume)."""
    # branch.on → a foreach-body agent id
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            {"type": "foreach", "id": "f", "items": ["a", "b"], "as": "i",
             "body": {"type": "agent", "id": "bodyleaf",
                      "spec": {"name": "b", "system_prompt": "p"}}},
            {"type": "branch", "on": "bodyleaf.choice",
             "cases": {"y": {"type": "agent", "id": "z", "spec": {"name": "z", "system_prompt": "p"}}}},
        ]}})
    assert any("foreach body" in p and "bodyleaf" in p for p in ei.value.problems)

    # inputs_from → a foreach-body agent id
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            {"type": "foreach", "id": "f", "items": ["a"], "as": "i",
             "body": {"type": "agent", "id": "bodyleaf",
                      "spec": {"name": "b", "system_prompt": "p"}}},
            {"type": "agent", "id": "writer", "inputs_from": ["bodyleaf"],
             "spec": {"name": "w", "system_prompt": "p"}},
        ]}})
    assert any("foreach body" in p and "bodyleaf" in p for p in ei.value.problems)

    # items_from → a foreach-body agent id
    with pytest.raises(WorkflowSpecError) as ei:
        WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
            {"type": "foreach", "id": "f", "items": ["a"], "as": "i",
             "body": {"type": "agent", "id": "bodyleaf",
                      "spec": {"name": "b", "system_prompt": "p"}}},
            {"type": "foreach", "id": "g", "items_from": "bodyleaf.list", "as": "j",
             "body": {"type": "agent", "id": "leaf2", "spec": {"name": "c", "system_prompt": "p"}}},
        ]}})
    assert any("foreach body" in p and "bodyleaf" in p for p in ei.value.problems)


def test_reference_to_foreach_aggregate_id_still_allowed():
    """The fix must NOT reject the valid pattern: referencing a top-level foreach's
    own aggregate id (which IS addressable/replayable) from inputs_from, items_from
    AND branch.on — all three ref kinds are symmetric."""
    spec = WorkflowSpec.from_json({"name": "w", "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "plan",
         "spec": {"name": "p", "system_prompt": "p"},
         "output_schema": {"name": "P", "schema": {"type": "object", "required": ["subtopics"],
            "properties": {"subtopics": {"type": "array", "items": {"type": "string"}}}}}},
        {"type": "foreach", "id": "research", "items_from": "plan.subtopics", "as": "t",
         "body": {"type": "agent", "id": "r", "spec": {"name": "r", "system_prompt": "p"}}},
        # items_from → a foreach AGGREGATE id (previously wrongly rejected as unknown)
        {"type": "foreach", "id": "expand", "items_from": "research.items", "as": "u",
         "body": {"type": "agent", "id": "e", "spec": {"name": "e", "system_prompt": "p"}}},
        # branch.on → a foreach AGGREGATE id
        {"type": "branch", "on": "expand.items",
         "cases": {"x": {"type": "agent", "id": "c", "spec": {"name": "c", "system_prompt": "p"}}}},
        {"type": "agent", "id": "write", "inputs_from": ["research"],
         "spec": {"name": "w", "system_prompt": "p"}},
    ]}})
    assert spec.name == "w"


def test_shared_budget_can_spawn_and_commit():
    b = SharedBudget(100)
    assert b.can_spawn()
    b.commit({"total_tokens": 100})
    assert not b.can_spawn()
    assert b.remaining == 0
