"""Loop and conditional graph validation."""
import pytest

from power_loop.workflow import WorkflowSpec, WorkflowSpecError


def test_when_validation():

    bad = {"name": "w", "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "x"},
         "when": {"ref": "b.pass", "op": "eq", "value": True}},
        {"type": "agent", "id": "b", "spec": {"name": "b", "system_prompt": "x"}},
    ]}}
    with pytest.raises(WorkflowSpecError) as exc:
        WorkflowSpec.from_json(bad)
    assert any("when references 'b'" in p for p in exc.value.problems)
    with pytest.raises(WorkflowSpecError):
        WorkflowSpec.from_json({"name": "w", "root": {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "x"},
                                               "when": {"ref": "a", "op": "between"}}})


def _loop_spec(max_iterations: int = 3, on_exhausted: str = "fail") -> dict:
    return {"name": "scene", "input": "做场景 pier", "root": {"type": "sequence", "steps": [
        {"type": "loop", "id": "pier", "max_iterations": max_iterations, "on_exhausted": on_exhausted,
         "continue_nodes": ["dev"], "until": {"ref": "review.pass"},
         "body": {"type": "sequence", "steps": [
             {"type": "agent", "id": "dev", "spec": {"name": "d", "system_prompt": "dev the scene"}, "input": "做 pier"},
             {"type": "agent", "id": "review", "spec": {"name": "r", "system_prompt": "review it"}, "inputs_from": ["dev"],
              "output_schema": {"name": "R", "schema": {"type": "object", "properties": {
                  "pass": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}}}}}},
         ]}},
        {"type": "agent", "id": "after", "spec": {"name": "a", "system_prompt": "after"}, "input": "pass={{review.pass}}"},
    ]}}


def test_loop_spec_roundtrip_and_validation():
    from power_loop.workflow.spec import WorkflowSpec

    s = WorkflowSpec.from_json(_loop_spec())
    again = WorkflowSpec.from_json(s.to_dict())
    assert again.to_dict() == s.to_dict()
    bad = _loop_spec()
    bad["root"]["steps"][0]["continue_nodes"] = ["nope"]
    bad["root"]["steps"][0]["until"] = {"ref": "ghost.pass"}
    with pytest.raises(WorkflowSpecError) as exc:
        WorkflowSpec.from_json(bad)
    text = " ".join(exc.value.problems)
    assert "continue_nodes 'nope'" in text and "ghost" in text
