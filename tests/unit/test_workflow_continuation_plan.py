"""Dependency closure is computed before any continuation can affect execution."""
from dataclasses import replace

import pytest

from power_loop.runtime.spec import AgentSpec
from power_loop.workflow.continuation import plan_continuation, replay_results
from power_loop.workflow.engine import WorkflowRunError
from power_loop.workflow.result import AgentResult
from power_loop.workflow.resume import continuation_replay
from power_loop.workflow.spec import (
    AgentNode,
    BranchNode,
    ForeachNode,
    LoopNode,
    ParallelNode,
    SequenceNode,
    WorkflowSpec,
)

pytestmark = pytest.mark.unit


def agent(name, **kwargs):
    return AgentNode(type="agent", id=name, spec=AgentSpec(name=name, system_prompt=name), **kwargs)


def sequence(*nodes, id=None):
    return SequenceNode(type="sequence", steps=nodes, id=id)


def parallel(*nodes, id=None):
    return ParallelNode(type="parallel", branches=nodes, id=id)


def spec(root):
    return WorkflowSpec(name="plan", root=root)


def result(name, status="completed"):
    return AgentResult(node_id=name, status=status, text="original", session_id=name+"-sid")


def test_closure_covers_sequence_tail_and_ancestors_without_invalidating_parallel_sibling():
    design = spec(sequence(parallel(sequence(agent("prefix"), agent("dev"), agent("review"), id="work"),
                                    agent("independent"), id="fanout"), agent("assemble"), id="root"))
    plan = plan_continuation(design, "dev")
    assert plan.invalidated == {"dev", "review", "work", "fanout", "assemble", "root"}
    assert plan.target_path == (0, 0, 1)
    assert plan.to_dict()["reasons"]["dev"] == ["target"]
    assert "sequence" in plan.to_dict()["reasons"]["review"]
    old = {name: result(name) for name in ["prefix", "dev", "review", "independent", "assemble"]}
    original = dict(old)
    invalidated = set()
    replay = continuation_replay(design, "dev", old, invalidated=invalidated)
    assert invalidated == plan.invalidated and set(replay) == {"prefix", "independent"}
    assert old == original and old["dev"].text == "original"
    assert plan == plan_continuation(design, "dev")
    plan.to_dict()["invalidated"].clear()
    assert "dev" in plan.invalidated  # inspection cannot change the frozen decision


@pytest.mark.parametrize("kind", ["inputs_from", "when", "template", "file", "branch", "foreach", "loop"])
def test_declared_references_propagate_through_containers_and_transitive_consumers(kind):
    producer = agent("source", output_file="out/source.txt")
    consumers = {
        "inputs_from": agent("consumer", inputs_from=("source",)),
        "when": agent("consumer", when={"ref": "source.ready"}),
        "template": agent("consumer", input="value={{ source.answer }}"),
        "file": agent("consumer", input="@@FILEREF:out/./source.txt[-3:]@@"),
        "branch": BranchNode(type="branch", id="choice", on="source.answer", cases={"a": agent("consumer")}, default=agent("alternate")),
        "foreach": ForeachNode(type="foreach", id="mapped", as_var="item", body=agent("consumer"), items_from="source.items"),
        "loop": LoopNode(type="loop", id="repeated", body=agent("consumer"), until={"ref": "source.ready"}),
    }
    # Programmatic specs are permitted to bypass parsing. Planning must still
    # find dependencies, even if parser ordering would reject this parallel ref.
    design = spec(parallel(producer, consumers[kind], agent("transitive", input="{{consumer.answer}}"), agent("untouched")))
    plan = plan_continuation(design, "source")
    assert {"source", "consumer", "transitive"} <= plan.invalidated
    assert "untouched" not in plan.invalidated
    if kind == "branch":
        assert {"choice", "alternate"} <= plan.invalidated
    if kind == "foreach":
        assert "mapped" in plan.invalidated
    if kind == "loop":
        assert "repeated" in plan.invalidated
    reason = {"foreach": "items_from", "loop": "until"}.get(kind, kind)
    assert any(reason in reasons for _, reasons in plan.reasons)


@pytest.mark.parametrize("reference", ["{{source}}", "{{unknown.answer}}", "@@FILEREF:external.txt@@", "{{source.too.deep}}"])
def test_only_resolved_template_or_known_output_file_dependencies_are_inferred(reference):
    design = spec(parallel(agent("source"), agent("other", input=reference)))
    assert plan_continuation(design, "source").invalidated == {"source"}


def test_inputs_from_uses_exact_keys_and_unknown_condition_is_not_guessed():
    design = spec(parallel(agent("source"), agent("other", inputs_from=("source.key",), when={"ref": "unknown.ready"})))
    assert plan_continuation(design, "source").invalidated == {"source"}


def test_duplicate_output_files_invalidate_consumers_of_either_known_producer():
    design = spec(parallel(agent("source", output_file="same.txt"), agent("other", output_file="same.txt"),
                           agent("consumer", input="@@FILEREF:same.txt@@")))
    assert plan_continuation(design, "source").invalidated == {"source", "consumer"}


def test_fallback_is_owned_evidence_and_cannot_be_targeted_as_an_independent_session():
    fallback = sequence(agent("fallback_prepare"), agent("fallback_finish"), id="fallback")
    design = spec(parallel(agent("dev", fallback=fallback), agent("reader", input="{{fallback_finish.answer}}")))
    assert plan_continuation(design, "dev").invalidated == {"dev", "fallback", "fallback_prepare", "fallback_finish", "reader"}
    with pytest.raises(WorkflowRunError, match="owning agent"):
        plan_continuation(design, "fallback_prepare")


def test_dependency_on_fallback_reaches_its_original_agent_and_all_substituted_evidence():
    fallback = agent("fallback", input="{{source.answer}}")
    design = spec(parallel(agent("source"), agent("dev", fallback=fallback), agent("reader", input="{{dev.answer}}")))
    assert plan_continuation(design, "source").invalidated == {"source", "fallback", "dev", "reader"}


def test_internal_loop_verdict_recomputes_aggregate_and_tail_but_preserves_prefix():
    loop = LoopNode(type="loop", id="loop", until={"ref": "review.pass"},
                    body=sequence(agent("prefix"), agent("dev"), agent("review")))
    design = spec(sequence(loop, agent("assemble")))
    plan = plan_continuation(design, "dev")
    assert plan.invalidated == {"dev", "review", "loop", "assemble"}
    assert "until" in dict(plan.reasons)["loop"]


def test_dependency_cycles_close_without_recursive_execution_or_hanging():
    design = spec(parallel(agent("source", input="{{other.answer}}"), agent("other", input="{{source.answer}}")))
    assert plan_continuation(design, "source").invalidated == {"source", "other"}


@pytest.mark.parametrize("target", ["missing", "container"])
def test_unknown_or_non_agent_target_is_refused(target):
    with pytest.raises(WorkflowRunError, match="unknown continuation agent"):
        plan_continuation(spec(sequence(agent("dev"), id="container")), target)


def test_programmatic_duplicate_ids_are_refused():
    with pytest.raises(WorkflowRunError, match="ambiguous"):
        plan_continuation(spec(parallel(agent("same"), agent("same"))), "same")


@pytest.mark.parametrize("nested", [False, True])
def test_foreach_target_requires_an_iteration_even_under_nested_containers(nested):
    child = sequence(agent("dev")) if nested else agent("dev")
    root = ForeachNode(type="foreach", as_var="item", items=(1, 2), body=child)
    with pytest.raises(WorkflowRunError, match="iteration-specific"):
        plan_continuation(spec(root), "dev")


@pytest.mark.parametrize("status", ["completed", "skipped", "failed", "cancelled", "context_checkpoint", "hit_round_limit", "pending", "running"])
def test_unaffected_terminal_evidence_is_preserved_without_retrying_its_side_effects(status):
    design = spec(parallel(agent("dev"), agent("other")))
    old = {"dev": result("dev"), "other": result("other", status), "outside": result("outside")}
    replay = replay_results(design, plan_continuation(design, "dev"), old)
    assert set(replay) == ({"other"} if status not in ("pending", "running") else set())
    if replay:
        assert replay["other"] is old["other"]


@pytest.mark.parametrize("kind", ["loop", "foreach"])
def test_replayed_aggregates_exclude_body_results_atomically(kind):
    aggregate = (LoopNode(type="loop", id="aggregate", body=agent("body"), until={"ref": "body.ok"}) if kind == "loop" else
                 ForeachNode(type="foreach", id="aggregate", as_var="item", items=(1,), body=agent("body")))
    design = spec(parallel(agent("dev"), aggregate))
    old = {name: result(name) for name in ("dev", "aggregate", "body")}
    assert set(replay_results(design, plan_continuation(design, "dev"), old)) == {"aggregate"}


def test_parser_still_rejects_explicit_parallel_sibling_dependencies():
    from power_loop.workflow.spec import WorkflowSpecError
    design = spec(parallel(agent("source"), agent("other", inputs_from=("source",))))
    with pytest.raises(WorkflowSpecError, match="not guaranteed"):
        WorkflowSpec.from_json(design.to_dict())


def test_external_until_does_not_invalidate_an_unrelated_loop():
    unrelated = LoopNode(type="loop", id="loop", body=agent("body"), until={"ref": "unknown.ready"})
    design = spec(parallel(agent("source"), unrelated))
    assert plan_continuation(design, "source").invalidated == {"source"}


def test_new_deep_edge_expands_a_container_already_marked_by_its_descendant():
    # The loop first becomes affected through a changed descendant, then through
    # an external changed condition. The latter must still expand its prefix.
    loop = LoopNode(type="loop", id="loop", body=sequence(agent("prefix"), agent("dev", input="{{source.answer}}")),
                    until={"ref": "source.ready"})
    design = spec(parallel(agent("source"), loop))
    assert plan_continuation(design, "source").invalidated == {"source", "loop", "prefix", "dev"}


def test_plan_covers_fallback_defaults_and_nested_aggregate_replay():
    inner = ForeachNode(type="foreach", id="map", as_var="item", items=(1,), body=agent("body"))
    outer = LoopNode(type="loop", id="loop", until={"ref": "outside.ready"}, body=inner)
    design = spec(parallel(agent("dev"), outer, agent("outside")))
    old = {name: result(name) for name in ("dev", "loop", "map", "body", "outside")}
    assert set(replay_results(design, plan_continuation(design, "dev"), old)) == {"loop", "outside"}
    changed = replace(design, root=parallel(agent("dev"), BranchNode(type="branch", on="dev.choice", cases={}, default=agent("default"))))
    assert plan_continuation(changed, "dev").invalidated == {"dev", "default"}


def test_changed_foreach_body_dependency_invalidates_all_iterations_and_prefixes():
    body = sequence(agent("prefix"), agent("consumer", input="{{source.answer}}"), agent("tail"))
    mapped = ForeachNode(type="foreach", id="map", as_var="item", items=(1, 2), body=body)
    design = spec(parallel(agent("source"), mapped, agent("independent")))
    plan = plan_continuation(design, "source")
    assert plan.invalidated == {"source", "map", "prefix", "consumer", "tail"}
    assert "independent" not in plan.invalidated


def test_missing_foreach_aggregate_never_replays_an_individual_iteration_result():
    mapped = ForeachNode(type="foreach", id="map", as_var="item", items=(1, 2), body=agent("body"))
    design = spec(parallel(agent("dev"), mapped))
    old = {"dev": result("dev"), "body": result("body")}
    assert not replay_results(design, plan_continuation(design, "dev"), old)
