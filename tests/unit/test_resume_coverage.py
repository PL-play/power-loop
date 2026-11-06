"""Coverage: crash-recovery replay/journal/budget invariants the existing suite doesn't exercise (probed clean; these lock the higher-arity / multi-restart / branch-decider / spec-budget paths)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_replayable_ids_deeply_nested_parallel_sequence_foreach_branch() -> None:
    from power_loop.workflow import WorkflowSpec, replayable_node_ids

    # s1 runs first (sequence), THEN a parallel of {foreach, branch} that both reference it — so the
    # references are reachable (s1 completes before the parallel). (Referencing s1 from one parallel
    # branch into a sibling would be a cross-sibling race, now rejected by reachability validation.)
    spec = WorkflowSpec.from_json({
        "name": "w", "input": "x",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "s1",
             "spec": {"name": "s1", "system_prompt": "p"},
             "output_schema": {"name": "S", "schema": {"type": "object"}}},
            {"type": "parallel", "branches": [
                {"type": "foreach", "id": "fe", "items_from": "s1.xs", "as": "v",
                 "body": {"type": "sequence", "steps": [
                     {"type": "agent", "id": "deep",
                      "spec": {"name": "d", "system_prompt": "{{v}}"}}]}},
                {"type": "branch", "on": "s1.k", "cases": {
                    "a": {"type": "agent", "id": "ca",
                          "spec": {"name": "ca", "system_prompt": "ca"}}},
                 "default": {"type": "agent", "id": "cd",
                             "spec": {"name": "cd", "system_prompt": "cd"}}},
            ]},
        ]},
    })
    # s1 + foreach aggregate 'fe' + branch case 'ca' + default 'cd' replayable;
    # 'deep' (inside the foreach body) is NOT individually replayable.
    assert replayable_node_ids(spec) == {"s1", "fe", "ca", "cd"}
