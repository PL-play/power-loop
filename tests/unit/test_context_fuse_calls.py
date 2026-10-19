"""Whole-call fuse: nested arguments, parallel ordering, audit isolation, no batch cap."""
import copy
import json
from types import SimpleNamespace

from power_loop import AgentLoopConfig, ProjectedRepresentation
from power_loop.core.pipeline import AgentPipeline


class Semantic(ProjectedRepresentation):
    def distill_tool_call(self, name, args, result, tool_registry):
        if name == "noop":
            return json.dumps(args) + result
        return f"{name}: wrote {len(args.get('ops', []))} operations; {result[:20]}"


def test_nested_candidates_do_not_starve_and_hot_tail_tracks_call_ids():
    p = AgentPipeline.__new__(AgentPipeline)
    p.config = AgentLoopConfig(system_prompt="t", representation=Semantic())
    p.send_index = 1
    p.sink = SimpleNamespace(_history_seqs=list(range(100)))
    p._send_start_len = 0
    calls = [{"id": str(i), "type": "function", "function": {
        "name": "noop" if i < 8 else "scene_pkg",
        "arguments": json.dumps({"ops": [{"nested": ["abc" * 500]}]})}}
        for i in range(28)]
    p.history = [{"role": "assistant", "tool_calls": calls}]
    # Results in reverse completion order; call 27 remains pending.
    p.history += [{"role": "tool", "tool_call_id": str(i), "content": "ok"}
                  for i in reversed(range(27))]
    original = copy.deepcopy(p.history)
    audit_calls = calls
    assert p._distill_oldest_tool_rows(1, 8) > 0
    after = p.history[0]["tool_calls"]
    assert all(after[i] == original[0]["tool_calls"][i] for i in range(8))
    assert all("_distilled" in after[i]["function"]["arguments"] for i in range(8, 20))
    assert after[20:] == original[0]["tool_calls"][20:]
    assert audit_calls == original[0]["tool_calls"]
    assert p._distill_oldest_tool_rows(1, 8) == 0

    # Each compacted call points at its own result, even with parallel results reversed.
    for i in range(8, 20):
        ri = next(n for n, row in enumerate(original)
                  if row.get("tool_call_id") == str(i))
        coordinate = f"recall_send(send_index=1, seq={ri})"
        assert coordinate in after[i]["function"]["arguments"]
        assert coordinate in p.history[ri]["content"]
