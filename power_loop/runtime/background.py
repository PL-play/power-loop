"""Durable identity and results of background tool calls; no schema migration needed."""
from __future__ import annotations

import json
from typing import Any

ACTIVE_STATUSES = frozenset({'queued', 'pending', 'running', 'stopping', 'stop_unconfirmed'})
RESULT_PREFIX = '[background_tool_results]\n'


def encode_results(tasks: list[dict[str, Any]]) -> str:
    return RESULT_PREFIX + json.dumps({'version': 1, 'tasks': tasks}, ensure_ascii=False)


async def load_record(store: Any, sid: str | None, task_id: str, *, reader=None, generation=None) -> dict[str, Any] | None:
    from power_loop.agent.runtime_store import RuntimeInputStore

    from .result_revisions import result_id
    port = RuntimeInputStore(store)
    reader = reader or port.db
    row = await reader.fetchone(f"SELECT j.job_id,j.generation,e.payload_json FROM {port.jobs} j "
        f"JOIN {port.events} e ON e.job_id=j.job_id AND e.action='create' "
        "WHERE j.job_id=? AND j.created_by_session_id=? AND j.kind IN ('background_tool','background_command')",
        (task_id, sid))
    if row is None:
        return None
    from power_loop.agent.runtime_types import InputRef
    start = await port.input(reader, InputRef(**json.loads(row["payload_json"])["request"]["input_ref"]))
    event = await reader.fetchone(f"SELECT payload_json FROM {port.events} WHERE event_id=?",
                                  (result_id(task_id, row["generation"] if generation is None else generation),))
    if event is None:
        return None
    result = json.loads(event["payload_json"])["result"]
    original = start.meta["descriptor"]["config"]["background"]
    return {"task_id": task_id, "tool": original["tool"], "args": original["arguments"],
            "status": result["status"], "result": result["final_text"], "result_ref": result["job_ref"]}


def summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    from power_loop.core.agent_context import get_current_loop

    loop = get_current_loop()
    config = getattr(loop, 'config', None)
    rep = getattr(config, 'representation', None)
    distill = getattr(rep, 'distill_tool_call', None)
    result = str(record.get('result') or '')
    from power_loop.runtime.tool_context import active_engine
    engine = active_engine.get()
    if engine is not None:  # tool context distills the canonical result itself
        return {k: record.get(k) for k in ('task_id', 'tool', 'status', 'result_ref')} | {
            'arguments': record.get('args') or {}, 'summary': result, 'representation': 'original'}
    summary = result[:750]
    if callable(distill):
        try:
            summary = distill(record['tool'], record.get('args') or {}, result,
                              getattr(loop, 'tool_registry', None))
        except Exception:
            # The canonical result stays available even if a host distiller fails.
            pass
    return {k: record.get(k) for k in ('task_id', 'tool', 'status', 'result_ref')} | {'summary': summary}
