"""Durable identity and results of background tool calls; no schema migration needed."""
from __future__ import annotations

import json
from typing import Any

ACTIVE_STATUSES = frozenset({'queued', 'pending', 'running'})
RESULT_PREFIX = '[background_tool_results]\n'


def record_key(task_id: str) -> str:
    return 'background_tool:' + task_id


def encode_results(tasks: list[dict[str, Any]]) -> str:
    return RESULT_PREFIX + json.dumps({'version': 1, 'tasks': tasks}, ensure_ascii=False)


async def load_record(store: Any, sid: str | None, task_id: str) -> dict[str, Any] | None:
    if store is None or sid is None or not hasattr(store, 'get_runtime_state'):
        return None
    value = await store.get_runtime_state(sid, record_key(task_id), default=None)
    return value if isinstance(value, dict) else None


def summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    from power_loop.core.agent_context import get_current_loop

    loop = get_current_loop()
    config = getattr(loop, 'config', None)
    rep = getattr(config, 'representation', None)
    distill = getattr(rep, 'distill_tool_call', None)
    result = str(record.get('result') or '')
    summary = result[:750]
    if callable(distill):
        try:
            summary = distill(record['tool'], record.get('args') or {}, result,
                              getattr(loop, 'tool_registry', None))
        except Exception:
            # The canonical result stays available even if a host distiller fails.
            pass
    return {k: record.get(k) for k in ('task_id', 'tool', 'status', 'result_ref')} | {'summary': summary}
