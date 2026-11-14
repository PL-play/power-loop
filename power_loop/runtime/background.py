"""Read models for background sessions and their ordinary input results."""
from __future__ import annotations

import json
from typing import Any

RESULT_PREFIX = "[background_tool_results]\n"


def encode_results(tasks: list[dict[str, Any]]) -> str:
    return RESULT_PREFIX + json.dumps({"version": 1, "tasks": tasks}, ensure_ascii=False)


async def load_record(store: Any, sid: str | None, task_id: str, *, reader=None) -> dict[str, Any] | None:
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_types import InputRef, plain

    port = RuntimeInputStore(store)
    reader = reader or port.db
    row = await reader.fetchone(
        f"SELECT * FROM {store.t.sessions} WHERE session_id=? AND created_by_session_id=? "
        "AND session_kind='background'",
        (task_id, sid),
    )
    if row is None or row["start_input_message_id"] is None:
        return None
    start = InputRef(store.store_id, task_id, row["start_input_message_id"])
    result = await port.get_result(start, actor=_actor(row, sid))
    original = json.loads(row["descriptor_json"])["config"]["background"]
    return {
        "task_id": task_id,
        "tool": original["tool"],
        "args": original["arguments"],
        "status": result.status if result else "accepted",
        "result": result.final_text if result else None,
        "result_ref": plain(start),
    }


def _actor(session, sid):
    from power_loop.agent.runtime_types import TrustedActor
    return TrustedActor(session["tenant_id"], session["principal_id"], sid)


def summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    from power_loop.core.agent_context import get_current_loop

    loop = get_current_loop()
    config = getattr(loop, "config", None)
    rep = getattr(config, "representation", None)
    distill = getattr(rep, "distill_tool_call", None)
    result = str(record.get("result") or "")
    from power_loop.runtime.tool_context import active_engine
    engine = active_engine.get()
    ref = record.get("result_ref")
    if hasattr(ref, "__dataclass_fields__"):
        from power_loop.agent.runtime_types import plain
        ref = plain(ref)
    if engine is not None:
        return {k: record.get(k) for k in ("task_id", "tool", "status")} | {
            "result_ref": ref,
            "arguments": record.get("args") or {},
            "summary": result,
            "representation": "original",
        }
    summary = result[:750]
    if callable(distill):
        try:
            summary = distill(
                record["tool"], record.get("args") or {}, result,
                getattr(loop, "tool_registry", None),
            )
        except Exception:
            pass
    return {k: record.get(k) for k in ("task_id", "tool", "status")} | {
        "result_ref": ref, "summary": summary,
    }
