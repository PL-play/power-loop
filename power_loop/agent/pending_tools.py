"""Read one interrupted tool batch from its original call and stored results."""
from __future__ import annotations

import json

from .runtime_types import RuntimeRequestError


async def read_pending_tools(store, executor, session_id, marker, next_seq):
    """Use the caller's executor and known upper bound; never replay or scan a send."""
    source = await executor.fetchone(
        f"SELECT id,seq,role,tool_call_id,send_index,round_index,tool_calls_json "
        f"FROM {store.t.messages} WHERE session_id=? AND seq=? AND role='assistant'",
        (session_id, marker["assistant_seq"]),
    )
    if source is None:
        raise RuntimeRequestError("pending_source_unavailable")
    try:
        calls = json.loads(source["tool_calls_json"])
    except (ValueError, TypeError) as exc:
        raise RuntimeRequestError("pending_source_unavailable") from exc
    if not isinstance(calls, list) or not calls or not any(call.get("id") for call in calls):
        raise RuntimeRequestError("pending_source_unavailable")
    results = await executor.fetchall(
        f"SELECT tool_call_id FROM {store.t.messages} WHERE session_id=? AND send_index=? "
        "AND seq>? AND seq<? AND role='tool' AND tool_call_id IS NOT NULL",
        (session_id, source["send_index"], source["seq"], next_seq),
    )
    completed = {row["tool_call_id"] for row in results}
    return source, [call for call in calls if str(call.get("id") or "") not in completed]
