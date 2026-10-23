"""Test-host output requirements; no provider response or business task changes."""
from __future__ import annotations

from dataclasses import replace

from power_loop.core.agent_context import get_session_id
from power_loop.core.pass_turn import PASS_TURN_TOOL, default_handler


def require_visible_response(registry, store, session_id):
    """Only this main session must write text after its latest input/result.

    A child may legitimately pass while waiting; its own completion protocol
    remains unchanged even if it inherits this registered handler.
    """
    async def pass_after_response(reason=""):
        if get_session_id() != session_id:
            return default_handler(reason)
        messages = await store.load_all_messages(session_id)
        boundary = max((m.seq for m in messages if m.role in ("user", "tool")), default=0)
        if not any(m.role == "assistant" and m.seq > boundary and isinstance(m.content, str) and m.content.strip()
                for m in messages):
            return ("This task requires a visible assistant result after the latest input/tool result. "
                "Write the final answer now using the saved evidence. Do not call any more work tools. "
                "A pass reason is not a user-visible answer.")
        return default_handler(reason)
    definition = replace(PASS_TURN_TOOL, description=(
        "End this main turn only after delivering its requested visible assistant response. "
        "An empty pass is rejected when the latest input/tool result has not received a visible answer. "
        "The reason is internal metadata and does not deliver the answer."))
    registry.register(definition, pass_after_response, overwrite=True)


async def consume_completed_result(loop, session_id, *, max_rounds=None, result_node=None):
    """The test already awaited completion; this main turn only reports it."""
    options = {"tools": [], "system_prompt": (
        "The supplied workflow completion has already settled. Extract its latest final result and write it as assistant text now. "
        "Use only the supplied evidence. Do not say WAITING, do not start or repair work, and do not call tools or pass_turn. "
        "For a numeric or VERSION result return that exact value as plain text without Markdown.")}
    if result_node is not None:
        options["system_prompt"] += (
            f" The workflow's final output is the result of node {result_node!r}. "
            "The summary lists nodes by name, not execution or output order. "
            "Other nodes may be prerequisites with different outputs; do not substitute their result."
        )
    if max_rounds is not None:
        options["max_rounds"] = max_rounds
    return await loop.flush_follow_ups(session_id, **options)
