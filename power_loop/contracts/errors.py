"""Public error hierarchy for power-loop.

All errors raised by the library inherit from :class:`PowerLoopError` so
callers can ``except PowerLoopError`` as a single catch-all.
"""

from __future__ import annotations

from typing import Any


class PowerLoopError(Exception):
    """Base for all power-loop raised exceptions."""


class SessionNotFoundError(PowerLoopError):
    """Raised when a ``session_id`` does not exist in the store."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"session not found: {session_id}")
        self.session_id = session_id


class SessionPendingError(PowerLoopError):
    """Raised when a session has unresolved tool_calls from a previous run.

    The previous loop crashed (or was killed) after the assistant emitted
    ``tool_calls`` but before all matching ``tool`` messages were appended.
    The OpenAI/Anthropic message protocol forbids us from sending the next
    LLM request in this state. The caller must explicitly choose:

      * ``StatefulAgentLoop.resume(sid)`` — finish executing the pending
        tool_calls and continue the loop, or
      * ``StatefulAgentLoop.abort_pending(sid)`` — append synthetic
        ``<aborted>`` tool messages, restoring protocol validity, then
        proceed with the new user input.
    """

    def __init__(
        self,
        session_id: str,
        *,
        assistant_seq: int,
        pending_tool_calls: list[dict[str, Any]],
    ) -> None:
        self.session_id = session_id
        self.assistant_seq = assistant_seq
        self.pending_tool_calls = pending_tool_calls
        names = ",".join(
            str((tc.get("function") or {}).get("name") or tc.get("name") or "?")
            for tc in pending_tool_calls
        )
        super().__init__(
            f"session {session_id} has {len(pending_tool_calls)} unresolved tool_calls"
            f" from round (assistant_seq={assistant_seq}): {names}"
        )
