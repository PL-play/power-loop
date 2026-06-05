"""MessageSink: persistence hook the pipeline calls on every state change.

The pipeline stays storage-agnostic. It calls these methods at well-defined
moments; a sink turns them into rows in the :class:`SessionStore`, or into
no-ops for an in-memory run.

Three concrete sinks ship here:

* :class:`NullSink` — the default, used when no persistence is wanted.
* :class:`SQLiteSink` — wraps a :class:`SessionStore` + ``session_id``.
* (Subagent sink, added in PR-3, also reuses :class:`SQLiteSink`.)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from power_loop.agent.types import LoopMessage
from power_loop.runtime.session_store import SessionStore


@runtime_checkable
class MessageSink(Protocol):
    """Persistence callbacks invoked by :class:`AgentPipeline`.

    Every method MUST be safe to call multiple times and MUST NOT raise on
    normal paths — sinks degrade gracefully and log internally if needed.
    """

    def on_round_started(self, round_index: int) -> None: ...
    def on_message_appended(self, message: LoopMessage, *, round_index: int | None) -> None: ...
    def on_assistant_tool_calls(
        self, *, assistant_seq: int, tool_calls: list[dict[str, Any]], round_index: int
    ) -> None: ...
    def on_round_ended(
        self, round_index: int, *, usage: dict[str, Any] | None = None
    ) -> None: ...


class NullSink:
    """No-op sink. Used when the pipeline runs without persistence."""

    def on_round_started(self, round_index: int) -> None: ...
    def on_message_appended(self, message: LoopMessage, *, round_index: int | None) -> None: ...
    def on_assistant_tool_calls(
        self, *, assistant_seq: int, tool_calls: list[dict[str, Any]], round_index: int
    ) -> None: ...
    def on_round_ended(
        self, round_index: int, *, usage: dict[str, Any] | None = None
    ) -> None: ...


class SQLiteSink:
    """Persist messages + pending-state to a :class:`SessionStore` row.

    Pending state machine
    ---------------------
    ``session_state.pending_json`` is set the moment the assistant emits
    ``tool_calls`` and is cleared once every matching ``tool`` message has
    been appended. Crash anywhere in between leaves the session in a
    *pending* state that the next :meth:`StatefulAgentLoop.send` will refuse
    until the caller picks resume/abort.
    """

    def __init__(self, store: SessionStore, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self._unresolved: set[str] = set()
        self._assistant_seq: int | None = None

    # ── messages ───────────────────────────────────────────────

    def on_round_started(self, round_index: int) -> None:
        self.store.set_round_index(self.session_id, round_index)

    def on_message_appended(
        self, message: LoopMessage, *, round_index: int | None
    ) -> None:
        role = message.get("role")
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            self.store.append_message(
                self.session_id,
                role="tool",
                content=_as_text(message.get("content")),
                tool_call_id=tool_call_id,
                name=message.get("name"),
                round_index=round_index,
            )
            # Auto-resolve pending: when the matching tool message lands,
            # drop it from the unresolved set and clear pending once empty.
            if tool_call_id and tool_call_id in self._unresolved:
                self._unresolved.discard(tool_call_id)
                if self._unresolved:
                    self.store.set_pending(
                        self.session_id,
                        {
                            "assistant_seq": self._assistant_seq,
                            "round_index": round_index,
                            "tool_call_ids": sorted(self._unresolved),
                        },
                    )
                else:
                    self.store.set_pending(self.session_id, None)
                    self._assistant_seq = None
            return
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            seq = self.store.append_message(
                self.session_id,
                role="assistant",
                content=_as_text(message.get("content")),
                tool_calls=list(tool_calls) if tool_calls else None,
                round_index=round_index,
            )
            if tool_calls:
                self._assistant_seq = seq
            return
        # user / system / anything else
        self.store.append_message(
            self.session_id,
            role=str(role or "user"),
            content=_as_text(message.get("content")),
            name=message.get("name"),
            round_index=round_index,
        )

    # ── pending state machine ──────────────────────────────────

    def on_assistant_tool_calls(
        self, *, assistant_seq: int, tool_calls: list[dict[str, Any]], round_index: int
    ) -> None:
        ids = [str(tc.get("id") or "") for tc in tool_calls if tc.get("id")]
        self._unresolved = set(ids)
        self._assistant_seq = assistant_seq
        self.store.set_pending(
            self.session_id,
            {
                "assistant_seq": assistant_seq,
                "round_index": round_index,
                "tool_call_ids": ids,
                "tool_calls": list(tool_calls),
            },
        )

    def on_round_ended(
        self, round_index: int, *, usage: dict[str, Any] | None = None
    ) -> None:
        if usage:
            self.store.record_usage(
                self.session_id,
                round_index=round_index,
                prompt_tokens=_int_or_none(usage.get("prompt_tokens") or usage.get("input")),
                completion_tokens=_int_or_none(
                    usage.get("completion_tokens") or usage.get("output")
                ),
                total_tokens=_int_or_none(usage.get("total_tokens")),
            )


def _as_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    # multimodal lists / dicts — preserve as JSON-ish string
    import json

    return json.dumps(content, ensure_ascii=False)


def _int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
