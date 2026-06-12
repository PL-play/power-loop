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
    def on_compaction(
        self,
        *,
        fold_start_idx: int,
        fold_end_idx: int,
        summary_text: str,
        before_tokens: int,
        after_tokens: int,
        round_index: int,
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
    def on_compaction(
        self,
        *,
        fold_start_idx: int,
        fold_end_idx: int,
        summary_text: str,
        before_tokens: int,
        after_tokens: int,
        round_index: int,
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
        self._tool_calls: list[dict[str, Any]] = []
        # Ordered seqs mirroring the pipeline's in-memory history. Initialized
        # by StatefulAgentLoop from the loaded active messages; appended to as
        # the pipeline emits new messages.
        self._history_seqs: list[int] = []

    def init_history_seqs(self, seqs: list[int]) -> None:
        """Called by :class:`StatefulAgentLoop` with the seqs of the loaded
        active messages, in the same order they sit in pipeline.history."""
        self._history_seqs = list(seqs)

    # ── messages ───────────────────────────────────────────────

    def on_round_started(self, round_index: int) -> None:
        self.store.set_round_index(self.session_id, round_index)

    def on_message_appended(
        self, message: LoopMessage, *, round_index: int | None
    ) -> None:
        role = message.get("role")
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            seq = self.store.append_message(
                self.session_id,
                role="tool",
                content=_as_text(message.get("content")),
                tool_call_id=tool_call_id,
                name=message.get("name"),
                round_index=round_index,
            )
            self._history_seqs.append(seq)
            # Auto-resolve pending: when the matching tool message lands,
            # drop it from the unresolved set and clear pending once empty.
            if tool_call_id and tool_call_id in self._unresolved:
                self._unresolved.discard(tool_call_id)
                if self._unresolved:
                    remaining_tool_calls = [
                        tc for tc in self._tool_calls
                        if str(tc.get("id") or "") in self._unresolved
                    ]
                    pending = {
                        "assistant_seq": self._assistant_seq,
                        "round_index": round_index,
                        "tool_call_ids": sorted(self._unresolved),
                        "tool_calls": remaining_tool_calls,
                    }
                    state = self.store.get_state(self.session_id)
                    prior = state.pending if state is not None and state.pending else {}
                    interactions = list(prior.get("pending_interactions") or [])
                    remaining_interactions = [
                        item for item in interactions
                        if str(item.get("tool_call_id") or "") in self._unresolved
                    ]
                    if remaining_interactions:
                        pending["pending_interactions"] = remaining_interactions
                    self.store.set_pending(self.session_id, pending)
                else:
                    self.store.set_pending(self.session_id, None)
                    self._assistant_seq = None
                    self._tool_calls = []
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
            self._history_seqs.append(seq)
            if tool_calls:
                self._assistant_seq = seq
            return
        # user / system / anything else
        seq = self.store.append_message(
            self.session_id,
            role=str(role or "user"),
            content=_as_text(message.get("content")),
            name=message.get("name"),
            round_index=round_index,
        )
        self._history_seqs.append(seq)

    # ── pending state machine ──────────────────────────────────

    def on_assistant_tool_calls(
        self, *, assistant_seq: int, tool_calls: list[dict[str, Any]], round_index: int
    ) -> None:
        ids = [str(tc.get("id") or "") for tc in tool_calls if tc.get("id")]
        self._unresolved = set(ids)
        self._assistant_seq = assistant_seq
        self._tool_calls = list(tool_calls)
        self.store.set_pending(
            self.session_id,
            {
                "assistant_seq": assistant_seq,
                "round_index": round_index,
                "tool_call_ids": ids,
                "tool_calls": list(tool_calls),
            },
        )

    def on_compaction(
        self,
        *,
        fold_start_idx: int,
        fold_end_idx: int,
        summary_text: str,
        before_tokens: int,
        after_tokens: int,
        round_index: int,
    ) -> None:
        """Persist a compaction: mark messages [fold_start_idx, fold_end_idx]
        in the in-memory history as ``compacted_out`` in the store, append the
        ``compact_note`` row, and rewrite ``_history_seqs`` to mirror the
        post-compaction in-memory history (so future appends keep the index
        invariant).
        """
        if not (0 <= fold_start_idx <= fold_end_idx < len(self._history_seqs)):
            return  # defensive: out-of-range indices → no-op
        from_seq = self._history_seqs[fold_start_idx]
        to_seq = self._history_seqs[fold_end_idx]
        _, note_seq = self.store.record_compaction(
            self.session_id,
            from_seq=from_seq,
            to_seq=to_seq,
            note_content=summary_text,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            round_index=round_index,
        )
        # In the in-memory history the cut range is replaced by ONE note
        # message; mirror that here.
        self._history_seqs = (
            self._history_seqs[:fold_start_idx]
            + [note_seq]
            + self._history_seqs[fold_end_idx + 1 :]
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
