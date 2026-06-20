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

import logging
from typing import Any, Protocol, runtime_checkable

from power_loop.agent.types import LoopMessage
from power_loop.runtime.store.store import SessionStore

logger = logging.getLogger(__name__)


@runtime_checkable
class MessageSink(Protocol):
    """Persistence callbacks invoked by :class:`AgentPipeline`.

    Every method MUST be safe to call multiple times and MUST NOT raise on
    normal paths — sinks degrade gracefully and log internally if needed.
    """

    async def on_round_started(self, round_index: int) -> None: ...
    async def on_message_appended(self, message: LoopMessage, *, round_index: int | None) -> None: ...
    def on_messages_inserted(self, *, index: int, count: int) -> None: ...  # pure (no I/O) → sync
    async def on_assistant_tool_calls(
        self, *, assistant_seq: int, tool_calls: list[dict[str, Any]], round_index: int
    ) -> None: ...
    async def on_compaction(
        self,
        *,
        fold_start_idx: int,
        fold_end_idx: int,
        summary_text: str,
        before_tokens: int,
        after_tokens: int,
        round_index: int,
        expected_history_len: int | None = None,
    ) -> None: ...
    async def on_round_ended(
        self, round_index: int, *, usage: dict[str, Any] | None = None
    ) -> None: ...


class NullSink:
    """No-op sink. Used when the pipeline runs without persistence."""

    async def on_round_started(self, round_index: int) -> None: ...
    async def on_message_appended(self, message: LoopMessage, *, round_index: int | None) -> None: ...
    def on_messages_inserted(self, *, index: int, count: int) -> None: ...
    async def on_assistant_tool_calls(
        self, *, assistant_seq: int, tool_calls: list[dict[str, Any]], round_index: int
    ) -> None: ...
    async def on_compaction(
        self,
        *,
        fold_start_idx: int,
        fold_end_idx: int,
        summary_text: str,
        before_tokens: int,
        after_tokens: int,
        round_index: int,
        expected_history_len: int | None = None,
    ) -> None: ...
    async def on_round_ended(
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
        # Count of compactions persisted during this sink's lifetime (one sink per send).
        # StatefulAgentLoop reads it after a run to decide whether its per-session row cache
        # can be extended with a cheap delta read (no fold this run) or must be invalidated
        # (a fold reshuffled the older active set, so the cached window is stale).
        self.compactions_applied = 0
        # Ordered seqs mirroring the pipeline's in-memory history, index-for-index.
        # Initialized by StatefulAgentLoop from the loaded active messages; grown by
        # on_message_appended; spliced by on_compaction. A ``None`` entry is a slot
        # for an in-memory-only message with no DB row — recalled ``memory_*``
        # messages injected by _maybe_recall (which never persist). Keeping these
        # placeholders is what preserves the index↔seq invariant so on_compaction
        # maps fold indices to the RIGHT rows (H1.1 / C1).
        #
        # ``_history_seqs`` holds each slot's *identity* seq (the DB row id, used to
        # mark the exact rows compacted_out). ``_history_ord`` holds each slot's
        # *logical* position, index-for-index. They differ for a ``compact_note``:
        # its identity is a fresh high seq, but it logically sits where its folded
        # range began. Tracking both keeps marking correct under a non-monotonic
        # identity map AND lets the note reload at the right position (C1 fix).
        self._history_seqs: list[int | None] = []
        self._history_ord: list[int | None] = []

    def init_history_seqs(
        self, seqs: list[int], ords: list[int] | None = None
    ) -> None:
        """Called by :class:`StatefulAgentLoop` with the seqs of the loaded
        active messages, in the same order they sit in pipeline.history.

        ``ords`` is the parallel list of logical positions (a ``compact_note``'s
        ``meta['ord']``, else the row's ``seq``). Omitting it mirrors ``seqs``
        (correct whenever no folded note is present)."""
        self._history_seqs = list(seqs)
        self._history_ord = list(ords) if ords is not None else list(seqs)

    def on_messages_inserted(self, *, index: int, count: int) -> None:
        """Record that ``count`` in-memory-only messages were spliced into
        ``pipeline.history`` at ``index`` without being persisted (recalled
        ``memory_*``). Insert matching ``None`` placeholders so ``_history_seqs``
        stays index-aligned with ``history`` and later folds map to the right rows."""
        if count <= 0:
            return
        idx = max(0, min(index, len(self._history_seqs)))
        self._history_seqs[idx:idx] = [None] * count
        self._history_ord[idx:idx] = [None] * count

    # ── messages ───────────────────────────────────────────────

    async def on_round_started(self, round_index: int) -> None:
        await self.store.set_round_index(self.session_id, round_index)

    async def on_message_appended(
        self, message: LoopMessage, *, round_index: int | None
    ) -> None:
        role = message.get("role")
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            seq = await self.store.append_message(
                self.session_id,
                role="tool",
                content=_as_text(message.get("content")),
                tool_call_id=tool_call_id,
                name=message.get("name"),
                round_index=round_index,
                meta=message.get("meta"),
                send_index=message.get("send_index"),
            )
            self._history_seqs.append(seq)
            self._history_ord.append(seq)
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
                    state = await self.store.get_state(self.session_id)
                    prior = state.pending if state is not None and state.pending else {}
                    interactions = list(prior.get("pending_interactions") or [])
                    remaining_interactions = [
                        item for item in interactions
                        if str(item.get("tool_call_id") or "") in self._unresolved
                    ]
                    if remaining_interactions:
                        pending["pending_interactions"] = remaining_interactions
                    await self.store.set_pending(self.session_id, pending)
                else:
                    await self.store.set_pending(self.session_id, None)
                    self._assistant_seq = None
                    self._tool_calls = []
            return
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            seq = await self.store.append_message(
                self.session_id,
                role="assistant",
                content=_as_text(message.get("content")),
                tool_calls=list(tool_calls) if tool_calls else None,
                round_index=round_index,
                meta=message.get("meta"),
                send_index=message.get("send_index"),
            )
            self._history_seqs.append(seq)
            self._history_ord.append(seq)
            if tool_calls:
                self._assistant_seq = seq
            return
        # user / system / anything else
        seq = await self.store.append_message(
            self.session_id,
            role=str(role or "user"),
            content=_as_text(message.get("content")),
            name=message.get("name"),
            round_index=round_index,
            meta=message.get("meta"),
            send_index=message.get("send_index"),
        )
        self._history_seqs.append(seq)
        self._history_ord.append(seq)

    # ── pending state machine ──────────────────────────────────

    async def on_assistant_tool_calls(
        self, *, assistant_seq: int, tool_calls: list[dict[str, Any]], round_index: int
    ) -> None:
        ids = [str(tc.get("id") or "") for tc in tool_calls if tc.get("id")]
        self._unresolved = set(ids)
        self._assistant_seq = assistant_seq
        self._tool_calls = list(tool_calls)
        await self.store.set_pending(
            self.session_id,
            {
                "assistant_seq": assistant_seq,
                "round_index": round_index,
                "tool_call_ids": ids,
                "tool_calls": list(tool_calls),
            },
        )

    async def on_compaction(
        self,
        *,
        fold_start_idx: int,
        fold_end_idx: int,
        summary_text: str,
        before_tokens: int,
        after_tokens: int,
        round_index: int,
        expected_history_len: int | None = None,
    ) -> None:
        """Persist a compaction: mark messages [fold_start_idx, fold_end_idx]
        in the in-memory history as ``compacted_out`` in the store, append the
        ``compact_note`` row, and rewrite ``_history_seqs`` to mirror the
        post-compaction in-memory history (so future appends keep the index
        invariant).

        Alignment safety net (H1.1): the fold indices are positions in
        ``pipeline.history``; we translate them through ``_history_seqs``, so the
        two MUST be the same length. If they are not — a desync from some message
        mutated outside the sink (a SESSION_START/ROUND_START hook replacing the
        list wholesale, C9), or a fold index that lands on a non-persisted
        ``None`` placeholder — we **skip persistence** rather than mark the wrong
        rows ``compacted_out``. The in-memory fold still stands; the un-persisted
        compaction simply re-triggers next round (active rows are untouched, so a
        resume is correct), trading a missed optimization for zero corruption.
        """
        if expected_history_len is not None and len(self._history_seqs) != expected_history_len:
            logger.warning(
                "skip compaction persistence for %s: _history_seqs (%d) misaligned with "
                "history (%d) — refusing to mark possibly-wrong rows compacted_out",
                self.session_id, len(self._history_seqs), expected_history_len,
            )
            return
        if not (0 <= fold_start_idx <= fold_end_idx < len(self._history_seqs)):
            return  # defensive: out-of-range indices → no-op
        from_seq = self._history_seqs[fold_start_idx]
        to_seq = self._history_seqs[fold_end_idx]
        if from_seq is None or to_seq is None:
            logger.warning(
                "skip compaction persistence for %s: fold boundary lands on a "
                "non-persisted (recalled) message — refusing to compact it out",
                self.session_id,
            )
            return
        # Mark the EXACT set of identity seqs being folded (skip None placeholders
        # for in-memory-only recalled messages). An explicit set — not a BETWEEN
        # range over the translated boundary seqs — is what stays correct when a
        # prior compact_note left ``_history_seqs`` non-monotonic (C1): a range
        # could invert (fold nothing in the DB while the in-memory fold proceeds)
        # or sweep active kept-tail rows.
        fold_slice = self._history_seqs[fold_start_idx : fold_end_idx + 1]
        fold_seqs = [s for s in fold_slice if s is not None]
        if not fold_seqs:
            # The folded range was entirely in-memory-only (recalled placeholders),
            # so there are no DB rows to mark and no note is persisted. But the
            # pipeline still replaced the range with ONE note message in its
            # in-memory history; mirror that as a single ``None`` placeholder so the
            # index maps stay length-aligned with ``history`` (otherwise the next
            # fold trips the expected_history_len safety net and stops persisting).
            self._history_seqs = (
                self._history_seqs[:fold_start_idx]
                + [None]
                + self._history_seqs[fold_end_idx + 1 :]
            )
            self._history_ord = (
                self._history_ord[:fold_start_idx]
                + [None]
                + self._history_ord[fold_end_idx + 1 :]
            )
            return
        # The note logically sits where the folded range began. Use the first
        # persisted slot's *logical* position (its ord), not its identity seq, so
        # a folded-in prior note contributes its logical position rather than its
        # high row id — and the new note reloads at the right place.
        ord_slice = [o for o in self._history_ord[fold_start_idx : fold_end_idx + 1]
                     if o is not None]
        order_key = ord_slice[0] if ord_slice else min(fold_seqs)
        _, note_seq = await self.store.record_compaction(
            self.session_id,
            from_seq=min(fold_seqs),
            to_seq=max(fold_seqs),
            note_content=summary_text,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            round_index=round_index,
            fold_seqs=fold_seqs,
            order_key=order_key,
        )
        # In the in-memory history the cut range is replaced by ONE note message;
        # mirror that in both parallel maps (identity = the new note row's seq;
        # logical position = where the folded range began).
        self._history_seqs = (
            self._history_seqs[:fold_start_idx]
            + [note_seq]
            + self._history_seqs[fold_end_idx + 1 :]
        )
        self._history_ord = (
            self._history_ord[:fold_start_idx]
            + [order_key]
            + self._history_ord[fold_end_idx + 1 :]
        )
        # A fold actually hit the DB (rows → compacted_out + a new note row), reshuffling the
        # durable active set. The loop's per-session row cache can no longer be extended with a
        # delta read and must be invalidated. (The in-memory-only-placeholder path above returns
        # before here precisely because it changes no durable rows.)
        self.compactions_applied += 1

    async def on_round_ended(
        self, round_index: int, *, usage: dict[str, Any] | None = None
    ) -> None:
        if usage:
            await self.store.record_usage(
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
