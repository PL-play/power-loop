"""Context compaction — protocol + default implementation (M1.7a).

Design contract (from ROADMAP §M1.7a / README §1):

* Triggered every ``round.start`` when estimated tokens >=
  ``max_tokens × trigger_ratio`` (or absolute ``CONTEXT_COMPACT_THRESHOLD``
  env override). Idempotent within a round.
* **Preserve** ``role=system`` messages (including prior ``compact_note``).
* **Preserve** the last ``keep_last_n`` exchanges. An exchange is a
  ``user / assistant(+optional tool_calls) / tool*`` triple — never split
  the atomic ``assistant(tool_calls)`` ↔ matching ``tool(tool_call_id=…)``
  pair.
* Summarize the cuttable middle via a separate LLM call (default = main
  LLM; injectable ``summary_llm`` for cheaper models).
* Insert one ``system / name=compact_note`` message in place of the cut
  range.
* Fail-soft: on summary error, return ``None`` plan → caller continues with
  uncompacted history; the pipeline then escalates to ``loop.degraded``
  only if the main LLM rejects on context-overflow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from llm_client.interface import LLMRequest
from power_loop.runtime.budget import estimate_tokens


@dataclass(frozen=True)
class CompactionPlan:
    """The output of a successful compaction round.

    ``fold_start_idx`` and ``fold_end_idx`` are **inclusive** indices into
    the pipeline's in-memory ``history`` list; everything between them is
    replaced by one ``compact_note`` message at ``fold_start_idx``.
    """

    fold_start_idx: int
    fold_end_idx: int
    summary_text: str
    before_tokens: int
    after_tokens: int


@runtime_checkable
class Compactor(Protocol):
    """A pluggable strategy that decides whether and how to compact."""

    async def maybe_compact(
        self,
        messages: list[dict[str, Any]],
        *,
        llm: Any,
        max_tokens: int,
        round_index: int,
    ) -> CompactionPlan | None: ...


# ── default implementation ──────────────────────────────────────────────


class DefaultCompactor:
    """Vendor-neutral compactor matching the M1.7a contract."""

    def __init__(
        self,
        *,
        trigger_ratio: float = 0.75,
        keep_last_n: int = 4,
        summary_max_tokens: int = 1024,
        summary_llm: Any | None = None,
        absolute_threshold: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        trigger_ratio
            Fire when ``estimate_tokens(history) ≥ max_tokens × trigger_ratio``.
            Default 0.75 leaves headroom for the next round's prompt + reply.
        keep_last_n
            Preserve the last N **user-bounded exchanges** verbatim (not
            summarized). This is the freshest context the model needs to
            answer the next turn well; folding it hurts reply quality
            dramatically. ``keep_last_n=1`` ≈ "aggressive — summarize
            everything but the last user turn"; the default 4 follows
            Anthropic's compaction guide.
        summary_max_tokens
            Token cap for the summary LLM call. Rule of thumb:
            ``summary_max_tokens ≈ max_tokens × 0.10–0.15``. 1024 is a
            balanced default for typical ``max_tokens=8000`` setups; bump
            it for very long sessions, lower it to cut summary cost.
        summary_llm
            Optional cheaper LLM dedicated to the summary call. Defaults
            to the main loop's LLM.
        absolute_threshold
            Absolute token count that overrides ``trigger_ratio`` when
            non-None. Env ``CONTEXT_COMPACT_THRESHOLD`` always wins over
            this if set.
        """
        self.trigger_ratio = float(trigger_ratio)
        self.keep_last_n = int(keep_last_n)
        self.summary_max_tokens = int(summary_max_tokens)
        self.summary_llm = summary_llm
        self.absolute_threshold = absolute_threshold

    # ── public ──────────────────────────────────────────────────────────

    async def maybe_compact(
        self,
        messages: list[dict[str, Any]],
        *,
        llm: Any,
        max_tokens: int,
        round_index: int,
    ) -> CompactionPlan | None:
        before = estimate_tokens(messages)
        if not self._should_trigger(before, max_tokens):
            return None
        span = self._compactable_span(messages)
        if span is None:
            return None
        start, end = span
        summary = await self._summarize_async(messages[start : end + 1], llm=llm)
        if summary is None:
            return None  # soft-fail; caller continues uncompacted
        after = estimate_tokens(
            [*messages[:start], _note(summary), *messages[end + 1 :]]
        )
        return CompactionPlan(
            fold_start_idx=start,
            fold_end_idx=end,
            summary_text=summary,
            before_tokens=before,
            after_tokens=after,
        )

    # ── trigger ─────────────────────────────────────────────────────────

    def _should_trigger(self, before_tokens: int, max_tokens: int) -> bool:
        # Env override (read lazily so monkeypatch in tests works).
        env = os.environ.get("CONTEXT_COMPACT_THRESHOLD")
        absolute = int(env) if env else self.absolute_threshold
        if absolute is not None:
            return before_tokens >= absolute
        if max_tokens <= 0:
            return False
        return before_tokens >= int(max_tokens * self.trigger_ratio)

    # ── span selection ──────────────────────────────────────────────────

    def _compactable_span(
        self, messages: list[dict[str, Any]]
    ) -> tuple[int, int] | None:
        """Return the inclusive index range we can safely fold, or None.

        Preserves all ``system`` messages at the front; preserves the tail
        ``keep_last_n`` exchanges; never splits an
        ``assistant(tool_calls) ↔ tool`` atomic pair.
        """
        n = len(messages)
        if n == 0:
            return None
        # Find end of leading system block.
        sys_end = 0
        while sys_end < n and messages[sys_end].get("role") == "system":
            sys_end += 1
        # Decide the tail boundary by counting exchanges from the end.
        tail_start = self._tail_start(messages, sys_end)
        if tail_start <= sys_end:
            return None  # nothing in the middle
        # Don't split a pending pair: if tail_start points at a `tool` whose
        # `assistant(tool_calls)` is in the cut range, expand tail to include
        # that assistant too.
        tail_start = self._expand_back_to_atomic(messages, tail_start)
        if tail_start <= sys_end:
            return None
        # And don't cut after the LAST element of an atomic pair without its
        # head: if messages[tail_start - 1] is a tool message, walk back so
        # the cut range ends on a non-tool boundary.
        end = tail_start - 1
        while end > sys_end and messages[end].get("role") == "tool":
            end -= 1
        if end < sys_end:
            return None
        return (sys_end, end)

    def _tail_start(self, messages: list[dict[str, Any]], sys_end: int) -> int:
        """Count exchanges from the tail; return the index of the start of
        the kept tail. An "exchange" begins at a ``user`` message."""
        n = len(messages)
        if n == sys_end:
            return n
        kept = 0
        i = n - 1
        boundary = n
        while i >= sys_end:
            if messages[i].get("role") == "user":
                kept += 1
                if kept >= self.keep_last_n:
                    boundary = i
                    break
            i -= 1
        else:
            boundary = sys_end  # not enough exchanges → keep everything after sys
        return boundary

    @staticmethod
    def _expand_back_to_atomic(
        messages: list[dict[str, Any]], tail_start: int
    ) -> int:
        """If ``messages[tail_start]`` is a ``tool`` and the corresponding
        ``assistant(tool_calls)`` is below the boundary, pull the boundary
        back so the pair stays together."""
        if tail_start >= len(messages):
            return tail_start
        msg = messages[tail_start]
        if msg.get("role") != "tool":
            return tail_start
        # walk back to the matching assistant
        j = tail_start - 1
        while j >= 0:
            m = messages[j]
            if m.get("role") == "assistant" and m.get("tool_calls"):
                return j
            j -= 1
        return tail_start

    # ── summarization ───────────────────────────────────────────────────

    async def _summarize_async(
        self, slice_msgs: list[dict[str, Any]], *, llm: Any
    ) -> str | None:
        if not slice_msgs:
            return None
        summary_llm = self.summary_llm or llm
        prompt = (
            "You are a conversation summarizer. Below is a slice of an "
            "agent's working transcript that needs to be compressed for "
            "context-window economy. Produce a concise but faithful summary "
            "that preserves: (1) decisions made, (2) facts established, "
            "(3) errors and how they were handled, (4) any pending intent "
            "the assistant was about to act on. Do NOT call tools. Wrap "
            "your summary in <summary>…</summary>.\n\n"
            "--- transcript slice ---\n"
            + _stringify_slice(slice_msgs)
        )
        try:
            response = await summary_llm.complete(
                LLMRequest(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.summary_max_tokens,
                    temperature=0.0,
                )
            )
        except Exception:
            return None
        text = (
            getattr(response, "raw_text", "")
            or getattr(response, "content_text", "")
            or ""
        ).strip()
        if not text:
            return None
        # Strip <summary>…</summary> if present.
        if text.startswith("<summary>") and "</summary>" in text:
            text = text[len("<summary>") :].split("</summary>")[0].strip()
        return text or None


# ── helpers ─────────────────────────────────────────────────────────────


def _stringify_slice(slice_msgs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for m in slice_msgs:
        role = m.get("role", "?")
        content = m.get("content")
        text = content if isinstance(content, str) else str(content or "")
        head = f"[{role}]"
        tool_calls = m.get("tool_calls")
        if tool_calls:
            head += f" tool_calls={len(tool_calls)}"
        lines.append(f"{head}\n{text}")
    return "\n\n".join(lines)


def _note(text: str) -> dict[str, Any]:
    return {"role": "system", "name": "compact_note", "content": text}


__all__ = ["Compactor", "CompactionPlan", "DefaultCompactor"]
