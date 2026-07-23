"""Context compaction — protocol + default implementation (M1.7a).

Design contract (from ROADMAP §M1.7a / README §1):

* Triggered every ``round.start`` when estimated tokens >=
  ``max_tokens × trigger_ratio`` (or absolute ``CONTEXT_COMPACT_THRESHOLD``
  env override). Idempotent within a round.
* **Preserve** the first ``role=system`` message (the original system_prompt)
	and ``memory_*`` messages. Old ``compact_note`` messages are foldable —
	the new summary merges them so at most one compact_note ever exists.
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

import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from power_loop._vendor.llm_client.interface import LLMRequest
from power_loop.runtime.budget import estimate_tokens

if TYPE_CHECKING:
    from power_loop.runtime.memory import MemoryProvider
    from power_loop.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompactionContext:
    """Optional coordinate handle passed to a compactor at fold time (H7 Phase 2).

    Lets a *custom* compactor capture must-keep detail into the **injected**
    ``MemoryProvider`` (or read store rows) before the fold drops it from the active
    window — e.g. ``await context.memory.remember(snapshot=...)`` with the slice
    ``messages[plan.fold_start_idx : plan.fold_end_idx + 1]``.

    Additive and optional: :class:`DefaultCompactor` ignores it, and the pipeline
    only passes it to compactors whose ``maybe_compact`` accepts a ``context``
    parameter (signature-checked), so pre-existing compactors keep working unchanged.
    """

    session_id: str | None = None
    memory: MemoryProvider | None = None
    round_index: int = 0
    # Read-only async accessor: (from_seq, to_seq) -> message dicts in that seq range,
    # including already-compacted rows. None when no store is attached. Async because the
    # store is async (the backend offloads I/O); ``await ctx.fetch_messages(a, b)``.
    fetch_messages: Callable[[int, int], Awaitable[list[dict[str, Any]]]] | None = None
    # The pipeline's incrementally-maintained token estimate of the current history
    # (SCALE-4). When provided, a compactor can use it for its trigger check instead
    # of re-scanning every message every round (O(history) → O(1)). None ⇒ scan.
    current_tokens: int | None = None


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
        context: CompactionContext | None = None,
    ) -> CompactionPlan | None: ...


# ── default implementation ──────────────────────────────────────────────


class DefaultCompactor(Compactor):
    """Vendor-neutral compactor matching the M1.7a contract."""

    def __init__(
        self,
        *,
        trigger_ratio: float = 0.75,
        keep_last_n: int = 4,
        summary_max_tokens: int = 5000,
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
            Token cap for the summary LLM call. Since at most one compact_note
            exists at any time (each compaction merges the old note into the
            new one), this can be generous — 5000 is a good default for
            preserving detailed context across many compaction rounds.
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
        context: CompactionContext | None = None,
    ) -> CompactionPlan | None:
        # The default compactor does not coordinate with memory/tools, but it DOES use
        # the pipeline's incremental token estimate when offered (SCALE-4) to avoid an
        # O(history) re-scan every round; it falls back to a full scan otherwise.
        before = (
            context.current_tokens
            if context is not None and context.current_tokens is not None
            else estimate_tokens(messages)
        )
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
        # Env override (read lazily so monkeypatch in tests works). Parse fail-soft: a malformed
        # value must NOT crash the whole send — fall back to the configured threshold (compaction-1).
        env = os.environ.get("CONTEXT_COMPACT_THRESHOLD")
        absolute = self.absolute_threshold
        if env:
            try:
                absolute = int(env)
            except (TypeError, ValueError):
                logger.warning(
                    "ignoring malformed CONTEXT_COMPACT_THRESHOLD=%r; using configured threshold", env
                )
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

        The original system_prompt is NOT in ``messages`` (it is passed to the LLM
        out-of-band), so the only leading ``system`` rows here are recalled
        ``memory_*`` rows (preserve — the MemoryProvider owns them) and prior
        ``compact_note`` rows. A leading ``compact_note`` is FOLDABLE: it must be
        merged into the new note so at most one ``compact_note`` ever exists (C7).
        Folding it out is safe even when recalled ``memory_*`` placeholders sit
        inside the span — the sink skips non-persisted rows when marking.
        """
        n = len(messages)
        if n == 0:
            return None
        # Find end of preserved system block. A leading compact_note is NOT
        # preserved (it must fold/merge); genuine injected system rows and
        # memory_* rows are.
        sys_end = 0
        if (
            sys_end < n
            and messages[sys_end].get("role") == "system"
            and (messages[sys_end].get("name") or "") != "compact_note"
        ):
            sys_end = 1
            # Preserve subsequent memory_* messages (they share system-region protection).
            while sys_end < n and messages[sys_end].get("role") == "system":
                name = messages[sys_end].get("name") or ""
                if name.startswith("memory_"):
                    sys_end += 1
                else:
                    break
        # Decide the tail boundary by counting exchanges from the end.
        tail_start = self._tail_start(messages, sys_end)
        if tail_start <= sys_end:
            return None
        # Don't split a pending pair.
        tail_start = self._expand_back_to_atomic(messages, tail_start)
        if tail_start <= sys_end:
            return None
        # ``tail_start`` is already a clean boundary (``_expand_back_to_atomic``
        # guarantees the kept tail never starts on an orphan ``tool``), so folding
        # [sys_end, tail_start-1] always folds complete assistant↔tool pairs.
        # Do NOT walk ``end`` back over trailing ``tool`` messages here: that would
        # fold an ``assistant(tool_calls)`` while leaving its ``tool`` replies in
        # the kept tail — a protocol-invalid orphan tool (HTTP 400 on the next call).
        end = tail_start - 1
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
            "context-window economy. The slice may include prior compact_notes — "
            "merge their content into your summary so there is at most ONE "
            "compact note at any time. Preserve: (1) decisions made, "
            "(2) facts established, (3) errors and how they were handled, "
            "(4) any pending intent the assistant was about to act on. "
            "Do NOT call tools. Wrap your summary in <summary>…</summary>.\n\n"
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
        return _strip_summary(text)


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


def _strip_summary(text: str | None) -> str | None:
    """Pull the ``<summary>…</summary>`` body if present, else return the trimmed text (or None)."""
    text = (text or "").strip()
    if not text:
        return None
    if "<summary>" in text and "</summary>" in text:
        text = text.split("<summary>", 1)[1].split("</summary>", 1)[0].strip()
    return text or None


# ── agentic, memory-aware compactor (opt-in) ────────────────────────────


#: System prompt for the compaction agent. Two-phase: extract durable facts to memory (notes)
#: via tools, THEN write a faithful compact summary. Planned so the model knows it is compacting
#: (not answering the user), what to keep, and to output the summary last without tool calls.
DEFAULT_COMPACTION_AGENT_PROMPT = (
    "You are the memory & compaction agent for a long-running AI assistant. You are given a "
    "SLICE of the assistant's working transcript that is about to be REMOVED from the live "
    "context window to save space. You are NOT answering the user — you are preserving what "
    "matters.\n\n"
    "Do this in order:\n"
    "1. EXTRACT durable facts worth remembering AFTER this slice leaves context — decisions "
    "made, stable user preferences/constraints, established facts, unresolved commitments, and "
    "hard-won fixes to errors — and SAVE each as a concise note via `note(action=add, ...)` (one "
    "fact per note). Use `note(action=update, ...)` to refine an existing note instead of "
    "duplicating. Save "
    "ONLY what will matter later; skip transient chatter and anything obvious from the system "
    "prompt. If nothing is worth remembering, save nothing.\n"
    "2. THEN write a faithful, compact summary of the slice for the context window. Preserve: "
    "(1) decisions, (2) facts established, (3) errors and how they were handled, (4) any pending "
    "intent the assistant was about to act on. Merge any prior compact-note content you see so "
    "there is one coherent summary. Do not invent. Keep it tight.\n\n"
    "Output the summary LAST, wrapped in <summary>…</summary>, in a message with NO tool calls."
)


class AgenticMemoryCompactor(DefaultCompactor):
    """Opt-in compactor that runs a bounded, memory-aware agent loop to compact a slice.

    Instead of one summarization call, it lets the model use a memory tool (by default the
    unified ``note`` tool) to persist durable facts into the **current
    session's notes** BEFORE compressing the rest into the ``compact_note`` summary. Reuses
    :class:`DefaultCompactor`'s trigger + span selection unchanged; only the summarize step is
    agentic.

    Safety: the agent loop is a FLAT, bounded tool-use loop (``max_rounds``) — not a nested
    ``StatefulAgentLoop`` — so it can never recurse into another compaction. Notes are written via
    the tool handlers, which resolve the active session/store from the loop's contextvars (set for
    the whole run, so they are live during compaction). On ANY failure (no tool support, malformed
    output, exception) it falls back to the plain single-call summary, so it never blocks a fold.

    Default-OFF: construct it explicitly and pass it as ``AgentLoopConfig.compactor`` to opt in.
    """

    def __init__(
        self,
        *,
        memory_tools: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_rounds: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._memory_tools = memory_tools
        self._system_prompt = system_prompt or DEFAULT_COMPACTION_AGENT_PROMPT
        self._max_rounds = max(1, int(max_rounds))

    def _registry(self) -> ToolRegistry:
        if self._memory_tools is not None:
            return self._memory_tools
        # Lazy (avoids an import cycle: tools → runtime). Default to the unified note tool.
        from power_loop.tools import create_default_tool_registry

        self._memory_tools = create_default_tool_registry(include=["note"])
        return self._memory_tools

    async def _summarize_async(self, slice_msgs: list[dict[str, Any]], *, llm: Any) -> str | None:
        if not slice_msgs:
            return None
        try:
            summary = await self._agentic_summarize(slice_msgs, llm=llm)
            if summary:
                return summary
        except Exception:
            logger.exception("agentic compaction failed; falling back to single-call summary")
        return await super()._summarize_async(slice_msgs, llm=llm)

    async def _agentic_summarize(self, slice_msgs: list[dict[str, Any]], *, llm: Any) -> str | None:
        registry = self._registry()
        tools = registry.to_openai_tools()
        convo: list[dict[str, Any]] = [
            {"role": "user", "content": "--- transcript slice to compact ---\n" + _stringify_slice(slice_msgs)}
        ]
        summary_llm = self.summary_llm or llm
        for _ in range(self._max_rounds):
            resp = await summary_llm.complete(
                LLMRequest(
                    messages=convo,
                    system_prompt=self._system_prompt,
                    tools=tools,
                    max_tokens=self.summary_max_tokens,
                    temperature=0.0,
                )
            )
            text = (getattr(resp, "raw_text", "") or getattr(resp, "content_text", "") or "").strip()
            tool_calls = resp.get_tool_calls()
            if not tool_calls:
                return _strip_summary(text)  # model finished (summary, or nudge-then-retry below)
            # Replay the assistant tool-call turn + execute each tool, feeding results back.
            convo.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
            for tc in tool_calls:
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = (fn or {}).get("name") or tc.get("name")
                raw_args = (fn or {}).get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except (TypeError, ValueError):
                    args = {}
                if not name:  # malformed tool_call with no name — answer it so the pair stays valid
                    result: Any = "error: tool call had no name"
                else:
                    try:
                        result = await registry.invoke_async(
                            str(name), args if isinstance(args, dict) else {}
                        )
                    except Exception as exc:  # a bad tool call must not abort compaction
                        result = f"error: {exc}"
                convo.append({"role": "tool", "tool_call_id": tc.get("id"), "content": str(result)})
        # Rounds exhausted while still tool-calling — one final, tool-free summary request.
        resp = await summary_llm.complete(
            LLMRequest(
                messages=[*convo, {"role": "user", "content": "Now output ONLY the <summary>…</summary>."}],
                system_prompt=self._system_prompt,
                max_tokens=self.summary_max_tokens,
                temperature=0.0,
            )
        )
        return _strip_summary(
            getattr(resp, "raw_text", "") or getattr(resp, "content_text", "") or ""
        )


__all__ = [
    "Compactor",
    "CompactionContext",
    "CompactionPlan",
    "DefaultCompactor",
    "AgenticMemoryCompactor",
    "DEFAULT_COMPACTION_AGENT_PROMPT",
]
