"""Context fold / compaction — the second of the two orthogonal context axes (power-loop 3.0).

A :class:`FoldStrategy` turns N already-recorded context records (``ProjectMessageRow``s) into ONE
``compact`` record. Its essence is "turn large text into small text"; it MAY have side effects
(write notes / memory) depending on the concrete strategy. It is orthogonal to
:mod:`power_loop.runtime.representation`: a strategy works on rows regardless of which
representation produced them (it re-renders them to text via ``context.representation.render``).

Two ship here:

* :class:`LLMSummaryFold` — the default. One LLM call, no tools/side effects.
* :class:`AgenticFold` — an LLM + a bounded tool loop that captures durable facts as notes
  (returned as :class:`NoteOp`\\ s the loop applies best-effort after the compact commits).

There is intentionally NO deterministic/no-LLM fold: concatenation/truncation isn't compaction.
Custom strategies implement the :class:`FoldStrategy` Protocol and are passed straight into config.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from power_loop._vendor.llm_client.interface import LLMRequest

if TYPE_CHECKING:
    from power_loop.runtime.memory import MemoryProvider
    from power_loop.runtime.representation import LoopMessage, Representation
    from power_loop.runtime.store.types import ProjectMessageRow
    from power_loop.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "FoldStrategy",
    "FoldContext",
    "FoldResult",
    "NoteOp",
    "LLMSummaryFold",
    "AgenticFold",
    "DEFAULT_FOLD_AGENT_PROMPT",
]


@dataclass(frozen=True)
class NoteOp:
    """A deferred note write an (agentic) fold captured. The loop applies these BEST-EFFORT after
    the compact commits (additive memory — not transactional with the compact; a rare crash between
    loses a note but never corrupts context). Returned in ``FoldResult.note_ops``."""

    op: str  # 'add' | 'update'
    content: str | None = None
    pinned: bool | None = None
    note_id: int | None = None


@dataclass(frozen=True)
class FoldContext:
    """Everything a strategy needs at fold time, for BOTH representations and ALL side-effect
    classes. The loop builds it INSIDE the active session + contextvar scope so any direct note/
    memory tool use resolves; the preferred channel is returning ``FoldResult.note_ops`` though."""

    session_id: str
    round_index: int
    representation: Representation  # re-render rows to text to summarize (substrate-agnostic)
    llm: Any | None = None  # main loop LLM; None only in a no-LLM unit harness
    summary_llm: Any | None = None  # optional cheaper dedicated summary model
    tool_registry: ToolRegistry | None = None  # for custom strategies that write directly
    memory: MemoryProvider | None = None  # for custom strategies that persist to memory
    max_tokens: int | None = None  # the budget (drives self-bounding of summary length)
    current_tokens: int | None = None  # loop's incremental rendered-prefix estimate


@dataclass(frozen=True)
class FoldResult:
    """One compact record + the SEND span it covers. The loop is the authority on the persisted
    compact's full ``from_send`` (it rolls a prior compact forward), so this exposes only the
    newly-folded ``folded_to_send`` cursor + the strategy's summary + any deferred note ops."""

    content: dict[str, Any]  # {"summary": "..."} — the compact row's content
    folded_to_send: int  # last folded send_index == the new read cursor
    rendered_text: str | None = None  # optional precomputed render (else representation.render)
    note_ops: tuple[NoteOp, ...] = ()  # applied best-effort after the compact commits (see NoteOp)


@runtime_checkable
class FoldStrategy(Protocol):
    """Turn a span of context records (an optional leading prior ``compact`` + the oldest live
    user/project sends) into ONE compact record. May have side effects per concrete strategy.

    ``keep_last_sends`` (>= 1): most-recent finished sends kept individually before older fold.
    ``trigger_ratio`` (0,1]: fold when rendered prefix tokens >= ``max_tokens × trigger_ratio``.
    ``fold_id``: stable id stamped on compact rows (``fold_version`` column) so a strategy switch
    is detectable per-row and re-foldable."""

    keep_last_sends: int
    trigger_ratio: float
    fold_id: str

    async def fold(
        self, rows: list[ProjectMessageRow], *, context: FoldContext
    ) -> FoldResult | None:
        """Fold ``rows`` (loop-filtered to the foldable span) into one compact. Return ``None`` to
        decline (nothing foldable / soft-fail). MUST NOT touch ``pl_messages`` or the store
        directly — the loop persists the compact (optimistic-concurrency commit), then applies any
        ``note_ops`` best-effort. MAY read text via ``context.representation.render(rows)``."""
        ...


# ── helpers ─────────────────────────────────────────────────────────────────


def _folded_sends(rows: list[ProjectMessageRow]) -> list[int]:
    """The send_indexes of the user/project rows actually being folded (a leading prior compact
    row is rolled forward but is not itself a newly-folded send)."""
    return [r.send_index for r in rows if r.kind in ("user", "project")]


def _stringify_messages(messages: list[LoopMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content")
        text = content if isinstance(content, str) else str(content or "")
        head = f"[{role}]"
        tool_calls = m.get("tool_calls")
        if tool_calls:
            head += f" tool_calls={len(tool_calls)}"
        lines.append(f"{head}\n{text}")
    return "\n\n".join(lines)


def _render_span_text(rows: list[ProjectMessageRow], representation: Representation) -> str:
    """Re-render the foldable rows (incl. a leading prior compact, rolled forward) to plain text
    for summarization — substrate-agnostic: works for verbatim OR projection rows."""
    return _stringify_messages(representation.render(rows))


def _strip_summary(text: str | None) -> str | None:
    """Pull the ``<summary>…</summary>`` body if present, else return the trimmed text (or None)."""
    text = (text or "").strip()
    if not text:
        return None
    if "<summary>" in text and "</summary>" in text:
        text = text.split("<summary>", 1)[1].split("</summary>", 1)[0].strip()
    return text or None


def _response_text(response: Any) -> str:
    return (
        getattr(response, "raw_text", "") or getattr(response, "content_text", "") or ""
    ).strip()


_SUMMARY_PROMPT = (
    "You are a conversation summarizer. Below is a slice of an agent's working transcript that "
    "needs to be compressed for context-window economy. The slice may include prior compact "
    "summaries — merge their content so there is at most ONE compact summary. Preserve: "
    "(1) decisions made, (2) facts established, (3) errors and how they were handled, (4) any "
    "pending intent the assistant was about to act on. Do NOT call tools. Wrap your summary in "
    "<summary>…</summary>.\n\n--- transcript slice ---\n"
)


async def _summarize(llm: Any, slice_text: str, *, max_tokens: int) -> str | None:
    """One summarization LLM call. Soft-fails to None on any error."""
    try:
        response = await llm.complete(
            LLMRequest(
                messages=[{"role": "user", "content": _SUMMARY_PROMPT + slice_text}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
        )
    except Exception:
        return None
    return _strip_summary(_response_text(response))


# ── LLM single-call fold (default) ───────────────────────────────────────────


@dataclass
class LLMSummaryFold:
    """Default fold strategy: a single LLM summarization call, no tools/side effects. Ports the
    old ``DefaultCompactor`` summarize step onto the representation-agnostic record interface."""

    keep_last_sends: int = 4
    trigger_ratio: float = 0.75
    summary_max_tokens: int = 5000
    summary_llm: Any | None = None  # optional default; context.summary_llm takes precedence
    fold_id: str = "llm_summary"

    def __post_init__(self) -> None:
        _validate_fold_params(
            keep_last_sends=self.keep_last_sends, trigger_ratio=self.trigger_ratio,
            summary_max_tokens=self.summary_max_tokens,
        )

    async def fold(
        self, rows: list[ProjectMessageRow], *, context: FoldContext
    ) -> FoldResult | None:
        folded = _folded_sends(rows)
        if not folded:
            return None  # nothing new to fold (e.g. only a prior compact row)
        llm = context.summary_llm or self.summary_llm or context.llm
        if llm is None:
            return None
        slice_text = _render_span_text(rows, context.representation)
        summary = await _summarize(llm, slice_text, max_tokens=self.summary_max_tokens)
        if summary is None:
            return None  # soft-fail; loop keeps the span unfolded this send
        return FoldResult(content={"summary": summary}, folded_to_send=max(folded))


# ── Agentic, memory-aware fold (LLM + tools → notes) ─────────────────────────


DEFAULT_FOLD_AGENT_PROMPT = (
    "You are the memory & compaction agent for a long-running AI assistant. You are given a SLICE "
    "of the assistant's working transcript that is about to be REMOVED from the live context "
    "window to save space. You are NOT answering the user — you are preserving what matters.\n\n"
    "Do this in order:\n"
    "1. EXTRACT durable facts worth remembering AFTER this slice leaves context — decisions made, "
    "stable user preferences/constraints, established facts, unresolved commitments, and hard-won "
    "fixes to errors — and SAVE each as a concise note via `note(action=add, ...)` (one fact per "
    "note). Use `note(action=update, ...)` to refine an existing note instead of duplicating. "
    "Save ONLY what "
    "will matter later; skip transient chatter. If nothing is worth remembering, save nothing.\n"
    "2. THEN write a faithful, compact summary of the slice for the context window. Preserve: "
    "(1) decisions, (2) facts established, (3) errors and how they were handled, (4) any pending "
    "intent the assistant was about to act on. Merge any prior compact content you see into one "
    "coherent summary. Do not invent. Keep it tight.\n\n"
    "Output the summary LAST, wrapped in <summary>…</summary>, in a message with NO tool calls."
)

#: The unified memory tool the agentic fold offers the model. Calls are CAPTURED into NoteOps (not
#: written here), so the strategy stays side-effect-free until the loop applies them best-effort.
_CAPTURE_NOTE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "note",
            "description": "Add or update a durable persistent note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "update"]},
                    "note_id": {"type": "integer", "description": "Required for update."},
                    "content": {"type": "string", "description": "The note text."},
                    "pinned": {"type": "boolean", "description": "Pin so it is never auto-evicted."},
                },
                "required": ["action"],
            },
        },
    },
]


@dataclass
class AgenticFold:
    """Fold strategy that runs a bounded, memory-aware agent loop: the model issues unified note
    add/update actions to persist durable facts, THEN writes the compact summary. Note calls are
    CAPTURED as :class:`NoteOp`\\ s (returned in ``FoldResult.note_ops``) and applied by the loop
    best-effort after the compact commits — so the strategy is side-effect-free + testable in
    isolation. On ANY failure it falls back to a plain single-call summary (KEEPING any notes
    already captured), so it never blocks a fold. Ports the old ``AgenticMemoryCompactor``."""

    keep_last_sends: int = 4
    trigger_ratio: float = 0.75
    summary_max_tokens: int = 5000
    max_rounds: int = 4
    system_prompt: str | None = None
    summary_llm: Any | None = None
    fold_id: str = "agentic"

    def __post_init__(self) -> None:
        _validate_fold_params(
            keep_last_sends=self.keep_last_sends, trigger_ratio=self.trigger_ratio,
            summary_max_tokens=self.summary_max_tokens,
        )
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1; got {self.max_rounds!r}")

    async def fold(
        self, rows: list[ProjectMessageRow], *, context: FoldContext
    ) -> FoldResult | None:
        folded = _folded_sends(rows)
        if not folded:
            return None
        llm = context.summary_llm or self.summary_llm or context.llm
        if llm is None:
            return None
        slice_text = _render_span_text(rows, context.representation)
        summary: str | None = None
        note_ops: tuple[NoteOp, ...] = ()
        try:
            summary, note_ops = await self._agentic_summarize(llm, slice_text)
        except Exception:
            logger.exception("agentic fold failed; falling back to single-call summary")
            note_ops = ()  # the agentic attempt raised mid-way → captured notes are unreliable
        if summary is None:
            # The agentic loop may have CAPTURED notes but produced no usable summary (exhausted
            # rounds / empty final). Keep those notes — only get the summary from a plain call.
            summary = await _summarize(llm, slice_text, max_tokens=self.summary_max_tokens)
        if summary is None:
            return None
        return FoldResult(
            content={"summary": summary}, folded_to_send=max(folded), note_ops=note_ops
        )

    async def _agentic_summarize(
        self, llm: Any, slice_text: str
    ) -> tuple[str | None, tuple[NoteOp, ...]]:
        system_prompt = self.system_prompt or DEFAULT_FOLD_AGENT_PROMPT
        convo: list[dict[str, Any]] = [
            {"role": "user", "content": "--- transcript slice to compact ---\n" + slice_text}
        ]
        captured: list[NoteOp] = []
        for _ in range(self.max_rounds):
            resp = await llm.complete(
                LLMRequest(
                    messages=convo,
                    system_prompt=system_prompt,
                    tools=_CAPTURE_NOTE_TOOLS,
                    max_tokens=self.summary_max_tokens,
                    temperature=0.0,
                )
            )
            text = _response_text(resp)
            tool_calls = resp.get_tool_calls()
            if not tool_calls:
                return _strip_summary(text), tuple(captured)
            convo.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
            for tc in tool_calls:
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = (fn or {}).get("name") or tc.get("name")
                raw_args = (fn or {}).get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except (TypeError, ValueError):
                    args = {}
                args = args if isinstance(args, dict) else {}
                result = self._capture(name, args, captured)
                convo.append({"role": "tool", "tool_call_id": tc.get("id"), "content": result})
        # Rounds exhausted while still tool-calling — one final, tool-free summary request.
        resp = await llm.complete(
            LLMRequest(
                messages=[*convo, {"role": "user", "content": "Now output ONLY the <summary>…</summary>."}],
                system_prompt=system_prompt,
                max_tokens=self.summary_max_tokens,
                temperature=0.0,
            )
        )
        return _strip_summary(_response_text(resp)), tuple(captured)

    @staticmethod
    def _capture(name: str | None, args: dict[str, Any], captured: list[NoteOp]) -> str:
        if name != "note":
            return f"error: unknown tool {name!r}"
        action = str(args.get("action") or "").strip().lower()
        if action == "add":
            content = args.get("content")
            if not content:
                return "error: note action=add requires content"
            captured.append(NoteOp(op="add", content=str(content), pinned=bool(args.get("pinned"))))
            return "ok: note captured"
        if action == "update":
            note_id = args.get("note_id")
            if note_id is None:
                return "error: note action=update requires note_id"
            try:
                nid = int(note_id)
            except (TypeError, ValueError):
                # A bad arg must NOT raise (that would abort the round + discard every captured
                # note); answer the tool with an error so the loop continues.
                return f"error: note_id must be an integer, got {note_id!r}"
            captured.append(
                NoteOp(
                    op="update",
                    note_id=nid,
                    content=(str(args["content"]) if args.get("content") is not None else None),
                    pinned=(bool(args["pinned"]) if "pinned" in args else None),
                )
            )
            return "ok: note update captured"
        return f"error: unsupported note action {action!r}"


def _validate_fold_params(
    *, keep_last_sends: int, trigger_ratio: float, summary_max_tokens: int | None = None
) -> None:
    if keep_last_sends < 1:
        raise ValueError(
            f"fold keep_last_sends must be >= 1 (a fold must keep the in-flight context "
            f"coherent; there is no never-fold); got {keep_last_sends!r}"
        )
    if not (trigger_ratio > 0 and trigger_ratio <= 1):
        raise ValueError(f"fold trigger_ratio must be in (0, 1]; got {trigger_ratio!r}")
    if summary_max_tokens is not None and summary_max_tokens < 1:
        raise ValueError(f"fold summary_max_tokens must be >= 1; got {summary_max_tokens!r}")
