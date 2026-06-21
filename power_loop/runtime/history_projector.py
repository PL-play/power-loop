"""Send-context history projection — the v2 derived-context layer.

A :class:`HistoryProjector` turns each FINISHED send's immutable ``pl_messages`` rows into a
compact, re-renderable projection (stored in ``pl_project_messages``), and renders stored
projections back into plain-text :data:`LoopMessage`\\ s for the LLM. ``pl_messages`` is never
modified; the projection is derived and rebuildable.

The library DEFAULT is **no projector** (``AgentLoopConfig.history_projector = None`` →
verbatim history + the in-place compactor, i.e. today's behavior). Set a projector to opt in.
Two ship here:

* :class:`IdentityProjector` — stores each send's rows verbatim and renders them back
  unchanged (history == today). For testing the seam; keeps full tool structure.
* :class:`DefaultDeterministicProjector` — a generic, no-LLM structured summary: each tool
  call is summarized via its optional ``ToolDefinition.project`` hook (else a truncating
  fallback), rendered to terse plain text (NO OpenAI ``tool_calls`` protocol shape — so a
  projected past send can never dangle a tool pair, and it is provider-agnostic).

Neither impl carries any application (DeepTalk) knowledge — a downstream projector subclass/
replacement supplies domain rendering (e.g. extracting a chat tool's delivered text).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from power_loop.runtime.store.types import MessageRow, ProjectMessageRow

if TYPE_CHECKING:
    from power_loop.tools.registry import ToolRegistry

LoopMessage = dict[str, Any]

#: Sentinel for "this tool call had NO result row at all" — distinct from a tool that
#: produced an empty (``""``) or null result. Kept private so it never leaks into stored
#: projection content (the projector turns it into an explicit ``<missing>`` marker).
_NO_RESULT = object()

__all__ = [
    "HistoryProjector",
    "IdentityProjector",
    "DefaultDeterministicProjector",
    "ProjectedRow",
    "ProjectedSend",
    "ProjectedCompact",
]


@dataclass
class ProjectedRow:
    """One row a projector wants written to ``pl_project_messages`` for a send."""

    kind: str  # 'user' | 'project'
    content: dict[str, Any]
    rendered_text: str | None = None


@dataclass
class ProjectedSend:
    """A projector's output for ONE finished send → rows + the source ``pl_messages`` span."""

    rows: list[ProjectedRow] = field(default_factory=list)
    source_seq_lo: int | None = None
    source_seq_hi: int | None = None


@dataclass
class ProjectedCompact:
    """A projector's fold of several projection rows into one ``compact`` row.

    ``from_send``/``to_send`` describe the span of the *newly folded* user/project sends
    ONLY — they intentionally do NOT extend over a rolled-forward prior ``compact`` row's
    own range. The loop is the authority on the persisted compact's full span: it computes
    ``compact_from_send`` from the prior compact (so nothing is lost) and ignores these
    fields. They exist for a standalone caller of ``compact()`` that has no prior context."""

    content: dict[str, Any]
    from_send: int
    to_send: int
    rendered_text: str | None = None


@runtime_checkable
class HistoryProjector(Protocol):
    """Strategy for the send-context projection layer. ``version`` is stamped on every row
    so a later projector change leaves prior rows intact (and is auditable). ``keep_last_sends``
    is how many most-recent finished sends stay individually projected before older ones are
    folded into a ``compact`` row (0 = never compact). ``trigger_ratio`` is the fraction of the
    loop's ``max_tokens`` the rendered projected prefix must reach before older sends fold
    (the token-driven fold trigger; the loop reads it via ``getattr(..., 0.75)`` so a custom
    projector that omits it still works, but declaring it keeps the contract explicit)."""

    version: int
    keep_last_sends: int
    trigger_ratio: float

    def project_send(
        self, send_rows: list[MessageRow], *, send_index: int, tool_registry: ToolRegistry | None
    ) -> ProjectedSend:
        """Project one finished send's ``pl_messages`` rows into projection rows."""
        ...

    def render(self, rows: list[ProjectMessageRow]) -> list[LoopMessage]:
        """Render stored projection rows (user/project/compact) into LLM messages."""
        ...

    def compact(self, rows: list[ProjectMessageRow]) -> ProjectedCompact | None:
        """Fold a span of projection rows into one ``compact`` row (or None = don't)."""
        ...


# ── helpers ───────────────────────────────────────────────────────────────────


def _truncate(s: Any, max_chars: int) -> str:
    t = "" if s is None else str(s)
    return t if len(t) <= max_chars else t[:max_chars] + "…"


def _parse_args(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw
    return raw


def _row_to_loop_dict(row: MessageRow) -> LoopMessage:
    """Verbatim MessageRow → LoopMessage (mirrors stateful_loop._row_to_loop_message; kept
    local to avoid an agent→runtime import cycle)."""
    msg: LoopMessage = {"role": row.role}
    if row.content is not None:
        msg["content"] = row.content
    if row.tool_calls:
        msg["tool_calls"] = list(row.tool_calls)
    if row.tool_call_id:
        msg["tool_call_id"] = row.tool_call_id
    if row.name:
        msg["name"] = row.name
    return msg


def _validate_projector_params(
    *, version: int, keep_last_sends: int, trigger_ratio: float
) -> None:
    """Fail-fast on the params shared by the shipped projectors, at construction (not deep in a
    later run). ``version`` >= 1 because it is the projection-compatibility key the reader uses
    to decide whether a stored row is faithfully renderable by the current projector (else it
    falls back to verbatim). A NaN ``trigger_ratio`` fails both comparisons → rejected, so it can
    never reach ``int(max_tokens * nan)`` (which would crash the fold)."""
    if version < 1:
        raise ValueError(
            f"projector version must be >= 1 (the projection-compatibility key); got {version!r}"
        )
    if keep_last_sends < 0:
        raise ValueError(f"projector keep_last_sends must be >= 0; got {keep_last_sends!r}")
    if not (trigger_ratio > 0 and trigger_ratio <= 1):
        raise ValueError(f"projector trigger_ratio must be in (0, 1]; got {trigger_ratio!r}")


# ── Identity (verbatim; for testing the seam) ──────────────────────────────────


@dataclass
class IdentityProjector:
    """Stores each send's rows verbatim and renders them back unchanged, so a session run
    with this projector sees byte-identical history to the no-projector default. Useful to
    prove the projection seam itself introduces no behavior change."""

    version: int = 1
    keep_last_sends: int = 0  # verbatim mode never folds
    trigger_ratio: float = 0.75  # unused (keep_last_sends==0 short-circuits folding); for Protocol parity

    def __post_init__(self) -> None:
        _validate_projector_params(
            version=self.version,
            keep_last_sends=self.keep_last_sends,
            trigger_ratio=self.trigger_ratio,
        )

    def project_send(
        self, send_rows: list[MessageRow], *, send_index: int, tool_registry: ToolRegistry | None
    ) -> ProjectedSend:
        seqs = [r.seq for r in send_rows]
        return ProjectedSend(
            rows=[ProjectedRow("project", {"messages": [_row_to_loop_dict(r) for r in send_rows]})],
            source_seq_lo=min(seqs) if seqs else None,
            source_seq_hi=max(seqs) if seqs else None,
        )

    def render(self, rows: list[ProjectMessageRow]) -> list[LoopMessage]:
        out: list[LoopMessage] = []
        for r in rows:
            out.extend((r.content or {}).get("messages") or [])
        return out

    def compact(self, rows: list[ProjectMessageRow]) -> ProjectedCompact | None:
        return None  # verbatim mode never folds


# ── Default deterministic (generic structured summary; no LLM) ──────────────────


@dataclass
class DefaultDeterministicProjector:
    """Generic, deterministic, no-LLM projector. Each send →
    ``user`` row: ``{"human": [<user inputs>]}`` (a LIST — folded follow-ups are preserved) +
    ``project`` row: ``{"tools": [...], "final_text": ...}``. Each tool call is summarized via
    its ``ToolDefinition.project`` hook when present, else a truncating fallback. Rendered to
    terse plain text with NO tool-protocol structure."""

    version: int = 1
    max_chars: int = 300  # per-field truncation budget
    keep_last_sends: int = 4  # most-recent sends ALWAYS kept individually (never folded)
    #: Fold older sends into a compact row once the rendered projected prefix reaches
    #: ``loop max_tokens × trigger_ratio`` (mirrors DefaultCompactor's policy) — token-driven,
    #: not send-count-driven. The most-recent ``keep_last_sends`` are always preserved.
    trigger_ratio: float = 0.75
    #: Hard cap (chars) on a rolled-forward ``compact`` row's rendered summary. This projector
    #: CONCATENATES (no LLM to summarize-to-shrink), so without a cap the compact — and thus the
    #: rendered context — would grow without bound over a long session, defeating the token fold.
    #: When the summary exceeds this, the OLDEST folded lines are dropped (their full detail stays
    #: in pl_messages, recoverable via recall_send). 0 disables the cap (unbounded — old behavior).
    max_compact_chars: int = 4000

    def __post_init__(self) -> None:
        _validate_projector_params(
            version=self.version,
            keep_last_sends=self.keep_last_sends,
            trigger_ratio=self.trigger_ratio,
        )
        if self.max_chars <= 0:
            raise ValueError(f"max_chars must be > 0; got {self.max_chars!r}")
        if self.max_compact_chars < 0:
            raise ValueError(
                f"max_compact_chars must be >= 0 (0 = unbounded); got {self.max_compact_chars!r}"
            )

    def project_send(
        self, send_rows: list[MessageRow], *, send_index: int, tool_registry: ToolRegistry | None
    ) -> ProjectedSend:
        users = [r for r in send_rows if r.role == "user"]
        # Tool results keyed by tool_call_id, preserving ORDER and DUPLICATES (a multimap of
        # FIFO queues): a malformed/imported/resumed transcript can repeat or omit an id, so a
        # plain dict would silently collapse two results onto one id (and drop the other). Each
        # assistant tool_call consumes the front of its id's queue; an empty/absent id falls
        # back to the "" bucket so positionally-paired empty-id calls still match.
        results: dict[str, deque[str | None]] = {}
        for r in send_rows:
            if r.role == "tool":
                results.setdefault(r.tool_call_id or "", deque()).append(r.content)
        tools: list[dict[str, Any]] = []
        final_text: str | None = None
        for r in send_rows:
            if r.role != "assistant":
                continue
            if r.content:
                final_text = r.content  # last non-empty assistant text is the send's output
            for tc in (r.tool_calls or []):
                fn = tc.get("function")
                fn = fn if isinstance(fn, dict) else {}
                name = fn.get("name") or tc.get("name")
                args = _parse_args(fn.get("arguments"))
                bucket = results.get(str(tc.get("id") or ""))
                # _NO_RESULT (no row) is distinct from a real None/"" result, so the projection
                # can show "<missing>" instead of masking an unfinished/failed call as empty.
                result = bucket.popleft() if bucket else _NO_RESULT
                tools.append(self._project_tool(name, args, result, tool_registry))
        seqs = [r.seq for r in send_rows]
        rows: list[ProjectedRow] = []
        if users:
            rows.append(ProjectedRow(
                "user", {"human": [_truncate(u.content, self.max_chars) for u in users]}
            ))
        rows.append(ProjectedRow("project", {
            "tools": tools,
            "final_text": _truncate(final_text, self.max_chars) if final_text else None,
        }))
        return ProjectedSend(
            rows=rows,
            source_seq_lo=min(seqs) if seqs else None,
            source_seq_hi=max(seqs) if seqs else None,
        )

    def _project_tool(
        self, name: str | None, args: Any, result: Any, tool_registry: ToolRegistry | None
    ) -> dict[str, Any]:
        # ``result`` is _NO_RESULT (no tool row found), or the real tool output (``str``/None/"").
        missing = result is _NO_RESULT
        result_str: str | None = None if missing else result
        rt = tool_registry.get(name) if (tool_registry is not None and name) else None
        proj = getattr(rt.definition, "project", None) if rt is not None else None
        if proj is not None:
            try:
                # Pass the result THROUGH (None when missing/null) so the hook can tell a
                # produced-but-empty result from one that never arrived. (Signature widened to
                # ``str | None``; the truncating fallback below handles None either way.)
                out = proj(args if isinstance(args, dict) else {}, result_str)
            except Exception:
                out = None  # a misbehaving tool projector must never break projection
            if isinstance(out, dict):
                return {"name": name, **out}
            if out is not None:
                return {"name": name, "summary": _truncate(out, self.max_chars)}
        if missing:
            return {"name": name, "result": "<missing>"}  # no result row — unfinished/failed call
        return {"name": name, "result": _truncate(result_str, self.max_chars)}

    # rendering ----------------------------------------------------------------
    def render(self, rows: list[ProjectMessageRow]) -> list[LoopMessage]:
        out: list[LoopMessage] = []
        for r in rows:
            if r.kind == "user":
                humans = (r.content or {}).get("human") or []
                out.append({"role": "user", "content": "\n".join(str(h) for h in humans)})
            elif r.kind == "project":
                out.append({"role": "assistant", "content": self._render_project(r.content)})
            elif r.kind == "compact":
                out.append({"role": "user", "content": self._render_compact(r.content)})
        return out

    def _render_tool(self, t: dict[str, Any]) -> str:
        name = t.get("name", "?")
        rest = {k: v for k, v in (t or {}).items() if k != "name"}
        if not rest:
            return str(name)
        body = ", ".join(f"{k}={v}" for k, v in rest.items())
        return f"{name}({body})"

    def _render_project(self, content: dict[str, Any] | None) -> str:
        content = content or {}
        parts: list[str] = []
        tools = content.get("tools") or []
        if tools:
            parts.append("[tools] " + "; ".join(self._render_tool(t) for t in tools))
        ft = content.get("final_text")
        if ft:
            parts.append(str(ft))
        return "\n".join(parts) if parts else "(no output)"

    def _render_compact(self, content: dict[str, Any] | None) -> str:
        return str((content or {}).get("summary") or "")

    # compaction ---------------------------------------------------------------
    def compact(self, rows: list[ProjectMessageRow]) -> ProjectedCompact | None:
        """Fold user/project rows (and an optional prior ``compact`` row, rolled forward so
        nothing is lost) into one summary. Deterministic concat, no LLM. ``from_send``/
        ``to_send`` are derived from the user/project send range; a leading prior-compact row's
        own range is preserved by the caller via ``compact_from_send``."""
        lines: list[str] = []
        folded_sends: list[int] = []
        for r in rows:
            if r.kind == "compact":
                prior = (r.content or {}).get("summary")
                if prior:
                    lines.append(str(prior))
            elif r.kind == "user":
                for h in (r.content or {}).get("human") or []:
                    lines.append(f"#{r.send_index} user: {h}")
                folded_sends.append(r.send_index)
            elif r.kind == "project":
                lines.append(f"#{r.send_index} agent: {self._render_project(r.content)}")
                folded_sends.append(r.send_index)
        if not folded_sends:
            return None
        summary = "\n".join(lines)
        if self.max_compact_chars and len(summary) > self.max_compact_chars:
            # No LLM to compress — bound the rolled-forward compact by keeping the most-recent
            # tail and marking the drop. The dropped sends remain in pl_messages (recall_send
            # recovers full detail), so this caps context size without losing the audit trail.
            summary = (
                "…[older folded sends omitted — use recall_send(#N) for full detail]\n"
                + summary[-self.max_compact_chars:]
            )
        return ProjectedCompact(
            content={"summary": summary},
            from_send=min(folded_sends),
            to_send=max(folded_sends),
        )
