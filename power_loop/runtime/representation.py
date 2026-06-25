"""Context representation — one of the two orthogonal axes of context handling (power-loop 3.0).

A :class:`Representation` decides HOW a finished send is recorded into ``pl_project_messages``
and rendered back to the LLM. It is **build + render only** — it NEVER decides what to drop
(that is :mod:`power_loop.runtime.fold`'s :class:`FoldStrategy`). The two axes are independent:
any representation composes with any fold strategy.

Two ship here:

* :class:`VerbatimRepresentation` (``kind="verbatim"``) — records each send's rows verbatim and
  renders them back unchanged, so history is byte-identical to a no-projection run (below the
  fold threshold). Its ``render`` ALSO renders ``compact`` rows (a fold can apply under verbatim).
* :class:`ProjectedRepresentation` (``kind="projection"``) — a generic, no-LLM per-send
  structured summary: each tool call is summarized via its optional ``ToolDefinition.project``
  hook (else a truncating fallback), rendered to terse plain text with NO tool-protocol shape.

Custom representations implement the :class:`Representation` Protocol (kind/version + project_send
+ render) and are passed straight into config — no inheritance required.
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

#: Sentinel for "this tool call had NO result row at all" — distinct from a tool that produced an
#: empty (``""``) or null result. Private so it never leaks into stored projection content.
_NO_RESULT = object()

__all__ = [
    "Representation",
    "VerbatimRepresentation",
    "ProjectedRepresentation",
    "ProjectedRow",
    "ProjectedSend",
]


@dataclass
class ProjectedRow:
    """One row a representation wants written to ``pl_project_messages`` for a send."""

    kind: str  # 'user' | 'project'
    content: dict[str, Any]
    rendered_text: str | None = None


@dataclass
class ProjectedSend:
    """A representation's output for ONE finished send → rows + the source ``pl_messages`` span."""

    rows: list[ProjectedRow] = field(default_factory=list)
    source_seq_lo: int | None = None
    source_seq_hi: int | None = None


@runtime_checkable
class Representation(Protocol):
    """How a finished send is recorded into ``pl_project_messages`` and rendered back. Build +
    render only; never folds. ``kind`` is stamped on every row (``representation_kind`` column);
    the reader renders a row only when its ``(kind, version)`` matches the active representation,
    else falls back to verbatim — so two representations never mis-render each other's rows.
    ``version`` (>= 1) is the secondary compat key within a kind."""

    kind: str
    version: int

    def project_send(
        self, send_rows: list[MessageRow], *, send_index: int, tool_registry: ToolRegistry | None
    ) -> ProjectedSend:
        """Record one finished send's ``pl_messages`` rows into projection rows."""
        ...

    def render(self, rows: list[ProjectMessageRow]) -> list[LoopMessage]:
        """Render stored rows (user/project/compact) into LLM messages. MUST handle
        ``kind='compact'`` (a fold can apply under any representation)."""
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
    """Verbatim MessageRow → LoopMessage (mirrors stateful_loop._row_to_loop_message; kept local
    to avoid an agent→runtime import cycle)."""
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


def _validate_representation_params(*, version: int, max_chars: int | None = None) -> None:
    if version < 1:
        raise ValueError(
            f"representation version must be >= 1 (the compatibility key); got {version!r}"
        )
    if max_chars is not None and max_chars <= 0:
        raise ValueError(f"max_chars must be > 0; got {max_chars!r}")


def _render_compact_row(row: ProjectMessageRow) -> LoopMessage:
    """A folded ``compact`` row → a single user-role text block. Shared by every representation
    (a fold can apply under any representation), so it is rendered uniformly."""
    text = row.rendered_text or (row.content or {}).get("summary") or ""
    return {"role": "user", "content": str(text)}


# ── Verbatim (byte-identical history; renders compact rows too) ──────────────────


@dataclass
class VerbatimRepresentation:
    """Records each send's rows verbatim and renders them back unchanged, so history is
    byte-identical to a no-projection run below the fold threshold. Unlike the old
    ``IdentityProjector`` it does NOT short-circuit folding (folding is an orthogonal concern now)
    and its ``render`` handles ``compact`` rows produced by whatever fold strategy is configured."""

    kind: str = "verbatim"
    version: int = 1

    def __post_init__(self) -> None:
        _validate_representation_params(version=self.version)

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
            if r.kind == "compact":
                out.append(_render_compact_row(r))
            else:  # user / project verbatim rows carry the raw message list
                out.extend((r.content or {}).get("messages") or [])
        return out


# ── Projection (generic structured summary; no LLM) ─────────────────────────────


@dataclass
class ProjectedRepresentation:
    """Generic, deterministic, no-LLM per-send projection. Each send →
    ``user`` row: ``{"input": [<user/trigger inputs, verbatim>]}`` (a LIST — folded follow-ups
    preserved; pre-3.3 rows used the key ``human``) +
    ``project`` row: ``{"tools": [...], "final_text": ...}``. Each tool call is summarized via its
    ``ToolDefinition.project`` hook when present, else a truncating fallback. Rendered to terse
    plain text with NO tool-protocol structure. (This is the old ``DefaultDeterministicProjector``
    MINUS its fold knobs/``compact()`` — folding now lives on :class:`FoldStrategy`.)"""

    kind: str = "projection"
    version: int = 1
    max_chars: int = 300  # per-field truncation budget

    def __post_init__(self) -> None:
        _validate_representation_params(version=self.version, max_chars=self.max_chars)

    def project_send(
        self, send_rows: list[MessageRow], *, send_index: int, tool_registry: ToolRegistry | None
    ) -> ProjectedSend:
        users = [r for r in send_rows if r.role == "user"]
        # Tool results keyed by tool_call_id, preserving ORDER and DUPLICATES (a multimap of FIFO
        # queues): a malformed/imported/resumed transcript can repeat or omit an id, so a plain
        # dict would silently collapse two results onto one id (and drop the other).
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
                result = bucket.popleft() if bucket else _NO_RESULT
                tools.append(self._project_tool(name, args, result, tool_registry))
        seqs = [r.seq for r in send_rows]
        rows: list[ProjectedRow] = []
        if users:
            # The INPUT side of a send (the user/trigger turn) is kept VERBATIM — it is the actual
            # conversation content, it is short relative to tool output, and truncating it would drop
            # context the model genuinely needs. Only the assistant's WORK (tool args/results +
            # final_text) is compressed, which is where the token savings actually are. Key is
            # ``input`` (the input turn — not necessarily a human; a multi-agent host feeds another
            # agent's message here); pre-3.3 rows used ``human`` and are still read (see render()).
            rows.append(ProjectedRow("user", {"input": [u.content for u in users]}))
        rows.append(
            ProjectedRow(
                "project",
                {
                    "tools": tools,
                    "final_text": _truncate(final_text, self.max_chars) if final_text else None,
                },
            )
        )
        return ProjectedSend(
            rows=rows,
            source_seq_lo=min(seqs) if seqs else None,
            source_seq_hi=max(seqs) if seqs else None,
        )

    def _project_tool(
        self, name: str | None, args: Any, result: Any, tool_registry: ToolRegistry | None
    ) -> dict[str, Any]:
        missing = result is _NO_RESULT
        result_str: str | None = None if missing else result
        rt = tool_registry.get(name) if (tool_registry is not None and name) else None
        proj = getattr(rt.definition, "project", None) if rt is not None else None
        if proj is not None:
            try:
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
        # Each rendered send is tagged with its ``#N`` send_index so the model can call
        # recall_send(send_index=N) on a folded/compacted earlier turn — the tool docstring and the
        # host's RECALL_SEND_NOTE both tell it to use "the #N the summary shows", so render MUST
        # actually emit them (else recall_send is undiscoverable). The folded compact carries its
        # covered range.
        out: list[LoopMessage] = []
        for r in rows:
            si = r.send_index
            if r.kind == "user":
                content = r.content or {}
                # ``input`` since 3.3; ``human`` is the pre-3.3 key — read both so old projection
                # rows still render correctly after upgrade.
                inputs = content.get("input")
                if inputs is None:
                    inputs = content.get("human") or []
                tag = f"[#{si}] " if si is not None else ""
                out.append({"role": "user", "content": tag + "\n".join(str(h) for h in inputs)})
            elif r.kind == "project":
                tag = f"#{si} " if si is not None else ""
                out.append({"role": "assistant", "content": tag + self._render_project(r.content)})
            elif r.kind == "compact":
                msg = _render_compact_row(r)
                lo, hi = r.compact_from_send, r.compact_to_send
                if lo is not None and hi is not None and hi >= lo > 0:
                    rng = f"#{lo}" if lo == hi else f"#{lo}–#{hi}"
                    msg = {
                        "role": "user",
                        "content": f"[older sends {rng} folded — recall_send(send_index=N) to "
                        f"expand]\n{msg['content']}",
                    }
                out.append(msg)
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
