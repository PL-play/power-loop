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
from dataclasses import fields as dataclass_fields
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

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
    "ProjectionRenderConfig",
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
    if max_chars <= 0 or len(t) <= max_chars:  # max_chars <= 0 → no truncation (unlimited)
        return t
    return t[:max_chars] + "…"


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


def _coerce_bool(v: Any) -> bool:
    """Strict-ish bool coercion for JSON-sourced config: 'true'/'1'/'yes'/'on' → True,
    'false'/'0'/'no'/'off'/'' → False (case-insensitive); ints via truthiness."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "on"}:
            return True
        if s in {"false", "0", "no", "off", ""}:
            return False
    return bool(v)


def _validate_representation_params(*, version: int, max_chars: int | None = None) -> None:
    if version < 1:
        raise ValueError(
            f"representation version must be >= 1 (the compatibility key); got {version!r}"
        )
    # max_chars <= 0 is now VALID and means "no truncation" (unlimited) — let a host disable the
    # library's per-field cap and do its own limiting. Any int is accepted.


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
class ProjectionRenderConfig:
    """Tunable format knobs for :meth:`ProjectedRepresentation.render` — they change the SHAPE of the
    rendered text only, never what is stored. Every field is a plain scalar so the whole config
    round-trips through JSON (a host can surface it in an admin UI and retune the rendered context
    live, then pass it back via ``ProjectedRepresentation(render_config=…)``). Placeholders: ``{n}`` in
    a tag → the send_index (empty tag or ``None`` index → no tag); ``{range}`` in ``fold_note`` → the
    folded ``#lo–#hi`` span. The defaults reproduce the built-in rendering exactly."""

    user_tag: str = "[#{n}] "  # prefix on a user/input row; {n}=send_index
    project_tag: str = "#{n} "  # prefix on a project/assistant row
    tools_header: str = "[tools] "  # before the tool list in a project row
    tool_sep: str = "; "  # between tools
    tool_arg_sep: str = ", "  # between a tool's k=v fields
    include_tools: bool = True  # render the tool list at all
    include_final_text: bool = True  # render the project row's trailing final_text
    empty_project: str = "(no output)"  # a project row that renders to nothing
    fold_note: str = "[older sends {range} folded — recall_send(send_index=N) to expand]"

    #: The two boolean knobs — coerced strictly from JSON so a string "false" doesn't read as True.
    _BOOL_FIELDS: ClassVar[frozenset[str]] = frozenset({"include_tools", "include_final_text"})

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ProjectionRenderConfig:
        """Build from a plain dict (e.g. JSON config), silently ignoring unknown keys. Known scalar
        fields are coerced to their declared types (projection-2): JSON serializes a bool as `true`,
        but a hand-written/templated config may carry the string "false", which is truthy — coerce
        the boolean knobs via a strict parser so they behave as written."""
        if not data:
            return cls()
        names = {f.name for f in dataclass_fields(cls)}
        coerced: dict[str, Any] = {}
        for k, v in data.items():
            if k not in names:
                continue
            coerced[k] = _coerce_bool(v) if k in cls._BOOL_FIELDS else str(v)
        return cls(**coerced)


@dataclass
class ProjectedRepresentation:
    """Generic, deterministic, no-LLM per-send projection. Each send →
    ``user`` row: ``{"input": [<user/trigger inputs before the first assistant turn, verbatim>]}``
    (a LIST; pre-3.3 rows used the key ``human``) +
    ``project`` row: ``{"tools": [...], "final_text": ...}``. Each tool call is summarized via its
    ``ToolDefinition.project`` hook when present, else a truncating fallback. MID-SEND user rows
    (durable reminder/finalize injections, drained steering follow-ups) appear in the ``tools``
    timeline as ``{"name": "__user__", "text": ...}`` entries at their chronological position
    (3.12 — previously they were folded into the input list, rewriting history). Rendered to terse
    plain text with NO tool-protocol structure. (This is the old ``DefaultDeterministicProjector``
    MINUS its fold knobs/``compact()`` — folding now lives on :class:`FoldStrategy`.)

    The render is customizable two ways without copy-pasting it: pass a :class:`ProjectionRenderConfig`
    (``render_config=`` — JSON-friendly format knobs, retunable from config/an admin UI) and/or
    override one of the small per-row methods (``render_row`` → ``render_user_row`` /
    ``render_project_row`` / ``render_compact_row``). Defaults reproduce the prior output exactly."""

    kind: str = "projection"
    version: int = 1
    max_chars: int = 300  # per-field truncation budget; <= 0 = no truncation (unlimited)
    #: Format knobs for render() — see :class:`ProjectionRenderConfig`. A plain dict is coerced (so a
    #: host can pass JSON config straight through). Subclasses can ALSO override the render_* methods.
    render_config: ProjectionRenderConfig = field(default_factory=ProjectionRenderConfig)
    #: How many of the most-recent sends a recency-aware ``render_project_row`` should render richly
    #: (older sends collapse to a stable one-liner). Set == the fold's ``keep_last_sends`` so the
    #: recency boundary and the fold boundary coincide. Consumed by :meth:`stamp_render_context`.
    hot_window: int = 4

    def __post_init__(self) -> None:
        _validate_representation_params(version=self.version, max_chars=self.max_chars)
        if isinstance(self.render_config, dict):
            self.render_config = ProjectionRenderConfig.from_dict(self.render_config)

    def project_send(
        self, send_rows: list[MessageRow], *, send_index: int, tool_registry: ToolRegistry | None
    ) -> ProjectedSend:
        # Tool results keyed by tool_call_id, preserving ORDER and DUPLICATES (a multimap of FIFO
        # queues): a malformed/imported/resumed transcript can repeat or omit an id, so a plain
        # dict would silently collapse two results onto one id (and drop the other).
        results: dict[str, deque[tuple[str | None, int | None]]] = {}
        for r in send_rows:
            if r.role == "tool":
                results.setdefault(r.tool_call_id or "", deque()).append((r.content, r.seq))
        tools: list[dict[str, Any]] = []
        inputs: list[str | None] = []
        final_text: str | None = None
        # One CHRONOLOGICAL pass: user rows BEFORE the first assistant turn are the send's
        # input; user rows appearing MID-SEND (durable LLM_BEFORE injections like starvation
        # reminders — 3.9.0 —, COMPLETE_DECIDE finalize prompts — 3.11.0 —, drained steering
        # follow-ups) are interleaved into the work timeline as ``__user__`` entries at their
        # real position. Folding them all into the input list (the pre-3.12 behavior) rewrote
        # history: the NEXT send's context showed every mid-run reminder as if it arrived
        # before the work started.
        seen_assistant = False
        for r in send_rows:
            if r.role == "user":
                if seen_assistant:
                    # Verbatim like the input (conversation content, not tool output).
                    tools.append({"name": "__user__", "text": r.content, "seq": r.seq})
                else:
                    inputs.append(r.content)
                continue
            if r.role != "assistant":
                continue
            seen_assistant = True
            if r.content:
                final_text = r.content  # last non-empty assistant text is the send's output
            for tc in (r.tool_calls or []):
                fn = tc.get("function")
                fn = fn if isinstance(fn, dict) else {}
                name = fn.get("name") or tc.get("name")
                args = _parse_args(fn.get("arguments"))
                bucket = results.get(str(tc.get("id") or ""))
                result, result_seq = bucket.popleft() if bucket else (_NO_RESULT, None)
                entry = self._project_tool(name, args, result, tool_registry)
                # Row coordinates (5.4.0): every projected tool entry carries the pl_messages seq of
                # its RESULT row (``seq``) and of the assistant row that issued the call
                # (``call_seq``), so a host renderer can print a self-contained
                # "recall_send(send_index=N, seq=S)" pointer on each line — the model should never
                # have to scan other sends to find where a dropped body lives.
                if isinstance(entry, dict):
                    entry.setdefault("seq", result_seq)
                    entry.setdefault("call_seq", r.seq)
                tools.append(entry)
        seqs = [r.seq for r in send_rows]
        rows: list[ProjectedRow] = []
        if inputs:
            # The INPUT side of a send (the user/trigger turn) is kept VERBATIM — it is the actual
            # conversation content, it is short relative to tool output, and truncating it would drop
            # context the model genuinely needs. Only the assistant's WORK (tool args/results +
            # final_text) is compressed, which is where the token savings actually are. Key is
            # ``input`` (the input turn — not necessarily a human; a multi-agent host feeds another
            # agent's message here); pre-3.3 rows used ``human`` and are still read (see render()).
            rows.append(ProjectedRow("user", {"input": inputs}))
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
    # Orchestration only; the per-row shapes live in small overridable methods (override exactly the
    # one you want instead of copy-pasting render) and the format lives in render_config (retune from
    # config, no code). Each rendered send is tagged with its ``#N`` send_index so the model can call
    # recall_send(send_index=N) on a folded earlier turn — the tool docstring + the host's
    # RECALL_SEND_NOTE tell it to use "the #N the summary shows", so the tags MUST be emitted (else
    # recall_send is undiscoverable); changing user_tag/project_tag away from a ``{n}`` form is at the
    # host's own risk.
    def stamp_render_context(
        self, rows: list[ProjectMessageRow], current_send_index: int | None
    ) -> None:
        """Attach transient RECENCY + cross-send-dedup context to project rows so a recency-aware
        ``render_project_row`` can (a) render the most-recent ``hot_window`` sends richly and collapse
        older ones, and (b) drop a stored tool entry whose dedup key ``k`` is superseded by the same
        ``k`` in a LATER send (e.g. a file re-read; the raw original stays in pl_messages, recoverable
        via recall_send).

        Recency is keyed on ABSOLUTE ``send_index`` vs ``current_send_index`` — NOT list position —
        so the fold's render of an ISOLATED old-span still classifies those sends as cold (else the
        old span would look 'recent', render richly, and the fold summary would never shrink). Rows
        are freshly hydrated per context build; these attrs never persist. Call ONCE over the full
        ordered project-row list before per-send rendering. Base ``render_project_row`` ignores these;
        only a recency-aware override consumes them (readers must ``getattr`` with a cold default)."""
        latest_key: dict[str, int] = {}
        for r in rows:
            if r.kind != "project" or not isinstance(r.content, dict) or r.send_index is None:
                continue
            for t in r.content.get("tools") or []:
                k = t.get("k") if isinstance(t, dict) else None
                if k and r.send_index > latest_key.get(k, -1):
                    latest_key[k] = r.send_index
        cur = None if current_send_index is None else int(current_send_index)
        for r in rows:
            if r.send_index is None or cur is None:
                r.recency = 10**9  # unknown position → cold (safe for the fold's isolated render)
            else:
                r.recency = max(0, cur - 1 - int(r.send_index))
            r.render_ctx = {"latest_key": latest_key, "hot_window": int(self.hot_window)}

    def render(self, rows: list[ProjectMessageRow]) -> list[LoopMessage]:
        out: list[LoopMessage] = []
        for r in rows:
            msg = self.render_row(r)
            if msg is not None:
                out.append(msg)
        return out

    def render_row(self, r: ProjectMessageRow) -> LoopMessage | None:
        """Dispatch one stored row to its per-kind renderer. Override to add a kind / change routing;
        an unhandled kind returns None (skipped)."""
        if r.kind == "user":
            return self.render_user_row(r)
        if r.kind == "project":
            return self.render_project_row(r)
        if r.kind == "compact":
            return self.render_compact_row(r)
        return None

    def render_user_row(self, r: ProjectMessageRow) -> LoopMessage:
        content = r.content or {}
        # ``input`` since 3.3; ``human`` is the pre-3.3 key — read both so old rows still render.
        inputs = content.get("input")
        if inputs is None:
            inputs = content.get("human") or []
        # The projector contract makes `input` a LIST, but a corrupt / custom-projector row may
        # carry a bare string — joining over it would iterate CHARACTER-by-character (projection-3).
        if not isinstance(inputs, list):
            inputs = [inputs]
        return {
            "role": "user",
            "content": self._send_tag(self.render_config.user_tag, r.send_index)
            + "\n".join(str(h) for h in inputs),
        }

    def render_project_row(self, r: ProjectMessageRow) -> LoopMessage:
        return {
            "role": "assistant",
            "content": self._send_tag(self.render_config.project_tag, r.send_index)
            + self._render_project(r.content),
        }

    def render_compact_row(self, r: ProjectMessageRow) -> LoopMessage:
        msg = _render_compact_row(r)
        lo, hi = r.compact_from_send, r.compact_to_send
        # Emit the recall hint whenever the compact covers at least one real send (hi >= 1),
        # regardless of lo. The old `lo > 0` also suppressed the note for a MATURE fold whose range
        # starts at send 0 (migration-seeded from_send=0), permanently hiding recall_send for those
        # folds (M-projection-1). The degenerate seed (hi == 0) is still suppressed.
        if lo is not None and hi is not None and hi >= lo and hi > 0:
            rng = f"#{lo}" if lo == hi else f"#{lo}–#{hi}"
            note = self.render_config.fold_note.replace("{range}", rng)
            return {"role": "user", "content": f"{note}\n{msg['content']}"}
        return msg

    @staticmethod
    def _send_tag(template: str, send_index: int | None) -> str:
        """A send_index tag from a ``{n}`` template; empty template or ``None`` index → no tag."""
        if send_index is None or not template:
            return ""
        return template.replace("{n}", str(send_index))

    def _render_tool(self, t: dict[str, Any]) -> str:
        name = t.get("name", "?")
        if name == "__user__":
            # A mid-send user injection (reminder / finalize / steering) at its
            # chronological position in the timeline — not a tool call.
            return f"[user] {t.get('text') or ''}"
        # Row coordinates (seq / call_seq, 5.4.0) are bookkeeping for host renderers, not part of
        # the generic "name(k=v, …)" line — a host that wants a recall pointer overrides the render.
        rest = {k: v for k, v in (t or {}).items() if k not in ("name", "seq", "call_seq")}
        if not rest:
            return str(name)
        body = self.render_config.tool_arg_sep.join(f"{k}={v}" for k, v in rest.items())
        return f"{name}({body})"

    def _render_project(self, content: dict[str, Any] | None) -> str:
        content = content or {}
        cfg = self.render_config
        parts: list[str] = []
        tools = content.get("tools") or []
        if tools and cfg.include_tools:
            parts.append(cfg.tools_header + cfg.tool_sep.join(self._render_tool(t) for t in tools))
        ft = content.get("final_text")
        if ft and cfg.include_final_text:
            parts.append(str(ft))
        return "\n".join(parts) if parts else cfg.empty_project
