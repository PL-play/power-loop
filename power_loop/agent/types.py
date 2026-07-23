from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

#: Distinguishes "caller passed nothing" from an explicit None on the deprecated 2.x kwargs.
_UNSET: Any = object()

if TYPE_CHECKING:
    from collections.abc import Callable

    from power_loop.runtime.fold import FoldStrategy
    from power_loop.runtime.memory import MemoryProvider
    from power_loop.runtime.notes import NotesPolicy
    from power_loop.runtime.representation import Representation
    from power_loop.runtime.retry import LLMRetryPolicy
    from power_loop.runtime.runtime_state import RuntimeProjector
    from power_loop.runtime.spec import AgentSpec

LoopStatus = Literal["completed", "pending_tools", "waiting_for_input", "cancelled", "hit_round_limit", "budget_exceeded", "degraded"]
LoopMessage = dict[str, Any]


def _default_representation() -> Representation:
    from power_loop.runtime.representation import VerbatimRepresentation
    return VerbatimRepresentation()


def _default_fold_strategy() -> FoldStrategy:
    from power_loop.runtime.fold import LLMSummaryFold
    return LLMSummaryFold()


def _fold_from_legacy_projector(proj: Any) -> FoldStrategy:
    """Seed the default LLM fold from a DEPRECATED ``history_projector``'s knobs so a legacy
    projector's ``keep_last_sends`` / ``trigger_ratio`` keep taking effect (e.g. DeepTalk's
    admin-configured projection settings). Without this the mapped fold would silently use
    ``LLMSummaryFold`` defaults (4 / 0.75) and ignore the operator's config."""
    from power_loop.runtime.fold import LLMSummaryFold
    # Only a MISSING/None keep falls back to 4; an explicit 0 means "keep ~none" (fold aggressively)
    # → clamp to the validator's floor of 1, NOT silently to 4 (B10). (A verbatim keep==0 projector is
    # routed to never-fold in _map_legacy_axes and never reaches here.)
    keep_raw = getattr(proj, "keep_last_sends", None)
    keep = 4 if keep_raw is None else max(1, int(keep_raw))
    trigger = float(getattr(proj, "trigger_ratio", 0.75) or 0.75)
    return LLMSummaryFold(keep_last_sends=keep, trigger_ratio=trigger)


def _default_runtime_projectors() -> tuple[RuntimeProjector, ...]:
    from power_loop.runtime.runtime_state import default_runtime_projectors
    return default_runtime_projectors()


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop."""

    system_prompt: str | None = None
    max_rounds: int = 24
    temperature: float | None = 0.0
    max_tokens: int | None = 8000
    #: Per-loop model override. When set, every LLM request of this loop carries
    #: it (the provider uses ``request.model`` over its own configured model), so
    #: a sub-agent / workflow step can run on a different model than the global
    #: one. ``None`` → use the LLM service's configured model.
    model: str | None = None
    #: OpenAI-compatible ``response_format`` (e.g. a json_schema dict from
    #: ``StructuredOutputSpec.to_openai_response_format()``), forwarded on the
    #: main generation call so the provider returns structured output. ``None``
    #: → free-form text.
    response_format: dict[str, Any] | None = None
    #: Hard per-run token budget (prompt + completion summed over the whole
    #: run, real provider usage — see ``ContextManager.usage_totals``). Checked
    #: at round boundaries: the round that crosses the budget still finishes
    #: (so no tool_calls are left dangling), then COMPLETE_DECIDE is consulted
    #: with reason="budget_exceeded" before the loop stops. A successful hook
    #: injection gets its bounded extra rounds even though the normal budget is
    #: exhausted. ``None`` or a non-positive value disables the budget.
    max_tokens_per_run: int | None = None
    #: Context handling = REPRESENTATION × FOLD_STRATEGY (power-loop 3.0), two orthogonal axes.
    #:
    #: ``representation`` — how each finished send is recorded/rendered:
    #:   * ``VerbatimRepresentation`` (default): full messages, byte-identical history.
    #:   * ``ProjectedRepresentation``: a per-send terse plain-text projection (send-context
    #:     projection), original detail kept in ``pl_messages`` (recall_send re-expands).
    #: Custom representations implement the ``Representation`` Protocol.
    representation: Representation = _UNSET  # resolved in __post_init__ (default or legacy-mapped)
    #: Recent-rows context cap (3.23, PROJECTION mode only). History normally runs from the fold
    #: compact to the newest message; a session whose fold lags — or whose sends are tiny and
    #: numerous — grows without bound. When set, the assembled history keeps at most this many
    #: rows: the compact summary (if any) is ALWAYS kept on top, the in-flight send is always kept
    #: in full, and older material drops in whole chunks from the oldest end (the legacy prefix
    #: first, then whole past sends) until the total fits — a chunk is never split, so a
    #: verbatim-fallback send can't orphan its tool rows. ``None``/``<=0`` disables the cap.
    #: Verbatim mode is unaffected (its window is bounded by the in-place compactor).
    max_context_rows: int | None = 300
    #: ``fold_strategy`` — how older history is compacted (N records → 1 compact) once over budget:
    #:   * ``LLMSummaryFold`` (default): one LLM summary call, no side effects.
    #:   * ``AgenticFold``: LLM + a bounded tool loop that persists durable facts as notes.
    #:   * custom: any ``FoldStrategy`` Protocol impl.
    #: Works under EITHER representation. Folds are always LLM-backed (no deterministic/never-fold).
    #: Trigger + keep-recent come from the strategy (``trigger_ratio`` / ``keep_last_sends``);
    #: the fold always keeps whole sends (never splits an atomic tool-call/result pair).
    fold_strategy: FoldStrategy = _UNSET  # resolved in __post_init__ (default or legacy-mapped)
    #: Wall-clock bound (seconds) on a single fold's LLM/agentic call. The fold runs OUTSIDE the
    #: store lock, but a hung provider would still stall the end-of-send path; on timeout the fold
    #: soft-fails (rows committed, no compact this send — retried next send). None disables.
    fold_timeout_s: float | None = 120.0
    #: On a representation/fold change for an existing session, fold the prior history into the new
    #: form ONCE (best-effort, never throws). Default True.
    migrate_history_on_switch: bool = True
    # ── deprecated 2.x kwargs (accepted + mapped onto the two axes in __post_init__) ──
    # The public API is representation × fold_strategy; these keep existing call sites + DeepTalk
    # working until migrated. A future major drops them.
    compactor: Any = _UNSET
    history_projector: Any = _UNSET
    migrate_history_on_projection_switch: Any = _UNSET
    #: History-repair backstop (the always-on prompt sanitizer in `align_tool_calls` realigns
    #: tool-call/result pairing before every LLM call regardless). When True, the orphan
    #: tool-result rows that sanitizer drops are ALSO physically deactivated in the store
    #: (state=DROPPED), so the corruption is repaired durably and not re-sanitized every load.
    #: Default False to keep `pl_messages` immutable; the prompt is kept valid either way.
    repair_corrupt_history: bool = False
    retry_policy: LLMRetryPolicy | None = None
    memory: MemoryProvider | None = None
    memory_budget_tokens: int = 1500
    #: Where the built-in MemoryRecallHook injects recalled memory into the
    #: per-call request: "tail" (default — after history, keeps the prior-history
    #: prefix byte-stable and prefix-cacheable) or "front" (after leading system
    #: messages — legacy position; breaks prefix caching when memory changes).
    memory_position: str = "tail"
    #: Auto-register the built-in MemoryRecallHook when ``memory`` is set. Turn
    #: off to inject memory yourself via an LLM_BEFORE hook.
    builtin_memory_hook: bool = True
    #: HOST seam for child-run configs (PROVISIONAL, 3.14). When set on a PARENT
    #: loop's config, every inline child spawned under it (``run_agent_spec``:
    #: spawn_agent / run_agent delegations and in-process workflow leaves) builds
    #: its default minimal AgentLoopConfig as usual, then passes it through
    #: ``factory(spec, default_child_config)`` — the returned config is used AS
    #: IS. Lets a host give heavy leaves its context strategy (representation /
    #: fold / microcompact) without forking ``run_agent_spec``. The seam is
    #: host-side by design: the LLM-authored ``AgentSpec`` cannot reach it; use
    #: ``spec`` (name/metadata) only to ROUTE. Prefer ``dataclasses.replace`` on
    #: the given default — it already carries the spec-derived fields
    #: (system_prompt / model / budgets / response_format) and the parent's
    #: retry_policy; overriding those is on you. Signature:
    #: ``(AgentSpec, AgentLoopConfig) -> AgentLoopConfig``.
    subagent_config_factory: Callable[[AgentSpec, AgentLoopConfig], AgentLoopConfig] | None = None

    # ── Microcompact (large tool-output spill-to-disk) ──
    #
    # A cheap, no-LLM per-round mechanism that replaces OLD oversized tool
    # outputs (older than the hot tail) with a short on-disk pointer, to save
    # context tokens — orthogonal to the LLM-summary fold/compactor. Verbatim
    # mode only (projection renders finished sends from the projection store).
    #
    # DEFAULT OFF as of 3.1.x: it only helps when those old outputs are never
    # needed again; otherwise the pointer just trades for a re-read. Projection
    # mode + fold + provider prefix-caching already cover context budget. Turn it
    # on for long verbatim sessions that read many large files and rarely revisit
    # the old ones. The thresholds default from the legacy CONTEXT_MICRO_* env
    # vars for back-compat; the config fields take precedence.
    microcompact_enabled: bool = False
    microcompact_size_limit: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_MICRO_SIZE_LIMIT", "1000"))
    )
    microcompact_hot_tail: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_MICRO_HOT_TAIL", "10"))
    )
    #: Where spilled outputs are written. None → the runtime home's ``.cache``.
    microcompact_spill_dir: str | None = None
    # ── Cross-process session leases (schema v7) ─────────────────────────────────────────
    # The in-process session lock only serializes ONE interpreter. Turn this on when several
    # processes share a store and could be handed the same session: each send takes a DB lease
    # first, renews it while it works, and a process that loses the race parks its steering in
    # the shared follow-up queue for the holder to drain (folding, across processes).
    #
    # DEFAULT OFF: it costs a lease write per send plus a renewal per round, and adds a failure
    # mode (a stalled holder's lease expiring) that a single-process host gains nothing from.
    # Only server-backed stores can honor it — SQLite is single-host by construction.
    distributed_sessions: bool = False
    #: How long a lease stays valid without renewal. Must comfortably exceed one round's wall
    #: time: too short and a slow round lets another process take the session away mid-run; too
    #: long and a genuinely dead holder blocks the session for that duration.
    session_lease_ttl_s: float = 90.0
    # Audit the EPHEMERAL context that LLM_BEFORE hooks inject per round (e.g. recalled memory),
    # which otherwise vanishes after the call. Recorded into the {prefix}hook_events store table,
    # linked to the round's assistant message; observability ONLY — never read back into history or
    # the LLM request, so it can't change context or prefix-caching.
    #   "off"      — do not capture (default; zero overhead).
    #   "metadata" — record name/source/char-count/position per injected item, NOT the text.
    #   "full"     — also record the injected content text. NOTE: stored VERBATIM with no per-item
    #                cap, so the audit table grows with large RAG/memory blocks — use "metadata" if
    #                volume is a concern.
    # ONE row is written per ROUND (the LLM_BEFORE hook runs each round; the builtin memory block is
    # memoized once per send but re-injected every round), so a multi-round send yields one audit row
    # per round. Assumes LLM_BEFORE handlers MUTATE ctx.messages in place (the builtin contract) and
    # captures only APPENDED injection, not in-place edits of existing messages. A handler that
    # REPLACES all or most of ctx.messages with fresh copies makes the per-injection diff
    # unresolvable — the row is then a small "inject_unresolved" marker (still never affects
    # context/cache).
    record_hook_events: str = "off"
    # Bounds for the unified note tool (agent-authored
    # notes). None → DEFAULT_NOTES_POLICY. See power_loop.runtime.notes.
    notes_policy: NotesPolicy | None = None
    skills_dir: str | None = None
    runtime_projectors: tuple[RuntimeProjector, ...] = field(default_factory=_default_runtime_projectors)

    # ── Tool catalog auto-injection (M1.10) ──
    #
    # When ``inject_tool_descriptions`` is True (default), the pipeline
    # automatically appends a human-readable tool catalog to the resolved
    # system prompt.  The catalog is generated from the live
    # ``ToolRegistry`` so the agent always knows which tools are
    # available — even when the user-supplied system prompt does not
    # mention them.
    #
    # The catalog lives inside ``self.system_prompt`` (a plain string
    # attribute on the pipeline), NOT in ``self.history``, so the
    # compactor never touches it.
    inject_tool_descriptions: bool = True
    tool_catalog_header: str = "# Available Tools"

    def effective_context_budget(self) -> int:
        """Fold/compaction budget after reserving headroom for the ephemeral
        memory block.

        Memory is injected at the per-call tail by the built-in hook and is NOT
        counted by the fold trigger (it isn't in ``self.history``). To keep
        ``history + memory`` within the model window, the fold threshold targets
        ``max_tokens − memory_budget_tokens`` so folding fires early enough.
        ``0``/``None`` max_tokens means "no explicit budget" → returned
        unchanged.
        """
        mt = int(self.max_tokens or 0)
        if mt <= 0:
            return mt
        if self.memory is not None and self.builtin_memory_hook:
            return max(1, mt - int(self.memory_budget_tokens or 0))
        return mt

    def __post_init__(self) -> None:
        self._map_legacy_axes()
        self._validate_context_config()
        # record_hook_events is a closed enum; normalize case and reject typos loudly (consistent
        # with the file's loud-config convention) rather than silently capturing nothing.
        rhe = str(self.record_hook_events or "off").strip().lower()
        if rhe not in ("off", "metadata", "full"):
            raise ValueError(
                "AgentLoopConfig: record_hook_events must be 'off' | 'metadata' | 'full'; "
                f"got {self.record_hook_events!r}"
            )
        object.__setattr__(self, "record_hook_events", rhe)
        # Mark init complete so __setattr__ starts re-validating reassignments (the dataclass
        # is mutable; a post-hoc reassignment of an axis or max_tokens must stay valid).
        object.__setattr__(self, "_initialized", True)

    def _map_legacy_axes(self) -> None:
        """Resolve representation/fold_strategy, mapping the deprecated 2.x ``history_projector`` /
        ``compactor`` / ``migrate_history_on_projection_switch`` kwargs onto them (NEW fields win
        when explicitly set). A legacy ``compactor`` (incl. ``None`` = no compaction) under verbatim
        is preserved EXACTLY via ``_legacy_verbatim_compactor`` so old behavior is unchanged; a
        legacy projector becomes the projection representation (its fold is now the fold_strategy)."""
        legacy_proj = self.history_projector
        legacy_comp = self.compactor
        fold_was_unset = self.fold_strategy is _UNSET
        if legacy_proj is not _UNSET or legacy_comp is not _UNSET or (
            self.migrate_history_on_projection_switch is not _UNSET
        ):
            warnings.warn(
                "AgentLoopConfig: history_projector / compactor / "
                "migrate_history_on_projection_switch are deprecated; use representation / "
                "fold_strategy / migrate_history_on_switch.",
                DeprecationWarning,
                stacklevel=3,
            )
        if self.representation is _UNSET:
            rep = legacy_proj if legacy_proj not in (_UNSET, None) else _default_representation()
            object.__setattr__(self, "representation", rep)
        if fold_was_unset:
            # Seed the fold from a legacy projector's knobs so its keep_last_sends / trigger_ratio
            # keep taking effect (DeepTalk admin config); else the library default.
            fs = (
                _fold_from_legacy_projector(legacy_proj)
                if legacy_proj not in (_UNSET, None)
                else _default_fold_strategy()
            )
            object.__setattr__(self, "fold_strategy", fs)
        # A legacy verbatim compactor (incl. an explicit None = no compaction) is preserved exactly
        # via resolve_compactor — but ONLY on the pure-legacy path (no projector AND no explicit
        # new fold_strategy). If the caller set fold_strategy explicitly, the new axis wins and a
        # stray legacy compactor= must NOT silently disable it.
        if legacy_comp is not _UNSET and legacy_proj in (_UNSET, None) and fold_was_unset:
            object.__setattr__(self, "_legacy_verbatim_compactor", legacy_comp)
        elif (
            legacy_proj not in (_UNSET, None)
            and fold_was_unset
            and getattr(legacy_proj, "kind", None) == "verbatim"
            and getattr(legacy_proj, "keep_last_sends", 1) == 0  # exact 0 (NOT `or 1`, which 0 defeats)
        ):
            # A legacy NEVER-FOLD projector (IdentityProjector: kind='verbatim', keep_last_sends==0)
            # maps to never-fold (compactor=None) — NOT a folding fold_strategy. Else it would fold
            # (the seeder coerces keep 0→positive) and, on the old projection path, drop the compact
            # (B7 data loss). Routes via resolve_compactor's verbatim branch (kind=='verbatim').
            object.__setattr__(self, "_legacy_verbatim_compactor", None)
        else:
            object.__setattr__(self, "_legacy_verbatim_compactor", _UNSET)
        if self.migrate_history_on_projection_switch is not _UNSET:
            object.__setattr__(
                self, "migrate_history_on_switch",
                bool(self.migrate_history_on_projection_switch),
            )
        # Clear the deprecated fields so they never leak into fingerprints / re-validation.
        object.__setattr__(self, "compactor", _UNSET)
        object.__setattr__(self, "history_projector", _UNSET)
        object.__setattr__(self, "migrate_history_on_projection_switch", _UNSET)

    def _validate_context_config(self) -> None:
        if self.representation is None or self.fold_strategy is None:
            raise ValueError("AgentLoopConfig: representation and fold_strategy are required")
        # max_tokens drives the fold trigger (max_tokens × trigger_ratio); a non-positive value
        # would make the fold misbehave (always/never fold), so reject it up front.
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(
                f"AgentLoopConfig: max_tokens must be > 0 (drives the fold trigger); "
                f"got {self.max_tokens!r}"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("representation", "fold_strategy", "max_tokens") and getattr(
            self, "_initialized", False
        ):
            old = getattr(self, name)
            super().__setattr__(name, value)
            try:
                self._validate_context_config()
            except Exception:
                # A rejected reassignment must not leave the config in the invalid state it was
                # about to enter (mutate-then-validate would otherwise corrupt it) — roll back.
                super().__setattr__(name, old)
                raise
        else:
            super().__setattr__(name, value)

    # ── internal resolution (3.0): map the two axes onto the loop's two mechanisms ──
    # Projection-style representations drive the derived-layer path (fold via fold_strategy at
    # end-of-send); verbatim drives the in-place compactor path (fold_strategy mapped to a
    # Compactor whose span selection already keeps atomic tool pairs / keep_last_n intact).
    @property
    def projection_representation(self) -> Any | None:
        """The representation when it's projection-style (renders per-send projections), else None
        (verbatim → in-place fold path)."""
        rep = self.representation
        return rep if getattr(rep, "kind", "projection") != "verbatim" else None

    def resolve_compactor(self) -> Any | None:
        """Verbatim mode → an in-place ``Compactor`` mapped from ``fold_strategy``; projection mode
        → ``None`` (projection folds at end-of-send via ``fold_strategy``). Constructed fresh per
        call (cheap)."""
        # Projection folds at end-of-send via fold_strategy — never an in-place compactor. Checked
        # FIRST so a post-init switch to a projection representation can't leave a stale legacy
        # verbatim compactor active (which would double-fold alongside the derived-layer fold).
        if getattr(self.representation, "kind", "projection") != "verbatim":
            return None
        # A deprecated verbatim ``compactor=`` (incl. None) is honored verbatim, so legacy
        # call sites keep their EXACT old compaction behavior.
        legacy = getattr(self, "_legacy_verbatim_compactor", _UNSET)
        if legacy is not _UNSET:
            return legacy
        from power_loop.runtime.fold import AgenticFold, LLMSummaryFold

        fs = self.fold_strategy
        if isinstance(fs, AgenticFold):
            from power_loop.runtime.compact import AgenticMemoryCompactor

            return AgenticMemoryCompactor(
                keep_last_n=fs.keep_last_sends,
                trigger_ratio=fs.trigger_ratio,
                summary_max_tokens=fs.summary_max_tokens,
                max_rounds=fs.max_rounds,
            )
        if isinstance(fs, LLMSummaryFold):
            from power_loop.runtime.compact import DefaultCompactor

            return DefaultCompactor(
                keep_last_n=fs.keep_last_sends,
                trigger_ratio=fs.trigger_ratio,
                summary_max_tokens=fs.summary_max_tokens,
            )
        # A custom FoldStrategy under verbatim → adapt it onto the in-place Compactor interface
        # (DefaultCompactor's span selection keeps atomic tool pairs / keep_last_n intact).
        from power_loop.runtime.fold_adapter import FoldStrategyCompactor

        return FoldStrategyCompactor(fs, max_tokens=self.max_tokens)


@dataclass
class AgentLoopResult:
    status: LoopStatus
    final_text: str = ""
    rounds: int = 0
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pending_interactions: list[dict[str, Any]] = field(default_factory=list)
    messages: list[LoopMessage] = field(default_factory=list)
    #: Cumulative token usage across every LLM call of this run:
    #: {prompt_tokens, completion_tokens, cache_read_tokens, reasoning_tokens,
    #:  total_tokens, calls}. Empty dict when the run never reached the LLM.
    usage: dict[str, int] = field(default_factory=dict)
    #: Tool invocations executed during this run.
    tool_calls: int = 0
