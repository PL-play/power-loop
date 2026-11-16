from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from power_loop.runtime.stop_policy import StopPolicy

if TYPE_CHECKING:
    from collections.abc import Callable

    from power_loop.runtime.fold import FoldStrategy
    from power_loop.runtime.memory import MemoryProvider
    from power_loop.runtime.notes import NotesPolicy
    from power_loop.runtime.representation import Representation
    from power_loop.runtime.retry import LLMRetryPolicy
    from power_loop.runtime.runtime_state import RuntimeProjector
    from power_loop.runtime.spec import AgentSpec

LoopStatus = Literal["completed", "pending_tools", "cancelled", "hit_round_limit", "budget_exceeded", "context_checkpoint", "degraded"]
LoopMessage = dict[str, Any]


def _default_representation() -> Representation:
    from power_loop.runtime.representation import ProjectedRepresentation
    return ProjectedRepresentation()


def _default_fold_strategy() -> FoldStrategy:
    from power_loop.runtime.fold import LLMSummaryFold
    return LLMSummaryFold()


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
    #: ``representation`` — how each finished send is recorded/rendered. The library default is
    #: ``ProjectedRepresentation``: a per-send terse plain-text projection (send-context
    #: projection), with the original detail kept in ``pl_messages`` (recall_send re-expands it).
    #: Custom representations implement the ``Representation`` Protocol.
    representation: Representation = field(default_factory=_default_representation)
    #: Optional history-row cap. The SDK default is unlimited (None); 0 also means unlimited.
    #: A positive value explicitly limits history, independently of token/fold budgets.
    #: History normally runs from the fold
    #: compact to the newest message; a session whose fold lags — or whose sends are tiny and
    #: numerous — grows without bound. When set, the assembled history keeps at most this many
    #: rows: the compact summary (if any) is ALWAYS kept on top, the in-flight send is always kept
    #: in full, and older material drops in whole chunks from the oldest end (the legacy prefix
    #: first, then whole past sends) until the total fits — a chunk is never split, so a
    #: verbatim-fallback send can't orphan its tool rows. ``None``/``<=0`` disables the cap.
    #: The current send and compact summary are retained in either representation.
    max_context_rows: int | None = None
    #: 6.10.0 上下文三旋钮（互相独立，都不再借用 ``max_tokens``——那是**每次请求的输出上限**）：
    #: ``context_budget_tokens``：折叠预算——投影前缀估算 token（≈字符/4）达到它就折叠更早的 send。
    #: None → 回退到 ``max_tokens``（历史行为，兼容）。
    context_budget_tokens: int | None = None
    #: ``context_checkpoint_tokens``：轮边界检查**上一轮供应商返回的真实 prompt_tokens**（=当前上下文
    #: 真实大小）≥ 它 → 以 ``context_checkpoint`` 优雅收尾本 send（COMPLETE_DECIDE 先给宿主收尾窗口）
    #: → 正常投影 → 宿主的续接机制在新 send 里接着干。与 ``max_tokens_per_run``（累计费用上限）正交。
    context_checkpoint_tokens: int | None = None
    #: Total prompt threshold for in-send semantic distillation. When enabled,
    #: context_checkpoint_tokens is ignored: fuse and cut are exclusive policies.
    tool_context: Any = None  # ToolContextConfig: opt-in async per-tool preparation + ledger
    #: Plain in-send fuse, used only WITHOUT ``tool_context`` (with it, the tool-context fuse
    #: ``trigger_tokens``/``target_tokens``/``hot_tail`` is the single fuse).
    insend_distill_tokens: int | None = None
    #: Deprecated compatibility field; does not limit the number of distilled calls.
    insend_distill_batch: int = 10
    insend_distill_hot_tail: int = 8
    #: 6.27.0 这些工具的**调用参数永不瘦身**（结果照旧）。给「模型自己说的话」用：send_message /
    #: post_update 这类工具的参数就是模型的发言，瘦成占位符后模型会照着占位符的样子再发一遍
    #: （真实事故：九条「⟨已移出上下文 N 字符…⟩」被当正文发进了聊天）。宿主按工具名登记。
    insend_distill_keep_tools: tuple[str, ...] = ()
    #: Cap on one tool result as written to history and the audit log (head + tail kept);
    #: None = no cap. recall_send can only return what was written.
    tool_result_max_chars: int | None = 20000
    #: 6.11.0 同轮并发：同一轮里 ≥2 个 ``async_capable`` 工具调用并发执行（上限 = 本值），结果仍按
    #: 原顺序回填；TOOL_BEFORE 仍按序先跑完（闸类 hook 语义不变）。非 async_capable 工具永远串行。
    #: 0/1 = 关。
    tool_batch_concurrency: int = 4
    #: 6.12.0 图片保留轮数：send 内的图片（图片工具 / 用户发图 的 attachment 块）只在进入上下文后的
    #: 这么多轮里以原图参与请求；之后在内存里换成一行占位文字（原行不动、pl_messages 不动）。
    #: 证据（conv-225）：三张图入上下文后每次调用 50–120s（文本部分全命中缓存、输出只有一两百 token），
    #: 入图前每轮 2–7s——供应商每次都重新预填图片。None/0 = 不撤（旧行为）。
    image_retention_rounds: int | None = 1
    #: ``fold_strategy`` — how older history is compacted (N records → 1 compact) once over budget:
    #:   * ``LLMSummaryFold`` (default): one LLM summary call, no side effects.
    #:   * ``AgenticFold``: LLM + a bounded tool loop that persists durable facts as notes.
    #:   * custom: any ``FoldStrategy`` Protocol impl.
    #: Folds are always LLM-backed (no deterministic/never-fold).
    #: Trigger + keep-recent come from the strategy (``trigger_ratio`` / ``keep_last_sends``);
    #: the fold always keeps whole sends (never splits an atomic tool-call/result pair).
    fold_strategy: FoldStrategy = field(default_factory=_default_fold_strategy)
    #: Wall-clock bound (seconds) on a single fold's LLM/agentic call. The fold runs OUTSIDE the
    #: store lock, but a hung provider would still stall the end-of-send path; on timeout the fold
    #: soft-fails (rows committed, no compact this send — retried next send). None disables.
    fold_timeout_s: float | None = 120.0
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
    #: Host-only delegation policy. single_level allows main/workflow nodes to
    #: delegate, but delegated agents cannot create another workflow or agent.
    delegation_policy: str = "legacy"

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
    #: Ownership expires after this duration without renewal. Shared coordinator
    #: implementations are provided by the application; the default is in memory.
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
    #: design/124 §8.4: bounded wind-down per kind of work once it is told to stop.
    stop_policy: StopPolicy = field(default_factory=StopPolicy)

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
        cb = int(self.context_budget_tokens or 0)
        mt = cb if cb > 0 else int(self.max_tokens or 0)   # 6.10.0：独立折叠预算优先
        if mt <= 0:
            return mt
        if self.memory is not None and self.builtin_memory_hook:
            return max(1, mt - int(self.memory_budget_tokens or 0))
        return mt

    def __post_init__(self) -> None:
        if self.delegation_policy not in ("legacy", "single_level"):
            raise ValueError("delegation_policy must be legacy or single_level")
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


@dataclass
class AgentLoopResult:
    status: LoopStatus
    final_text: str = ""
    rounds: int = 0
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[LoopMessage] = field(default_factory=list)
    #: Cumulative token usage across every LLM call of this run:
    #: {prompt_tokens, completion_tokens, cache_read_tokens, reasoning_tokens,
    #:  total_tokens, calls}. Empty dict when the run never reached the LLM.
    usage: dict[str, int] = field(default_factory=dict)
    #: Tool invocations executed during this run.
    tool_calls: int = 0
    #: design/188：这一轮是不是调 pass_turn 明确结束的、理由；停的那一刻是否有刚折进来的新输入。
    passed: bool = False
    pass_reason: str | None = None
    new_input_after_pass: bool = False
    #: 每一步写的文字（不只是最后一段）：[{step, text, tools}]，按顺序。
    texts: list[dict[str, Any]] = field(default_factory=list)
