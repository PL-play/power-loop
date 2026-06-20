from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from power_loop.runtime.compact import Compactor
    from power_loop.runtime.history_projector import HistoryProjector
    from power_loop.runtime.memory import MemoryProvider
    from power_loop.runtime.notes import NotesPolicy
    from power_loop.runtime.retry import LLMRetryPolicy
    from power_loop.runtime.runtime_state import RuntimeProjector

LoopStatus = Literal["completed", "pending_tools", "waiting_for_input", "cancelled", "hit_round_limit", "budget_exceeded", "degraded"]
LoopMessage = dict[str, Any]


def _default_compactor() -> Compactor:
    from power_loop.runtime.compact import DefaultCompactor
    return DefaultCompactor()


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
    #: (so no tool_calls are left dangling), then the loop stops with
    #: status="budget_exceeded". ``None`` disables.
    max_tokens_per_run: int | None = None
    compactor: Compactor | None = field(default_factory=_default_compactor)
    #: Opt-in send-context projection (v2). When set, the loop feeds the LLM a per-send
    #: PLAIN-TEXT projection of FINISHED sends (from pl_project_messages) plus the in-flight
    #: send verbatim, instead of the full verbatim history; finished sends are projected at
    #: end-of-send into pl_project_messages (pl_messages stays the immutable audit log).
    #: MUTUALLY EXCLUSIVE with ``compactor`` (set ``compactor=None``) — see __post_init__.
    #: ``None`` (default) → today's behavior (verbatim history + in-place compactor).
    history_projector: HistoryProjector | None = None
    retry_policy: LLMRetryPolicy | None = None
    memory: MemoryProvider | None = None
    memory_budget_tokens: int = 1500
    # Bounds for the note_add/note_update/note_delete tools (agent-authored
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

    def __post_init__(self) -> None:
        self._validate_projector_compactor_exclusion()
        # Mark init complete so __setattr__ starts re-validating reassignments (the dataclass
        # is mutable; the reader at _run_loop assumes compactor is None whenever a projector is
        # set, so a post-hoc reassignment of either field must not silently break that).
        object.__setattr__(self, "_initialized", True)

    def _validate_projector_compactor_exclusion(self) -> None:
        # The projection layer REPLACES in-place compaction; running both would let the
        # compactor insert compact_note rows whose logical-ord reordering breaks the
        # reader's send_index partitioning. Force the caller to disable one.
        if self.history_projector is not None and self.compactor is not None:
            raise ValueError(
                "AgentLoopConfig: history_projector and compactor are mutually exclusive — "
                "the projection layer replaces in-place compaction. Set compactor=None when "
                "using a history_projector."
            )

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in ("history_projector", "compactor") and getattr(self, "_initialized", False):
            self._validate_projector_compactor_exclusion()


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
