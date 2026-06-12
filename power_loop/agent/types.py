from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from power_loop.runtime.compact import Compactor
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
    #: Hard per-run token budget (prompt + completion summed over the whole
    #: run, real provider usage — see ``ContextManager.usage_totals``). Checked
    #: at round boundaries: the round that crosses the budget still finishes
    #: (so no tool_calls are left dangling), then the loop stops with
    #: status="budget_exceeded". ``None`` disables.
    max_tokens_per_run: int | None = None
    compactor: Compactor | None = field(default_factory=_default_compactor)
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
