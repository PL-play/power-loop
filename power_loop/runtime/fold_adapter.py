"""Adapt a custom :class:`~power_loop.runtime.fold.FoldStrategy` onto the in-place ``Compactor``
interface, so a custom fold works under VERBATIM representation too (power-loop 3.0).

The shipped strategies (``LLMSummaryFold`` / ``AgenticFold``) map directly to the tested in-place
compactors (``DefaultCompactor`` / ``AgenticMemoryCompactor``) — see ``AgentLoopConfig.resolve_compactor``.
This adapter covers the remaining case: an arbitrary custom ``FoldStrategy`` selected with the
verbatim representation. It reuses ``DefaultCompactor``'s trigger + span selection (which keeps
atomic ``assistant(tool_calls)``↔``tool`` pairs and ``keep_last_n`` intact — so the kept tail is
always provider-valid) and only swaps the SUMMARIZE step to call the custom strategy's
``fold(rows, context)`` over the cuttable span (wrapped as one verbatim record).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from power_loop.runtime.compact import DefaultCompactor

if TYPE_CHECKING:
    from power_loop.runtime.fold import FoldStrategy


class FoldStrategyCompactor(DefaultCompactor):
    def __init__(self, fold_strategy: FoldStrategy, *, max_tokens: int | None = None) -> None:
        super().__init__(
            keep_last_n=fold_strategy.keep_last_sends,
            trigger_ratio=fold_strategy.trigger_ratio,
            summary_max_tokens=int(getattr(fold_strategy, "summary_max_tokens", 5000) or 5000),
        )
        self._fold_strategy = fold_strategy
        self._max_tokens = max_tokens

    async def _summarize_async(self, slice_msgs: list[dict[str, Any]], *, llm: Any) -> str | None:
        if not slice_msgs:
            return None
        # Lazy imports avoid a runtime import cycle (representation/fold ← store ← …).
        from power_loop.runtime.fold import FoldContext
        from power_loop.runtime.representation import VerbatimRepresentation
        from power_loop.runtime.store.types import ProjectMessageRow

        rep = VerbatimRepresentation()
        # Wrap the cuttable verbatim span as ONE verbatim 'project' record so the strategy can
        # render+summarize it substrate-agnostically (rep.render turns it back into messages).
        rows = [
            ProjectMessageRow(
                session_id="", send_index=1, kind="project",
                content={"messages": slice_msgs}, rendered_text=None,
                source_seq_lo=None, source_seq_hi=None, compact_from_send=None,
                compact_to_send=None, projector_version=0, token_estimate=None, created_at=0,
            )
        ]
        ctx = FoldContext(
            session_id="", round_index=0, representation=rep, llm=llm,
            max_tokens=self._max_tokens,
        )
        try:
            result = await self._fold_strategy.fold(rows, context=ctx)
        except Exception:
            # Mirror DefaultCompactor's fail-soft: fall back to the single-call summary.
            return await super()._summarize_async(slice_msgs, llm=llm)
        if result is None:
            return None
        # Note-side-effects (result.note_ops) are written by AgenticFold via its own capture path
        # only in the projection (derived) flow; in the verbatim in-place flow a custom strategy's
        # note_ops are dropped here (the in-place path has no transactional note channel) — custom
        # strategies that need notes under verbatim should write them in fold() directly.
        return (result.content or {}).get("summary")
