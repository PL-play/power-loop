"""power-loop 3.0 — the orthogonal context config: representation × fold_strategy.

Covers AgentLoopConfig's resolution of the two axes onto the loop's mechanisms (projection
derived-fold vs verbatim in-place compactor), the deprecated 2.x kwarg mapping (history_projector
/ compactor / migrate_history_on_projection_switch), and the custom-fold-in-verbatim adapter.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pytest

from power_loop.agent.types import AgentLoopConfig
from power_loop.runtime.compact import AgenticMemoryCompactor, DefaultCompactor
from power_loop.runtime.fold import AgenticFold, FoldContext, FoldResult, LLMSummaryFold
from power_loop.runtime.fold_adapter import FoldStrategyCompactor
from power_loop.runtime.history_projector import DefaultDeterministicProjector
from power_loop.runtime.representation import ProjectedRepresentation, VerbatimRepresentation

# ── defaults + the four combos resolve correctly ──────────────────────────


def test_default_is_verbatim_llm_summary():
    c = AgentLoopConfig()
    assert c.representation.kind == "verbatim"
    assert isinstance(c.fold_strategy, LLMSummaryFold)
    assert isinstance(c.resolve_compactor(), DefaultCompactor)  # verbatim → in-place compactor
    assert c.projection_representation is None


def test_verbatim_agentic_resolves_to_agentic_compactor():
    c = AgentLoopConfig(representation=VerbatimRepresentation(), fold_strategy=AgenticFold())
    comp = c.resolve_compactor()
    assert isinstance(comp, AgenticMemoryCompactor)


def test_projection_resolves_to_no_inplace_compactor():
    for fs in (LLMSummaryFold(), AgenticFold()):
        c = AgentLoopConfig(representation=ProjectedRepresentation(), fold_strategy=fs)
        assert c.resolve_compactor() is None  # projection folds at end-of-send, not in place
        assert c.projection_representation is not None
        assert c.projection_representation.kind == "projection"


def test_fold_params_propagate_to_verbatim_compactor():
    c = AgentLoopConfig(fold_strategy=LLMSummaryFold(keep_last_sends=7, trigger_ratio=0.5,
                                                     summary_max_tokens=1234))
    comp = c.resolve_compactor()
    assert comp.keep_last_n == 7 and comp.trigger_ratio == 0.5 and comp.summary_max_tokens == 1234


# ── custom fold strategy ───────────────────────────────────────────────────


@dataclass
class _CustomFold:
    keep_last_sends: int = 3
    trigger_ratio: float = 0.6
    summary_max_tokens: int = 999
    fold_id: str = "custom"

    async def fold(self, rows, *, context: FoldContext) -> FoldResult | None:
        return FoldResult(content={"summary": "CUSTOM SUMMARY"}, folded_to_send=1)


def test_custom_fold_under_verbatim_uses_adapter():
    c = AgentLoopConfig(fold_strategy=_CustomFold())
    comp = c.resolve_compactor()
    assert isinstance(comp, FoldStrategyCompactor)
    assert comp.keep_last_n == 3 and comp.trigger_ratio == 0.6 and comp.summary_max_tokens == 999


@pytest.mark.asyncio
async def test_fold_adapter_delegates_summarize_to_strategy():
    adapter = FoldStrategyCompactor(_CustomFold(), max_tokens=8000)
    out = await adapter._summarize_async(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}], llm=None
    )
    assert out == "CUSTOM SUMMARY"


@pytest.mark.asyncio
async def test_fold_adapter_falls_back_on_strategy_error():
    @dataclass
    class _Boom:
        keep_last_sends: int = 2
        trigger_ratio: float = 0.75
        fold_id: str = "boom"

        async def fold(self, rows, *, context):
            raise RuntimeError("nope")

    class _StubLLM:
        async def complete(self, request):
            from power_loop._vendor.llm_client.interface import LLMResponse
            return LLMResponse(raw_text="<summary>fallback</summary>")

    adapter = FoldStrategyCompactor(_Boom(), max_tokens=8000)
    out = await adapter._summarize_async([{"role": "user", "content": "x"}], llm=_StubLLM())
    assert out == "fallback"  # strategy raised → single-call summary fallback


# ── deprecated 2.x kwarg mapping ───────────────────────────────────────────


@contextmanager
def _no_warn() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def test_legacy_compactor_none_preserved_exactly():
    with _no_warn():
        c = AgentLoopConfig(compactor=None)
    # compactor=None historically = "no compaction"; preserved (resolve returns None).
    assert c.resolve_compactor() is None
    assert c.representation.kind == "verbatim"


def test_legacy_compactor_instance_preserved_exactly():
    comp = DefaultCompactor(keep_last_n=9)
    with _no_warn():
        c = AgentLoopConfig(compactor=comp)
    assert c.resolve_compactor() is comp  # the EXACT instance, untouched


def test_legacy_projector_becomes_projection_representation():
    proj = DefaultDeterministicProjector(keep_last_sends=2)
    with _no_warn():
        c = AgentLoopConfig(history_projector=proj, compactor=None)
    assert c.projection_representation is proj
    assert c.resolve_compactor() is None  # projection → no in-place fold


def test_legacy_migrate_flag_alias():
    with _no_warn():
        c = AgentLoopConfig(history_projector=DefaultDeterministicProjector(),
                            compactor=None, migrate_history_on_projection_switch=False)
    assert c.migrate_history_on_switch is False


def test_new_axes_win_over_legacy_when_both_present():
    with _no_warn():
        c = AgentLoopConfig(
            representation=VerbatimRepresentation(),
            fold_strategy=AgenticFold(),
            history_projector=DefaultDeterministicProjector(),  # ignored — new wins
        )
    assert c.representation.kind == "verbatim"
    assert isinstance(c.fold_strategy, AgenticFold)


def test_legacy_kwargs_emit_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AgentLoopConfig(compactor=None)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


# ── validation ─────────────────────────────────────────────────────────────


def test_max_tokens_must_be_positive():
    with pytest.raises(ValueError):
        AgentLoopConfig(max_tokens=0)
    with pytest.raises(ValueError):
        AgentLoopConfig(max_tokens=-1)


def test_max_tokens_reassignment_rolls_back():
    c = AgentLoopConfig(max_tokens=8000)
    with pytest.raises(ValueError):
        c.max_tokens = 0
    assert c.max_tokens == 8000  # rolled back


# ── review fixes: legacy mapping must NOT silently drop fold knobs / disable an explicit fold ──


def test_legacy_projector_seeds_fold_keep_trigger():
    # A deprecated history_projector's keep_last_sends / trigger_ratio must carry into the mapped
    # fold (else DeepTalk's admin-configured projection knobs become no-ops). Review #2/#11.
    with _no_warn():
        c = AgentLoopConfig(
            history_projector=DefaultDeterministicProjector(keep_last_sends=2, trigger_ratio=0.4),
            compactor=None,
        )
    assert c.fold_strategy.keep_last_sends == 2
    assert c.fold_strategy.trigger_ratio == 0.4


def test_legacy_compactor_none_does_not_disable_explicit_fold():
    # If the caller sets a NEW fold_strategy explicitly, a stray legacy compactor=None must NOT
    # silently disable it (new axis wins). Review #15.
    with _no_warn():
        c = AgentLoopConfig(
            representation=VerbatimRepresentation(), fold_strategy=AgenticFold(), compactor=None,
        )
    assert isinstance(c.resolve_compactor(), AgenticMemoryCompactor)


def test_post_init_switch_to_projection_clears_legacy_compactor():
    # A legacy verbatim compactor + a later switch to a projection representation must NOT leave a
    # stale in-place compactor active (double-fold). resolve_compactor checks projection first. #16.
    with _no_warn():
        c = AgentLoopConfig(compactor=DefaultCompactor())
    assert isinstance(c.resolve_compactor(), DefaultCompactor)
    c.representation = ProjectedRepresentation()
    assert c.resolve_compactor() is None  # projection → no in-place compactor (no double-fold)
