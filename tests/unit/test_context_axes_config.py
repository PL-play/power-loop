"""The context config contract after design 216: projection-only history.

The two axes remain ``representation`` × ``fold_strategy``. The library default is
``ProjectedRepresentation`` (finished sends reach the model through their projection) and
``LLMSummaryFold`` (folds happen at end-of-send). The removed verbatim representation and its
in-place compactor are not aliased, mapped, or implicitly converted: passing the old 2.x kwargs
fails loudly.
"""

from __future__ import annotations

import dataclasses

import pytest

from power_loop.agent.types import AgentLoopConfig
from power_loop.runtime.fold import AgenticFold, LLMSummaryFold
from power_loop.runtime.representation import ProjectedRepresentation


def test_default_is_projection_llm_summary():
    c = AgentLoopConfig()
    assert isinstance(c.representation, ProjectedRepresentation)
    assert c.representation.kind == "projection"
    assert isinstance(c.fold_strategy, LLMSummaryFold)


def test_custom_representation_and_fold_axes_are_kept():
    rep = ProjectedRepresentation(version=2)
    fold = AgenticFold(keep_last_sends=3)
    c = AgentLoopConfig(representation=rep, fold_strategy=fold)
    assert c.representation is rep
    assert c.fold_strategy is fold


def test_replace_keeps_the_axes():
    c = AgentLoopConfig(representation=ProjectedRepresentation(version=2),
                        fold_strategy=LLMSummaryFold(keep_last_sends=6))
    replaced = dataclasses.replace(c, system_prompt="per-send", max_rounds=3)
    assert replaced.representation is c.representation
    assert replaced.fold_strategy is c.fold_strategy


def test_removed_legacy_kwargs_are_rejected():
    with pytest.raises(TypeError):
        AgentLoopConfig(compactor=None)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        AgentLoopConfig(history_projector=ProjectedRepresentation())  # type: ignore[call-arg]


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