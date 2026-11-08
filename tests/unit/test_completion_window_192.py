"""Bounded remaining rounds and original deadlines; no retired wait-handoff loop.

Automatic required-wait collection and its staged final-answer proof were retired.
Actual late result/resume behavior is exercised in test_late_node_results_208;
original paid allowance is covered in test_workflow_continuation_budget_192.
"""
import asyncio
from types import SimpleNamespace

import pytest

from power_loop.runtime.completion import CompletionWindow

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_reentry_keeps_spent_rounds_and_original_deadline():
    window = CompletionWindow(5, 2, round_ceiling=3)
    window.bind('leaf')
    deadline = window.deadline
    assert window.consume(SimpleNamespace(rounds=1), window.allowance({'rounds':1}))
    window.bind('leaf')
    assert window.remaining == 1 and window.deadline == deadline
    assert window.consume(SimpleNamespace(rounds=1), window.allowance({'rounds':2}))
    assert window.allowance({'rounds':3}) == 0


async def test_reentry_cannot_refresh_time_before_another_paid_round():
    window = CompletionWindow(.01)
    window.bind('leaf')
    await asyncio.sleep(.02)
    window.bind('leaf')
    with pytest.raises(TimeoutError):
        window.require_time()


async def test_zero_reported_rounds_are_charged_against_the_original_ceiling():
    window = CompletionWindow(5, 30, round_ceiling=3)
    window.bind('leaf')
    for _ in range(2):
        assert window.consume(SimpleNamespace(rounds=0), window.allowance({'rounds':1}))
    assert window.remaining == 28 and window.allowance({'rounds':1}) == 0


async def test_excessive_or_negative_round_evidence_is_rejected():
    window = CompletionWindow(5, 2)
    with pytest.raises(ValueError, match='invalid_completion_round_evidence'):
        window.consume(SimpleNamespace(rounds=-1),2)
    assert not window.consume(SimpleNamespace(rounds=3),2)


@pytest.mark.parametrize('options', [{'timeout_s':0},{'timeout_s':float('inf')},{'max_rounds':-1},
    {'max_rounds':True},{'round_ceiling':True},{'multi_session':'yes'}])
async def test_invalid_completion_budget_is_rejected(options):
    with pytest.raises(ValueError,match='invalid_completion'):
        CompletionWindow(**options)


async def test_scope_rebinding_requires_the_owning_engine_policy():
    window = CompletionWindow()
    window.bind('leaf')
    with pytest.raises(ValueError,match='scope_changed'):
        window.bind('other')
    window = CompletionWindow(multi_session=True)
    window.bind('leaf')
    deadline = window.deadline
    window.bind('retry')
    assert window.deadline == deadline
