"""Direct model calls use the original budget without a separate cost path."""
import pytest

from power_loop import LLMRequest
from power_loop.runtime.model_calls import ModelCalls
from power_loop.runtime.task_runtime import _SEND

from .test_fold_calls_192 import SummaryModel, _prepared, fold_case, registry_store

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
__all__ = ['fold_case', 'registry_store']


async def test_auxiliary_call_uses_normal_llm_accounting(fold_case):
    c = fold_case
    model = SummaryModel()
    c.host.service = model
    token = _SEND.set(c.binding)
    try:
        response = await ModelCalls(session_id=c.sid).wrap(model).complete(LLMRequest(
            messages=[{'role': 'user', 'content': 'compare these questions'}], max_tokens=20))
    finally:
        _SEND.reset(token)
    assert response.raw_text == '<summary>folded</summary>' and model.calls == 1
    prepared, = await _prepared(c)
    assert 'purpose' not in prepared
    assert c.host.purposes == [None] and c.host.reserve_calls == c.host.apply_calls == 1
