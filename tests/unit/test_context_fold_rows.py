"""Row pressure must summarize old sends before the next prompt trims them."""
import pytest

from power_loop import AgentLoopConfig, LLMResponse, StatefulAgentLoop
from power_loop.runtime.fold import LLMSummaryFold
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.projection_fold import read_state

from .test_job_registry_192 import registry_store
from .test_stateful_loop import _Scripted

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_row_limit_folds_old_sends_before_the_token_limit(registry_store):
    store, _ = registry_store
    summary = _Scripted([LLMResponse(raw_text='<summary>FIRST-FACT SECOND-FACT</summary>'),
                         LLMResponse(raw_text='<summary>FIRST-FACT SECOND-FACT THIRD-FACT FOURTH-FACT</summary>')])
    model = _Scripted([LLMResponse(raw_text=f'answer {n}') for n in range(1, 7)])
    loop = StatefulAgentLoop(store=store, llm=model, config=AgentLoopConfig(
        representation=ProjectedRepresentation(), context_budget_tokens=10**6, max_context_rows=8,
        fold_strategy=LLMSummaryFold(keep_last_sends=1, trigger_ratio=.75, summary_llm=summary)))
    sid = await loop.new_session()
    try:
        for fact in ['FIRST-FACT', 'SECOND-FACT']:
            await loop.send(fact, sid)
        assert not summary.calls
        assert (await read_state(store, store._db, sid))['rows'] == 4
        await loop.send('THIRD-FACT', sid)
        compact = await store.latest_project_compact(sid)
        assert compact.compact_to_send == 2
        assert all(text in str(summary.calls[0]) for text in ['FIRST-FACT', 'SECOND-FACT', 'answer 1', 'answer 2'])
        await loop.send('FOURTH-FACT', sid)
        assert 'FIRST-FACT SECOND-FACT' in str(model.calls[-1])
        state = await read_state(store, store._db, sid)
        assert state['tokens'] < 750_000 and state['rows'] == 4
        assert len(summary.calls) == 1
        await loop.send('FIFTH-FACT', sid)
        assert (await store.latest_project_compact(sid)).compact_to_send == 4
        assert all(text in str(summary.calls[1]) for text in ['FIRST-FACT SECOND-FACT', 'THIRD-FACT', 'FOURTH-FACT'])
        await loop.send('SIXTH-FACT', sid)
        assert 'FIRST-FACT SECOND-FACT THIRD-FACT FOURTH-FACT' in str(model.calls[-1])
        assert (await read_state(store, store._db, sid))['rows'] == 4
        assert len(await store.load_all_messages(sid)) == 12
    finally:
        await loop.aclose(drain_timeout_s=.1)
