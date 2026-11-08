"""Paid provider smoke: send(history=False) reuses one session but each run starts blank."""
import pytest

from power_loop import AgentLoopConfig, RequestOptions, SessionStore
from power_loop.runtime.representation import ProjectedRepresentation

from ._llm import make_llm


@pytest.mark.asyncio
async def test_real_history_off_runs_do_not_see_each_other(runtime_agent):
    llm = make_llm(max_tokens=2048)
    store = await SessionStore.open(':memory:')
    loop = await runtime_agent(llm=llm, store=store, config=AgentLoopConfig(
        system_prompt='Answer briefly.', max_tokens=2048, max_rounds=3,
        representation=ProjectedRepresentation()))
    try:
        sid = loop.target.session_id
        first = await loop.send('My secret word is PELICAN-42. Reply OK.', options=RequestOptions(history=False, metadata={'agent': 'a'}))
        assert first.status == 'completed'
        second = await loop.send('What is my secret word? If you were not told one, reply exactly UNKNOWN.', options=RequestOptions(history=False, metadata={'agent': 'b'}))
        assert second.status == 'completed'
        assert 'PELICAN' not in second.final_text.upper()
        assert not await store.load_project_messages(sid)
        third = await loop.send('What is my secret word? Reply with it only.')
        assert third.status == 'completed' and 'PELICAN-42' in third.final_text.upper()
    finally:
        await loop.aclose()
        await store.close()
        await llm.close()
