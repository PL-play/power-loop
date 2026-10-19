"""One paid summary: preserve critical evidence and reuse it without another request."""
import pytest

from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine, ToolSource
from tests.real._llm import make_llm


@pytest.mark.asyncio
async def test_real_summary_is_metered_and_reused():
    llm = make_llm(max_tokens=4096)
    store = await SessionStore.open(':memory:')
    try:
        sid = await store.create_session()
        config = ToolContextConfig(overrides={'bash': {'between_sends': 'llm_summary', 'timeout_s': 90}})
        engine = ToolContextEngine(config=config, store=store, owner_session_id=sid,
                                   owner_send_index=2, llm=llm, model=None)
        await engine.start()
        source = ToolSource(sid, 1, 3, 'call', 'bash', {'command': 'pytest critical.py',
            'description': 'Check critical deployment blocker'},
            'normal initialization succeeded\n' * 100 + '\nFAIL critical.py: secret key is absent; exit code 42. Do not deploy until fixed.')
        from power_loop.runtime.context_worker import SummaryWorker
        await engine.submit(source)
        job = await engine.cache.claim(engine.executor_key)
        await SummaryWorker(store).execute(job, engine)
        await engine.cache.reconcile()
        summary = await engine.distill(source, 'between_sends')
        assert summary is not None and 'critical.py' in summary and '42' in summary
        assert await engine.distill(source, 'between_sends') == summary
        stats = await store.get_session_stats(sid)
        assert stats.llm_calls == 1 and stats.total_tokens > 0
    finally:
        await store.close()
        await llm.close()
