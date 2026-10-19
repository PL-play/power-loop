"""One paid summary: preserve critical evidence and reuse it without another request."""
import pytest

from power_loop._vendor.llm_client.interface import LLMResponse
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


@pytest.mark.asyncio
async def test_real_workflow_summary_is_attributed_after_node_completion():
    import asyncio
    import json

    from power_loop import AgentLoopConfig, StatefulAgentLoop
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.runtime.representation import ProjectedRepresentation
    from power_loop.runtime.store.context_cache import ContextCache
    from power_loop.tools.registry import ToolRegistry
    from power_loop.workflow import WorkflowSpec
    from power_loop.workflow.engine import WorkflowEngine
    from tests.unit.test_stateful_loop import _Scripted, _tool_resp

    summary_llm = make_llm(max_tokens=4096)
    store = await SessionStore.open(':memory:')
    registry = ToolRegistry()
    registry.register(ToolDefinition('inspect_evidence', 'inspect fixed evidence'),
        lambda: 'successful checks\n' * 200 + 'critical.py failed: missing secret; exit 42; deployment blocked')
    config = ToolContextConfig(summary_llm=summary_llm,
        overrides={'inspect_evidence': {'between_sends': 'llm_summary', 'timeout_s': 90}})
    loop = StatefulAgentLoop(store=store, tool_registry=registry,
        llm=_Scripted([_tool_resp('c', 'inspect_evidence', '{}'), LLMResponse(raw_text='evidence recorded')]),
        config=AgentLoopConfig(compactor=None, representation=ProjectedRepresentation(), tool_context=config))
    spec = WorkflowSpec.from_json({'name': 'live-summary', 'root': {'type': 'agent', 'id': 'review',
        'spec': {'name': 'review', 'system_prompt': 'inspect', 'tools': ['inspect_evidence']}}})
    try:
        result = await WorkflowEngine(loop, run_id='real-summary-workflow').run(spec)
        assert result.status == 'completed'
        leaf = result.results['review'].session_id
        cache = ContextCache(store)
        async def finished():
            while True:
                rows = await cache.db.fetchall(f"SELECT * FROM {cache.attempts} WHERE outcome!='started'")
                if rows:
                    return rows
                await asyncio.sleep(.05)
        rows = await asyncio.wait_for(finished(), 100)
        assert len(rows) == 1
        record = json.loads(rows[0]['record_json'])
        assert record['outcome'] == 'success' and not record['usage_unknown']
        assert record['owner_session_id'] == leaf and record['workflow_run_id'] == 'real-summary-workflow'
        assert record['node_id'] == 'review' and record['usage']['total_tokens'] > 0
        summary = await cache.get(rows[0]['cache_key'])
        assert summary and 'critical.py' in summary and '42' in summary
        assert await cache.get(rows[0]['cache_key']) == summary
        await cache.reconcile()
        assert (await store.get_session_stats(leaf)).total_tokens >= record['usage']['total_tokens']
    finally:
        await store.close()
        await summary_llm.close()
