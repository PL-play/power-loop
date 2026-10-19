from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from power_loop._vendor.llm_client.interface import LLMResponse, LLMTokenUsage
from power_loop.contracts.context_policy import ToolContextPolicy
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.context_worker import SummaryWorker
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine, ToolSource
from power_loop.tools.registry import ToolRegistry
from tests.unit.test_stateful_loop import _Scripted, _tool_resp


@pytest.fixture
async def env():
    store = await SessionStore.open(':memory:')
    sid = await store.create_session()
    registry = ToolRegistry()
    registry.register(ToolDefinition('bash', '', project=lambda a, r: a.get('description', 'ok')),
                      lambda **kw: 'result' * 300)
    llm = _Scripted([LLMResponse(raw_text='verified exact-file.txt', token_usage=LLMTokenUsage(
        prompt_tokens=100, completion_tokens=20, total_tokens=120, prompt_cached_tokens=50))])
    config = ToolContextConfig(overrides={'bash': {'between_sends': 'llm_summary'}})
    engine = ToolContextEngine(config=config, store=store, owner_session_id=sid, owner_send_index=2,
                               registry=registry, llm=llm, model='test')
    await engine.start()
    source = ToolSource(sid, 1, 3, 'call', 'bash', {'command': 'long ' * 300, 'description': 'checks'}, 'ok ' * 300)
    yield store, sid, registry, llm, engine.config, engine, source
    await store.close()


async def generate(engine, source):
    await engine.submit(source)
    job = await engine.cache.claim(engine.executor_key)
    assert job is not None
    await SummaryWorker(engine.store).execute(job, engine)
    await engine.cache.reconcile()
    return job


def test_metadata_not_in_schema_and_field_override():
    policy = ToolContextPolicy(insend_fuse='preserve')
    tool = ToolDefinition('load_skill', '', context_policy=policy)
    assert 'context_policy' not in tool.to_openai_tool()['function']
    assert policy.override({'between_sends': 'llm_summary'}).insend_fuse == 'preserve'
    with pytest.raises((ValueError, TypeError)):
        policy.override({'typo': 1})


async def test_read_pending_is_nonblocking_and_no_model_request(env):
    _, _, _, llm, _, engine, source = env
    assert await engine.submit(source)
    assert (await engine.distill(source, 'between_sends')).startswith('checks')
    assert not llm.calls


async def test_completed_shared_cache_and_local_recall(env):
    store, sid, _, llm, _, engine, source = env
    await generate(engine, source)
    other = replace(source, seq=99, call_id='other')
    assert not await engine.submit(other)
    text = await engine.distill(other, 'between_sends')
    assert 'verified exact-file' in text and 'seq=99' in text
    assert len(llm.calls) == 1
    stats = await store.get_session_stats(sid)
    assert stats.total_tokens == 120 and stats.llm_calls == 1 and stats.sends == 0


async def test_concurrent_enqueue_and_claim_are_single_flight(env):
    _, _, _, _, _, engine, source = env
    inserted = await asyncio.gather(*(engine.submit(source) for _ in range(10)))
    assert sum(inserted) == 1
    claims = await asyncio.gather(*(engine.cache.claim(engine.executor_key) for _ in range(10)))
    assert sum(job is not None for job in claims) == 1


async def test_hash_arguments_prompt_and_scope(env):
    _, _, _, _, _, engine, source = env
    policy = engine.policy(source)
    key = engine.cache_key(source, policy)
    assert key == engine.cache_key(replace(source, seq=9, call_id='new'), policy)
    assert key != engine.cache_key(replace(source, arguments={'command': 'different'}), policy)
    assert key != engine.cache_key(source, replace(policy, prompt='different'))
    engine.cache_scope = 'other-user'
    assert key != engine.cache_key(source, policy)


async def test_late_summary_does_not_rewrite_selected_view(env):
    _, _, _, _, _, engine, source = env
    before = await engine.distill(source, 'between_sends')
    await generate(engine, source)
    assert await engine.distill(source, 'between_sends') == before
    assert 'verified exact-file' in await engine.distill(replace(source, seq=4), 'between_sends')


async def test_keep_only_current_send(env):
    _, _, _, _, _, engine, source = env
    source = replace(source, arguments={**source.arguments, 'context_policy': 'keep'})
    await generate(engine, source)
    assert await engine.distill(source, 'insend_fuse') is None
    assert await engine.distill(source, 'age_retirement') is None
    assert 'verified exact-file' in await engine.distill(source, 'between_sends')


async def test_discarded_summary_billed_once_and_fallback(env):
    store, sid, _, llm, _, engine, source = env
    llm.responses[0].raw_text = 'too large ' * 3000
    await generate(engine, source)
    assert (await engine.distill(source, 'between_sends')).startswith('checks')
    assert not await engine.submit(source)
    await engine.cache.reconcile()
    stats = await store.get_session_stats(sid)
    assert stats.total_tokens == 120 and stats.llm_calls == 1


async def test_failed_summary_unknown_and_durable_outbox(env):
    _, _, _, _, _, engine, source = env
    class Broken:
        async def complete(self, request):
            raise TimeoutError()
    engine.llm = Broken()
    await generate(engine, source)
    deliveries = []
    async def receiver(record):
        deliveries.append(record)
        if len(deliveries) == 1:
            raise RuntimeError('offline')
    await engine.cache.reconcile(receiver)
    await engine.cache.reconcile(receiver)
    await engine.cache.reconcile(receiver)
    assert len(deliveries) == 2
    assert deliveries[0]['attempt_id'] == deliveries[1]['attempt_id']
    assert deliveries[0]['usage_unknown'] and deliveries[0]['owner_send_index'] == 2
    assert deliveries[0]['source_send_index'] == 1


async def test_expired_started_lease_is_uncertain_not_retried(env):
    _, _, _, _, _, engine, source = env
    await engine.submit(source)
    job = await engine.cache.claim(engine.executor_key)
    record = {'attempt_id': 'lost', 'owner_session_id': engine.sid, 'outcome': 'started', 'usage': None}
    await engine.cache.begin_attempt(job, record)
    await engine.cache.db.execute(f'UPDATE {engine.cache.cache} SET lease_until=0')
    assert await engine.cache.claim(engine.executor_key) is None
    await engine.cache.reconcile()
    rows = await engine.cache.db.fetchall(f'SELECT outcome,record_json FROM {engine.cache.attempts}')
    assert rows[0]['outcome'] == 'interrupted'
    assert json.loads(rows[0]['record_json'])['usage_unknown']


async def test_eviction_keeps_ledger_and_frozen_view(env):
    store, sid, _, _, _, engine, source = env
    await generate(engine, source)
    selected = await engine.distill(source, 'between_sends')
    assert await engine.cache.prune(max_entries=0, max_bytes=0) == 1
    assert await engine.cache.get(engine.cache_key(source, engine.policy(source))) is None
    assert await engine.distill(source, 'between_sends') == selected
    assert (await store.get_session_stats(sid)).total_tokens == 120
    assert len(await engine.cache.db.fetchall(f'SELECT * FROM {engine.cache.attempts}')) == 1


async def test_background_delegates_target_keep(env):
    from power_loop.runtime.background import encode_results, record_key
    store, sid, _, llm, _, engine, source = env
    record = {'task_id': 'task-1', 'tool': 'bash', 'args': {**source.arguments, 'context_policy': 'keep'},
              'result': source.result, 'status': 'completed'}
    await store.set_runtime_state(sid, record_key('task-1'), record)
    wrapped = replace(source, name='background_run', arguments={'action': 'check', 'task_id': 'task-1'},
                      result=encode_results([record]))
    assert await engine.submit(wrapped)
    job = await engine.cache.claim(engine.executor_key)
    await SummaryWorker(store).execute(job, engine)
    assert await engine.distill(wrapped, 'insend_fuse') is None
    assert 'verified exact-file' in await engine.distill(wrapped, 'between_sends')
    assert len(llm.calls) == 1


async def test_send_returns_while_summary_running(env):
    from power_loop.agent.stateful_loop import StatefulAgentLoop
    from power_loop.agent.types import AgentLoopConfig
    from power_loop.runtime.representation import ProjectedRepresentation
    store, _, registry, _, config, _, _ = env
    blocked = asyncio.Event()
    class Slow:
        async def complete(self, request):
            await blocked.wait()
            return LLMResponse(raw_text='summary')
    config.summary_llm = Slow()
    main = _Scripted([_tool_resp('c', 'bash', '{}'), LLMResponse(raw_text='done'), LLMResponse(raw_text='next')])
    loop = StatefulAgentLoop(llm=main, store=store, tool_registry=registry,
        config=AgentLoopConfig(tool_context=config, representation=ProjectedRepresentation()))
    sid = await loop.new_session()
    try:
        first = await asyncio.wait_for(loop.send('test', sid), 2)
        second = await asyncio.wait_for(loop.send('next', sid), 2)
        assert first.final_text == 'done' and second.final_text == 'next'
        assert not blocked.is_set()
    finally:
        blocked.set()
        await loop.aclose()


async def test_capacity_applies_backpressure_without_model(env):
    _, _, _, llm, _, engine, source = env
    engine.config.cache_max_entries = 1
    assert await engine.submit(source)
    assert not await engine.submit(replace(source, arguments={'command': 'second'}))
    assert not llm.calls


async def test_ttl_and_approximate_lru(env):
    _, _, _, _, _, engine, source = env
    await generate(engine, source)
    key = engine.cache_key(source, engine.policy(source))
    await engine.cache.db.execute(f'UPDATE {engine.cache.cache} SET expires_at=0 WHERE cache_key=?', (key,))
    assert await engine.cache.get(key) is None
    await engine.cache.prune(max_entries=100, max_bytes=1000000)
    assert await engine.submit(source)


async def test_explicit_503_backoff_distinct_attempts(env):
    _, _, _, _, _, engine, source = env
    class ServiceError(Exception):
        status_code = 503
    class Busy:
        async def complete(self, request):
            raise ServiceError()
    engine.llm = Busy()
    await generate(engine, source)
    assert await engine.cache.claim(engine.executor_key) is None
    await engine.cache.db.execute(f'UPDATE {engine.cache.cache} SET retry_at=0')
    job = await engine.cache.claim(engine.executor_key)
    await SummaryWorker(engine.store).execute(job, engine)
    assert await engine.cache.claim(engine.executor_key) is None
    rows = await engine.cache.db.fetchall(f'SELECT attempt_id FROM {engine.cache.attempts}')
    assert len(rows) == 2 and rows[0]['attempt_id'] != rows[1]['attempt_id']


async def test_cancelled_job_stays_uncertain_after_cache_eviction(env):
    _, _, _, _, _, engine, source = env
    class Cancel:
        async def complete(self, request):
            raise asyncio.CancelledError()
    engine.llm = Cancel()
    await engine.submit(source)
    job = await engine.cache.claim(engine.executor_key)
    with pytest.raises(asyncio.CancelledError):
        await SummaryWorker(engine.store).execute(job, engine)
    await engine.cache.prune(max_entries=0, max_bytes=0)
    assert not await engine.submit(source)
