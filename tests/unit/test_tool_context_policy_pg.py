import asyncio
import os
from urllib.parse import urlparse

import pytest

from power_loop.runtime.store.factory import open_store


@pytest.mark.asyncio
async def test_auxiliary_stats_atomic_replay_pg():
    dsn = os.getenv('POWER_LOOP_TEST_PG_DSN')
    if not dsn:
        pytest.skip('requires explicit isolated PG DSN')
    assert urlparse(dsn).path.endswith('_test')
    store = await open_store(dsn, table_prefix='plcontexttest_')
    sid = await store.create_session()
    try:
        usage = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}
        await asyncio.gather(*(store.record_auxiliary_usage(sid, 'attempt', usage) for _ in range(8)))
        stats = await store.get_session_stats(sid)
        assert stats.llm_calls == 1 and stats.total_tokens == 120 and stats.sends == 0
    finally:
        await store.close_session(sid)
        await store.close()


@pytest.mark.asyncio
async def test_auxiliary_stats_revision_order_pg():
    dsn = os.getenv('POWER_LOOP_TEST_PG_DSN')
    if not dsn:
        pytest.skip('requires explicit isolated PG DSN')
    assert urlparse(dsn).path.endswith('_test')
    store = await open_store(dsn, table_prefix='plcontexttest_')
    sid = await store.create_session()
    try:
        await store.record_auxiliary_usage(sid, 'corrected', {}, revision=1)
        known = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}
        await asyncio.gather(*(store.record_auxiliary_usage(sid, 'corrected',
            known if n % 2 else {}, revision=2 if n % 2 else 1) for n in range(8)))
        stats = await store.get_session_stats(sid)
        assert stats.llm_calls == 1 and stats.total_tokens == 120
    finally:
        await store.close_session(sid)
        await store.close()
