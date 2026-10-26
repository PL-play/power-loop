"""Flat summary scheduling preserves global and per-session admission limits."""
import inspect
import re

import pytest

from power_loop.runtime.store.context_cache import ContextCache

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_flat_claim_skips_a_busy_owner_without_starving_another_owner(registry_store):
    store, _ = registry_store
    a,b = await store.create_session(), await store.create_session()
    cache = ContextCache(store)
    for key,owner in [('first-a',a),('second-a',a),('first-b',b)]:
        assert await cache.enqueue(key,'executor',{'owner_session_id':owner,'timeout_s':30},60)
    for index,key in enumerate(['first-a','second-a','first-b']):
        await cache.db.execute(f'UPDATE {cache.cache} SET created_at=? WHERE cache_key=?', (index,key))
    first = await cache.claim('executor')
    assert first['owner_session_id'] == a
    second = await cache.claim('executor')
    assert second['owner_session_id'] == b
    assert await cache.claim('executor') is None
    pending = await cache.db.fetchone(f'SELECT status FROM {cache.cache} WHERE cache_key=?', ('second-a',))
    assert pending['status'] == 'pending'
    await cache.db.execute(f"UPDATE {cache.cache} SET status='ready' WHERE cache_key=?", ('first-a',))
    assert (await cache.claim('executor'))['cache_key'] == 'second-a'


async def test_zero_per_session_admission_never_claims_and_capacity_has_an_index_path(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    cache = ContextCache(store)
    assert await cache.enqueue('original','executor',{'owner_session_id':sid,'timeout_s':30},60)
    assert await cache.claim('executor',per_session=0) is None
    assert (await cache.claim('executor'))['cache_key'] == 'original'
    source = inspect.getsource(ContextCache.claim)
    assert not re.search(r'\(\s*SELECT\b',source,re.I)
    if cache.db.dialect.name == 'sqlite':
        rows = await cache.db.fetchall(f'EXPLAIN QUERY PLAN SELECT COUNT(*),SUM(size_bytes) FROM {cache.cache}')
        assert any('idx_context_capacity' in row['detail'] for row in rows)
