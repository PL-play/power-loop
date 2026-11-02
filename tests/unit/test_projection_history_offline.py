"""Stopped-store conversion and per-send raw fallback on three real engines."""
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager

import pytest

from power_loop import AgentLoopConfig, LLMResponse, StatefulAgentLoop
from power_loop.runtime.representation import ProjectedRepresentation
from scripts.migrate_projection_history import convert

from .test_job_registry_192 import registry_store
from .test_stateful_loop import _Scripted

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def original_session(store, *, note=False):
    loop = StatefulAgentLoop(store=store, llm=_Scripted([LLMResponse(raw_text=f'answer {n}') for n in range(1, 4)]))
    sid = await loop.new_session(metadata={'projection_migrated': 0})
    try:
        for n in range(1, 4):
            await loop.send(f'original fact {n}', sid)
    finally:
        await loop.aclose(drain_timeout_s=.1)
    if note:
        seqs = [r.seq for r in await store.load_active_messages(sid, send_index=1)]
        await store.record_compaction(sid, from_seq=min(seqs), to_seq=max(seqs),
            note_content='PREVIOUS SUMMARY original fact 1 answer 1', before_tokens=None, after_tokens=None,
            round_index=0, fold_seqs=seqs, order_key=0)
    return sid


async def cli(store, dsn, sid, *flags):
    child = await asyncio.create_subprocess_exec(sys.executable, '-m', 'scripts.migrate_projection_history',
        '--dsn', dsn, '--prefix', store.table_prefix, '--session-id', sid, *flags,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(child.communicate(), 30)
    assert child.returncode == 0, err.decode()+out.decode()
    return json.loads(out)


@pytest.mark.parametrize('note', [False, True])
async def test_real_cli_import_is_idempotent_and_runtime_reads_missing_or_stale_send(registry_store, tmp_path, monkeypatch, note):
    store, _ = registry_store
    backend = store._db.dialect.name
    dsn = str(tmp_path/'registry.db') if backend == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if backend == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    sid = await original_session(store, note=note)
    before = await store._db.fetchall(f'SELECT * FROM {store.t.messages} WHERE session_id=? ORDER BY id', (sid,))
    assert not (await cli(store, dsn, sid))['applied']
    assert not await store.load_project_messages(sid)
    assert (await cli(store, dsn, sid, '--apply', '--writers-stopped'))['project_rows'] == 6 - 2*int(note)
    projected = await store.load_project_messages(sid)
    assert len(projected) == 6 - int(note)
    if note:
        assert (await store.latest_project_compact(sid)).compact_to_send == 0
    assert (await cli(store, dsn, sid, '--apply', '--writers-stopped'))['project_rows'] == 0
    assert await store.load_project_messages(sid) == projected
    assert 'projection_migrated' not in (await store.get_session(sid)).metadata
    await store._db.execute(f'DELETE FROM {store.t.project_messages} WHERE session_id=? AND send_index=2', (sid,))
    await store._db.execute(f'UPDATE {store.t.project_messages} SET projector_version=0 WHERE session_id=? AND send_index=3', (sid,))
    reads, load = [], store.load_active_messages
    async def raw(session_id, **kw):
        reads.append(kw.get('send_index'))
        return await load(session_id, **kw)
    monkeypatch.setattr(store, 'load_active_messages', raw)
    model = _Scripted([LLMResponse(raw_text='new answer')])
    loop = StatefulAgentLoop(store=store, llm=model, config=AgentLoopConfig(
        representation=ProjectedRepresentation(), context_budget_tokens=10**6))
    try:
        await loop.send('continue', sid)
        assert {2, 3} <= set(reads) and None not in reads
        assert all(s in str(model.calls[0]) for s in ['original fact 1', 'original fact 2', 'answer 2', 'original fact 3', 'answer 3'])
        if note:
            assert 'PREVIOUS SUMMARY' in str(model.calls[0])
        current = await store._db.fetchall(f'SELECT * FROM {store.t.messages} WHERE session_id=? AND id<=? ORDER BY id', (sid, before[-1]['id']))
        assert current == before
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def test_offline_partial_write_rolls_back_originals_metadata_and_projections(registry_store, monkeypatch):
    store, _ = registry_store
    sid = await original_session(store)
    before = await store.get_session(sid)
    originals = await store.load_all_messages(sid)
    with pytest.raises(ValueError, match='maintenance_required'):
        await convert(store, sid, apply=True)
    with pytest.raises(ValueError, match='session_not_found'):
        await convert(store, 'missing')
    transaction = store._db.transaction
    @asynccontextmanager
    async def broken():
        async with transaction() as tx:
            class Writer:
                def __getattr__(self, key):
                    return getattr(tx, key)
                async def execute(self, sql, params=()):
                    result = await tx.execute(sql, params)
                    if sql.startswith('INSERT INTO '+store.t.project_messages):
                        raise OSError('process lost before projection transaction committed')
                    return result
            yield Writer()
    with monkeypatch.context() as patch:
        patch.setattr(store._db, 'transaction', broken)
        with pytest.raises(OSError):
            await convert(store, sid, apply=True, writers_stopped=True)
    assert await store.get_session(sid) == before
    assert await store.load_all_messages(sid) == originals
    assert not await store.load_project_messages(sid)
    assert (await convert(store, sid, apply=True, writers_stopped=True))['project_rows'] == 6
