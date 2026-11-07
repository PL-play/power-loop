"""P5c maintenance cost: fixed pending size, history and cumulative batch accounting."""
import json
import os
import time
from pathlib import Path

import pytest

from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_retention import NOW, OLD, insert, root
from tests.unit.test_send_sql_profile import Profiled, _counts
from tests.write_meter import WriteMeter

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('history', [0, 10000])
@pytest.mark.parametrize('pending', [False, True])
async def test_maintenance_cost(registry_store, history, pending):
    store, _ = registry_store
    db, p = store._db, store.table_prefix
    retained, sid, _ = await root(store, 'history', at=NOW)
    async with db.transaction() as tx:
        for i in range(history):
            await insert(tx, p+'job_events', event_id=f'history-{i}', job_id=retained,
                event_type='trace', job_version=1, payload_json='{}', delivery_status='handled', created_at=NOW)
        if pending:
            await insert(tx, store.t.follow_up_queue, session_id=sid, item_id='pending', content='work',
                kind='user', mode='queue', status='pending', created_at=NOW)
    meter = await WriteMeter.install(db, p)
    profiled = Profiled(db, meter)
    store._db = profiled
    output = []
    async def sample(name):
        await meter.reset()
        profiled.records.clear()
        profiled.tx_sites.clear()
        started = time.monotonic()
        counts = await store.purge_expired(NOW, limit=1000)
        elapsed = time.monotonic()-started
        writes = await meter.report()
        sql = _counts(profiled.records, profiled.tx_sites)
        sql['total'] = len(profiled.records)
        output.append(dict(stage=os.environ.get('P5C_STAGE', 'N'), db=db.dialect.name,
            scenario=name, history=history, pending=int(pending), limit=1000,
            sql=sql, transactions=len(profiled.tx_sites), writes=writes,
            records=list(profiled.records), tx_sites=list(profiled.tx_sites),
            elapsed_s=elapsed, deleted=counts))
        if os.environ.get('P5C_STAGE', 'N') == 'N':
            assert writes['affected_rows'] <= 1000
            assert writes['max_tx_rows'] <= 1000
        return writes['affected_rows']
    try:
        assert await sample('no_expired_work') == 0
        # One expired tree larger than the allowance, a small tree and live work coexist.
        large, _, _ = await root(store, 'expired-large')
        await root(store, 'expired-small')
        await root(store, 'live', ended=False)
        async with db.transaction() as tx:
            for i in range(1100):
                await insert(tx, p+'job_events', event_id=f'expired-{i}', job_id=large,
                    event_type='trace', job_version=1, payload_json='{}', delivery_status='handled', created_at=OLD)
        for batch in range(4):
            affected = await sample(f'expired_batch_{batch}')
            output[-1]['remaining_expired_events'] = (await db.fetchone(
                f'SELECT COUNT(*) AS n FROM {p}job_events WHERE job_id=?', (large,)))['n']
            if affected == 0:
                break
        assert await db.fetchone(f'SELECT job_id FROM {p}jobs WHERE job_id=?', (retained,))
        assert not await db.fetchone(f'SELECT job_id FROM {p}jobs WHERE job_id=?', (large,))
        if pending:
            assert await db.fetchone(f'SELECT id FROM {store.t.follow_up_queue} WHERE status=?', ('pending',))
        path = os.environ.get('P5C_MAINTENANCE_OUT')
        if path:
            with Path(path).open('a') as stream:
                for row in output:
                    stream.write(json.dumps(row)+'\n')
    finally:
        store._db = db
        await meter.close()


@pytest.mark.parametrize('pending', [False, True])
async def test_close_cost(registry_store, pending):
    from power_loop.contracts.errors import SessionBusy
    store, _ = registry_store
    db = store._db
    _, sid, _ = await root(store, 'close-cost', at=NOW)
    if pending:
        await insert(db, store.t.follow_up_queue, session_id=sid, item_id='pending', content='work',
                     kind='user', mode='queue', status='pending', created_at=NOW)
    else:
        async with db.transaction() as tx:
            for seq in range(1,1101):
                await insert(tx, store.t.messages, session_id=sid, seq=seq, role='user', content='history', created_at=OLD)
    meter = await WriteMeter.install(db, store.table_prefix)
    profiled = Profiled(db, meter)
    store._db = profiled
    reports = []
    try:
        for operation in ('close', 'delete-0', 'delete-1', 'delete-2'):
            await meter.reset()
            profiled.records.clear()
            profiled.tx_sites.clear()
            started = time.monotonic()
            if operation == 'close':
                result = await store.close_if_idle(sid)
                assert result == ('busy' if pending else 'closed')
            elif pending:
                with pytest.raises(SessionBusy):
                    await store.close_session_tree(sid)
                result = 'busy'
            else:
                result = await store.close_session_tree(sid)
            duration = time.monotonic()-started
            writes = await meter.report()
            assert writes['affected_rows'] <= 1000 and writes['max_tx_rows'] <= 1000
            reports.append({'db':db.dialect.name,'pending':pending,'operation':operation,'result':result,
                'sql':{**_counts(profiled.records,profiled.tx_sites),'total':len(profiled.records)},
                'transactions':len(profiled.tx_sites),'records':list(profiled.records),'tx_sites':list(profiled.tx_sites),
                'writes':writes,'elapsed_s':duration})
            if pending and operation!='close':
                break
        if path := os.environ.get('P5C_CLOSE_OUT'):
            with Path(path).open('a') as stream:
                for report in reports:
                    stream.write(json.dumps(report)+'\n')
    finally:
        store._db = db
        await meter.close()
