import asyncio
from contextvars import ContextVar
from types import SimpleNamespace

import pytest

from power_loop.contracts.tools import ToolDefinition
from power_loop.core import agent_context
from power_loop.tools import default_tools as dt


@pytest.mark.asyncio
async def test_queue_is_fifo_per_session_and_cancel_does_not_invoke(monkeypatch):
    sid = ContextVar('queue_sid', default='a')
    events = {i: asyncio.Event() for i in range(24)}
    started = []
    active = {}
    peaks = {}
    rows = {}
    transitions = []

    class Store:
        async def upsert_background_task(self, session, **kw):
            rows[session, kw['task_id']] = SimpleNamespace(**kw)
            transitions.append((session, kw['task_id'], kw['status']))

        async def list_background_tasks(self, session):
            return [v for (s, _), v in rows.items() if s == session]

        async def get_background_task(self, session, tid):
            return rows.get((session, tid))

    store = Store()

    class Registry:
        def get(self, name):
            return ToolDefinition(name=name, description='test', async_capable=True)

        async def invoke_async(self, name, args):
            session, i = sid.get(), args['i']
            started.append(i)
            active[session] = active.get(session, 0) + 1
            peaks[session] = max(peaks.get(session, 0), active[session])
            try:
                await events[i].wait()
                if i == 0:
                    raise ValueError('test failure releases slot')
                return str(i)
            finally:
                active[session] -= 1

    monkeypatch.setattr(agent_context, 'get_current_loop', lambda: SimpleNamespace(tool_registry=Registry()))
    monkeypatch.setattr(dt, '_current_store_and_session', lambda: (store, sid.get()))
    manager = dt.BackgroundManager()
    notifications = []

    async def notified(session, tid, status):
        notifications.append((session, tid, status))

    monkeypatch.setattr(dt, '_TOOL_TASK_ON_COMPLETE', notified)

    async def submit(i):
        receipt = await manager.run_tool('image', {'i': i})
        return receipt.split('task_id=')[1].split('（')[0]

    try:
        ids = await asyncio.gather(*(submit(i) for i in range(12)))
        assert started == list(range(8))
        assert [rows['a', tid].status for tid in ids] == ['running'] * 8 + ['queued'] * 4
        assert await manager.cancel_task(ids[9]) == 'stopped'
        assert rows['a', ids[9]].status == 'cancelled'
        assert not notifications
        # A second sid has its own eight slots even in the same manager/store.
        token = sid.set('b')
        more = await asyncio.gather(*(submit(i) for i in range(12, 21)))
        sid.reset(token)
        assert started == list(range(8)) + list(range(12, 20))
        assert rows['b', more[-1]].status == 'queued'
        # Failure releases capacity without any check or completion collection.
        events[0].set()
        for _ in range(20):
            await asyncio.sleep(0)
        assert started[-1] == 8
        assert rows['a', ids[0]].status == 'failed'
        assert rows['a', ids[10]].status == 'queued'
        for event in events.values():
            event.set()
        await asyncio.gather(*list(manager._workers.values()), return_exceptions=True)
        await asyncio.sleep(0)
        assert peaks == {'a': 8, 'b': 8}
        assert 9 not in started
        assert [i for i in started if i < 12][-3:] == [8, 10, 11]
        assert not manager._tool_slots
        assert not manager._stop_tokens
        assert ('a', ids[10], 'queued') in transitions
        assert ('a', ids[10], 'running') in transitions
        assert ('a', ids[10], 'completed') in transitions
    finally:
        for worker in list(manager._workers.values()):
            worker.cancel()
        await asyncio.gather(*list(manager._workers.values()), return_exceptions=True)


@pytest.mark.asyncio
async def test_wait_includes_queued_tasks_without_a_store(monkeypatch):
    manager = dt.BackgroundManager()
    manager.tasks['queued'] = {'status': 'queued', 'command': 'tool:image'}
    monkeypatch.setattr(dt, '_current_store_and_session', lambda: (None, None))

    async def finish():
        await asyncio.sleep(0.03)
        manager.tasks['queued'].update(status='completed', result='image ready')

    worker = asyncio.create_task(finish())
    manager._workers['queued'] = worker
    output = await manager.wait(timeout_s=1)
    assert worker.done()
    assert '[completed] image ready' in output
    assert '都做完了' in output
