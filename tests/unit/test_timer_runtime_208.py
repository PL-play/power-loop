"""Current-runtime timer and atomic board behavior, using actual store backends."""
import asyncio
import time

import pytest

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeRequestError,
    StoreBlackboard,
    TimerContext,
    TimerRunner,
    create_default_tool_registry,
)

from .test_job_registry_192 import registry_store
from .test_runtime_messaging_203 import fixture
from .test_timers import _Scripted

__all__ = ['registry_store']


async def composition(store):
    _, target, actor = await fixture(store)
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(_Scripted(), create_default_tool_registry(preset='core',bind=False), AgentLoopConfig())
    factory.register('test.agent@1', build)
    runtime = Runtime(store=store, factory=factory, node_id='timers')
    await runtime.worker.start()
    return runtime, target, actor, TimerContext(store.store_id, target.session_id, actor)


async def test_timer_two_drivers_and_cancel(registry_store):
    store, _ = registry_store
    runtime, target, actor, source = await composition(store)
    try:
        timer = await runtime.timer_backend.create(source, due_at_ms=1, note='wake')
        second = TimerRunner(store=store, client=runtime.client)
        assert sum(await asyncio.gather(runtime.timer_driver.scan_once(), second.scan_once())) == 1
        row = await store.get_timer(target.session_id, timer.timer_id)
        assert row.status == 'fired' and row.fire_count == 1
        from power_loop import InputRef
        result = await runtime.client.wait(InputRef(store.store_id, target.session_id, f'timer-{timer.timer_id}-{timer.created_at}-0'),
            actor=actor, deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed'
        periodic = await runtime.timer_backend.create(source, due_at_ms=1, note='repeat', interval_s=60)
        await store.transition_timer(target.session_id, periodic.timer_id, from_status='armed', to_status='firing')
        assert await runtime.timer_backend.cancel(source, periodic.timer_id)
        assert not await store.finish_firing_timer(target.session_id, periodic.timer_id)
        assert (await store.get_timer(target.session_id, periodic.timer_id)).status == 'cancelled'
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_board_first_writers_and_limit(registry_store):
    store, connect = registry_store
    other = await connect()
    a, b = StoreBlackboard(store, max_entries=20), StoreBlackboard(other, max_entries=20)
    entries = await asyncio.gather(*( (a if i%2 else b).post('shared', text=str(i)) for i in range(20)))
    assert len({e.id for e in entries}) == 20
    assert {e.text for e in await b.read('shared')} == {str(i) for i in range(20)}
    from power_loop import BlackboardError
    with pytest.raises(BlackboardError, match='full'):
        await a.post('shared', text='overflow')
    await asyncio.gather(a.update('shared',entries[0].id,text='changed'),
                         b.update('shared',entries[0].id,status='done'))
    updated=next(e for e in await b.read('shared') if e.id==entries[0].id)
    assert updated.text=='changed' and updated.status=='done'
    removed = await asyncio.gather(a.remove('shared',entries[0].id),b.remove('shared',entries[0].id),return_exceptions=True)
    assert sum(x is True for x in removed)==1
    assert sum(isinstance(x,BlackboardError) and 'no board entry' in str(x) for x in removed)==1
    assert await a.read('isolated')==[]
    await a.delete('shared')
    assert await b.read('shared') == []
    assert await store._db.fetchone(f'SELECT 1 FROM {store.t.sessions}') is None


@pytest.mark.parametrize('code', ['capability_unavailable','resource_unavailable','definition_unavailable'])
async def test_factory_code_survives_query(registry_store, code):
    store, _ = registry_store
    _, target, actor = await fixture(store)
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        raise RuntimeRequestError(code, 'missing shell@1')
    factory.register('test.agent@1', build)
    runtime = Runtime(store=store, factory=factory, node_id='codes', timers_enabled=False)
    await runtime.worker.start()
    try:
        receipt = await runtime.client.send(target, 'test', message_id='missing', actor=actor)
        result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=int(time.time()*1000)+10000)
        assert result.error_code == code
        assert (await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=int(time.time()*1000)+10000)).error_code == code
        assert 'shell@1' in result.error
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()


@pytest.mark.parametrize('directive,outcome', [('continue','fired'), ('skip','fired'), ('break','cancelled'), ('postpone','armed')])
async def test_timer_hooks(registry_store, directive, outcome):
    from power_loop import AgentHooks, HookDirective, HookPoint
    store, _ = registry_store
    runtime, target, actor, source = await composition(store)
    hooks = AgentHooks()
    async def hook(ctx):
        if directive == 'postpone':
            ctx.postpone_s = 300
        else:
            ctx.directive = HookDirective(directive)
        ctx.message = 'hook-modified-message'
    hooks.register(HookPoint.TIMER_FIRE, hook)
    runner = TimerRunner(store=store, client=runtime.client, hooks=hooks)
    try:
        timer = await runtime.timer_backend.create(source, due_at_ms=1, note='hook')
        await runner.scan_once()
        assert (await store.get_timer(target.session_id, timer.timer_id)).status == outcome
        rows = await store._db.fetchall(f'SELECT content FROM {store.t.follow_up_queue} WHERE session_id=?', (target.session_id,))
        assert len(rows) == (1 if directive == 'continue' else 0)
        if rows:
            assert 'hook-modified-message' in rows[0]['content']
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_timer_backend_authorization_and_closed_target(registry_store):
    from dataclasses import replace

    from power_loop import TrustedActor
    store, _ = registry_store
    runtime, target, actor, source = await composition(store)
    try:
        stranger = replace(source, actor=TrustedActor('tenant', 'other', None))
        for action in (lambda: runtime.timer_backend.create(stranger, delay_s=10, note='bad'),
                       lambda: runtime.timer_backend.list(stranger),
                       lambda: runtime.timer_backend.cancel(stranger, 1)):
            with pytest.raises(RuntimeRequestError, match='forbidden'):
                await action()
        await store._db.execute(f"UPDATE {store.t.session_state} SET lifecycle='closed' WHERE session_id=?", (target.session_id,))
        with pytest.raises(RuntimeRequestError, match='session_closed'):
            await runtime.timer_backend.create(source, delay_s=10, note='closed')
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_timer_driver_required_and_profiles_preflight(registry_store):
    store, _ = registry_store
    from power_loop import StoreTimerBackend
    factory = RegistryRuntimeFactory()
    with pytest.raises(RuntimeRequestError, match='timer_driver'):
        Runtime(store=store, factory=factory, node_id='bad', timer_backend=StoreTimerBackend(store))
    runtime = Runtime(store=store, factory=factory, node_id='missing', required_profiles=['shell@1'])
    with pytest.raises(RuntimeRequestError, match='missing profiles'):
        await runtime.worker.start()
    assert not runtime.client._accepting
    await runtime.worker.shutdown()
    await runtime.client.aclose()
    await runtime.transport.aclose()


async def test_registered_resource_check_precedes_accepting_and_can_retry(registry_store, tmp_path):
    store, _ = registry_store
    factory = RegistryRuntimeFactory()
    checked = []
    async def mount_ready():
        checked.append('mount')
        if not (tmp_path/'mounted').exists():
            raise RuntimeRequestError('resource_unavailable','workspace@1 mount unavailable')
    factory.register_profile('workspace@1', preflight=mount_ready)
    runtime = Runtime(store=store,factory=factory,node_id='mount',timers_enabled=False,
        required_profiles=['workspace@1'])
    try:
        with pytest.raises(RuntimeRequestError,match='resource_unavailable'):
            await runtime.worker.start()
        assert not runtime.client._accepting and not runtime.worker._subscriptions
        (tmp_path/'mounted').touch()
        await runtime.worker.start()
        assert runtime.status()['accepting'] and checked==['mount','mount']
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()
