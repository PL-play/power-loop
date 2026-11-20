import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop import MemoryMessageTransport, RegistryRuntimeFactory, Runtime, RuntimeClient, RuntimeRequestError
from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator


async def child(store):
    kind = store._db.dialect.name
    dsn = store._db.path if kind == 'sqlite' else os.environ['POWER_LOOP_TASK_PG_DSN' if kind == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.process_guard_worker_208',
        env={**os.environ, 'P7A_DB_DSN': dsn, 'P7A_TABLE_PREFIX': store.table_prefix},
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    async with asyncio.timeout(20):
        line = (await process.stdout.readline()).decode().strip()
    return process, line


async def finish(process):
    if process.returncode is None:
        process.stdin.write(b'close\n')
        await process.stdin.drain()
    async with asyncio.timeout(15):
        _, error = await process.communicate()
    assert process.returncode == 0, error.decode()


async def test_two_default_processes_and_release(registry_store):
    store, _ = registry_store
    first, status = await child(store)
    try:
        assert status == 'ready'
        second, status = await child(store)
        assert status == 'invalid_options'
        await second.wait()
        assert second.returncode == 0, (await second.stderr.read()).decode()
        client = RuntimeClient(store=store, transport=MemoryMessageTransport())
        assert client._accepting
        await client.aclose()
    finally:
        await finish(first)
    third, status = await child(store)
    assert status == 'ready'
    await finish(third)


async def test_shared_memory_domain_refcount(registry_store):
    store, connect = registry_store
    other = await connect()
    coordinator, transport = MemorySessionOwnershipCoordinator(), MemoryMessageTransport()
    a = Runtime(store=other, factory=RegistryRuntimeFactory(), node_id='a', coordinator=coordinator, transport=transport, timers_enabled=False)
    b = Runtime(store=store, factory=RegistryRuntimeFactory(), node_id='b', coordinator=coordinator, transport=transport, timers_enabled=False)
    await a.worker.start()
    await b.worker.start()
    await a.worker.shutdown()
    # A guard shared by live workers must not pin the first worker's data pool.
    await asyncio.wait_for(other.close(), 2)
    c = Runtime(store=store, factory=RegistryRuntimeFactory(), node_id='c', timers_enabled=False)
    with pytest.raises(RuntimeRequestError, match='coordination domain'):
        await c.worker.start()
    await b.worker.shutdown()
    await c.worker.start()
    await c.worker.shutdown()
    for runtime in (a,b,c):
        await runtime.client.aclose()
    await transport.aclose()
    await c.transport.aclose()


async def _input_state(port, ref):
    return await port.db.fetchone(
        f"SELECT {port._now_sql()} AS db_now,s.execution_state,s.active_send_index,"
        "s.owner_node_id,s.owner_epoch,s.owner_lease_until_ms,"
        "q.status AS input_status,q.terminal_status,q.completed_at "
        f"FROM {port.state} s JOIN {port.q} q ON q.session_id=s.session_id "
        "WHERE s.session_id=? AND q.item_id=?",
        (ref.session_id, ref.message_id),
    )


@pytest.mark.parametrize('active', [False, True])
async def test_lock_connection_loss_stops_admission(registry_store, active):
    from power_loop import AgentLoopConfig, ExecutionEnvironment, LLMResponse, LLMService
    from power_loop.agent.runtime_store import now_ms
    from power_loop.agent.runtime_types import InputRef, plain
    from power_loop.core.execution_context import get_execution_context

    from .session_native_fixture import fixture

    store, _ = registry_store
    port, target, actor = await fixture(store)
    entered, stopped = asyncio.Event(), asyncio.Event()
    next_text = 'next input done after takeover'

    class Model(LLMService):
        def __init__(self, *, block):
            self.block = block
            self.inputs = []

        async def complete(self, request, **kwargs):
            self.inputs.append(get_execution_context().input_ref)
            if self.block:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()
            return LLMResponse(raw_text=next_text)

        async def close(self):
            pass

    def make_runtime(node, model, **settings):
        factory = RegistryRuntimeFactory()

        async def build(descriptor, context):
            return ExecutionEnvironment(model, None, AgentLoopConfig(context_budget_tokens=10**9))

        factory.register('test.agent@215', build)
        return Runtime(store=store, factory=factory, node_id=node, timers_enabled=False, **settings)

    old_model, next_model = Model(block=True), Model(block=False)
    old = make_runtime('lost', old_model, **({'session_lease_ttl_s': 15} if active else {}))
    successor = None
    evidence = {'db': store._db.dialect.name, 'old_lease_ttl_s': old.session_lease_ttl_s}
    try:
        await old.worker.start()
        if active:
            receipt = await old.client.send(target, 'hold', message_id='hold', actor=actor)
            ref = receipt.input_ref
            evidence['original_receipt'] = plain(receipt)
            assert receipt.newly_accepted and ref == InputRef(store.store_id, target.session_id, 'hold')
            await asyncio.wait_for(entered.wait(), 5)
            running = evidence['before_loss'] = await _input_state(port, ref)
            assert running['execution_state'] == 'running' and running['owner_node_id'] == old.node_id
            assert running['input_status'] == 'delivered' and running['terminal_status'] is None

        guard = old.process_guard.entry
        if guard.kind == 'sqlite':
            guard.file.close()
        elif guard.kind == 'postgres':
            await guard.connection.close()
        else:
            guard.connection.close()
        monitor = old.worker._guard_monitor
        await asyncio.wait_for(asyncio.shield(monitor), 10)
        assert monitor.done() and old.worker.last_error.startswith('store_lock_lost')
        assert not old.client._accepting
        with pytest.raises(RuntimeRequestError, match='session_closed') as refused:
            await old.client.send(target, 'new', message_id='after-loss', actor=actor)
        if not active:
            return

        evidence['local_stop'] = {
            'model_stopped': stopped.is_set(), 'monitor_done': monitor.done(),
            'last_error': old.worker.last_error, 'accepting': old.client._accepting,
            'refusal_code': refused.value.code, 'shutdown_result': plain(old.worker.shutdown_result),
        }
        assert stopped.is_set() and ref in old.worker.shutdown_result.interrupted_inputs
        stopped_state = evidence['after_local_stop'] = await _input_state(port, ref)
        expires = stopped_state['owner_lease_until_ms']
        assert stopped_state['owner_epoch'] == running['owner_epoch']
        assert stopped_state['db_now'] < expires
        assert stopped_state['terminal_status'] is None

        successor = make_runtime('successor', next_model, recovery_interval_s=1)
        await successor.worker.start()
        started = evidence['successor_started'] = await _input_state(port, ref)
        evidence['successor_recovery_interval_s'] = successor.recovery_interval_s
        assert successor.client._accepting and successor.client is not old.client
        assert successor.transport is not old.transport and successor.coordinator is not old.coordinator
        assert started['owner_epoch'] == running['owner_epoch'] and started['owner_node_id'] == old.node_id
        assert started['owner_lease_until_ms'] == expires and started['db_now'] < expires
        assert started['execution_state'] == 'running' and started['active_send_index'] == running['active_send_index']
        assert started['input_status'] == 'delivered' and started['terminal_status'] is None
        assert old_model.inputs == [ref] and next_model.inputs == []
        evidence['calls_at_successor_start'] = {'old': len(old_model.inputs), 'successor': len(next_model.inputs)}

        # SQLite's database clock is truncated to whole seconds. Wait once using
        # the remaining database lease, then give public wait its original 5 s.
        await asyncio.sleep((expires - started['db_now']) / 1000)
        observation = evidence['at_wait_start'] = await _input_state(port, ref)
        assert observation['db_now'] >= expires
        deadline = now_ms() + 5000
        evidence['wait_deadline_ms'] = deadline
        result = await successor.client.wait(ref, actor=actor, deadline_ms=deadline)
        evidence['original_result'] = plain(result)
        recovered = evidence['after_recovery'] = await _input_state(port, ref)
        assert result.input_ref == ref and result.target == target and result.status == 'interrupted'
        assert recovered['execution_state'] == 'idle' and recovered['active_send_index'] is None
        assert recovered['owner_epoch'] > running['owner_epoch'] and recovered['owner_node_id'] is None
        assert recovered['terminal_status'] == 'interrupted' and recovered['completed_at'] >= expires
        assert old_model.inputs == [ref] and next_model.inputs == []
        evidence['calls_after_recovery'] = {'old': len(old_model.inputs), 'successor': len(next_model.inputs)}

        next_receipt = await successor.client.send(target, 'continue after takeover', message_id='next', actor=actor)
        next_ref = next_receipt.input_ref
        evidence['next_receipt'] = plain(next_receipt)
        assert next_receipt.newly_accepted and next_ref == InputRef(store.store_id, target.session_id, 'next')
        next_result = await successor.client.wait(next_ref, actor=actor, deadline_ms=now_ms() + 5000)
        evidence['next_result'] = plain(next_result)
        assert next_result.input_ref == next_ref and next_result.target == target
        assert next_result.status == 'completed' and next_result.final_text == next_text
        assert old_model.inputs == [ref] and next_model.inputs == [next_ref]
    except Exception as exc:
        evidence['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        evidence['model_inputs'] = {'old': plain(old_model.inputs), 'successor': plain(next_model.inputs)}
        try:
            for runtime in (successor, old):
                if runtime is not None:
                    await runtime.worker.shutdown(timeout_s=2, cancel_running=True)
                    await runtime.client.aclose()
                    await runtime.transport.aclose()
        finally:
            if active and (out := os.environ.get('R15_OUT')):
                path = Path(out) / 'guard' / f"takeover-{evidence['db']}.json"
                with path.open('x') as stream:
                    json.dump(evidence, stream, indent=2)
                    stream.write('\n')
