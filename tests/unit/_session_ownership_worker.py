"""Drive the same Runtime in two independent interpreters and real shared SQL."""
import asyncio
import json
import os
import signal
import sys

from power_loop import (
    AgentLoopConfig,
    DatabasePollingTransport,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    SessionLeaseLost,
    ToolDefinition,
)
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import InputRef, TargetRef, TrustedActor
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.factory import open_store
from power_loop.testing.db_session_ownership import DatabaseSessionOwnershipCoordinator
from power_loop.tools.registry import ToolRegistry


def report(**data):
    print(json.dumps(data), flush=True)


async def main():
    dsn, prefix, sid, path, role = sys.argv[1:]
    store = await open_store(dsn, table_prefix=prefix, schema='tasks')
    if role.startswith('acquire-'):
        from power_loop.runtime.session_ownership import hold_session_ownership

        from .test_session_ownership_tasks import holder
        class Delayed(DatabaseSessionOwnershipCoordinator):
            grants = []
            async def acquire(self, key, **kwargs):
                if role == 'acquire-old' and not self.grants:
                    report(event='acquire_pending')
                    assert (await asyncio.to_thread(sys.stdin.readline)).strip() == 'resume'
                grant = await super().acquire(key, **kwargs)
                self.grants.append(grant)
                return grant
        coordinator = Delayed(path)
        try:
            async with hold_session_ownership(holder(store, coordinator, role), sid) as lease:
                report(event='installed', epoch=lease.epoch, grants=len(coordinator.grants),
                       first_released=coordinator.grants[0].released)
        finally:
            await store.close()
        return
    registry = ToolRegistry()
    stopped = asyncio.Event()

    async def hold():
        report(event='running')
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            denied = []
            for action in (
                lambda: store.append_message(sid, role='assistant', content='late-old-write'),
                lambda: store.set_pending(sid, None),
                lambda: store.set_round_index(sid, 999),
                lambda: store.set_runtime_state(sid, 'late-state', {'value': 'old'}),
            ):
                try:
                    await action()
                except SessionLeaseLost:
                    denied.append(True)
                else:
                    denied.append(False)
            report(event='old_write_denied', denied=denied)
            stopped.set()
            raise

    registry.register(ToolDefinition('hold', 'Wait', input_schema={'type': 'object', 'properties': {}},
                                     pricing=Free('test')), hold)

    class Model(LLMService):
        calls = 0
        async def complete(self, request, **kwargs):
            self.calls += 1
            if role == 'old':
                return LLMResponse(raw_text='', tool_calls=[{'id': 'original-call', 'type': 'function',
                    'function': {'name': 'hold', 'arguments': '{}'}}])
            assert any(m.get('content') == 'original' for m in request.messages)
            assert any(m.get('role') == 'tool' and m.get('tool_call_id') == 'original-call'
                       and '未知' in m.get('content', '') for m in request.messages)
            return LLMResponse(raw_text='new owner finished')
        def stream(self, request):
            raise AssertionError('nonstream test')
        async def close(self):
            pass

    model = Model()
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(model, registry, AgentLoopConfig(compactor=None, max_rounds=2))
    factory.register('test.agent@1', build)
    runtime = Runtime(store=store, factory=factory, node_id=role, boot_id=role,
        coordinator=DatabaseSessionOwnershipCoordinator(path),
        transport=DatabasePollingTransport(RuntimeInputStore(store), interval_s=.03, control_interval_s=.03),
        session_lease_ttl_s=.6, recovery_interval_s=None, shared_execution=True)
    target = TargetRef(store.store_id, sid)
    actor = TrustedActor('tenant', 'principal', None)
    try:
        await runtime.worker.start()
        if role == 'old':
            await runtime.client.send(target, 'original', message_id='old-input', actor=actor)
            assert (await asyncio.to_thread(sys.stdin.readline)).strip() == 'pause'
            report(event='pausing')
            os.kill(os.getpid(), signal.SIGSTOP)
            await asyncio.wait_for(stopped.wait(), 10)
            await runtime.worker.shutdown(timeout_s=3, cancel_running=True)
            report(event='stopped')
        else:
            report(event='ready')
            receipt = None
            while (command := (await asyncio.to_thread(sys.stdin.readline)).strip()) != 'quit':
                if command == 'try':
                    receipt = await runtime.client.send(target, 'competing', message_id='new-input', actor=actor)
                    await asyncio.sleep(.15)
                    report(event='busy', calls=model.calls)
                elif command == 'takeover':
                    await runtime.worker.recover()
                    result = await runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=now_ms()+10000)
                    interrupted = await runtime.client.wait(InputRef(store.store_id, sid, 'old-input'),
                        actor=actor, deadline_ms=now_ms()+1000)
                    assert interrupted.status == 'interrupted' and interrupted.job_ref.generation == 1
                    state = await store.get_state(sid)
                    report(event='healed', status=result.status, pending=state.pending, calls=model.calls)
                else:
                    raise AssertionError(command)
    finally:
        await runtime.aclose(timeout_s=3, cancel_running=True)


if __name__ == '__main__':
    asyncio.run(main())
