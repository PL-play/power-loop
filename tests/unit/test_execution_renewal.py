"""Session renewal is the only execution clock; hooks cannot defeat its loss."""
import asyncio
from contextlib import asynccontextmanager

import pytest

from power_loop import (
    AgentHooks,
    AgentLoopConfig,
    ExecutionEnvironment,
    HookDirective,
    HookPoint,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    SessionLeaseLost,
    ToolDefinition,
)
from power_loop.runtime.pricing import Free
from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_runtime_messaging_203 import fixture

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class WaitingModel(LLMService):
    def __init__(self):
        self.entered, self.cancelled, self.release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.calls = 0
    async def complete(self, request, **kwargs):
        self.calls += 1
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return LLMResponse(raw_text='done')
    def stream(self, request):
        raise AssertionError('nonstream test')
    async def close(self):
        pass


@asynccontextmanager
async def running(store, coordinator, *, hooks=None):
    model, factory = WaitingModel(), RegistryRuntimeFactory()
    async def build(descriptor, context):
        assert context.owner_epoch > 0 and context.owner_token
        return ExecutionEnvironment(model, ToolRegistry(), AgentLoopConfig(compactor=None))
    factory.register('test.agent@1', build)
    runtime = Runtime(store=store, factory=factory, node_id='test', coordinator=coordinator,
        session_lease_ttl_s=.3, recovery_interval_s=None, hooks=hooks, hook_timeout_s=.03)
    _, target, actor = await fixture(store)
    await runtime.worker.start()
    try:
        await runtime.client.send(target, 'start', message_id='input', actor=actor)
        await asyncio.wait_for(model.entered.wait(), 5)
        yield runtime, model, target, actor
    finally:
        model.release.set()
        await runtime.worker.shutdown(timeout_s=2, cancel_running=True)
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_only_session_renews_without_writing_job_heartbeats(registry_store):
    store, _ = registry_store
    class Counting(MemorySessionOwnershipCoordinator):
        renewals = 0
        async def renew(self, lease, **kwargs):
            self.renewals += 1
            return await super().renew(lease, **kwargs)
    coordinator = Counting()
    async with running(store, coordinator) as (runtime, model, target, actor):
        before = await store._db.fetchall(f'SELECT * FROM {store.table_prefix}jobs')
        await asyncio.sleep(.45)
        after = await store._db.fetchall(f'SELECT * FROM {store.table_prefix}jobs')
        assert after == before and coordinator.renewals >= 2
        assert not model.cancelled.is_set() and model.calls == 1


@pytest.mark.parametrize('failure', ['false', 'exception', 'timeout'])
async def test_renewal_failure_cancels_before_broken_lost_hook(registry_store, failure, caplog):
    store, _ = registry_store
    class Broken(MemorySessionOwnershipCoordinator):
        broken = False
        async def renew(self, lease, **kwargs):
            if not self.broken:
                return await super().renew(lease, **kwargs)
            if failure == 'exception':
                raise OSError('shared coordinator unavailable')
            if failure == 'timeout':
                await asyncio.Event().wait()
            return False
    hooks, noticed = AgentHooks(), asyncio.Event()
    async def lost(context):
        noticed.set()
        await asyncio.Event().wait()
    hooks.register(HookPoint.EXECUTION_LOST, lost)
    async with running(store, Broken(), hooks=hooks) as (runtime, model, target, actor):
        runtime.coordinator.broken = True
        await asyncio.wait_for(model.cancelled.wait(), 2)
        await asyncio.wait_for(noticed.wait(), 2)
        await asyncio.sleep(.06)
        assert model.calls == 1
        assert 'execution notification failed' in caplog.text
        assert not runtime.coordinator._leases


@pytest.mark.parametrize('directive', [HookDirective.SKIP, HookDirective.SHORT_CIRCUIT])
async def test_tool_error_hook_never_replaces_or_retries_lease_loss(registry_store, directive):
    store, _ = registry_store
    calls, invoked = [], []
    class Model(WaitingModel):
        async def complete(self, request, **kwargs):
            self.calls += 1
            return LLMResponse(raw_text='', tool_calls=[{'id': 'lost', 'type': 'function',
                'function': {'name': 'lose', 'arguments': '{}'}}])
    async def lose():
        calls.append('tool')
        raise SessionLeaseLost('test')
    def replace_error(ctx):
        invoked.append(ctx.error)
        ctx.output = 'success'
        return directive
    hooks, registry, model = AgentHooks(), ToolRegistry(), Model()
    hooks.register(HookPoint.TOOL_ERROR, replace_error)
    registry.register(ToolDefinition('lose', 'lose permission', input_schema={'type': 'object', 'properties': {}},
                                     pricing=Free('test')), lose)
    agent = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None), hooks=hooks).start()
    try:
        result = await agent.send('go')
        assert result.status == 'failed' and result.error == 'SessionLeaseLost', result
        assert calls == ['tool'] and not invoked and model.calls == 1
    finally:
        await agent.aclose()
