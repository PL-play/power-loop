"""Explicit local Runtime composition shared by runnable examples.

The host owns the provider for this demo; workers borrow it per execution.
Deployment factories should resolve their own credentials/resources by profile.
"""
from contextlib import asynccontextmanager

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    SessionStore,
    StatefulAgentLoop,
    TrustedActor,
)


class BorrowedModel:
    def __init__(self, provider):
        self.provider = provider

    def __getattr__(self, name):
        return getattr(self.provider, name)

    async def close(self):
        pass


@asynccontextmanager
async def demo_runtime(model, registry=None, *, config=None, command=None, store=None, runtime_env=None, hooks=None, event_bus=None, capacity=1, target=None):
    store = store or await SessionStore.open(':memory:', schema='tasks')
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(BorrowedModel(model), registry, config or AgentLoopConfig(compactor=None),
            command=command if 'child_spec' not in descriptor.config else None, runtime_env=runtime_env, hooks=hooks, event_bus=event_bus)
    factory.register('example@1', build, profiles=('example.tools@1',))
    runtime = Runtime(store=store, factory=factory, node_id='example', capacity=capacity,
                      required_profiles=['example@1', 'example.tools@1'])
    facade = StatefulAgentLoop(runtime=runtime)
    try:
        await runtime.worker.start()
        actor = TrustedActor('example', 'host', None)
        target = target or await facade.new_session(RuntimeDescriptor(1, 'agent', 'example@1', None,
            'example.tools@1', 'builtin.none@1', 'builtin.none@1', {}, {}), actor=actor)
        yield runtime, facade, target, actor
    finally:
        await facade.aclose()
        await runtime.aclose(cancel_running=True)
        await model.close()
