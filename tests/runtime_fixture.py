"""Test composition of the public message runtime; no direct execution entry."""
import asyncio
from dataclasses import fields

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    RegistryRuntimeFactory,
    RequestOptions,
    Runtime,
    RuntimeDescriptor,
    StatefulAgentLoop,
    TrustedActor,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import plain


class RunningAgent:
    def __init__(self, store, model, registry=None, config=None, *, task_runtime=None, hooks=None):
        self.config = config or AgentLoopConfig(compactor=None)
        self.executions = {}
        factory = RegistryRuntimeFactory()
        async def build(descriptor, context):
            self.executions[context.input_ref] = context.task
            return ExecutionEnvironment(model, registry, self.config, hooks=hooks, task_runtime=task_runtime)
        factory.register('fixture.agent@1', build)
        self.runtime = Runtime(store=store, factory=factory, node_id='fixture', boot_id='boot',
                               capacity=1, recovery_interval_s=None)
        self.loop = StatefulAgentLoop(runtime=self.runtime)
        self.actor = TrustedActor(task_runtime.tenant_id if task_runtime else 'test',
                                  task_runtime.principal_id if task_runtime else 'host', None)
        options = {f.name: getattr(self.config, f.name) for f in fields(RequestOptions)
                   if hasattr(self.config, f.name)}
        if options.get('model'):
            options['allowed_models'] = [options['model']]
        self.descriptor = RuntimeDescriptor(1, 'agent', 'fixture.agent@1', None,
            'fixture.tools@1', 'fixture.hooks@1', 'fixture.resources@1', plain(options), {})
        self.counter = 0

    async def start(self):
        await self.runtime.worker.start()
        self.target = await self.loop.new_session(self.descriptor, actor=self.actor)
        return self

    async def send(self, body, *, options=None, message_id=None):
        self.counter += 1
        result = await self.loop.send(self.target, body, message_id=message_id or f'm{self.counter}',
                                     actor=self.actor, deadline_ms=now_ms() + 120000, options=options)
        # Include resource close, ownership release and final activation check in
        # performance measurements, not just the result transaction.
        await self.finish(result.input_ref)
        return result

    async def finish(self, input_ref):
        task = self.executions.pop(input_ref, None)
        if task:
            await asyncio.gather(task)

    async def drain(self):
        while self.runtime.worker._tasks:
            await asyncio.gather(*tuple(self.runtime.worker._tasks))

    async def aclose(self):
        await self.runtime.worker.shutdown(timeout_s=2, cancel_running=True)
        await self.loop.aclose()
        await self.runtime.client.aclose()
        await self.runtime.transport.aclose()
        self.executions.clear()
