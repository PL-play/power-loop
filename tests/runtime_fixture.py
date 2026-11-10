"""Test composition of the public message runtime; no direct execution entry."""
import asyncio
from dataclasses import fields

from power_loop import (
    AgentHooks,
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
    def __init__(self, store, model, registry=None, config=None, *, task_runtime=None, hooks=None, runtime_env=None, process_control=None, capacity=1, command=None, event_bus=None, allowed_models=None, coordinator=None, transport=None):
        self.owns_transport = transport is None
        self.config = config or AgentLoopConfig(compactor=None)
        self.hooks = hooks or AgentHooks()
        self.model = model
        self.registry = registry
        self.task_runtime = task_runtime
        self.executions = {}
        self.execution_started = asyncio.Condition()
        factory = RegistryRuntimeFactory()
        async def build(descriptor, context):
            self.executions[context.input_ref] = asyncio.current_task()
            async with self.execution_started:
                self.execution_started.notify_all()
            return ExecutionEnvironment(model, registry, self.config, hooks=self.hooks, task_runtime=task_runtime,
                runtime_env=runtime_env, process_control=process_control, event_bus=event_bus,
                command=command if "child_spec" not in descriptor.config else None)
        factory.register('fixture.agent@1', build)
        self.runtime = Runtime(store=store, factory=factory, node_id='fixture', boot_id='boot',
                               capacity=capacity, recovery_interval_s=None, coordinator=coordinator, transport=transport)
        self.loop = StatefulAgentLoop(runtime=self.runtime)
        self.actor = TrustedActor(task_runtime.tenant_id if task_runtime else 'test',
                                  task_runtime.principal_id if task_runtime else 'host', None)
        options = {f.name: getattr(self.config, f.name) for f in fields(RequestOptions)
                   if hasattr(self.config, f.name)}
        if options.get('model'):
            options['allowed_models'] = allowed_models if allowed_models is not None else [options['model']]
        self.descriptor = RuntimeDescriptor(1, 'agent', 'fixture.agent@1', None,
            'fixture.tools@1', 'fixture.hooks@1', 'fixture.resources@1', plain(options), {})
        self.counter = 0

    async def start(self, *, target=None):
        await self.runtime.worker.start()
        self.target = target if target is not None else await self.loop.new_session(self.descriptor, actor=self.actor)
        return self

    async def send(self, body, *, options=None, message_id=None):
        self.counter += 1
        result = await self.loop.send(self.target, body, message_id=message_id or f'm{self.counter}',
                                     actor=self.actor, deadline_ms=now_ms() + 120000, options=options)
        # Include resource close, ownership release and final activation check in
        # performance measurements, not just the result transaction.
        await self.finish(result.input_ref, failed=result.status in {"failed", "unknown", "cancelled"})
        return result

    async def finish(self, input_ref, *, failed=False):
        task = self.executions.pop(input_ref, None)
        if task:
            await asyncio.gather(task, return_exceptions=failed)

    async def drain(self, *, after_input=None):
        if after_input is not None:
            async with asyncio.timeout(15), self.execution_started:
                await self.execution_started.wait_for(lambda: after_input in self.executions)
            await self.finish(after_input)
        while self.runtime.worker._tasks:
            await asyncio.wait(tuple(self.runtime.worker._tasks))
        self.executions = {ref: task for ref, task in self.executions.items() if not task.done()}

    async def aclose(self):
        await self.runtime.worker.shutdown(timeout_s=2, cancel_running=True)
        await self.loop.aclose()
        await self.runtime.client.aclose()
        if self.owns_transport:
            await self.runtime.transport.aclose()
        self.executions.clear()
