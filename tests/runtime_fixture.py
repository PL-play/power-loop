"""Test composition of the public message runtime; no direct execution entry."""
import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, fields, replace
from typing import Any

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
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import plain


@dataclass
class ScriptedModel(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done")
        response = self.responses[self._idx]
        self._idx += 1
        return response

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return empty()

    async def close(self) -> None:
        return None


class RunningAgent:
    def __init__(self, store, model, registry=None, config=None, *, hooks=None, runtime_env=None, process_control=None, capacity=1, command=None, event_bus=None, allowed_models=None, coordinator=None, transport=None, options=None, fuse_local_send=True):
        self.owns_transport = transport is None
        self.config = config or AgentLoopConfig(context_budget_tokens=10**9)
        self.hooks = hooks or AgentHooks()
        self.model = model
        self.registry = registry
        self.executions = {}
        self.execution_started = asyncio.Condition()
        factory = RegistryRuntimeFactory()
        async def build(descriptor, context):
            self.executions[context.input_ref] = asyncio.current_task()
            async with self.execution_started:
                self.execution_started.notify_all()
            return ExecutionEnvironment(model, registry, self.config, hooks=self.hooks,
                runtime_env=runtime_env, process_control=process_control, event_bus=event_bus,
                command=command if "child_spec" not in descriptor.config else None)
        factory.register('fixture.agent@215', build)
        self.runtime = Runtime(store=store, factory=factory, node_id='fixture', boot_id='boot',
                               capacity=capacity, recovery_interval_s=None, coordinator=coordinator,
                               transport=transport, fuse_local_send=fuse_local_send)
        self.loop = StatefulAgentLoop(runtime=self.runtime)
        self.actor = TrustedActor('test', 'host', None)
        descriptor_config = {f.name: getattr(self.config, f.name) for f in fields(RequestOptions)
                             if hasattr(self.config, f.name)}
        if options is not None:
            descriptor_config.update({k: v for k, v in plain(options).items() if v is not None})
        self.descriptor = RuntimeDescriptor(2, 'agent', 'fixture.agent@215', None,
            'fixture.tools@215', 'fixture.hooks@215', 'fixture.resources@215', plain(descriptor_config), {})
        self.counter = 0

    async def start(self, *, target=None):
        await self.runtime.worker.start()
        self.target = target if target is not None else await self.loop.new_session(self.descriptor, actor=self.actor)
        return self

    async def new_session(self, *, options):
        descriptor = replace(self.descriptor, config={**dict(self.descriptor.config),
            **{k: v for k, v in plain(options).items() if v is not None}})
        self.target = await self.loop.new_session(descriptor, actor=self.actor)
        return self.target

    async def send(self, body, *, message_id=None, priority="normal"):
        self.counter += 1
        result = await self.loop.send(self.target, body, message_id=message_id or f'm{self.counter}',
                                     actor=self.actor, deadline_ms=now_ms() + 120000, priority=priority)
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
