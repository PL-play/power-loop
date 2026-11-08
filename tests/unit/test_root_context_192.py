"""Host provenance is reconstructed from the accepted descriptor, never caller task locals."""
import asyncio
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    TrustedActor,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import plain
from power_loop.runtime import root_context
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.task_runtime import _SEND, TaskRuntime

from .test_job_registry_192 import registry_store

__all__ = ['registry_store', 'root_case']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
NAMESPACE = "deeptalk.agent-run.v1"


class Model(LLMService):
    def __init__(self):
        self.calls, self.before = [], None
    async def complete(self, request, **kwargs):
        binding = _SEND.get()
        if self.before is not None:
            await self.before(binding)
        self.calls.append(binding.lease.job_id)
        return LLMResponse(raw_text='DONE', token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
    async def close(self):
        pass


@pytest.fixture
async def root_case(registry_store):
    store, connect = registry_store
    manager, model = JobManager(store), Model()
    task_runtime = TaskRuntime(manager, tenant_id='root-context', principal_id='host', executor_id='executor', boot_id='boot')
    factory = RegistryRuntimeFactory()
    scopes = []
    class Resources:
        def __init__(self, scope):
            self.scope = scope
        async def aclose(self):
            self.scope.close()
    async def build(descriptor, context):
        scope = ExitStack()
        if 'origin' in descriptor.config:
            origin = descriptor.config['origin']
            scope.enter_context(task_runtime.root_context(context.session_id,
                namespace=origin['namespace'], payload=plain(origin['payload'])))
        scopes.append(scope)
        return ExecutionEnvironment(model, None, config=AgentLoopConfig(max_rounds=2, compactor=None),
            task_runtime=task_runtime, resources=Resources(scope))
    factory.register('root@1', build)
    worker = Runtime(store=store, factory=factory, node_id='root', boot_id='boot')
    actor = TrustedActor('root-context', 'host', None)
    descriptor = RuntimeDescriptor(1, 'agent', 'root@1', None, 'builtin.empty@1', 'builtin.none@1', 'builtin.none@1', {}, {})
    await worker.worker.start()
    target = await worker.open_session(descriptor, actor=actor)
    counter = 0
    async def submit(body='original work', *, source=None, message_id=None, target_override=None):
        nonlocal counter
        counter += 1
        frozen = replace(descriptor, config={'origin': source} if source else {})
        return await worker.client.send(target_override or target, body, message_id=message_id or f'root-{counter}', actor=actor, descriptor=frozen)
    async def send(body='original work', *, source=None, message_id=None):
        receipt = await submit(body, source=source, message_id=message_id)
        result = await worker.client.wait(receipt.input_ref, actor=actor, deadline_ms=now_ms()+15000)
        return result
    case = SimpleNamespace(store=store, connect=connect, manager=manager, runtime=task_runtime,
        worker=worker, model=model, actor=task_runtime.actor(target.session_id), trusted_actor=actor,
        target=target, sid=target.session_id, descriptor=descriptor, send=send, submit=submit)
    try:
        yield case
    finally:
        await worker.worker.shutdown(timeout_s=2, cancel_running=True)
        await worker.client.aclose()
        await worker.transport.aclose()


def source(run_id):
    return {'namespace':NAMESPACE, 'payload':{'origin_run_id':run_id}}


async def test_original_root_is_saved_before_provider_and_not_replaced_by_next_run(root_case):
    c, observed = root_case, []
    async def before(binding):
        observed.append(await root_context.read(c.manager, c.actor, binding.lease.job_id, namespace=NAMESPACE))
    c.model.before = before
    metadata = source(101)
    receipt = await c.submit(source=metadata)
    metadata['payload']['origin_run_id'] = 999
    first = await c.worker.client.wait(receipt.input_ref, actor=c.trusted_actor, deadline_ms=now_ms()+15000)
    assert first.status == 'completed'
    assert (await c.send(source=source(202))).status == 'completed'
    assert [value['payload']['origin_run_id'] for value in observed] == [101,202]
    assert observed[0]['root_job_id'] != observed[1]['root_job_id']
    assert not observed[0]['execution_authority']
    reopened = await c.connect()
    try:
        saved = await root_context.read(JobManager(reopened), c.actor, c.model.calls[0])
        assert saved == observed[0]
        saved['payload']['origin_run_id'] = 777
        assert (await root_context.read(c.manager, c.actor, c.model.calls[0]))['payload']['origin_run_id'] == 101
    finally:
        await reopened.close()
    assert len(c.model.calls) == 2


async def test_missing_context_and_foreign_actor_are_explicit(root_case):
    c = root_case
    assert (await c.send()).status == 'completed'
    job = c.model.calls[0]
    assert await root_context.read(c.manager, c.actor, job, required=False) is None
    with pytest.raises(JobError, match='root_context_missing'):
        await root_context.read(c.manager, c.actor, job)
    assert (await c.send(source=source(101))).status == 'completed'
    with pytest.raises(JobError, match='access_denied'):
        await root_context.read(c.manager, JobActor('root-context','foreign',c.sid), c.model.calls[-1])
    with pytest.raises(JobError, match='namespace_mismatch'):
        await root_context.read(c.manager, c.actor, c.model.calls[-1], namespace='different')
    assert len(c.model.calls) == 2


async def test_model_cannot_replace_host_context_and_next_input_does_not_inherit_it(root_case):
    c = root_case
    async def before(binding):
        with pytest.raises(JobError, match='root_context_requires_host'):
            with c.runtime.root_context(c.sid, **source(999)):
                pytest.fail('model scope cannot bind host attribution')
    c.model.before = before
    assert (await c.send(source=source(101))).status == 'completed'
    assert (await root_context.read(c.manager, c.actor, c.model.calls[0]))['payload'] == source(101)['payload']
    assert (await c.send()).status == 'completed'
    assert await root_context.read(c.manager, c.actor, c.model.calls[1], required=False) is None
    assert root_context.for_send(c.runtime, c.sid) is None


async def test_cancelled_input_retains_saved_origin_and_allows_new_input(root_case):
    c = root_case
    entered = asyncio.Event()
    async def before(binding):
        entered.set()
        await asyncio.Event().wait()
    c.model.before = before
    receipt = await c.submit(source=source(101))
    await asyncio.wait_for(entered.wait(), 5)
    await c.worker.client.cancel(receipt.input_ref, actor=c.trusted_actor, reason='user stop')
    result = await c.worker.client.wait(receipt.input_ref, actor=c.trusted_actor, deadline_ms=now_ms()+15000)
    assert result.status == 'cancelled'
    assert (await root_context.read(c.manager, c.actor, result.job_ref.job_id))['payload'] == source(101)['payload']
    c.model.before = None
    assert (await c.send(source=source(202))).status == 'completed'
    assert root_context.for_send(c.runtime, c.sid) is None


@pytest.mark.parametrize("payload", [{}, {1: "ambiguous"}, {"n": float("nan")}, {"n": 1.0},
    {"n": b"not-json"}, {"n": {1}}, {"n": "\ud800"}, {"n": "x\x00y"},
    {"n": "x" * 16385}, {"n": 2**63}])
async def test_invalid_metadata_is_rejected_before_job_creation(payload):
    runtime = SimpleNamespace(actor=lambda sid: JobActor("tenant", "host", sid))
    with pytest.raises(JobError, match="invalid_root_context_payload"):
        root_context.bind(runtime, "main", namespace=NAMESPACE, payload=payload)


@pytest.mark.parametrize("namespace", ["", " leading", "BAD", "a/parent", "x" * 129, True, None])
async def test_namespace_is_an_exact_host_contract(namespace):
    runtime = SimpleNamespace(actor=lambda sid: JobActor("tenant", "host", sid))
    with pytest.raises(JobError, match="invalid_root_context_namespace"):
        root_context.bind(runtime, "main", namespace=namespace, payload={"run_id": 101})


async def test_recursive_context_never_enters_the_host_binding():
    runtime = SimpleNamespace(actor=lambda sid: JobActor("tenant", "host", sid))
    payload = {}
    payload["cycle"] = payload
    with pytest.raises(JobError, match="invalid_root_context_payload"):
        root_context.bind(runtime, "main", namespace=NAMESPACE, payload=payload)
    assert root_context.for_send(runtime, "main") is None
