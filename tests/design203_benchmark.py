"""P0 fixed-provider benchmark, run with python -m tests.design203_benchmark.

No WriteMeter/Profiled while timing. Warm 10, three paired 34-request batches.
Each arm is the same old implementation in P0; later compare N to frozen B.
Setup/warmup/cleanup excluded; operation includes result persistence/consumption.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import statistics
import time
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    StatefulAgentLoop,
    ToolDefinition,
    record_billing_receipt,
)
from power_loop.core import agent_context
from power_loop.runtime import delegation
from power_loop.runtime.budget_contracts import BudgetConfig, BudgetPlan, LocalBudgetScope
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import BillingReceipt, Free, Money
from power_loop.runtime.response_capture import ResponseCaptureReceipt
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.schema import _STORE_TABLES
from power_loop.runtime.store.task_schema import TASK_TABLES
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow import create_workflow
from tests.unit.test_model_budgets_192 import Host
from tests.unit.test_nested_tool_billing_196 import Paid

SCENARIOS = ('send', 'paid', 'append', 'cancel', 'child', 'workflow', 'stateless', 'same_session')


def rss():
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


class Model(LLMService):
    def __init__(self, rounds=1):
        self.rounds, self.calls = rounds, 0
        self.hold = False
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.cancel_seen = None

    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.hold and self.calls == 1:
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancel_seen = time.perf_counter()
                raise
        calls = None
        if self.calls < self.rounds:
            calls = [{'id': f'call-{self.calls}', 'type': 'function',
                      'function': {'name': 'echo', 'arguments': '{"text":"ok"}'}}]
        return LLMResponse(raw_text='done' if calls is None else '', tool_calls=calls,
                           token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))

    def stream(self, request):
        raise AssertionError('fixed nonstream model')

    async def close(self):
        pass


@asynccontextmanager
async def simple(store, lane, scenario):
    paid = scenario == 'paid'
    model, registry, host = Model(4 if paid else 1), ToolRegistry(), Host()

    async def capture(packet):
        return ResponseCaptureReceipt(packet.capture_id, packet.fingerprint, 'fixture-original')

    async def tool_plan(context, call, *, pricing):
        return BudgetPlan(0, {'attempt_id': context['attempt_id'], 'origin': context}) if paid else None

    async def echo(text):
        if paid:
            await record_billing_receipt(BillingReceipt(provider_amount=Money(Decimal('1'), 'CNY')))
        return text

    host.service, host.capture_response, host.prepare_tool = model, capture, tool_plan
    registry.register(ToolDefinition('echo', 'Return input', input_schema={
        'type': 'object', 'properties': {'text': {'type': 'string'}}},
        async_capable=True, pricing=Paid() if paid else Free('fixture')), echo)
    config = BudgetConfig('send-budget', (LocalBudgetScope('shared', max_calls=100000,
        max_tokens=10000000, max_inflight=64),), host) if paid else None
    runtime = TaskRuntime(JobManager(store), tenant_id='profile-'+scenario, principal_id='host',
        executor_id=f'bench-{lane}-'+os.urandom(4).hex(), boot_id=os.urandom(8).hex(), lease_ms=300000, boot_contract_hash='a' * 64,
        budget_config=config)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=5, max_tokens=20, compactor=None, delegation_policy='single_level'))
    sid = await loop.new_session()
    try:
        async def operation():
            if scenario != "same_session":
                model.calls = 0
            if scenario in ('append', 'cancel'):
                model.hold, model.entered, model.release = True, asyncio.Event(), asyncio.Event()
                token = CancellationToken()
                task = asyncio.create_task(loop.send('run', sid, stop_event=token))
                entered = asyncio.create_task(model.entered.wait())
                done, _ = await asyncio.wait((task, entered), timeout=10, return_when=asyncio.FIRST_COMPLETED)
                if task in done:
                    entered.cancel()
                    await task
                    raise AssertionError("send ended before model")
                assert entered in done, "model entry timed out"
                if scenario == 'append':
                    accepted = await loop.follow_up('extra', sid, item_id=f'append-{time.time_ns()}')
                    assert accepted.accepted == 1
                    model.release.set()
                    assert (await task).status == 'completed'
                    assert model.calls == 2
                    return None
                started = time.perf_counter()
                token.cancel('benchmark')
                result = await asyncio.wait_for(task, 5)
                assert result.status == 'cancelled', result.status
                assert model.cancel_seen is not None
                return (model.cancel_seen - started) * 1000
            result = await loop.send('run', sid, history=scenario != 'stateless')
            assert result.status == 'completed'
            assert scenario == 'same_session' or model.calls == model.rounds
            if paid:
                assert sum(key.startswith('toolattempt_') for key in host.applied) == 3 * host.completed_requests + 3
                host.completed_requests += 1
            return None
        host.completed_requests = 0
        yield operation
    finally:
        if runtime._lifecycle is not None:
            await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=0.1)


@asynccontextmanager
async def composite(store, scenario):
    # These fixtures belong only to the historical P0 runner. The current
    # runtime benchmark imports Model/batch/rss without using that old API.
    from tests.unit.test_task_runtime_192 import AdapterModel
    from tests.unit.test_workflow_task_runtime_192 import WorkflowModel, leaf

    manager, registry = JobManager(store), ToolRegistry()
    runtime = TaskRuntime(manager, tenant_id='composite', principal_id='host',
        executor_id='bench-'+os.urandom(4).hex(), boot_id=os.urandom(8).hex(), lease_ms=300000)
    model = AdapterModel() if scenario == 'child' else WorkflowModel()
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy='single_level', max_rounds=4, compactor=None))
    sid = await loop.new_session()
    model.root = sid
    state = {'handles': [], 'workflows': [], 'notifications': [], 'spec': {
        'name': 'registered-workflow', 'root': {'type': 'parallel', 'branches': [leaf('BUILD'), leaf('REVIEW')]}}}

    async def launch():
        workflow = create_workflow(state['spec'], parent_loop=agent_context.get_current_loop(),
                                   parent_session_id=agent_context.get_session_id())
        state['workflows'].append(workflow)
        handle = await workflow.start(detached=True)
        state['handles'].append(handle)
        return json.dumps({'run_id': handle.run_id, 'job_id': handle.job_id,
                           'operation_id': handle.operation_id, 'status': 'queued'})

    registry.register(ToolDefinition('launch', 'Start a workflow', input_schema={'type': 'object'},
                                    pricing=Free('fixture')), launch)
    try:
        async def operation():
            if scenario == 'child':
                model.root_calls = model.child_calls = 0
                model.release.set()
                assert (await loop.send('delegate', sid)).status == 'completed'
                await delegation.await_children(store, sid, timeout_s=120)
                assert model.child_calls == 1
                assert (await loop.flush_follow_ups(sid)).status == 'completed'
            else:
                model.hold = set()
                model.counts.clear()
                state['handles'].clear()
                state['workflows'].clear()
                state['notifications'].clear()
                state['spec']['root'] = {'type': 'parallel', 'branches': [leaf('BUILD'), leaf('REVIEW')]}
                assert (await loop.send('run workflow', sid)).status == 'completed'
                await asyncio.gather(*(handle.task for handle in state['handles']))
                saved = await manager.get_job(runtime.actor(sid), state['handles'][0].job_id)
                assert saved.outcome == 'succeeded', saved.result
                assert model.counts['BUILD'] == model.counts['REVIEW'] == 1
                assert (await loop.flush_follow_ups(sid)).status == 'completed'
            return None
        yield operation
    finally:
        if runtime._lifecycle is not None:
            await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=0.1)


async def batch(operations, count):
    latencies, controls = [], []
    queue = asyncio.Queue()
    for _ in range(count):
        queue.put_nowait(None)
    peak, initial = rss(), rss()
    done = asyncio.Event()

    async def sample():
        nonlocal peak
        while not done.is_set():
            peak = max(peak, rss())
            await asyncio.sleep(0.005)

    async def lane(operation):
        while not queue.empty():
            queue.get_nowait()
            start = time.perf_counter()
            control = await asyncio.wait_for(operation(), 120)
            latencies.append((time.perf_counter() - start) * 1000)
            if control is not None:
                controls.append(control)

    monitor = asyncio.create_task(sample())
    start, cpu = time.perf_counter(), time.process_time()
    try:
        await asyncio.gather(*(lane(op) for op in operations))
        elapsed, cpu = time.perf_counter() - start, time.process_time() - cpu
    finally:
        done.set()
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
    assert len(latencies) == count
    gc.collect()
    return {'completed': count, 'errors': 0, 'backlog': queue.qsize(),
        'seconds': elapsed, 'p50_ms': statistics.median(latencies), 'p95_ms': percentile(latencies, .95),
        'throughput_rps': count / elapsed, 'cpu_seconds': cpu,
        'rss_before': initial, 'rss_peak': peak, 'rss_after_gc': rss(),
        'rss_increase': max(0, peak - initial), 'latencies_ms': latencies,
        'control_p50_ms': statistics.median(controls) if controls else None,
        'control_p95_ms': percentile(controls, .95) if controls else None}


async def run(args):
    from contextlib import AsyncExitStack
    dsn = args.dsn
    if '://' in dsn and not urlparse(dsn).path.endswith('_test'):
        raise ValueError('benchmark requires an isolated _test database')
    if os.environ.get('P0_WRITE_METER'):
        raise ValueError('disable write instrumentation for timing')
    prefix = 'p0b_' + os.urandom(4).hex() + '_'
    store = await open_store(dsn, table_prefix=prefix, schema='tasks')
    report = {'scenario': args.scenario, 'db': store._db.dialect.name, 'concurrency': args.concurrency,
              'warmup': 10, 'batch_size': args.samples, 'pairs': [],
              'head': args.head, 'python': os.sys.version, 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    try:
        # Alternate reference/repeat order by pair; both are B code in P0.
        for pair in range(3):
            arms = ['B', 'repeat'] if pair % 2 == 0 else ['repeat', 'B']
            for arm in arms:
                async with AsyncExitStack() as stack:
                    operations = []
                    for lane in range(1 if args.scenario == 'same_session' else args.concurrency):
                        cm = composite(store, args.scenario) if args.scenario in ('child', 'workflow') else simple(store, lane, args.scenario)
                        operations.append(await stack.enter_async_context(cm))
                    if args.scenario == 'same_session':
                        # One actual session, requests queue in its SDK send guard.
                        operations = [operations[0]] * args.concurrency
                    await batch(operations, 10)
                    result = await batch(operations, args.samples)
                    report['pairs'].append({'pair': pair, 'arm': arm, **result})
                print(json.dumps({'scenario': args.scenario, 'pair': pair, 'arm': arm,
                                  'p50_ms': result['p50_ms'], 'rps': result['throughput_rps']}), flush=True)
                gc.collect()
                report['pairs'][-1]['rss_after_close_gc'] = rss()
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    finally:
        if store._db.dialect.name != 'sqlite':
            await store._db.execute(f'DROP VIEW IF EXISTS {prefix}usage_rounds')
            for table in (*TASK_TABLES, *_STORE_TABLES, 'schema_migrations'):
                await store._db.execute(f'DROP TABLE IF EXISTS {prefix}{table}')
        await store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--scenario', choices=SCENARIOS, required=True)
    parser.add_argument('--concurrency', type=int, choices=(1, 8), required=True)
    parser.add_argument('--samples', type=int, default=34)
    parser.add_argument('--head', required=True)
    parser.add_argument('--out', required=True)
    asyncio.run(run(parser.parse_args()))
