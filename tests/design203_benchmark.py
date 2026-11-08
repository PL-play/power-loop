"""P0 fixed-provider benchmark, run with python -m tests.design203_benchmark.

No WriteMeter/Profiled while timing. Warm 10, three paired 34-request batches.
Each arm runs the current Runtime. Frozen B is retained in the P0 artifacts.
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
from pathlib import Path
from urllib.parse import urlparse

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    ToolDefinition,
)
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.schema import _STORE_TABLES
from power_loop.runtime.store.task_schema import TASK_TABLES
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

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
async def simple(store, lane, scenario, *, coordinator=None, transport=None):
    from tests.run_design203_runtime_benchmark import composition
    async with composition(store, lane, scenario, coordinator=coordinator, transport=transport) as operation:
        yield operation


@asynccontextmanager
async def composite(store, scenario, *, coordinator=None, transport=None):
    from power_loop.agent.runtime_store import now_ms
    from power_loop.workflow import submit_workflow
    from tests.runtime_fixture import RunningAgent
    registry, handles = ToolRegistry(), []
    class CompositeModel(LLMService):
        def __init__(self):
            self.counts = {}
        async def complete(self, request, **kwargs):
            role = next((tag for tag in ('CHILD', 'BUILD', 'REVIEW') if (request.system_prompt or '').startswith(tag)), 'MAIN')
            self.counts[role] = self.counts.get(role, 0) + 1
            calls = None
            if role == 'MAIN' and self.counts[role] == 1:
                name, args = ('spawn_agent', {'task':'work', 'system_prompt':'CHILD', 'tools':[]}) if scenario == 'child' else ('launch', {})
                calls = [{'id':'launch', 'type':'function', 'function':{'name':name, 'arguments':json.dumps(args)}}]
            return LLMResponse(raw_text='' if calls else role+' done', tool_calls=calls)
        async def close(self):
            pass
    model = CompositeModel()
    register_spawn_agent(registry)
    async def launch():
        spec = {'name':'registered-workflow', 'root':{'type':'parallel', 'branches':[
            {'type':'agent', 'id':tag.lower(), 'spec':{'name':tag.lower(), 'system_prompt':tag, 'tools':[]}}
            for tag in ('BUILD','REVIEW')]}}
        handle = (await submit_workflow({'spec': spec})).handle
        handles.append(handle)
        return json.dumps({'run_id':handle.run_id})
    registry.register(ToolDefinition('launch', 'Start workflow', input_schema={'type':'object','properties':{}}, pricing=Free('fixture')), launch)
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None),
        coordinator=coordinator, transport=transport).start()
    try:
        async def operation():
            model.counts.clear()
            handles.clear()
            assert (await runner.send('run')).status == 'completed'
            for handle in handles:
                result = await handle.wait(deadline_ms=now_ms()+120000)
                assert result.status == 'completed', result
            await runner.drain()
            if scenario == 'child':
                assert model.counts['CHILD'] == 1
            else:
                assert len(handles) == 1 and model.counts.get('BUILD') == model.counts.get('REVIEW') == 1, (len(handles), model.counts, await store.load_all_messages(runner.target.session_id))
            return None
        yield operation
    finally:
        await runner.aclose()


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
        # Alternate current/repeat order; never label this checkout as frozen B.
        for pair in range(3):
            arms = ['current', 'repeat'] if pair % 2 == 0 else ['repeat', 'current']
            for arm in arms:
                async with AsyncExitStack() as stack:
                    from power_loop import MemoryMessageTransport
                    from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator
                    coordinator, transport = MemorySessionOwnershipCoordinator(), MemoryMessageTransport()
                    operations = []
                    for lane in range(1 if args.scenario == 'same_session' else args.concurrency):
                        cm = composite(store, args.scenario, coordinator=coordinator, transport=transport) if args.scenario in ('child', 'workflow') else simple(store, lane, args.scenario, coordinator=coordinator, transport=transport)
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
