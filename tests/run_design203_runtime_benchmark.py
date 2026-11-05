"""P1 completed-request timing against frozen P0 B. No write instrumentation."""
import argparse
import asyncio
import gc
import json
import os
import subprocess
import sys
import time
from contextlib import AsyncExitStack, asynccontextmanager
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

from power_loop import AgentLoopConfig, RequestOptions, ToolDefinition, record_billing_receipt
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.budget_contracts import BudgetConfig, BudgetPlan, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.pricing import BillingReceipt, Free, Money
from power_loop.runtime.response_capture import ResponseCaptureReceipt
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.schema import _STORE_TABLES
from power_loop.runtime.store.task_schema import TASK_TABLES
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.design203_benchmark import Model, batch, rss
from tests.runtime_fixture import RunningAgent
from tests.unit.test_model_budgets_192 import Host
from tests.unit.test_nested_tool_billing_196 import Paid

SCENARIOS = ('send', 'paid', 'append', 'cancel', 'stateless', 'same_session')


@asynccontextmanager
async def composition(store, lane, scenario):
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
    accounting = TaskRuntime(JobManager(store), tenant_id='profile-'+scenario, principal_id='host',
        executor_id=f'bench-{lane}-'+os.urandom(4).hex(), boot_id=os.urandom(8).hex(),
        lease_ms=300000, boot_contract_hash='a'*64, budget_config=config)
    agent = await RunningAgent(store, model, registry,
        AgentLoopConfig(max_rounds=5, max_tokens=20, compactor=None, delegation_policy='single_level'),
        task_runtime=accounting).start()
    serial = 0
    try:
        async def operation():
            nonlocal serial
            serial += 1
            mid = f'operation-{serial}'
            if scenario != 'same_session':
                model.calls = 0
            if scenario in ('append', 'cancel'):
                model.hold, model.entered, model.release = True, asyncio.Event(), asyncio.Event()
                receipt = await agent.runtime.client.send(agent.target, 'run', message_id=mid, actor=agent.actor)
                await asyncio.wait_for(model.entered.wait(), 10)
                if scenario == 'append':
                    extra = await agent.runtime.client.append(agent.target, 'extra', message_id=mid+'-append', actor=agent.actor)
                    assert extra.newly_accepted
                    model.release.set()
                else:
                    started = time.perf_counter()
                    await agent.runtime.client.cancel(receipt.input_ref, actor=agent.actor, reason='benchmark')
                result = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+120000)
                await agent.finish(receipt.input_ref)
                assert result.status == ('completed' if scenario == 'append' else 'cancelled'), result
                if scenario == 'append':
                    assert model.calls == 2
                    return None
                assert model.cancel_seen is not None
                return (model.cancel_seen-started)*1000
            result = await agent.send('run', options=RequestOptions(history=scenario != 'stateless'))
            assert result.status == 'completed'
            assert scenario == 'same_session' or model.calls == model.rounds
            return None
        yield operation
    finally:
        await agent.aclose()
        await accounting.shutdown(timeout_s=1)
        assert not agent.runtime.worker._tasks and not agent.runtime.worker._active
        assert not agent.runtime.transport._subscriptions


async def run_case(dsn, scenario, concurrency, output):
    if output.exists():
        raise ValueError(f'refusing to replace an existing measurement: {output}')
    if '://' in dsn and not urlparse(dsn).path.endswith('_test'):
        raise ValueError('isolated _test database required')
    prefix = 'p1b_'+os.urandom(4).hex()+'_'
    store = await open_store(dsn, table_prefix=prefix)
    report = {'scenario': scenario, 'db': store._db.dialect.name, 'concurrency': concurrency,
              'warmup': 10, 'batch_size': 34, 'pairs': [], 'python': os.sys.version,
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'code': 'P1 working tree; see delivery.md'}
    try:
        for repeat in range(3):
            async with AsyncExitStack() as stack:
                operations = [await stack.enter_async_context(composition(store, lane, scenario))
                              for lane in range(1 if scenario == 'same_session' else concurrency)]
                if scenario == 'same_session':
                    operations *= concurrency
                await batch(operations, 10)
                result = await batch(operations, 34)
                report['pairs'].append({'pair': repeat, 'arm': 'N', **result})
            gc.collect()
            report['pairs'][-1]['rss_after_close_gc'] = rss()
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'completed': output.name, 'requests': 102, 'errors': 0}), flush=True)
    finally:
        if store._db.dialect.name != 'sqlite':
            await store._db.execute(f'DROP VIEW IF EXISTS {prefix}usage_rounds')
            for table in (*TASK_TABLES, *_STORE_TABLES, 'schema_migrations'):
                await store._db.execute(f'DROP TABLE IF EXISTS {prefix}{table}')
        await store.close()


def main(args):
    if os.environ.get('P0_WRITE_METER'):
        raise ValueError('timing must run without instrumentation')
    args.out.mkdir(parents=True, exist_ok=True)
    for backend in args.backends:
        dsn = str(args.out/'timing_test.db') if backend == 'sqlite' else os.environ[
            'POWER_LOOP_TASK_PG_DSN' if backend == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
        for scenario in args.scenarios:
            for concurrency in ([8] if scenario == 'same_session' else [1, 8]):
                print(json.dumps({'starting': [backend, scenario, concurrency]}), flush=True)
                subprocess.run([sys.executable, '-m', 'tests.run_design203_runtime_benchmark',
                    '--case', '--dsn', dsn, '--scenario', scenario, '--concurrency', str(concurrency),
                    '--out', str(args.out/f'{backend}-{scenario}-c{concurrency}.json')], check=True)
        if backend == 'sqlite':
            for suffix in ('', '-wal', '-shm'):
                Path(dsn+suffix).unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', action='store_true')
    parser.add_argument('--dsn')
    parser.add_argument('--scenario', choices=SCENARIOS)
    parser.add_argument('--concurrency', type=int, choices=(1, 8))
    parser.add_argument('--backends', nargs='+', default=['sqlite', 'postgres', 'mysql'])
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    args = parser.parse_args()
    if args.case:
        asyncio.run(run_case(args.dsn, args.scenario, args.concurrency, args.out))
    else:
        main(args)
