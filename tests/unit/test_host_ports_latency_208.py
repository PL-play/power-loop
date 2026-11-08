"""Small isolated latency samples for the newly changed ports, without SQL probes."""
import json
import math
import os
import resource
import statistics
import time
from pathlib import Path

import pytest

from power_loop import RuntimeClient, StoreBlackboard, StoreTimerBackend, TimerContext, TimerRunner
from power_loop.agent.runtime_transport import MemoryMessageTransport
from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator
from power_loop.runtime.store.process_guard import ProcessStoreGuard

from .test_host_ports_cost_profile_208 import registry_store
from .test_runtime_messaging_203 import fixture

__all__=['registry_store']


@pytest.mark.parametrize('operation',['timer_create_cancel','timer_fire_acceptance','timer_scan_empty',
    'blackboard_post_update_delete','startup_lock'])
async def test_port_latency(registry_store,operation):
    store,_=registry_store
    _,target,actor=await fixture(store)
    transport=MemoryMessageTransport()
    client=RuntimeClient(store=store,transport=transport)
    timers=StoreTimerBackend(store)
    context=TimerContext(store.store_id,target.session_id,actor)
    board=StoreBlackboard(store)
    driver=TimerRunner(store=store,client=client)
    driver._last_recovery=time.monotonic()
    async def once(index):
        if operation=='timer_create_cancel':
            timer=await timers.create(context,delay_s=3600,note='latency timer')
            assert await timers.cancel(context,timer.timer_id)
        elif operation=='timer_fire_acceptance':
            await timers.create(context,delay_s=-1,note='latency timer')
            assert await driver.scan_once()==1
        elif operation=='timer_scan_empty':
            assert await driver.scan_once()==0
        elif operation=='blackboard_post_update_delete':
            await board.post('latency-board',text='first')
            await board.update('latency-board',1,text='updated')
            await board.delete('latency-board')
        else:
            guard=ProcessStoreGuard(store,MemorySessionOwnershipCoordinator(),transport)
            await guard.acquire()
            await guard.check()
            await guard.release()
    try:
        for index in range(3):
            await once(index)
        timings=[]
        usage_before=resource.getrusage(resource.RUSAGE_SELF)
        cpu=time.process_time()
        wall=time.monotonic()
        for index in range(20):
            start=time.monotonic()
            await once(index+3)
            timings.append((time.monotonic()-start)*1000)
        elapsed=time.monotonic()-wall
        cpu=time.process_time()-cpu
        usage_after=resource.getrusage(resource.RUSAGE_SELF)
        data=dict(db=store._db.dialect.name,operation=operation,warmup=3,samples=20,concurrency=1,
            scope='complete named port operation; timer firing ends at durable input acceptance, no model execution',
            p50_ms=statistics.median(timings),p95_ms=sorted(timings)[math.ceil(.95*len(timings))-1],
            throughput_per_s=20/elapsed,elapsed_s=elapsed,cpu_process_s=cpu,
            rss_peak_kib_before=usage_before.ru_maxrss,rss_peak_kib_after=usage_after.ru_maxrss,
            rss_note='pytest process lifetime high-water mark, not database/server memory or operation allocation',
            completed=20,errors=0,accepted_inputs_unexecuted=23 if operation=='timer_fire_acceptance' else 0,
            latency_ms=timings)
        if output:=os.environ.get('P7_PORT_LATENCY_OUT'):
            with Path(output).open('a') as file:
                file.write(json.dumps(data)+'\n')
    finally:
        await client.aclose()
        await transport.aclose()
