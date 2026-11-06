"""Own a real source process,then die around its three continuation commits."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import control_requests, subagent_admission
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.store.factory import open_store


async def main():
    dsn, prefix, raw, mode, actual, marker, output = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname=='127.0.0.1' and address.path.endswith('_test')
    context = json.loads(raw)
    phase, committed = mode.split(':')
    committed = committed=='1'
    store = await open_store(dsn, table_prefix=prefix, schema='compatible')
    manager = JobManager(store, now_ms=lambda:context['now'])
    actor = JobActor(**context['actor'])
    source = await manager.register_job(actor, kind='send', session_id=actor.session_id,
        system_key='continuation-source', request={'send_index':3})
    caller = await manager.claim_job(actor, source.job.job_id, executor_id=context['executor_id'],
        boot_id=context['boot_id'], expected_version=source.job.version)
    seq = await store.append_message(actor.session_id, role='assistant', send_index=3, round_index=0,
        tool_calls=[{'id':'resume-worker-call', 'type':'function', 'function':{'name':'spawn_agent', 'arguments':'{}'}}])
    ref = await store.message_ref(actor.session_id, seq)
    key = manager._origin_key(actor, 'nudge', ref, 'resume-worker-call', None)
    Path(actual).write_text(json.dumps({'caller':asdict(caller), 'request_id':control_requests.identity(subagent_admission.ACTION,key)}))
    transaction = manager.db.transaction

    def die():
        Path(marker).write_text(mode)
        os._exit(79)

    @asynccontextmanager
    async def boundary():
        async with transaction() as tx:
            yield tx
            rows = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='task_control_request'")
            accepted = [row for row in rows if (lambda data:data['phase']=='accepted' and data['action']==subagent_admission.ACTION)(
                json.loads(row['payload_json']))]
            current = None
            if accepted:
                if any(row['event_id']==control_requests.completion_id(accepted[0]['event_id']) for row in rows):
                    current = 'completed'
                elif await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE action='nudge'"):
                    current = 'target'
                else:
                    current = 'accepted'
            hit = current==phase
            if hit and not committed:
                die()
        if hit:
            die()

    manager.db.transaction = boundary
    try:
        result, _ = await manager.resume_subagent(actor, context['job_id'], caller=caller, call_ref=ref,
            tool_call_id='resume-worker-call', session_id=context['sid'], message='continue once', expected_generation=context['generation'],
            executor_id=context['executor_id'], boot_id=context['boot_id'], lease_ms=300000,
            contract=await store.get_runtime_state(context['sid'], 'workflow_execution_contract'))
        Path(output).write_text(json.dumps(result.result))
    finally:
        await store.close()


asyncio.run(main())
