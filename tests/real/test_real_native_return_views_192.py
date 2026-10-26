"""A real main reads a saved native return through the actual built-in jobs."""
import asyncio
import json

import pytest

from power_loop import register_jobs_tools
from power_loop.runtime.pending_calls import recover_runtime
from tests.unit._owned_pending_case_192 import owned_pending_case
from tests.unit.test_controller_handoff_192 import new_main
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_owned_pending_review_192 import cleanup

from .test_real_async_boundaries_192 import BoundedProvider

__all__ = ['registry_store']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_main_reads_original_return_with_builtin_jobs(registry_store, record_property):
    provider = BoundedProvider(limit=8)
    c = loop = runtime = None
    try:
        async with asyncio.timeout(180):
            c = await owned_pending_case(registry_store, 'subagent', provider=provider,
                native_tail='inspect_original_note', native_first=True)
            loop, runtime = await new_main(c, registry_store)
            register_jobs_tools(loop.tool_registry)
            source = c.state.source
            actor = runtime.actor(c.state.root)
            assert (await recover_runtime(runtime, actor, source))['state'] == 'queued_for_review'
            [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='native_callback_returned'", (source,))
            original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY id')
            response = await loop.send(
                f'Read the exact original callback data using the jobs tool: action=get, job_id={source}, '
                f'view=returns, event_id={event["event_id"]}. Call this once via native function calling, '
                'then acknowledge the supplied original outcome in one short sentence. Do not execute or stop any task.',
                c.state.root, tools=['jobs'])
            assert response.status == 'completed'
            rows = await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} WHERE session_id=? ORDER BY id", (c.state.root,))
            actual_calls = []
            for row in rows:
                for call in json.loads(row['tool_calls_json'] or '[]'):
                    if call['function']['name'] == 'jobs':
                        actual_calls.append(call)
            assert len(actual_calls) == 1
            call = actual_calls[0]
            args = json.loads(call['function']['arguments'])
            assert args['view'] == 'returns' and args['event_id'] == event['event_id'] and args['job_id'] == source
            [receipt] = [row for row in rows if row['role'] == 'tool' and row['tool_call_id'] == call['id']]
            public = json.loads(receipt['content'])
            assert public['read_only'] and not public['execution_authority']
            value = json.loads(public['text'])
            assert value['return_text'] == 'Original native completion' and not value['resource_stop_proven']
            for row in original:
                assert await c.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (row['id'],)) == row
            assert c.state.native_calls == 1 and c.state.executed == c.state.prepared == 0
    finally:
        if c is not None:
            await cleanup(c, loop, runtime)
        provider.report(record_property)
        await provider.close()
