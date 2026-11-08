"""Concurrent resume inputs cannot both create the next generation."""
import asyncio

import pytest

from power_loop.agent.runtime_store import now_ms

from .test_job_registry_192 import registry_store
from .test_subagent_resume_192 import ready
from .test_task_runtime_192 import managed

__all__ = ['registry_store', 'managed', 'ready']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_concurrent_resume_inputs_keep_one_generation_and_explicit_other_result(ready,monkeypatch):
    c=ready
    client,actor=c.loop.runtime.client,c.loop.actor
    acquire=c.loop.runtime.coordinator.acquire
    enabled=False
    async def held(key,**kwargs):
        if key[1]==c.sid and not enabled:
            return None
        return await acquire(key,**kwargs)
    monkeypatch.setattr(c.loop.runtime.coordinator,'acquire',held)
    receipts=await asyncio.gather(*(client.resume(c.first.job_ref,body='same child next turn',
        message_id='concurrent-'+str(i),actor=actor) for i in range(2)))
    assert len({r.input_ref for r in receipts})==2
    enabled=True
    c.model.release.set()
    await c.loop.runtime.worker.recover()
    results=await asyncio.gather(*(client.wait(r.input_ref,actor=actor,deadline_ms=now_ms()+15000) for r in receipts))
    await c.loop.drain()
    assert sorted(r.status for r in results)==['completed','failed']
    winner=next(r for r in results if r.status=='completed')
    loser=next(r for r in results if r.status=='failed')
    assert winner.job_ref.generation==2 and winner.usage['total_tokens']==10
    assert loser.error_code=='stale_generation' or loser.error=='stale_generation'
    assert c.model.child_calls==2 and len(await c.store.list_children(c.parent))==1
    assert await client.wait(c.first.job_ref,actor=actor,deadline_ms=now_ms()+1000)==c.first
