"""Child output belongs to the current input; empty output is an explicit failure."""
import asyncio
import json

import pytest

from power_loop import JobRef
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import get_session_id

from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import finish, jobs, managed, receipt

__all__ = ['registry_store', 'managed']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('ending', ['text', 'empty', 'pass', 'recover', 'provider_error', 'cancel'])
async def test_child_result_has_fresh_output_or_explicit_failure_without_business_replay(managed, monkeypatch, ending):
    store, manager, runtime, agent, model, parent, _ = managed
    original = model.complete
    requests = []
    async def complete(request, **kwargs):
        response = await original(request, **kwargs)
        if get_session_id() == parent and response.tool_calls and ending in ('pass', 'recover'):
            arguments = json.loads(response.tool_calls[0]['function']['arguments'])
            arguments['max_rounds'] = 4
            response.tool_calls[0]['function']['arguments'] = json.dumps(arguments)
        if get_session_id() != parent:
            requests.append(request)
            if ending == 'provider_error':
                raise RuntimeError('output provider failed')
            if ending in ('empty', 'pass') or (ending == 'recover' and len(requests) == 1):
                response.raw_text = ''
                response.tool_calls = ([{'id':'empty-pass','type':'function','function':{
                    'name':'pass_turn','arguments':'{"reason":"no answer"}'}}] if ending != 'empty' else None)
        return response
    monkeypatch.setattr(model, 'complete', complete)
    await agent.send('delegate')
    await asyncio.wait_for(model.entered.wait(), 5)
    accepted = await receipt(managed)
    ref = JobRef(**accepted['job_ref'])
    if ending == 'cancel':
        current = await agent.runtime.client.inspect(ref, actor=agent.actor)
        await agent.runtime.client.cancel(current.job_ref, actor=agent.actor, reason='stop original child')
    await finish(managed)
    result = await agent.runtime.client.wait(ref, actor=agent.actor, deadline_ms=now_ms()+1000)
    [child] = [j for j in await jobs(managed) if j.kind == 'subagent']
    assert model.child_calls == ({'empty':3, 'pass':4, 'recover':2}.get(ending, 1))
    if ending in ('pass', 'recover'):
        assert not requests[1].tools
    assert child.resource_state == 'stopped'
    if ending in ('text', 'recover'):
        assert result.status == 'completed' and result.final_text == 'CHILD-PROOF-42'
        assert result.usage['total_tokens'] == (20 if ending == 'recover' else 10)
    elif ending == 'cancel':
        assert result.status == 'cancelled'
    else:
        assert result.status == ('interrupted' if ending in ('empty','pass') else 'failed')
        if ending in ('empty', 'pass'):
            assert result.error == 'hit_round_limit'
            assert result.usage['total_tokens'] == (30 if ending == 'empty' else 40)
            assert result.final_text.strip() == '[hit_round_limit]'
    assert (await store.inbox_counts(parent))['pending'] == 0
