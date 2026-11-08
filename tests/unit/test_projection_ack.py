"""A completed background result stays visible after a failed parent model call."""
import asyncio
import hashlib
import json

import pytest

from power_loop import InputRef, JobRef, LLMResponse, LLMService, ToolDefinition, create_default_tool_registry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.pricing import Free
from power_loop.runtime.result_revisions import result_id


@pytest.mark.parametrize('output', ['BUILD OK', 'image.webp'])
async def test_original_background_result_survives_failed_parent_and_is_not_reinjected(runtime_agent, tmp_path, output):
    release = asyncio.Event()
    calls = []
    class Model(LLMService):
        failed = False
        async def complete(self, request, **kwargs):
            calls.append(request.messages)
            if len(calls) == 1:
                return LLMResponse(raw_text='', tool_calls=[{'id':'background','type':'function','function':{
                    'name':'background_run','arguments':'{"action":"tool","tool":"build","args":{}}'}}])
            if output in str(request.messages) and not self.failed:
                self.failed = True
                raise ConnectionError('parent model reply unknown')
            return LLMResponse(raw_text='parent done')
        async def close(self):
            pass
    invoked = []
    async def build():
        invoked.append(True)
        await release.wait()
        return output
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    registry.register(ToolDefinition('build','build',async_capable=True,pricing=Free('fixture')), build)
    agent = await runtime_agent(model=Model(), registry=registry)
    try:
        assert (await agent.send('build')).status == 'completed'
        rows = await agent.runtime.store.load_all_messages(agent.target.session_id)
        ref = JobRef(**json.loads(next(row.content for row in rows if row.tool_call_id=='background'))['job_ref'])
        release.set()
        child = await agent.runtime.client.wait(ref, actor=agent.actor, deadline_ms=now_ms()+10000)
        assert child.status == 'completed' and child.final_text == output
        parent_ref = InputRef(ref.store_id, agent.target.session_id,
            hashlib.sha256(('result:'+result_id(child.job_ref.job_id, child.job_ref.generation)).encode()).hexdigest())
        result = await agent.runtime.client.wait(parent_ref, actor=agent.actor, deadline_ms=now_ms()+10000)
        assert result.status == 'failed' and 'reply unknown' in result.error
        await agent.finish(parent_ref, failed=True)
        assert (await agent.send('continue from saved result')).status == 'completed'
        assert output in str(calls[-1]) and invoked == [True]
        notices = [r for r in await agent.runtime.store.load_all_messages(agent.target.session_id)
                   if '[background_tool_results]' in str(r.content)]
        assert len(notices) == 1 and output in notices[0].content
        assert await agent.runtime.client.wait(ref, actor=agent.actor, deadline_ms=now_ms()+1000) == child
        assert all('_ack' not in row and '_persist' not in row for request in calls for row in request)
    finally:
        release.set()
