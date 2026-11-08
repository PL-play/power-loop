"""Workflow history keeps raw/projection/export behavior through the current tool."""
import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.env import RuntimeEnv
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.workflow.facade import register_workflow_tool
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_stateful_loop import _Scripted, _tool_resp

__all__=['registry_store']
pytestmark=[pytest.mark.unit,pytest.mark.asyncio]


@pytest.mark.parametrize('view',['messages','projection','export','outside'])
async def test_current_workflow_logs_preserve_selected_source_and_safe_export(registry_store,tmp_path,view):
    store,_=registry_store
    model=_Scripted([LLMResponse(raw_text='ANSWER-42')])
    registry=ToolRegistry()
    register_workflow_tool(registry)
    agent=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None,
        representation=ProjectedRepresentation()),runtime_env=RuntimeEnv(workspace_dir=tmp_path)).start()
    try:
        handle=await agent.runtime.client.workflow(agent.target,{'name':'history','root':{
            'type':'agent','id':'calculate','input':'calculate','spec':{'name':'calculate',
                'system_prompt':'Return ANSWER-42.','tools':[],'max_rounds':2}}},message_id='original',actor=agent.actor)
        result=await handle.wait(deadline_ms=now_ms()+10000)
        assert result.status=='completed'
        await agent.drain()
        sid=result.results['calculate'].session_id
        before=await store.load_all_messages(sid)
        options={'action':'logs','run_id':handle.run_id,'node':'calculate','limit':20}
        if view=='projection':
            options['source']='projection'
        elif view in {'export','outside'}:
            options['output_file']='audit/result.jsonl' if view=='export' else '../outside.jsonl'
        model.responses.extend([_tool_resp('read-original','workflow',json.dumps(options)),LLMResponse(raw_text='read')])
        model._idx=len(model.responses)-2
        reply=await agent.send('Read the original history once.')
        assert reply.status=='completed'
        [raw]=[r for r in await store.load_all_messages(agent.target.session_id) if r.role=='tool' and r.name=='workflow']
        value=json.loads(raw.content)
        if view=='outside':
            assert value.get('error') and not (tmp_path.parent/'outside.jsonl').exists()
        elif view=='export':
            assert value['status']=='exported' and value['records']==len(before)
            lines=[json.loads(line) for line in (tmp_path/'audit/result.jsonl').read_text().splitlines()]
            assert all(row['session_id']==sid for row in lines) and 'ANSWER-42' in str(lines)
        else:
            assert value['status']=='ok' and value['source']==view and 'ANSWER-42' in str(value)
        assert await store.load_all_messages(sid)==before
        assert (await store._db.fetchone(f"SELECT COUNT(*) AS n FROM {store.table_prefix}jobs WHERE kind='workflow_node'"))['n']==1
    finally:
        await agent.aclose()
