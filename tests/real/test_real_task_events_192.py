"""Real task result, receipt and export reads never repeat original work."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, JobRef, create_default_tool_registry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.env import RuntimeEnv
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from ._bounded import BoundedProvider

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


async def latest_tool(agent,name):
    return [r for r in await agent.runtime.store.load_all_messages(agent.target.session_id) if r.role=='tool' and r.name==name][-1]


async def ask(agent,name,args):
    result=await agent.send('Call '+name+' exactly once with '+json.dumps(args)+'. Then acknowledge the saved result in assistant text. No other tool calls.')
    assert result.status=='completed'
    row=await latest_tool(agent,name)
    return row,json.loads(row.content)


async def test_real_saved_result_stop_receipt_and_export_do_not_reexecute(runtime_agent,tmp_path,record_property):
    provider,accepted=BoundedProvider(limit=20),[]
    registry=create_default_tool_registry(include=['jobs','recall_send'],workspace_dir=tmp_path)
    register_workflow_tool(registry,ports=WorkflowToolPorts(on_accepted=lambda value:accepted.append(value)))
    agent=await runtime_agent(model=provider,registry=registry,runtime_env=RuntimeEnv(workspace_dir=tmp_path),
        config=AgentLoopConfig(compactor=None,max_rounds=4,max_tokens=2048))
    provider.root_sid,provider.hold_children=agent.target.session_id,True
    store=agent.runtime.store
    spec={'name':'audit','root':{'type':'agent','id':'calculate','input':'19+23','spec':{
        'name':'calculate','system_prompt':'Return only the arithmetic answer in assistant text.','tools':[],'max_rounds':2}}}
    try:
        _,receipt=await ask(agent,'workflow',{'action':'create','spec':spec})
        assert receipt['status']=='accepted' and len(accepted)==1
        await asyncio.wait_for(provider.child_entered.wait(),5)
        provider.release_children.set()
        handle=accepted[0].handle
        result=await handle.wait(deadline_ms=now_ms()+120000)
        assert result.status=='completed' and '42' in result.results['calculate'].text
        await agent.drain()
        sid=result.results['calculate'].session_id
        before=await store.load_all_messages(sid)
        event='result_'+handle.job_id
        _,value=await ask(agent,'jobs',{'action':'get','job_id':handle.job_id,'view':'results','event_id':event,'max_chars':16000})
        assert value['status']=='ok' and '42' in value['text']
        stopped,stop=await ask(agent,'jobs',{'action':'stop','job_id':handle.job_id,'expected_generation':1})
        assert stop['status']=='accepted'
        _,recalled=await ask(agent,'recall_send',{'send_index':stopped.send_index,'seq':stopped.seq,'char_limit':16000})
        assert handle.job_id in str(recalled) and 'accepted' in str(recalled)
        _,exported=await ask(agent,'jobs',{'action':'get','job_id':handle.job_id,'view':'results','event_id':event,'output_file':'audit/result.jsonl'})
        assert exported['status']=='exported' and exported['records']==1
        lines=[json.loads(line) for line in (tmp_path/'audit/result.jsonl').read_text().splitlines()]
        assert len(lines)==1 and lines[0]['event_id']==event and '42' in str(lines)
        assert await store.load_all_messages(sid)==before and len(accepted)==1
        assert (await handle.wait(deadline_ms=now_ms()+1000)).results['calculate']==result.results['calculate']
    finally:
        provider.release_children.set()
        provider.report(record_property)


async def test_real_active_input_receipt_reads_do_not_consume_child_input(runtime_agent,tmp_path,record_property):
    provider=BoundedProvider(limit=20)
    registry=create_default_tool_registry(include=['jobs','recall_send'],workspace_dir=tmp_path)
    register_spawn_agent(registry)
    agent=await runtime_agent(model=provider,registry=registry,capacity=2,
        config=AgentLoopConfig(compactor=None,max_rounds=4,max_tokens=2048))
    provider.root_sid,provider.hold_children=agent.target.session_id,True
    store=agent.runtime.store
    try:
        _,created=await ask(agent,'spawn_agent',{'task':'Compute 19+23.','tools':[],'max_rounds':3,
            'system_prompt':'Calculate the arithmetic, follow the latest correction, and return only its number in assistant text.'})
        await asyncio.wait_for(provider.child_entered.wait(),5)
        row,receipt=await ask(agent,'spawn_agent',{'action':'nudge','job_id':created['job_id'],'expected_generation':1,
            'message':'Correction: compute 20+23 and return 43.'})
        original_input=receipt['input_ref']
        before=await store._db.fetchone(f'SELECT status FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?',
            (original_input['session_id'],original_input['message_id']))
        assert before['status'] in {'pending','claimed','delivered'}
        _,queried=await ask(agent,'jobs',{'action':'get','job_id':created['job_id']})
        assert 'error' not in queried
        _,recalled=await ask(agent,'recall_send',{'send_index':row.send_index,'seq':row.seq,'char_limit':16000})
        assert original_input['message_id'] in str(recalled)
        assert not (await agent.runtime.client.inspect(JobRef(**created['job_ref']),actor=agent.actor)).result_available
        provider.release_children.set()
        result=await agent.runtime.client.wait(JobRef(**created['job_ref']),actor=agent.actor,deadline_ms=now_ms()+120000)
        assert result.status=='completed' and result.final_text.strip()=='43'
        await agent.drain()
        assert len(await store.list_children(agent.target.session_id))==1
    finally:
        provider.release_children.set()
        provider.report(record_property)
