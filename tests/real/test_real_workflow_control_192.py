"""A model cancels the accepted workflow; no replacement run or invented stop proof."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig
from power_loop.agent.runtime_store import now_ms
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_workflow_cancel_uses_original_admission_without_replacement_or_false_stop_claim(runtime_agent, record_property):
    provider = BoundedProvider(limit=10)
    accepted=[]
    registry=ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=accepted.append))
    agent=await runtime_agent(model=provider, registry=registry, capacity=2,
        config=AgentLoopConfig(max_rounds=5,max_tokens=2048,compactor=None,
            system_prompt='Follow the requested workflow sequence exactly. Create only once. A cancel receipt is not proof of physical stop or billing settlement. Do not poll or create replacement work.'))
    sid=agent.target.session_id
    provider.root_sid,provider.hold_children=sid,True
    spec={'name':'cancel-original','root':{'type':'sequence','steps':[
        {'type':'agent','id':'calculate','input':'19+23','spec':{'name':'calculate','system_prompt':'Return the number.','tools':[],'max_rounds':2}},
        {'type':'agent','id':'review','input':'Review previous result.','inputs_from':['calculate'],'spec':{'name':'review','system_prompt':'Review supplied result.','tools':[],'max_rounds':2}}]}}
    try:
        async with asyncio.timeout(180):
            await agent.send('Call workflow action=create exactly once with spec='+json.dumps(spec)+'. Then finish with WAITING and no further tools.')
            [admission]=accepted
            handle=admission.handle
            await asyncio.wait_for(provider.child_entered.wait(),10)
            result=await agent.send("Cancel the original workflow with workflow action=cancel, run_id='"+handle.run_id+"'. Acknowledge the request in assistant text; do not claim physical stop or create replacement work.")
            assert result.status=='completed'
            rows=[m for m in await agent.runtime.store.load_all_messages(sid) if m.role=='tool' and m.name=='workflow']
            assert len(rows)==2
            creation,stopped=[json.loads(m.content) for m in rows]
            assert creation['run_id']==handle.run_id
            assert stopped['job_ref']['job_id']==handle.job_id and stopped['state']=='applied'
            assert 'resource_stop_confirmed' not in stopped
            saved=await handle.wait(deadline_ms=now_ms()+15000)
            assert saved.status=='cancelled'
            await agent.drain()
            jobs=await agent.runtime.store._db.fetchall(f'SELECT kind FROM {agent.runtime.store.table_prefix}jobs')
            assert sum(j['kind']=='workflow' for j in jobs)==1
            assert sum(j['kind']=='workflow_node' for j in jobs)==1
            assert len(accepted)==1
    finally:
        provider.release_children.set()
        await agent.aclose()
        provider.report(record_property)
