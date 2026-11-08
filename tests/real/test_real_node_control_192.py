"""Real models stop one original node while its sibling and review continue."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict

import pytest

from power_loop import AgentLoopConfig
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.jobs import JobManager
from power_loop.tools.jobs import register_jobs_tools
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


class NodeProvider(BoundedProvider):
    def __init__(self):
        super().__init__(limit=14)
        self.entered = defaultdict(asyncio.Event)

    async def complete(self, request, **kwargs):
        if get_session_id() != self.root_sid:
            for role in ("CALCULATE", "AUDIT", "REVIEW"):
                if "ROLE-"+role in (request.system_prompt or ""):
                    self.entered[role].set()
        return await super().complete(request, **kwargs)


@pytest.mark.parametrize('tool', ['workflow', 'jobs'])
async def test_real_node_cancel_preserves_original_parallel_sibling_review_and_completion(runtime_agent, record_property, tool):
    provider=NodeProvider()
    accepted=[]
    registry=ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=accepted.append))
    register_jobs_tools(registry)
    agent=await runtime_agent(model=provider, registry=registry, capacity=3,
        config=AgentLoopConfig(max_rounds=5,max_tokens=2048,compactor=None,
            system_prompt='Follow the requested tool call once, then answer in assistant text. Do not poll, repeat, resume or replace work.'))
    sid=agent.target.session_id
    provider.root_sid,provider.hold_children=sid,True
    def leaf(role,task,**extra):
        return {'type':'agent','id':role,'input':task,'spec':{'name':role,'system_prompt':'ROLE-'+role+'. Compute supplied arithmetic and return the number.','tools':[],'max_rounds':2},**extra}
    spec={'name':'one-node-stop','root':{'type':'sequence','steps':[
        {'type':'parallel','branches':[leaf('CALCULATE','19+23'),leaf('AUDIT','7+5')]},
        leaf('REVIEW','Verify the AUDIT arithmetic and return its number.',inputs_from=['AUDIT'])]}}
    try:
        async with asyncio.timeout(180):
            await agent.send('Call workflow action=create exactly once with spec='+json.dumps(spec)+'. Then finish with WAITING and no further tools.')
            [admission]=accepted
            handle=admission.handle
            await asyncio.wait_for(asyncio.gather(provider.entered['CALCULATE'].wait(),provider.entered['AUDIT'].wait()),10)
            store=agent.runtime.store
            manager=JobManager(store)
            [node]=await manager.db.fetchall(f"SELECT job_id,generation FROM {manager.jobs} WHERE kind='workflow_node' AND title='CALCULATE'")
            args={'action':'cancel','run_id':handle.run_id,'node_job_id':node['job_id']} if tool=='workflow' else {'action':'stop','job_id':node['job_id']}
            await agent.send("Call jobs once with action=get and job_id='"+node['job_id']+"'. Then finish without more tools.")
            await agent.send('Stop only CALCULATE, preserving AUDIT and the original workflow. Call '+tool+' once with arguments='+json.dumps(args)+'. Then acknowledge without more tools.')
            [raw]=[m for m in reversed(await store.load_all_messages(sid)) if m.role=='tool' and m.name==tool][:1]
            receipt=json.loads(raw.content)
            control=receipt['receipt'] if tool=='jobs' else receipt
            assert control['job_ref']['job_id']==node['job_id'] and control['state']=='applied'
            assert not (await agent.runtime.client.inspect(handle.job_ref,actor=agent.actor)).cancel_requested
            provider.release_children.set()
            from power_loop.agent.runtime_store import now_ms
            result=await handle.wait(deadline_ms=now_ms()+90000)
            await agent.drain()
            assert result.results['CALCULATE'].status=='cancelled'
            assert result.results['AUDIT'].ok and result.results['AUDIT'].text.strip()=='12'
            assert result.results['REVIEW'].ok and provider.entered['REVIEW'].is_set()
            assert len(accepted)==1
    finally:
        provider.release_children.set()
        await agent.aclose()
        provider.report(record_property)
