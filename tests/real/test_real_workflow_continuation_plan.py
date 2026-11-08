"""Explicit resume invalidates dependent results and retains unrelated work."""
from collections import Counter

import pytest

from power_loop import AgentLoopConfig, JobRef, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.tools.spawn_agent import register_spawn_agent

from ._bounded import BoundedProvider

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


async def test_real_resume_recomputes_review_and_assembly_with_current_answer(runtime_agent,record_property):
    roles=Counter()
    class Provider(BoundedProvider):
        async def complete(self,request,**kwargs):
            role=next((r for r in ('CALCULATE','REVIEW','SIBLING','ASSEMBLY') if 'ROLE-'+r in (request.system_prompt or '')),'MAIN')
            roles[role]+=1
            return await super().complete(request,**kwargs)
    provider=Provider(limit=14)
    agent=await runtime_agent(model=provider,config=AgentLoopConfig(compactor=None,max_tokens=2048))
    def leaf(name,role,**extra):
        return {'type':'agent','id':name,'spec':{'name':name,'tools':[],'max_rounds':2,
            'system_prompt':'ROLE-'+role+'. Copy the latest supplied VERSION code verbatim. Reply only with that code, without tools.'},**extra}
    spec={'name':'repair','root':{'type':'sequence','steps':[
        {'type':'parallel','branches':[
            {'type':'sequence','steps':[leaf('calculate','CALCULATE',input='Reply VERSION-42.'),
                leaf('review','REVIEW',input='Copy the current supplied calculation.',inputs_from=['calculate'])]},
            leaf('sibling','SIBLING',input='Reply VERSION-INDEPENDENT.')]},
        leaf('assembly','ASSEMBLY',input='Copy the current reviewed code.',inputs_from=['review'])]}}
    try:
        handle=await agent.runtime.client.workflow(agent.target,spec,message_id='original',actor=agent.actor)
        first=await handle.wait(deadline_ms=now_ms()+120000)
        assert first.status=='completed' and first.results['assembly'].text.strip()=='VERSION-42'
        await agent.drain()
        resumed=await handle.resume(generation=1,node_id='calculate',message='Correction: return VERSION-43.',message_id='correction')
        second=await resumed.wait(deadline_ms=now_ms()+120000)
        assert second.status=='completed'
        assert all(second.results[n].text.strip()=='VERSION-43' for n in ('calculate','review','assembly'))
        assert second.results['sibling']==first.results['sibling'] and roles['SIBLING']==1
        assert second.results['calculate'].session_id==first.results['calculate'].session_id
        assert second.results['review'].session_id!=first.results['review'].session_id
        assert roles['CALCULATE']==roles['REVIEW']==roles['ASSEMBLY']==2
        assert (await handle.wait(deadline_ms=now_ms()+1000)).results['assembly'].text.strip()=='VERSION-42'
    finally:
        provider.report(record_property)


async def test_real_child_empty_pass_repairs_answer_in_original_session(runtime_agent,record_property):
    import asyncio
    import json
    provider,registry=BoundedProvider(limit=9),ToolRegistry()
    register_spawn_agent(registry)
    agent=await runtime_agent(model=provider,registry=registry,config=AgentLoopConfig(compactor=None,max_rounds=4,max_tokens=2048))
    provider.root_sid,provider.hold_children=agent.target.session_id,True
    try:
        first=await agent.send("Call spawn_agent exactly once with tools=[], max_rounds=4, task='Report result code 42', "
            "system_prompt='The code is 42. FIRST call pass_turn once without answer text. When asked for a final answer, return 42 without tools.'. "
            'Then reply WAITING. When the result arrives report it without more tools.')
        assert first.status=='completed'
        await asyncio.wait_for(provider.child_entered.wait(),5)
        store=agent.runtime.store
        [receipt]=[json.loads(r.content) for r in await store.load_all_messages(agent.target.session_id) if r.role=='tool' and r.name=='spawn_agent']
        provider.release_children.set()
        result=await agent.runtime.client.wait(JobRef(**receipt['job_ref']),actor=agent.actor,deadline_ms=now_ms()+120000)
        assert result.status=='completed' and result.final_text.strip()=='42' and result.usage['calls']>=2
        rows=await store.load_all_messages(receipt['session_id'])
        assert len([r for r in rows if r.role=='tool' and r.name=='pass_turn'])==1
        await agent.drain()
        assert len(await store.list_children(agent.target.session_id))==1
    finally:
        provider.release_children.set()
        provider.report(record_property)
