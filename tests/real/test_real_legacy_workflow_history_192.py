"""Real model reads and exports current workflow nodes without executing them again."""
import json

import pytest

from power_loop import AgentLoopConfig, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.env import RuntimeEnv
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.workflow.facade import register_workflow_tool

from ._bounded import BoundedProvider

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


@pytest.mark.parametrize('layout',['single','foreach'])
async def test_real_current_workflow_history_projection_and_export(runtime_agent,tmp_path,record_property,layout):
    provider,registry=BoundedProvider(limit=18),ToolRegistry()
    register_workflow_tool(registry)
    agent=await runtime_agent(model=provider,registry=registry,runtime_env=RuntimeEnv(workspace_dir=tmp_path),
        config=AgentLoopConfig(compactor=None,max_rounds=4,max_tokens=2048,representation=ProjectedRepresentation(),
            system_prompt='Make exactly the requested workflow call once, then reply in assistant text. Do not create, resume or poll work.'))
    leaf={'type':'agent','id':'calculate','input':'Compute 19+23.',
        'spec':{'name':'calculate','system_prompt':'Return only the arithmetic answer in assistant text.','tools':[],'max_rounds':2}}
    root=leaf if layout=='single' else {'type':'foreach','id':'each','as':'item','items':['a','b'],'max_concurrency':2,'body':leaf}
    store=agent.runtime.store
    try:
        handle=await agent.runtime.client.workflow(agent.target,{'name':'history','root':root},message_id='original',actor=agent.actor)
        assert (await handle.wait(deadline_ms=now_ms()+120000)).status=='completed'
        await agent.drain()
        nodes=await store._db.fetchall(f"SELECT job_id,session_id FROM {store.table_prefix}jobs WHERE kind='workflow_node' ORDER BY job_id")
        originals={n['session_id']:await store.load_all_messages(n['session_id']) for n in nodes}
        node=nodes[0]
        for source,output in [('messages',None),('projection',None),('messages','audit/node.jsonl')]:
            options={'action':'logs','run_id':handle.run_id,'node_job_id':node['job_id'],'source':source,'limit':20}
            if output:
                options['output_file']=output
            result=await agent.send('Call workflow exactly once with '+json.dumps(options)+'. Then report the returned history or export path. No other tools.')
            assert result.status=='completed'
            tool=[r for r in await store.load_all_messages(agent.target.session_id) if r.role=='tool' and r.name=='workflow'][-1]
            value=json.loads(tool.content)
            if output:
                assert value['status']=='exported'
                lines=[json.loads(line) for line in (tmp_path/output).read_text().splitlines()]
                assert len(lines)==value['records'] and all(row['session_id']==node['session_id'] for row in lines)
                assert '42' in str(lines)
            else:
                assert value['status']=='ok' and value['source']==source and '42' in str(value)
        for sid,history in originals.items():
            assert await store.load_all_messages(sid)==history
        assert (await store._db.fetchone(f"SELECT COUNT(*) AS n FROM {store.table_prefix}jobs WHERE kind='workflow_node'"))['n']==len(nodes)
    finally:
        provider.report(record_property)
