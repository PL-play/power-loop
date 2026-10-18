"""Small real-provider acceptance for opt-in leaf scheduling (no world mutations)."""
import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.workflow import create_workflow

from ._llm import make_llm

pytestmark = pytest.mark.real_llm


async def test_real_leaf_mode_with_independent_reviewer(tmp_path):
    llm = make_llm(max_tokens=4096)
    loop = StatefulAgentLoop(llm=llm, db_path=str(tmp_path/'workflow.db'),
                             config=AgentLoopConfig(system_prompt='Coordinate the checks.', max_tokens=4096))
    def leaf(name, prompt, inputs=None):
        return {'type':'agent','id':name,'spec':{'name':name,'system_prompt':prompt,'max_tokens':4096,'tools':[]},
                **({'inputs_from':inputs} if inputs else {})}
    spec = {'name':'leaf-mode-acceptance','input':'Compute 7+5.', 'root':{
        'type':'parallel','max_concurrency':1,'concurrency_scope':'leaf','branches':[
            {'type':'sequence','steps':[leaf('dev','Return the sum of 7 and 5 as one number.'),
                                       leaf('review','Check the supplied result of 7+5. Reply PASS if it is 12, otherwise FAIL.',['dev'])]},
            leaf('independent','Reply READY only.')]}}
    try:
        result = await create_workflow(spec,parent_loop=loop).run()
        assert result.status == 'completed', result.errors
        assert '12' in result.results['dev'].text
        assert 'PASS' in result.results['review'].text.upper()
        assert result.usage['total_tokens'] > 0
    finally:
        await llm.close()
        await (await loop.ensure_store()).close()
