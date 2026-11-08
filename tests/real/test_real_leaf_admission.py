"""Small real-provider acceptance for opt-in leaf scheduling (no world mutations)."""
import pytest

from power_loop import AgentLoopConfig, SessionStore
from power_loop.agent.runtime_store import now_ms

from ._llm import make_llm

pytestmark = pytest.mark.real_llm


async def test_real_leaf_mode_with_independent_reviewer(tmp_path, runtime_agent):
    llm = make_llm(max_tokens=4096)
    store = await SessionStore.open(str(tmp_path/'workflow.db'))
    loop = await runtime_agent(model=llm, store=store, capacity=1,
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
        handle = await loop.runtime.client.workflow(loop.target, spec, message_id='workflow', actor=loop.actor)
        result = await handle.wait(deadline_ms=now_ms()+150000)
        assert result.status == 'completed', result.errors
        assert '12' in result.results['dev'].text
        assert 'PASS' in result.results['review'].text.upper()
        assert result.usage['total_tokens'] > 0
    finally:
        await loop.aclose()
        await llm.close()
        await store.close()
