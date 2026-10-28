"""Workflow ancestry metadata of linked child sessions."""
from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.runtime.store.store import SessionStore
from tests.unit.test_stateful_loop import _Scripted


async def test_plain_subagent_inherits_workflow_ancestry(tmp_path):
    from power_loop.core.agent_context import reset_session_id, set_session_id
    from power_loop.runtime.spec import AgentSpec, run_agent_spec
    store = await SessionStore.open(str(tmp_path / 'child.db'))
    loop = StatefulAgentLoop(store=store, llm=_Scripted([LLMResponse(raw_text='done')]),
                             config=AgentLoopConfig(compactor=None))
    parent = await loop.new_session(metadata={'workflow_run_id': 'outer', 'workflow_node_id': 'node', 'attempt': 2})
    token = set_session_id(parent)
    try:
        result = await run_agent_spec(AgentSpec(name='helper', system_prompt='help', lifecycle='linked'),
                                      'check', parent_loop=loop)
        child = await store.get_session(result['session_id'])
        assert child.metadata['workflow_ancestry'] == [
            {'session_id': parent, 'workflow_run_id': 'outer', 'node_id': 'node', 'node_attempt': 2}]
    finally:
        reset_session_id(token)
        await store.close()
