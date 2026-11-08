"""Real-provider workflows through a current factory and durable handle."""
from power_loop import AgentLoopConfig, create_llm_service_from_env
from power_loop.agent.runtime_store import now_ms


async def run_workflow(runtime_agent, spec):
    agent = await runtime_agent(model=create_llm_service_from_env(),
        config=AgentLoopConfig(system_prompt='orchestrator', max_rounds=4, compactor=None))
    handle = await agent.runtime.client.workflow(agent.target, spec, message_id='workflow', actor=agent.actor)
    result = await handle.wait(deadline_ms=now_ms()+300000)
    await agent.drain()
    return result
