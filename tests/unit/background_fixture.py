"""Start a background tool using the public session runtime."""
from power_loop import AgentLoopConfig, create_default_tool_registry
from tests.runtime_fixture import RunningAgent

from .test_background_sessions_215 import BackgroundParent


async def start_background(store, registry, tool, arguments, *, runtime_env):
    registered = create_default_tool_registry(bind=False).get('background_run')
    registry.register(registered.definition, registered.handler)
    runner = await RunningAgent(store, BackgroundParent(arguments={
        'action':'tool', 'tool':tool, 'args':arguments}), registry,
        AgentLoopConfig(context_budget_tokens=10**9), runtime_env=runtime_env).start()
    await runner.send('start')
    row = next(m for m in await store.load_all_messages(runner.target.session_id)
               if m.tool_call_id == 'background')
    return runner, row
