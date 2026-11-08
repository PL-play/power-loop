"""06 · A trusted factory command delegates with a frozen AgentSpec.

Waiting host commands release capacity; no ContextVar injection or bare loop.
"""
import asyncio
import json
import time

from _runtime import demo_runtime

from power_loop import (
    AgentSpec,
    LLMResponse,
    LLMService,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
    run_agent_spec,
)
from power_loop.runtime.pricing import Free


class MathModel(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text='126')

    async def close(self):
        pass


async def delegate_math():
    context = get_tool_runtime_context(required=True)
    return await run_agent_spec(AgentSpec(name='math', system_prompt='Compute and return a number.',
        tools=[], max_rounds=1, lifecycle='linked'), '(17 + 25) * 3', parent_loop=context.loop)


async def main():
    registry = ToolRegistry()
    registry.register(ToolDefinition('math_command','Host-selected math delegation', pricing=Free('composite')), delegate_math)
    async with demo_runtime(MathModel(), registry, command=('math_command', {})) as (runtime, loop, target, actor):
        result = await loop.send(target, 'run math command', message_id='math', actor=actor,
                                 deadline_ms=int(time.time()*1000)+10000)
        child = json.loads(result.final_text)
        assert result.status == child['status'] == 'completed' and child['final_text'] == '126'
        print(child['final_text'])
        return child['final_text']


if __name__ == '__main__':
    asyncio.run(main())
