"""05 · Align an interrupted tool response as unknown; never repeat the action."""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService, ToolDefinition, ToolRegistry
from power_loop.runtime.pricing import Free


class ContinueModel(LLMService):
    async def complete(self, request, **kwargs):
        assert '结果未知' in str(request.messages)
        return LLMResponse(raw_text='The previous result is unknown. New input can proceed.')

    async def close(self):
        pass


async def main():
    registry = ToolRegistry()
    def action():
        raise AssertionError('An unknown action must not be repeated')
    registry.register(ToolDefinition('external_action','External action', pricing=Free('demo')), action)
    async with demo_runtime(ContinueModel(), registry) as (runtime, loop, target, actor):
        calls = [{'id':'original','type':'function','function':{'name':'external_action','arguments':'{}'}}]
        seq = await runtime.store.append_message(target.session_id, role='assistant', tool_calls=calls)
        await runtime.store.set_pending(target.session_id,
            {'assistant_seq':seq,'round_index':0,'tool_call_ids':['original'],'tool_calls':calls})
        result = await loop.send(target, 'Continue with the unknown outcome', message_id='new-input', actor=actor,
                                 deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed'
        print(result.final_text)
        return result.final_text


if __name__ == '__main__':
    asyncio.run(main())
