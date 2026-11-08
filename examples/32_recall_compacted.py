"""32 · Retrieve original detail after folding with recall_send.

Projected history folds by send index. Original messages remain available through
recall_send; no manual execution context or old loop constructor is needed.
"""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    LLMSummaryFold,
    ProjectedRepresentation,
    create_default_tool_registry,
)

ACCESS_CODE = 'ZX7741QORN'

class RecallModel(LLMService):
    async def complete(self, request, **kwargs):
        if any('Retrieve rack serial now' in str(m.get('content')) for m in request.messages):
            outputs = [m['content'] for m in request.messages if m.get('tool_call_id') == 'recall']
            if outputs:
                assert ACCESS_CODE in outputs[-1]
                return LLMResponse(raw_text=ACCESS_CODE)
            return LLMResponse(raw_text='', tool_calls=[{'id':'recall','type':'function','function':{
                'name':'recall_send','arguments':'{"send_index":1}'}}])
        return LLMResponse(raw_text='Work progressed; exact rack details remain in the original record.')
    async def close(self):
        pass

async def main():
    config = AgentLoopConfig(max_tokens=100, context_budget_tokens=40,
        representation=ProjectedRepresentation(), fold_strategy=LLMSummaryFold(keep_last_sends=1))
    async with demo_runtime(RecallModel(), create_default_tool_registry(include=['recall_send']), config=config) as (runtime, loop, target, actor):
        for index, text in enumerate([f'Rack serial {ACCESS_CODE}. '+ 'context '*80, 'more context '*80, 'continue '*80]):
            result = await loop.send(target, text, message_id=f'seed-{index}', actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
        compact = await runtime.store.latest_project_compact(target.session_id)
        assert compact and compact.compact_to_send >= 1 and ACCESS_CODE not in str(compact.content)
        result = await loop.send(target, 'Retrieve rack serial now', message_id='recall', actor=actor, deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed' and ACCESS_CODE in result.final_text, result
        print(result.final_text)
        return result.final_text

if __name__ == '__main__':
    asyncio.run(main())
