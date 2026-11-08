"""40 · Projected history folds by input; recall_send reads the original detail."""
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


class FruitModel(LLMService):
    async def complete(self, request, **kwargs):
        if any(m.get('tool_call_id') == 'recall' for m in request.messages):
            text = next(m['content'] for m in request.messages if m.get('tool_call_id') == 'recall')
            assert 'apple1' in text
            return LLMResponse(raw_text='apple1')
        if any(m.get('content') == 'Recall the first fruit' for m in request.messages):
            return LLMResponse(raw_text='', tool_calls=[{'id':'recall','type':'function','function':{
                'name':'recall_send','arguments':'{"send_index":1}'}}])
        return LLMResponse(raw_text='Fruit noted.')
    async def close(self):
        pass

async def main():
    representation = ProjectedRepresentation()
    config = AgentLoopConfig(max_tokens=100, context_budget_tokens=40, representation=representation,
        fold_strategy=LLMSummaryFold(keep_last_sends=2))
    async with demo_runtime(FruitModel(), create_default_tool_registry(include=['recall_send']), config=config) as (runtime, loop, target, actor):
        for i, fruit in enumerate(['apple1','banana2','cherry3','date4']):
            result = await loop.send(target, fruit + ' context'*80, message_id=f'fruit-{i}', actor=actor,
                deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
        compact = await runtime.store.latest_project_compact(target.session_id)
        assert compact is not None
        rows = await runtime.store.load_project_messages(target.session_id, after_send_index=compact.compact_to_send)
        projected = representation.render([compact]+rows)
        assert not any('tool_calls' in row or 'tool_call_id' in row for row in projected)
        originals = await runtime.store.load_all_messages(target.session_id)
        assert len(originals) == 8 and 'apple1' in originals[0].content
        result = await loop.send(target, 'Recall the first fruit', message_id='recall', actor=actor,
            deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed' and result.final_text == 'apple1'
        print(compact.content, result.final_text)
        return {'has_compact':True, 'pl_messages_intact':True, 'projection_is_plain_text':True, 'recall_recovers_send1':True}

if __name__ == '__main__':
    asyncio.run(main())
