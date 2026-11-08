"""04 · Fold old projected sends; preserve the project code in the summary."""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMSummaryFold, ProjectedRepresentation


class SummaryModel(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text='Project code is BLUE-37.')

    async def close(self):
        pass


async def main():
    config = AgentLoopConfig(max_rounds=2, max_tokens=200, context_budget_tokens=40,
        representation=ProjectedRepresentation(), fold_strategy=LLMSummaryFold(keep_last_sends=1))
    async with demo_runtime(SummaryModel(), config=config) as (runtime, loop, target, actor):
        for i in range(3):
            result = await loop.send(target, 'Remember BLUE-37. ' + 'context '*80, message_id=f'context-{i}',
                actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
        compact = await runtime.store.latest_project_compact(target.session_id)
        assert compact is not None and 'BLUE-37' in str(compact.content)
        print(compact.content)
        return result.final_text


if __name__ == '__main__':
    asyncio.run(main())
