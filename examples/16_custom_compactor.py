"""16 · A custom FoldStrategy that deliberately discards old details.

For semantic summaries use LLMSummaryFold. This example's policy reports how
many records it drops; it does not pretend truncation preserves their facts.
"""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import AgentLoopConfig, FoldResult, LLMResponse, LLMService, ProjectedRepresentation


class CountOnlyFold:
    keep_last_sends = 1
    trigger_ratio = .5
    fold_id = 'example.count-only@1'

    async def fold(self, rows, *, context):
        return FoldResult(content={'summary':f'{len(rows)} old records discarded by explicit host policy.'},
                          folded_to_send=max(row.send_index for row in rows))


class ReplyModel(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text='Acknowledged.')

    async def close(self):
        pass


async def main():
    config = AgentLoopConfig(representation=ProjectedRepresentation(), fold_strategy=CountOnlyFold(),
                             context_budget_tokens=40, max_tokens=100, max_rounds=1)
    async with demo_runtime(ReplyModel(), config=config) as (runtime, loop, target, actor):
        for index in range(4):
            result = await loop.send(target, 'fact '*80, message_id=f'fact-{index}', actor=actor,
                                     deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
        compact = await runtime.store.latest_project_compact(target.session_id)
        assert compact is not None and 'discarded' in str(compact.content)
        print(compact.content)


if __name__ == '__main__':
    asyncio.run(main())
