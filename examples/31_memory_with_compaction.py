"""31 · Recalled memory stays ephemeral while persisted sends are folded."""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMSummaryFold, ProjectedRepresentation


class StaticMemory:
    """A trivial MemoryProvider that always recalls the same two facts and never
    persists. ``recall`` returns plain messages; the framework auto-tags them as
    ``role=system, name=memory_*``."""

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        return [
            {"content": "Known fact: the user's name is Alan."},
            {"content": "Known fact: the user's favorite number is 37."},
        ]

    async def remember(self, *, snapshot, session_id):
        return None  # this demo's memory is static


class MemoryModel(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text='Alan / 37' if 'favorite number is 37' in str(request.messages) else 'Earlier project work.')
    async def close(self):
        pass

async def main():
    config = AgentLoopConfig(max_tokens=100, context_budget_tokens=40, memory_budget_tokens=10,
        representation=ProjectedRepresentation(), fold_strategy=LLMSummaryFold(keep_last_sends=1), memory=StaticMemory())
    async with demo_runtime(MemoryModel(), config=config) as (runtime, loop, target, actor):
        for i in range(3):
            result = await loop.send(target, 'project context '*80, message_id=f'memory-{i}', actor=actor,
                                     deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed' and result.final_text == 'Alan / 37'
        compact = await runtime.store.latest_project_compact(target.session_id)
        rows = await runtime.store.load_all_messages(target.session_id)
        assert compact and not any((row.name or '').startswith('memory_') for row in rows)
        assert not any('Known fact:' in row.content for row in rows)
        print(result.final_text, compact.content)
        return result.final_text

if __name__ == '__main__':
    asyncio.run(main())
