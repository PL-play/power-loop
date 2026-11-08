"""33 · A FoldStrategy stores useful detail in a host memory provider before folding.

The memory write is a normal external action, not an exact-recovery guarantee.
The factory supplies that provider again for the next session.
"""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMSummaryFold, MemorySnapshot, ProjectedRepresentation

CODENAME = 'BLUEHERON'

class CapturingMemory:
    """A tiny cross-session memory. ``remember`` with status='compaction' captures
    the about-to-be-folded slice; ``recall`` injects captured snippets back."""

    def __init__(self) -> None:
        self.captured: list[str] = []

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        if not self.captured:
            return []
        return [{"content": "Context captured from earlier (now-compacted) turns:\n" + "\n".join(self.captured)}]

    async def remember(self, *, snapshot: MemorySnapshot, session_id):
        if snapshot.status == "compaction":
            text = " ".join((m.get("content") or "") for m in snapshot.messages)
            self.captured.append(text[:2000])


class CoordinatingFold(LLMSummaryFold):
    async def fold(self, rows, *, context):
        result = await super().fold(rows, context=context)
        if result and context.memory:
            messages = context.representation.render(rows)
            await context.memory.remember(snapshot=MemorySnapshot(session_id=context.session_id,
                messages=messages, status='compaction'), session_id=context.session_id)
        return result

class FactModel(LLMService):
    async def complete(self, request, **kwargs):
        text = str(request.messages)
        return LLMResponse(raw_text=CODENAME if CODENAME in text else 'Project progressed.')
    async def close(self):
        pass

async def main():
    memory = CapturingMemory()
    config = AgentLoopConfig(max_tokens=100, context_budget_tokens=40, memory_budget_tokens=10,
        representation=ProjectedRepresentation(), fold_strategy=CoordinatingFold(keep_last_sends=1), memory=memory)
    async with demo_runtime(FactModel(), config=config) as (runtime, loop, target, actor):
        for i in range(3):
            result = await loop.send(target, f'Codename {CODENAME}. '+ 'context '*80, message_id=f'capture-{i}',
                                     actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
        assert any(CODENAME in text for text in memory.captured)
    async with demo_runtime(FactModel(), config=AgentLoopConfig(compactor=None, memory=memory)) as (runtime, loop, target, actor):
        result = await loop.send(target, 'Recall the codename', message_id='recall', actor=actor, deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed' and result.final_text == CODENAME
        print(result.final_text)
        return result.final_text

if __name__ == '__main__':
    asyncio.run(main())
