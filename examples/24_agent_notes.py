"""24 · Notes use the current execution's store; a later input recalls them."""
import asyncio
import json
import time

from _runtime import demo_runtime

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    NoteMemory,
    NotesPolicy,
    SessionStore,
    create_default_tool_registry,
)


class NotesModel(LLMService):
    def __init__(self):
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(raw_text='', tool_calls=[{'id':'note','type':'function','function':{
                'name':'note','arguments':json.dumps({'action':'add','content':'BLUEBIRD deploys to ap-southeast-1'})}}])
        assert 'BLUEBIRD' in str(request.messages)
        return LLMResponse(raw_text='BLUEBIRD / ap-southeast-1')

    async def close(self):
        pass

async def main():
    store = await SessionStore.open(':memory:')
    policy = NotesPolicy(max_notes=20)
    config = AgentLoopConfig(compactor=None, memory=NoteMemory(store, policy=policy), notes_policy=policy)
    async with demo_runtime(NotesModel(), create_default_tool_registry(include=['note']), store=store, config=config) as (runtime, loop, target, actor):
        for index, text in enumerate(['Record my project', 'Recall my project']):
            result = await loop.send(target, text, message_id=f'note-{index}', actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
            print(result.final_text)
        notes = await store.list_notes(target.session_id)
        assert len(notes) == 1 and 'BLUEBIRD' in notes[0].content
        await store.close_session(target.session_id)
        assert not await store.list_notes(target.session_id)

if __name__ == '__main__':
    asyncio.run(main())
