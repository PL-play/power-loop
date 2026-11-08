"""Current-runtime real-test fixture; the test owns its shared provider lifetime."""
from contextlib import asynccontextmanager

from power_loop import LLMService, SessionStore
from tests.runtime_fixture import RunningAgent


class BorrowedProvider(LLMService):
    def __init__(self,provider):
        self.provider=provider
    async def complete(self,request,**kwargs):
        return await self.provider.complete(request,**kwargs)
    def stream(self,request):
        return self.provider.stream(request)
    async def close(self):
        pass


@asynccontextmanager
async def live_agent(tmp_path,model,registry=None,config=None,**options):
    store=await SessionStore.open(str(tmp_path/'live_test.db'))
    runner=await RunningAgent(store,BorrowedProvider(model),registry,config,**options).start()
    try:
        yield runner
    finally:
        await runner.aclose()
        await model.close()
        await store.close()


async def tool_receipt(runner,name):
    import json
    rows=await runner.runtime.store.load_all_messages(runner.target.session_id)
    calls=[call['id'] for row in rows for call in row.tool_calls or [] if call['function']['name']==name]
    assert calls, f'model did not invoke {name}'
    return json.loads(next(row.content for row in rows if row.tool_call_id==calls[-1]))


async def settle(runner):
    import asyncio
    async with asyncio.timeout(120):
        while runner.runtime.worker._tasks or runner.runtime.worker._pending:
            await asyncio.sleep(.02)
