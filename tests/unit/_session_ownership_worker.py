"""Drive an actual SDK send in each independent interpreter."""
import asyncio
import json
import os
import signal
import sys

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    SessionBusy,
    SessionLeaseLost,
    StatefulAgentLoop,
    ToolDefinition,
)
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.factory import open_store
from power_loop.tools.registry import ToolRegistry

from ._shared_session_coordinator import SharedSessionCoordinator


def report(**data):
    print(json.dumps(data), flush=True)


async def main():
    dsn, prefix, sid, path, role, managed = sys.argv[1:]
    store = await open_store(dsn, table_prefix=prefix, schema='compatible')
    coordinator = SharedSessionCoordinator(path)
    registry = ToolRegistry()

    async def hold():
        report(event='running')
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            denied = []
            for action in (
                lambda: store.append_message(sid, role='assistant', content='late-old-write'),
                lambda: store.set_pending(sid, None),
                lambda: store.set_round_index(sid, 999),
                lambda: store.set_runtime_state(sid, 'late-state', {'value': 'old'}),
            ):
                try:
                    await action()
                except SessionLeaseLost:
                    denied.append(True)
                else:
                    denied.append(False)
            report(event='old_write_denied', denied=denied)
            raise

    registry.register(ToolDefinition('hold', 'Wait', input_schema={'type': 'object', 'properties': {}},
                                     pricing=Free('test')), hold)

    class Model(LLMService):
        calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            if role == 'old':
                return LLMResponse(raw_text='', tool_calls=[{'id': 'original-call', 'type': 'function',
                    'function': {'name': 'hold', 'arguments': '{}'}}])
            messages = request.messages
            assert any(m.get('content') == 'original' for m in messages)
            assert any(m.get('role') == 'tool' and m.get('tool_call_id') == 'original-call'
                       and '未知' in m.get('content', '') for m in messages)
            return LLMResponse(raw_text='new owner finished')

        def stream(self, request):
            raise AssertionError('nonstream test')

        async def close(self):
            return None

    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.task_runtime import TaskRuntime
    runtime = (TaskRuntime(JobManager(store), tenant_id='ownership', principal_id='host',
                           executor_id=role, boot_id=role) if managed == 'yes' else None)
    model = Model()
    loop = StatefulAgentLoop(llm=model, store=store, tool_registry=registry, task_runtime=runtime,
        session_coordinator=coordinator, config=AgentLoopConfig(compactor=None, max_rounds=2,
                                                               session_lease_ttl_s=.6, delegation_policy="single_level"))
    try:
        if role == 'old':
            async def pause():
                assert (await asyncio.to_thread(sys.stdin.readline)).strip() == 'pause'
                report(event='pausing')
                os.kill(os.getpid(), signal.SIGSTOP)
            pause_task = asyncio.create_task(pause())
            try:
                await loop.send('original', sid)
            except SessionLeaseLost:
                report(event='stopped')
            await pause_task
        else:
            await loop.prewarm(sid)
            report(event='warmed', next_seq=loop._session_cache[sid].next_seq)
            while (command := (await asyncio.to_thread(sys.stdin.readline)).strip()) != 'quit':
                if command == 'try':
                    try:
                        await loop.send('competing', sid)
                    except SessionBusy:
                        report(event='busy', calls=model.calls)
                elif command == 'takeover':
                    result = await loop.send('takeover', sid)
                    state = await store.get_state(sid)
                    report(event='healed', status=result.status, pending=state.pending, calls=model.calls)
                else:
                    raise AssertionError(command)
    finally:
        await loop.aclose()
        await store.close()


if __name__ == '__main__':
    asyncio.run(main())
