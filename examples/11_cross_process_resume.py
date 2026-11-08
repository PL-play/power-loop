"""11 · Sequential processes reopen the same store and TargetRef.

The first worker shuts down and releases the default store lock before the
second starts. Independent simultaneous workers require shared adapters.
"""
import asyncio
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService, SessionStore, TargetRef


class RecallModel(LLMService):
    async def complete(self, request, **kwargs):
        assert 'Alan' in str(request.messages) and '37' in str(request.messages)
        return LLMResponse(raw_text='Alan / 37')

    async def close(self):
        pass


async def phase(db_path, sid=None):
    store = await SessionStore.open(db_path)
    target = TargetRef(store.store_id, sid) if sid else None
    async with demo_runtime(RecallModel(), store=store, target=target) as (runtime, loop, target, actor):
        result = await loop.send(target, 'What did I tell you?' if sid else 'My name is Alan and my number is 37.',
            message_id='phase-2' if sid else 'phase-1', actor=actor, deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed' and result.final_text == 'Alan / 37'
        print(result.final_text)
        return target.session_id


def main():
    if len(sys.argv) == 4 and sys.argv[1] == 'phase2':
        asyncio.run(phase(sys.argv[2], sys.argv[3]))
        return
    with tempfile.TemporaryDirectory(prefix='power_loop_reopen_test_') as directory:
        path = str(Path(directory)/'sessions.db')
        sid = asyncio.run(phase(path))
        subprocess.run([sys.executable, __file__, 'phase2', path, sid], check=True)


if __name__ == '__main__':
    main()
