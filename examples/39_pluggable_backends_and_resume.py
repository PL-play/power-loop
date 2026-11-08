"""39 · Rebuild a Runtime from a store DSN, stable profiles and TargetRef.

open_store accepts SQLite, PostgreSQL or MySQL. Install the matching
postgres/mysql extra and use a provisioned database for those backends. This
example uses a disposable SQLite file and verifies its schema before reopening.
"""
import asyncio
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService, SchemaPolicy, StoreSchemaError, open_store


class ResumeModel(LLMService):
    async def complete(self, request, **kwargs):
        assert 'BLUEBIRD-7' in str(request.messages)
        return LLMResponse(raw_text='BLUEBIRD-7')
    async def close(self):
        pass

async def main():
    with TemporaryDirectory(prefix='power_loop_resume_test_') as directory:
        path = Path(directory) / 'store.db'
        store = await open_store(str(path))
        async with demo_runtime(ResumeModel(), store=store) as (runtime, loop, target, actor):
            result = await loop.send(target, 'Remember BLUEBIRD-7', message_id='remember', actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed'
        store = await open_store(str(path), schema=SchemaPolicy.VERIFY)
        async with demo_runtime(ResumeModel(), store=store, target=target) as (runtime, loop, target, actor):
            result = await loop.send(target, 'Recall the code', message_id='recall', actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed' and result.final_text == 'BLUEBIRD-7'
            print(result.final_text)
        try:
            await open_store(str(Path(directory)/'empty.db'), schema=SchemaPolicy.VERIFY)
        except StoreSchemaError as exc:
            assert exc.ddl
            print('Missing schema rejected; provisioning statements:', len(exc.ddl))
        else:
            raise AssertionError('VERIFY must reject a missing schema')
        return result.final_text

if __name__ == '__main__':
    asyncio.run(main())
