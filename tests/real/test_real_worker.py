"""A real provider is rebuilt in a separate current-runtime worker process."""
import asyncio
import json
import sys

import pytest

from power_loop import SessionStore

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


async def isolated(tmp_path,mode):
    path=tmp_path/'isolated_test.db'
    process=await asyncio.create_subprocess_exec(sys.executable,'-m','tests.real._runtime_worker_203',
        stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    try:
        async with asyncio.timeout(150):
            output,error=await process.communicate(json.dumps({'path':str(path),'mode':mode}).encode())
        assert process.returncode==0,error.decode()
        return path,json.loads(output.decode().splitlines()[-1])
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_isolated_worker_real_llm_from_env(tmp_path):
    path,saved=await isolated(tmp_path,'send')
    result=saved['result']
    assert result['status']=='completed' and 'paris' in result['final_text'].lower()
    assert result['usage']['total_tokens']>0
    store=await SessionStore.open(str(path),schema='verify')
    try:
        rows=await store.load_all_messages(saved['root']['session_id'])
        assert any(row.role=='assistant' and 'paris' in str(row.content).lower() for row in rows)
    finally:
        await store.close()
