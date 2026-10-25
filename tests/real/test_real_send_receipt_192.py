"""One real provider call, then reopen/recall the exact original return value."""
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop import SessionStore
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.send_receipt import read

from .test_real_model_attempts_192 import model_runtime

__all__ = ["model_runtime"]
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_completed_send_receipt_survives_reopen_without_paid_replay(model_runtime, tmp_path, record_property):
    store, manager, runtime, provider, loop, sid = model_runtime
    source = {"namespace": "test.original-run", "payload": {"run_id": 1}}
    with runtime.root_context(sid, **source):
        result = await loop.send("Return only the integer 42. Do not use tools.", sid)
    assert result.status == "completed" and "42" in result.final_text and provider.calls == 1
    [job] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
    receipt = await read(manager, runtime.actor(sid), job["job_id"], expected_host_context=source)
    assert receipt["result"] == asdict(result)
    assert receipt["result"]["usage"]["total_tokens"] > 0
    # Use the actual original SQLite path and table namespace, with no provision.
    reopened = await SessionStore.open(str(tmp_path / "model-attempts.db"),
        table_prefix=store.table_prefix, schema="compatible")
    try:
        assert await read(JobManager(reopened), runtime.actor(sid), job["job_id"], expected_host_context=source) == receipt
    finally:
        await reopened.close()
    child = await asyncio.create_subprocess_exec(sys.executable,
        str(Path(__file__).with_name("_send_receipt_reader_192.py")),
        str(tmp_path / "model-attempts.db"), store.table_prefix,
        json.dumps(asdict(runtime.actor(sid))), job["job_id"], json.dumps(source),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    record_property("original_receipt_reader_pid", child.pid)
    try:
        stdout, stderr = await asyncio.wait_for(child.communicate(), 30)
        assert child.returncode == 0, stderr.decode()[-2000:]
        observed = json.loads(stdout)
        assert observed["receipt"] == receipt and observed["real_provider_requests"] == 0
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
    assert provider.calls == 1
    assert not receipt["execution_authority"] and not receipt["billing_settled"]
