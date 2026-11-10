"""One real provider call. An ordinary reply stays in messages, not a task row."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from power_loop import SessionStore

from .test_real_model_attempts_192 import model_runtime

__all__ = ["model_runtime"]
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


def _assistant(rows):
    return [row.content for row in rows if row.role == "assistant"]


async def test_real_completed_send_receipt_survives_reopen_without_paid_replay(model_runtime, tmp_path, record_property):
    store, manager, runtime, provider, loop, sid = model_runtime
    source = {"namespace": "test.original-run", "payload": {"run_id": 1}}
    with runtime.root_context(sid, **source):
        result = await loop.send("Return only the integer 42. Do not use tools.", sid)
    assert result.status == "completed" and "42" in result.final_text and provider.calls == 1
    assert result.usage["total_tokens"] > 0
    assert not await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs}")
    saved = _assistant(await store.load_all_messages(sid))
    assert any(text and "42" in text for text in saved)
    # Use the actual original SQLite path and table namespace, with no provision.
    reopened = await SessionStore.open(str(tmp_path / "model-attempts.db"),
        table_prefix=store.table_prefix, schema="verify")
    try:
        assert _assistant(await reopened.load_all_messages(sid)) == saved
    finally:
        await reopened.close()
    child = await asyncio.create_subprocess_exec(sys.executable,
        str(Path(__file__).with_name("_send_receipt_reader_192.py")),
        str(tmp_path / "model-attempts.db"), store.table_prefix, sid,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    record_property("original_receipt_reader_pid", child.pid)
    try:
        stdout, stderr = await asyncio.wait_for(child.communicate(), 30)
        assert child.returncode == 0, stderr.decode()[-2000:]
        observed = json.loads(stdout)
        assert observed["assistant"] == saved and observed["real_provider_requests"] == 0
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
    assert provider.calls == 1
