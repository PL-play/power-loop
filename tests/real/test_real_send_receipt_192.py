"""A real provider result remains readable through the original InputRef."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from power_loop import SessionStore
from power_loop.agent.runtime_store import RuntimeInputStore

from .test_real_model_attempts_192 import model_runtime

__all__ = ["model_runtime"]
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


def _assistant(rows):
    return [row.content for row in rows if row.role == "assistant"]


async def test_real_completed_send_receipt_survives_reopen_without_paid_replay(
    model_runtime, tmp_path, record_property,
):
    store, provider, loop, sid = model_runtime
    result = await loop.send("Return only the integer 42. Do not use tools.")
    assert result.status == "completed" and "42" in result.final_text and provider.calls == 1
    assert result.usage["total_tokens"] > 0
    saved = _assistant(await store.load_all_messages(sid))
    assert any(text and "42" in text for text in saved)

    reopened = await SessionStore.open(
        str(tmp_path / "model-attempts.db"), table_prefix=store.table_prefix, schema="verify",
    )
    try:
        assert _assistant(await reopened.load_all_messages(sid)) == saved
        cold = await RuntimeInputStore(reopened).get_result(result.input_ref, actor=loop.actor)
        assert cold == result
    finally:
        await reopened.close()

    child = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).with_name("_send_receipt_reader_192.py")),
        str(tmp_path / "model-attempts.db"),
        store.table_prefix,
        sid,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
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
