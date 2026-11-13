"""Real session results are inspectable without replaying the provider request."""
import pytest

from power_loop.agent.runtime_store import RuntimeInputStore

from .test_real_model_attempts_192 import model_runtime

__all__ = ["model_runtime"]
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_session_inspection_reads_original_result_without_replay(model_runtime):
    store, provider, loop, _ = model_runtime
    result = await loop.send("Reply exactly 42. Do not use any tools.")
    assert result.status == "completed"

    port = RuntimeInputStore(store)
    status = await port.inspect(result.input_ref, actor=loop.actor)
    assert status.input_state == "delivered" and status.terminal_status == "completed"
    assert status.send_index == result.send_index and status.result_available
    target_status = await port.inspect(loop.target, actor=loop.actor)
    assert target_status.execution_state == "idle" and target_status.send_index == result.send_index
    page = await port.messages(loop.target, actor=loop.actor)
    assert any(row.role == "assistant" and "42" in (row.content or "") for row in page.items)
    assert provider.calls == 1
