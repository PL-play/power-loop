"""A legal schema-shaped text never turns a negative loop result into structured success.

The original 192 test drove ``runtime_worker._InputStart.completed``. That class was
deleted by the session-native rewrite, so this version drives the same decision through
the current worker: the deterministic loop entry returns a fixed LoopStatus, and the
production code that follows it (``_map_result``, the ``status == "completed"`` parse
guard, ``final`` persistence and the public result read) runs unchanged. Negative
results must not be parsed, must keep ``structured_output is None``, and must read back
with the mapped failure status and reason from the public result entry.
"""

from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, RequestOptions
from power_loop.agent.stateful_loop import _ExecutionLoop
from tests.runtime_fixture import RunningAgent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'integer'}}, 'required': ['answer']}
LEGAL_JSON = '{"answer":42}'
RESPONSE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'Answer', 'schema': SCHEMA}}


def _fake_loop_status(monkeypatch, status):
    async def run(self, *args, **kwargs):
        return SimpleNamespace(status=status, final_text=LEGAL_JSON, result_seq=None,
                               usage={}, structured_output=None)
    monkeypatch.setattr(_ExecutionLoop, '_run_session_send', run)


async def _run_one(monkeypatch, store, loop_status):
    _fake_loop_status(monkeypatch, loop_status)
    agent = await RunningAgent(
        store, None, config=AgentLoopConfig(context_budget_tokens=10**9),
        options=RequestOptions(response_format=RESPONSE_FORMAT),
    ).start()
    try:
        return await agent.send(LEGAL_JSON)
    finally:
        await agent.aclose()


@pytest.mark.parametrize('loop_status,status,error_code', [
    ('failed', 'failed', 'failed'),
    ('cancelled', 'cancelled', 'cancelled'),
    ('hit_round_limit', 'failed', 'hit_round_limit'),
    ('degraded', 'failed', 'degraded'),
    ('timed_out', 'failed', 'timed_out'),
])
async def test_schema_shaped_negative_result_does_not_become_success(
    registry_store, monkeypatch, loop_status, status, error_code,
):
    store, _ = registry_store
    result = await _run_one(monkeypatch, store, loop_status)

    assert result.status == status
    assert result.structured_output is None
    assert result.error_code == error_code
    # The mapped failure keeps the worker's reason text (the legal JSON body here)
    # instead of silently dropping it.
    assert result.error == LEGAL_JSON


async def test_schema_shaped_completed_result_parses_once(registry_store, monkeypatch):
    """Control: the real parse branch still produces structured output on success."""
    store, _ = registry_store
    result = await _run_one(monkeypatch, store, 'completed')

    assert result.status == 'completed'
    assert dict(result.structured_output) == {'answer': 42}
    assert result.final_text == LEGAL_JSON
    assert result.error is None and result.error_code is None