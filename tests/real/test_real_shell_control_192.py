"""A real model starts one command; cancellation uses its original shell target."""
import json

import pytest

from power_loop import AgentLoopConfig, InputRef, RuntimeEnv, SessionStore, ToolDefinition, create_default_tool_registry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.pricing import Free
from power_loop.tools.default_tools import run_background
from tests.unit.test_shell_process_control_192 import OriginalBackend, _file, _gone

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_model_probe_original_stop_and_transcript(tmp_path, record_property, runtime_agent):
    provider, backend = BoundedProvider(limit=3), OriginalBackend()
    store = await SessionStore.open(str(tmp_path/'original.db'))
    registry = create_default_tool_registry(include=['bash'], workspace_dir=tmp_path)
    pidfile, accepted = tmp_path/'child', []
    async def launch_probe():
        message = await run_background(action='run', command=f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 60' & wait")
        accepted.append(json.loads(message))
        return message
    registry.register(ToolDefinition('launch_probe', 'Start the local test probe once.',
        input_schema={'type':'object','properties':{}}, pricing=Free('fixture')), launch_probe)
    loop = await runtime_agent(model=provider, store=store, registry=registry,
        runtime_env=RuntimeEnv(workspace_dir=tmp_path, shell_backend=backend),
        config=AgentLoopConfig(max_rounds=3, max_tokens=2048, compactor=None))
    try:
        result = await loop.send('Call launch_probe exactly once. Then reply STARTED. When its eventual result arrives, acknowledge STOPPED without starting anything again.')
        assert result.status == 'completed' and len(accepted) == 1 and provider.calls == 2
        original = InputRef(**accepted[0]['input_ref'])
        pid = await _file(pidfile)
        record_property('owned_process_pid', pid)
        await loop.runtime.client.cancel(original, actor=loop.actor)
        stopped = await loop.runtime.client.wait(original, actor=loop.actor, deadline_ms=now_ms()+10000)
        assert stopped.status == 'cancelled'
        await _gone(pid)
        await loop.drain()
        messages = await store.load_all_messages(loop.target.session_id)
        assert any(m.role == 'tool' and accepted[0]['job_id'] in (m.content or '') for m in messages)
        assert backend.launches == 1 and provider.calls == 3 and provider.refused_calls == 0
        record_property('original_task_id', stopped.job_ref.job_id)
    finally:
        for receipt in accepted:
            await loop.runtime.client.cancel(InputRef(**receipt['input_ref']), actor=loop.actor)
        await loop.aclose()
        provider.report(record_property)
        await store.close()
