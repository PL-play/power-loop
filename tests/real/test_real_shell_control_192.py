"""A real model starts one local probe; original mechanical control stops it."""
from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.pricing import Free
from power_loop.tools import default_tools as dt
from power_loop.tools.registry import ToolRegistry
from tests.unit.test_shell_process_control_192 import OriginalBackend, _file, _gone

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_model_probe_original_stop_and_transcript(tmp_path, monkeypatch, record_property):
    provider = BoundedProvider(limit=2)
    store = await SessionStore.open(str(tmp_path / 'original.db'))
    manager, backend = dt.BackgroundManager(), OriginalBackend()
    monkeypatch.setattr(dt, 'BG', manager)
    registry = ToolRegistry()
    pidfile = tmp_path / 'child'
    accepted = []

    async def launch_probe():
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, shell_backend=backend)):
            message = await manager.run(f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 60' & wait")
        accepted.append(message.split()[2])
        return message

    registry.register(ToolDefinition(name='launch_probe', description='Start the local test probe once.',
        input_schema={'type': 'object', 'properties': {}}, pricing=Free("test_fixture")), launch_probe)
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=3, max_tokens=2048, compactor=None))
    try:
        sid = await loop.new_session()
        result = await loop.send('Call launch_probe exactly once. Then reply STARTED. Do not wait or call other tools.', sid)
        assert result.status == 'completed' and len(accepted) == 1
        tid = accepted[0]
        pid = await _file(pidfile)
        record_property('owned_process_pid', pid)
        record_property('owned_process_pid', manager.tasks[tid]['proc'].pid)
        assert (await loop.stop_job(sid, 'bg:' + tid))['result'] == 'stopped'
        assert (await store.get_background_task(sid, tid)).status == 'cancelled'
        await _gone(pid)
        messages = await store.load_all_messages(sid)
        assert any(m.role == 'tool' and tid in (m.content or '') for m in messages)
        assert backend.launches == 1 and provider.calls == 2
        record_property('original_task_id', tid)
    finally:
        provider.report(record_property)
        for tid in accepted:
            await manager.cancel_task(tid, shell_term_s=0)
        await loop.aclose(drain_timeout_s=0.1)
        await provider.close()
        await store.close()
