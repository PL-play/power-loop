"""Background commands use the factory's explicit shell and process-control ports."""
import json
import os
from collections.abc import Hashable
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, RuntimeEnv, create_default_tool_registry
from power_loop.agent.runtime_store import now_ms
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_timers import _Scripted, _tool_resp

__all__ = ['registry_store']

class _MarkerBackend:
    """A stand-in sandbox backend: tags the launch env with a unique marker and
    records that its launch hooks were consulted."""

    def launch_argv(self, workspace_dir: Path) -> list[str]:
        return ["/bin/bash", "--norc", "--noprofile"]

    def launch_cwd(self, workspace_dir: Path) -> str | None:
        return str(workspace_dir)

    def launch_env(self, workspace_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["PL_SANDBOX_MARKER"] = "sandbox-123"
        return env

    def session_key(self, workspace_dir: Path) -> Hashable:
        return ("marker", str(workspace_dir))



@pytest.mark.parametrize('bound',[False,True])
@pytest.mark.parametrize('invalid_control',[False,True])
async def test_background_uses_factory_environment_and_control(registry_store,tmp_path,invalid_control,bound):
    store,_=registry_store
    model=_Scripted(responses=[_tool_resp('background','background_run',json.dumps({
        'action':'run','command':'echo marker=$PL_SANDBOX_MARKER; touch ran'}))])
    registry=create_default_tool_registry(preset='core',include=['background_run','bash'],bind=bound,workspace_dir=tmp_path)
    environment=RuntimeEnv(workspace_dir=tmp_path,shell_backend=_MarkerBackend())
    agent=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None),runtime_env=environment,
        process_control=(lambda *args,**kw: object()) if invalid_control else None).start()
    try:
        assert (await agent.send('run in sandbox')).status=='completed'
        rows=await store.load_active_messages(agent.target.session_id)
        receipt=json.loads(next(r.content for r in rows if r.role=='tool'))
        from power_loop import JobRef
        ref=JobRef(**receipt['job_ref'])
        result=await agent.runtime.client.wait(ref,actor=agent.actor,deadline_ms=now_ms()+10000)
        if invalid_control:
            assert result.status=='failed' and result.error == 'TypeError' and 'ShellProcessControl' in result.final_text
            assert not (tmp_path/'ran').exists()
        else:
            assert result.status=='completed' and 'marker=sandbox-123' in result.final_text
            assert (tmp_path/'ran').exists()
    finally:
        await agent.aclose()
