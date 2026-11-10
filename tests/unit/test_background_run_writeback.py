"""Background results use persisted jobs and current runtime lifecycle."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, RuntimeEnv, create_default_tool_registry
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import JobRef
from power_loop.runtime.background import load_record
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_timers import _Scripted, _tool_resp

__all__ = ['registry_store']
pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'timeout', 'cancel'])
async def test_background_command_result_survives_worker_shutdown(registry_store, tmp_path, outcome):
    store, _ = registry_store
    command = 'echo command-finished' if outcome == 'success' else 'sleep 30; echo never'
    model = _Scripted([_tool_resp('bg', 'background_run', json.dumps({'action':'run','command':command})),
        LLMResponse(raw_text='parent returned')])
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    env = RuntimeEnv(workspace_dir=tmp_path, background_timeout_s=.2 if outcome == 'timeout' else 30)
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None), runtime_env=env).start()
    try:
        assert (await runner.send('start command')).status == 'completed'
        rows = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(row.content for row in rows if row.tool_call_id == 'bg'))
        ref = JobRef(**receipt['job_ref'])
        if outcome == 'cancel':
            async with asyncio.timeout(10):
                while True:
                    status = await runner.runtime.client.inspect(ref, actor=runner.actor)
                    if status.job_ref.generation > 0:
                        break
                    await asyncio.sleep(.01)
            await runner.runtime.client.cancel(status.job_ref, actor=runner.actor)
        result = await runner.runtime.client.wait(ref, actor=runner.actor, deadline_ms=now_ms()+15000)
        assert result.status == {'success':'completed','timeout':'failed','cancel':'cancelled'}[outcome]
        if outcome == 'success':
            assert 'command-finished' in result.final_text
        elif outcome == 'timeout':
            assert 'timed out' in result.final_text and 'never' not in result.final_text
        await runner.drain()
        await runner.runtime.worker.shutdown()
        saved = await load_record(store, runner.target.session_id, ref.job_id)
        assert saved['status'] == result.status and saved['result'] == result.final_text
        assert (await runner.runtime.client.wait(ref, actor=runner.actor, deadline_ms=now_ms()+1000)) == result
    finally:
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('allowed', [True, False])
async def test_action_tool_failure_and_async_permission(registry_store, tmp_path, allowed):
    from power_loop import ToolDefinition
    store, _ = registry_store
    invoked = []
    async def image():
        invoked.append(True)
        raise RuntimeError('provider down')
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    registry.register(ToolDefinition('image', 'generate', async_capable=allowed, pricing=Free('fixture')), image)
    model = _Scripted([_tool_resp('bg', 'background_run', '{"action":"tool","tool":"image","args":{}}')])
    runner = await RunningAgent(store, model, registry).start()
    try:
        await runner.send('start image')
        rows = await store.load_all_messages(runner.target.session_id)
        output = next(row.content for row in rows if row.tool_call_id == 'bg')
        jobs = await store._db.fetchall(f"SELECT job_id FROM {store.table_prefix}jobs WHERE kind='background_tool'")
        if allowed:
            ref = JobRef(**json.loads(output)['job_ref'])
            result = await runner.runtime.client.wait(ref, actor=runner.actor, deadline_ms=now_ms()+10000)
            assert result.status == 'failed' and 'provider down' in result.final_text
            assert invoked == [True] and len(jobs) == 1
        else:
            assert 'does not allow background execution' in output and not invoked and not jobs
    finally:
        await runner.aclose()


def test_async_capable_suffix_only_with_background_run_mounted():
    from power_loop.contracts.tools import ToolDefinition

    async def _h(**kw):
        return "x"

    def _mk(*defs):
        r = ToolRegistry()
        for d in defs:
            from power_loop.tools.default_tools import DEFAULT_TOOL_HANDLERS
            r.register(d, DEFAULT_TOOL_HANDLERS["background_run"] if d.name == "background_run" else _h)
        return r

    gen = ToolDefinition(name="generate_image", description="生成图片。", async_capable=True, pricing=Free("test_fixture"))
    bg = ToolDefinition(name="background_run", description="bg。", pricing=Free("test_fixture"))
    plain = ToolDefinition(name="write_file", description="写文件。", pricing=Free("test_fixture"))

    with_bg = {t["function"]["name"]: t["function"]["description"]
               for t in _mk(gen, bg, plain).to_openai_tools()}
    assert "可异步" in with_bg["generate_image"]
    assert "可异步" not in with_bg["write_file"], "未标记的工具不加后缀"
    assert "可异步" not in with_bg["background_run"]

    without_bg = {t["function"]["name"]: t["function"]["description"]
                  for t in _mk(gen, plain).to_openai_tools()}
    assert "⏳" not in without_bg["generate_image"], "background_run 不在场就别教模型调它"


def test_fast_readonly_tools_get_no_background_suffix_but_stay_async_capable():
    """read_file/grep/glob/recall_send 从没被 background_run 跑过：不追加后缀，
    但 async_capable 不动（同轮并发靠它）。后缀本身不再教模型去 check 取结果。"""
    from power_loop.contracts.tools import ToolDefinition

    async def _h(**kw):
        return "x"

    r = ToolRegistry()
    for d in (
        ToolDefinition(name="read_file", description="读。", async_capable=True, background_hint=False, pricing=Free("test_fixture")),
        ToolDefinition(name="generate_image", description="生成图片。", async_capable=True, pricing=Free("test_fixture")),
        ToolDefinition(name="background_run", description="bg。", pricing=Free("test_fixture")),
    ):
        from power_loop.tools.default_tools import DEFAULT_TOOL_HANDLERS
        r.register(d, DEFAULT_TOOL_HANDLERS["background_run"] if d.name == "background_run" else _h)
    out = {t["function"]["name"]: t["function"]["description"] for t in r.to_openai_tools()}
    assert "可异步" not in out["read_file"]
    assert r.get("read_file").definition.async_capable is True
    assert "可异步" in out["generate_image"]
    assert "action=\"check\")" not in out["generate_image"], "后缀不该教它去 check 取结果"


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['enable', 'execution'])
async def test_cancel_after_claim_before_callback_records_original_result(registry_store, monkeypatch, boundary, tmp_path):
    from contextlib import asynccontextmanager

    from power_loop import ToolDefinition
    from power_loop.runtime import model_attempts
    from power_loop.runtime.task_runtime import TaskRuntime

    entered, release, called = asyncio.Event(), asyncio.Event(), []
    async def barrier(binding):
        evidence = getattr(binding, "resource_evidence", None) or {}
        if evidence.get("lightweight") and not evidence.get("upgraded"):
            return  # Ordinary send has no task row until it starts a background job.
        if binding.runtime.manager._joined_write.get() is not None:
            return  # Main admission holds its transaction; this barrier is after child claim.
        row = await binding.runtime.manager.get_job(binding.actor, binding.lease.job_id)
        if row.kind == 'background_tool':
            entered.set()
            await release.wait()
    if boundary == 'enable':
        original = model_attempts.enable
        async def enable(binding):
            await barrier(binding)
            return await original(binding)
        monkeypatch.setattr(model_attempts, 'enable', enable)
    else:
        original = TaskRuntime.execution
        @asynccontextmanager
        async def execution(runtime, binding, **kwargs):
            await barrier(binding)
            async with original(runtime, binding, **kwargs) as result:
                yield result
        monkeypatch.setattr(TaskRuntime, 'execution', execution)
    async def operation():
        called.append('unexpected')
        return 'effect'
    registry = create_default_tool_registry(workspace_dir=tmp_path)
    registry.register(ToolDefinition('operation', 'one finite action', async_capable=True, pricing=Free('fixture')), operation)
    model = _Scripted([_tool_resp('bg', 'background_run', '{"action":"tool","tool":"operation","args":{}}'), LLMResponse(raw_text='queued')])
    runner = await RunningAgent(registry_store[0], model, registry).start()
    try:
        await runner.send('launch')
        await asyncio.wait_for(entered.wait(), 10)
        [row] = await registry_store[0]._db.fetchall(f"SELECT * FROM {registry_store[0].table_prefix}jobs WHERE kind='background_tool'")
        ref = JobRef(registry_store[0].store_id, row['job_id'], row['generation'])
        assert ref.generation == 1
        await runner.runtime.client.cancel(ref, actor=runner.actor)
        release.set()
        result = await runner.runtime.client.wait(ref, actor=runner.actor, deadline_ms=now_ms()+15000)
        assert result.status == 'cancelled' and result.error_code == 'cancel_requested'
        await runner.drain()
        assert not called
        assert (await load_record(registry_store[0], runner.target.session_id, ref.job_id))['status'] == 'cancelled'
    finally:
        release.set()
        await runner.aclose()
