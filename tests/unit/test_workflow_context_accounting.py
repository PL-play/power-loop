"""Real in-process workflow execution with deterministic provider fault injection.

These are integration contracts, not task-quality or real-provider cost experiments.
"""
import asyncio
import json

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMResponse, LLMTokenUsage
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.context_cache import ContextCache
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.tool_context import ToolContextConfig
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine
from tests.unit.test_stateful_loop import _Scripted, _tool_resp


def spec(node, tools):
    return WorkflowSpec.from_json({'name': node, 'input': 'check artifact', 'root': {
        'type': 'agent', 'id': node, 'spec': {'name': node, 'system_prompt': node, 'tools': tools}}})


async def test_nested_workflow_late_summary_keeps_owner_ancestry_and_raw_evidence(tmp_path):
    store = await SessionStore.open(str(tmp_path / 'nested.db'))
    gate, started = asyncio.Event(), asyncio.Event()
    receipts, nested = [], []

    class SlowSummary:
        async def complete(self, request):
            started.set()
            await gate.wait()
            return LLMResponse(raw_text='critical.py failed with exit 42', token_usage=LLMTokenUsage(
                prompt_tokens=100, completion_tokens=20, total_tokens=120))

    async def deliver(record):
        receipts.append(dict(record))

    registry = ToolRegistry()
    registry.register(ToolDefinition('inspect_artifact', 'inspect artifact', project=lambda a, r: 'checked'),
                      lambda **kw: 'details\n' * 300 + 'critical.py failed exit 42')

    async def run_nested():
        result = await WorkflowEngine(get_current_loop(), run_id='inner-run').run(spec('inner', ['inspect_artifact']))
        nested.append(result)
        return result.status

    registry.register(ToolDefinition('nested', 'run inner workflow'), run_nested)
    llm = _Scripted([_tool_resp('nested-call', 'nested', '{}'),
                     _tool_resp('bash-call', 'inspect_artifact', '{"description":"critical inspection"}'),
                     LLMResponse(raw_text='inner done'), LLMResponse(raw_text='outer done')])
    cfg = ToolContextConfig(summary_min_chars=0, overrides={'inspect_artifact': {'general': 'llm_summary'}},
                            summary_llm=SlowSummary(), deliver_usage=deliver)
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=registry,
        config=AgentLoopConfig(tool_context=cfg, representation=ProjectedRepresentation(), compactor=None))
    try:
        result = await asyncio.wait_for(WorkflowEngine(loop, run_id='outer-run').run(spec('outer', ['nested', 'inspect_artifact'])), 5)
        assert result.status == 'completed' and nested[0].status == 'completed'
        await asyncio.wait_for(started.wait(), 2)
        inner_sid = nested[0].results['inner'].session_id
        outer_sid = result.results['outer'].session_id
        snapshot = dict(result.usage)
        raw = await store.load_all_messages(inner_sid)
        inner_before = await store.get_session_stats(inner_sid)
        outer_before = await store.get_session_stats(outer_sid)
        gate.set()
        cache = ContextCache(store)
        for _ in range(200):
            await cache.reconcile(deliver)
            rows = await cache.db.fetchall(f"SELECT record_json FROM {cache.attempts} WHERE outcome='success'")
            if rows and receipts:
                break
            await asyncio.sleep(.01)
        assert len(rows) == 1
        receipt = json.loads(rows[0]['record_json'])
        assert receipt['owner_session_id'] == inner_sid
        assert receipt['workflow_run_id'] == 'inner-run' and receipt['node_id'] == 'inner'
        assert receipt['workflow_ancestry'][-1] == {
            'session_id': outer_sid, 'workflow_run_id': 'outer-run', 'node_id': 'outer', 'node_attempt': 1}
        assert result.usage == snapshot  # never mutate a result after returning
        assert await store.load_all_messages(inner_sid) == raw  # recall source remains intact
        source = next(r for r in raw if r.seq == receipt['source_seq'])
        assert 'critical.py' in json.dumps(source.content)
        assert (await store.get_session_stats(inner_sid)).total_tokens == inner_before.total_tokens + 120
        assert (await store.get_session_stats(outer_sid)) == outer_before
        assert len({r['attempt_id'] for r in receipts}) == 1
        resumed_spec = WorkflowSpec.from_json({'name': 'resumed', 'root': {'type': 'sequence', 'steps': [
            {'type': 'agent', 'id': name, 'spec': {'name': name, 'system_prompt': name}}
            for name in ['inner', 'tail']]}})
        resumed = await WorkflowEngine(loop, run_id='inner-run', replay={'inner': nested[0].results['inner']}).run(resumed_spec)
        tail = await store.get_session(resumed.results['tail'].session_id)
        assert tail.metadata['workflow_ancestry'] == receipt['workflow_ancestry']

    finally:
        gate.set()
        await store.close()


async def _crash_workflow(path):
    """Child-process entry: exit after a real workflow completes, without cleanup."""
    import os
    store = await SessionStore.open(path)
    started = asyncio.Event()
    class PendingProvider:
        async def complete(self, request):
            started.set()
            await asyncio.Event().wait()
    registry = ToolRegistry()
    registry.register(ToolDefinition('inspect_artifact', 'inspect'), lambda: 'critical.py exit42\n' * 200)
    config = ToolContextConfig(summary_min_chars=0, overrides={'inspect_artifact': {'general': 'llm_summary'}},
                               summary_llm=PendingProvider())
    loop = StatefulAgentLoop(store=store, tool_registry=registry,
        llm=_Scripted([_tool_resp('inspect', 'inspect_artifact', '{}'), LLMResponse(raw_text='finished')]),
        config=AgentLoopConfig(tool_context=config, compactor=None, representation=ProjectedRepresentation()))
    result = await WorkflowEngine(loop, run_id='hard-exit-run').run(spec('review', ['inspect_artifact']))
    assert result.status == 'completed'
    await asyncio.wait_for(started.wait(), 2)
    os._exit(23)


async def test_completed_workflow_hard_exit_reconciles_without_reexecuting(tmp_path):
    import subprocess
    import sys
    path = str(tmp_path / 'crash.db')
    script = ('import asyncio,sys; from tests.unit.test_workflow_context_accounting import _crash_workflow; '
              'asyncio.run(_crash_workflow(sys.argv[1]))')
    process = await asyncio.to_thread(subprocess.run, [sys.executable, '-c', script, path],
                                     capture_output=True, text=True, timeout=20)
    assert process.returncode == 23, process.stderr
    store = await SessionStore.open(path)
    cache = ContextCache(store)
    delivered = []
    async def receiver(record):
        delivered.append(record)
    try:
        rows = await cache.db.fetchall(f'SELECT * FROM {cache.attempts}')
        assert len(rows) == 1 and rows[0]['outcome'] == 'started'
        before = json.loads(rows[0]['record_json'])
        owner = before['owner_session_id']
        original = await store.load_all_messages(owner)
        old_stats = await store.get_session_stats(owner)
        await cache.db.execute(f'UPDATE {cache.cache} SET lease_until=0')
        await cache.reconcile(receiver)
        await cache.reconcile(receiver)
        assert len(delivered) == 1 and delivered[0]['usage_unknown']
        assert delivered[0]['workflow_run_id'] == 'hard-exit-run'
        assert delivered[0]['node_id'] == 'review' and delivered[0]['node_attempt'] == 1
        assert delivered[0]['owner_session_id'] == owner and delivered[0]['source_send_index'] == 1
        stats = await store.get_session_stats(owner)
        assert stats.llm_calls == old_stats.llm_calls + 1 and stats.total_tokens == old_stats.total_tokens
        assert await store.load_all_messages(owner) == original
        job = await cache.db.fetchone(f'SELECT executor_key FROM {cache.cache}')
        assert await cache.claim(job['executor_key']) is None
    finally:
        await store.close()


async def test_plain_subagent_inherits_workflow_ancestry(tmp_path):
    from power_loop.core.agent_context import reset_session_id, set_session_id
    from power_loop.runtime.spec import AgentSpec, run_agent_spec
    store = await SessionStore.open(str(tmp_path / 'child.db'))
    loop = StatefulAgentLoop(store=store, llm=_Scripted([LLMResponse(raw_text='done')]),
                             config=AgentLoopConfig(compactor=None))
    parent = await loop.new_session(metadata={'workflow_run_id': 'outer', 'workflow_node_id': 'node', 'attempt': 2})
    token = set_session_id(parent)
    try:
        result = await run_agent_spec(AgentSpec(name='helper', system_prompt='help', lifecycle='linked'),
                                      'check', parent_loop=loop)
        child = await store.get_session(result['session_id'])
        assert child.metadata['workflow_ancestry'] == [
            {'session_id': parent, 'workflow_run_id': 'outer', 'node_id': 'node', 'node_attempt': 2}]
    finally:
        reset_session_id(token)
        await store.close()
