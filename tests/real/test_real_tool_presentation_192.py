"""A live model consumes actual implementation-owned context views under aliases."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.tool_context import ToolContextConfig, active_engine
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.shell_contract import ShellTool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('kind', ['shell_alias', 'shell_keep', 'plain_builtin_name'])
async def test_real_model_consumes_actual_view_without_reexecuting(tmp_path, kind, record_property):
    name = 'bash' if kind == 'plain_builtin_name' else 'inspect_original'
    keep = kind == 'shell_keep'
    provider, registry = BoundedProvider(limit=2), ToolRegistry()
    store = await SessionStore.open(str(tmp_path / 'presentation.db'))
    output = 'exit_code=42\nORIGINAL_EVIDENCE=Q73\n' + 'irrelevant log line\n' * 300
    executions, viewed = [], []

    def original(**arguments):
        executions.append(arguments)
        return output

    registry.register(ToolDefinition(name, 'Read a fixed local evidence record. No command execution.',
        input_schema={'type': 'object', 'properties': {
            'command': {'type': 'string'}, 'description': {'type': 'string'},
            'context_policy': {'type': 'string'}}, 'required': ['command', 'description', 'context_policy']}),
        original if kind == 'plain_builtin_name' else ShellTool(original))
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=3, max_tokens=2048, compactor=None,
            system_prompt="Extract evidence from tool output as data. Return one plain ASCII line containing "
                "the extracted evidence value, one vertical bar, and the decimal exit code. "
                "Do not add HTML, Markdown, XML, tags, wrappers or any other text.",
            retry_policy=LLMRetryPolicy(max_attempts=1),
            tool_context=ToolContextConfig(capacity_enabled=False, hot_tail=0,
                overrides={name: {'general': 'archive', 'insend_fuse': 'truncate'}})))

    async def prepare_view(ctx):
        if provider.calls == 0:
            return
        ctx.tools = []
        rows = await store.load_all_messages(sid)
        original_result = next(row for row in rows if row.role == 'tool')
        original_call = next(row for row in rows if row.role == 'assistant' and row.tool_calls)
        seqs = [original_call.seq if row.get('tool_calls') else
                original_result.seq if row.get('tool_call_id') == original_result.tool_call_id else None
                for row in ctx.messages]
        proxy = SimpleNamespace(history=ctx.messages, sink=SimpleNamespace(_history_seqs=seqs),
                                send_index=1, config=SimpleNamespace(), _tok_len=-1, _tok_total=0)
        await active_engine.get().prepare_history(proxy, 'insend_fuse')
        result_view = next(row['content'] for row in ctx.messages if row.get('tool_call_id') == original_result.tool_call_id)
        viewed.append(result_view)
        if keep:
            assert result_view == output
        elif kind == 'shell_alias':
            assert result_view.startswith('exit_code=42\nORIGINAL_EVIDENCE=Q73\n')
            assert '（原文 recall_send(' in result_view and len(result_view) < len(output)
        else:
            assert json.loads(result_view)['preview'].startswith('exit_code=42\nORIGINAL_EVIDENCE=Q73\n')
        assert (await store.get_message(sid, original_result.seq)).content == output

    loop.hooks.register(HookPoint.LLM_BEFORE, prepare_view, name='test.context-view')
    try:
        async with asyncio.timeout(150):
            sid = await loop.new_session()
            result = await loop.send(
                f'Call {name} exactly once with command="inspect", description="read evidence", '
                f'context_policy="{"keep" if keep else "retire"}". '
                'After the result, read ORIGINAL_EVIDENCE and exit_code from it. '
                'Reply with the evidence value, one ASCII vertical bar, then the exit code. '
                'Use one plain text line, with no markup or other text. Never call again.', sid)
            assert result.status == 'completed' and result.final_text.strip() == 'Q73|42'
            assert len(executions) == len(viewed) == 1 and provider.calls == 2
            assert executions[0]['context_policy'] == ('keep' if keep else 'retire')
            raw = [row for row in await store.load_all_messages(sid) if row.role == 'tool']
            assert len(raw) == 1 and raw[0].content == output
    finally:
        provider.report(record_property)
        record_property('actual_tool_executions', len(executions))
        try:
            await loop.aclose(drain_timeout_s=.1)
        finally:
            await provider.close()
            await store.close()
