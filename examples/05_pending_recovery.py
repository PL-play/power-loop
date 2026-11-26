"""05 · Align an interrupted tool response as unknown; never repeat the action."""
import asyncio
import time

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService, ToolDefinition, ToolRegistry
from power_loop.runtime.pricing import Free


class ContinueModel(LLMService):
    def __init__(self):
        self.seen_messages = []

    async def complete(self, request, **kwargs):
        messages = list(request.messages)
        self.seen_messages.append(messages)
        assistant_calls = [
            tc
            for msg in messages
            if msg.get('role') == 'assistant'
            for tc in (msg.get('tool_calls') or [])
            if tc.get('id') == 'original'
        ]
        tool_results = [
            msg
            for msg in messages
            if msg.get('role') == 'tool' and msg.get('tool_call_id') == 'original'
        ]
        assert len(assistant_calls) == 1, messages
        assert len(tool_results) == 1, messages
        tool_msg = tool_results[0]
        content = str(tool_msg.get('content') or '')
        assert tool_msg.get('name') == 'external_action', tool_msg
        assert (
            content.startswith('<interrupted:')
            and '结果未知' in content
            and 'interrupted before result was persisted' in content
        ), tool_msg
        return LLMResponse(raw_text='The previous result is unknown. New input can proceed.')

    async def close(self):
        pass


async def main():
    action_calls = 0
    registry = ToolRegistry()

    def action():
        nonlocal action_calls
        action_calls += 1
        raise AssertionError('An unknown action must not be repeated')

    registry.register(ToolDefinition('external_action', 'External action', pricing=Free('demo')), action)
    model = ContinueModel()
    async with demo_runtime(model, registry) as (runtime, loop, target, actor):
        # Simulate an interrupted Turn 1 that emitted external_action(id="original")
        # and recorded pending_tool_seq before the worker lost execution.
        _, batch, _ = await runtime.client.inputs.accept_and_claim(
            target,
            'Run external_action',
            message_id='interrupted-turn',
            actor=actor,
            node_id='crashed-node',
            boot_id='crashed-boot',
            ttl_s=1.0,
        )
        assert batch is not None
        calls = [{'id': 'original', 'type': 'function', 'function': {'name': 'external_action', 'arguments': '{}'}}]
        seq = await runtime.store.append_message(
            target.session_id,
            role='assistant',
            tool_calls=calls,
            send_index=batch.send_index,
            round_index=0,
        )
        await runtime.store.set_pending(
            target.session_id,
            {'assistant_seq': seq, 'round_index': 0, 'tool_call_ids': ['original'], 'tool_calls': calls},
        )
        await runtime.client.inputs.final(
            batch,
            status='interrupted',
            error='worker interrupted before tool result was persisted',
            next_seq=seq + 1,
        )

        # Heal the dangling tool call as unknown without re-executing external_action.
        repaired = await runtime.client.repair(target, mode='heal_broken', actor=actor)
        assert repaired.mode == 'heal_broken'
        assert repaired.repaired_messages == 1
        assert repaired.repaired_projections == 1
        assert action_calls == 0

        result = await loop.send(
            target,
            'Continue with the unknown outcome',
            message_id='new-input',
            actor=actor,
            deadline_ms=int(time.time() * 1000) + 10000,
        )
        assert action_calls == 0
        assert len(model.seen_messages) == 1
        assert result.status == 'completed'
        print(result.final_text)
        return result.final_text


if __name__ == '__main__':
    asyncio.run(main())
