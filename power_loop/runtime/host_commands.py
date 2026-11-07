"""Factory-resolved host commands execute on the ordinary input's worker.

The input is the command's authority. It is never disguised as an assistant
tool call. Internal model/paid calls retain this send's original budget owner.
"""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
from dataclasses import asdict

from power_loop.agent.runtime_types import RuntimeRequestError, canonical
from power_loop.agent.stateful_loop import StatefulResult
from power_loop.contracts.hook_contexts import SessionEndCtx, SessionStartCtx
from power_loop.contracts.hooks import HookPoint
from power_loop.core import agent_context

from . import tool_attempts
from .control_runtime import current
from .jobs import _hash
from .native_returns import _Receipt
from .nested_tools import join_children
from .pricing import Free, PricedCall
from .send_receipt import KEY, capture
from .task_runtime import _SEND
from .tool_billing import receipt_context


async def run(loop, ctx, start, command, send_index):
    name, arguments = command
    registered = loop.tool_registry.get(name)
    if registered is None or ctx.options.tools is not None and name not in ctx.options.tools:
        raise RuntimeRequestError('forbidden', 'host command outside the frozen tool grant')
    # This outer router is free; supplier/model calls use the regular priced
    # inner-call interface. A paid tool requires its own real supplier receipt.
    if type(registered.definition.pricing) is not Free:
        raise RuntimeRequestError('invalid_options', 'host command router must be explicitly free')
    loop.tool_registry._raise_if_invalid(name, arguments)
    store, sid = loop.store, ctx.session_id
    loop._host_command = {'next_child': 0, 'consumed': [], 'send_index': send_index}
    async with store.send_read_scope(sid), loop._runner.session_async(session_id=sid):
        with ExitStack() as scopes:
            for setter, value, reset in (
                (agent_context.set_current_loop, loop, agent_context.reset_current_loop),
                (agent_context.set_effective_tools, frozenset(ctx.options.tools) if ctx.options.tools is not None else None,
                 agent_context.reset_effective_tools),
                (agent_context.set_current_cancel_token, ctx.stop, agent_context.reset_current_cancel_token),
            ):
                scopes.callback(reset, setter(value))
            async with loop.task_runtime.model_send(loop, sid, send_index,
                    input_start=start, cancel_token=ctx.stop) as evidence:
                await loop.hooks.run_typed_async(HookPoint.SESSION_START,
                    SessionStartCtx(scope='host_command', stop_event=ctx.stop))
                binding = _SEND.get()
                ref = await store.message_ref(sid, start.next_seq - 1)
                control = current(binding)
                pricing = registered.definition.pricing
                key = 'toolattempt_' + _hash([asdict(ctx.input_ref), asdict(binding.lease), 'host-command'])[:48]
                prepared = dict(version=1, attempt_id=key, scope=asdict(binding.lease), session_id=sid,
                    call_ref=ref.to_dict(), call_id=ctx.input_ref.message_id, request_hash=_hash(arguments),
                    tool_name=name, pricing_rule={'rule_id': pricing.rule_id, 'rule_version': pricing.version()})
                value = control.begin_attempt(key, 'tool', prepared)
                value['budget_required'] = False
                control.record('attempt', key, value)
                control.dispatch(key)
                attempt = tool_attempts.ToolAttempt(binding, key, prepared,
                    PricedCall(name, arguments, 'main'), pricing, asyncio.current_task())
                slot = _Receipt(binding, ref, ctx.input_ref.message_id, _hash(command), _hash(name),
                    asyncio.current_task(), budget_attempt=attempt)
                cache = binding.runtime._native_returns
                cache.slots[cache.key(slot)] = slot
                error, text = None, ''
                async def stop_command():
                    await ctx.stop.wait()
                    await asyncio.sleep(loop.config.stop_policy.finish_tool_s)
                    ctx.task.cancel('host command cancellation grace elapsed')
                stopper = asyncio.create_task(stop_command())
                try:
                    ctx.stop.raise_if_cancelled()
                    with receipt_context(slot):
                        output = await loop.tool_registry.invoke_async(name, arguments)
                    await join_children(slot)
                    text = output if isinstance(output, str) else canonical(output)
                    await store.append_message(sid, role='assistant', content=text,
                        send_index=send_index, round_index=0, meta={'host_command_result': name})
                    result = StatefulResult(sid, 'completed', text,
                        usage=dict(binding.resource_evidence.get('model_usage_totals', {})))
                    evidence.update(exit=result.status, usage=result.usage, rounds=0, new_input_after_pass=False)
                    evidence[KEY] = capture(result)
                except BaseException as exc:
                    error = exc
                    await join_children(slot, cancel=True)
                    raise
                finally:
                    stopper.cancel()
                    await asyncio.gather(stopper, return_exceptions=True)
                    tool_attempts.capture(attempt, None, error)
                    await tool_attempts.finish(attempt)
                    slot.finished = True
                    cache.slots.pop(cache.key(slot))
                    await loop.hooks.run_typed_async(HookPoint.SESSION_END,
                        SessionEndCtx(scope='host_command', reason='failed' if error else 'completed', final_text=text))


async def consume_result(loop, result):
    """An inline host command has consumed this exact original child result."""
    import hashlib

    from power_loop.agent.runtime_types import InputRef

    from .result_revisions import result_id
    ctx = loop._runtime_worker_context
    key = result_id(result.job_ref.job_id, result.job_ref.generation)
    ref = InputRef(ctx.store_id, ctx.session_id, hashlib.sha256(('result:' + key).encode()).hexdigest())
    messages = await loop._runtime_worker.drain_append(ctx, only=[ref])
    state = loop._host_command
    if len(state['consumed']) + len(messages) > 1000:
        from power_loop.agent.runtime_types import CapacityError
        raise CapacityError()
    for message in messages:
        seq = await loop.store.append_message(ctx.session_id, role=message['role'],
            content=message['content'], name=message.get('name'), inbox=message['_inbox'],
            send_index=state['send_index'], round_index=0)
        consumed = {'seq': seq, 'role': message['role'], 'content': message['content'],
            'item_ids': message['_inbox']['item_ids']}
        state['consumed'].append(consumed)
        from .result_consumption import observe
        await observe(_SEND.get(), [consumed], round_index=0, send_index=state['send_index'])
