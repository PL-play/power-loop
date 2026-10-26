"""Actual owned managed call exits; its original raw tool write fails."""
from power_loop.runtime import delegation
from power_loop.runtime.managed_effects import (
    FunctionManagedTool,
    ManagedToolContract,
    ManagedToolPreparation,
    ManagedToolResult,
)

from .test_original_owner_review_192 import original_owner_case, wait_owned


async def owned_pending_case(registry_store, kind, *, unknown=False, claim_failure=None, provider=None,
                             raw_fail_at=1, owner_exit_gate=None, effect_calls=1, shared_calls=12,
                             native_tail=None, native_first=False, configure=None, flush_checkpoints=True):
    c = await original_owner_case(registry_store, kind, stage=None, allowed_calls=1,
                                  max_tokens=4096 if provider is not None else 2048, provider=provider,
                                  effect_calls=effect_calls, shared_calls=shared_calls,
                                  native_tail=native_tail, native_first=native_first)
    c.state.prepared = c.state.executed = c.state.lookups = c.state.raw_failures = 0
    c.state.raw_writes = 0
    c.state.resolved = not unknown
    async def prepare(context, arguments):
        c.state.prepared += 1
        return ManagedToolPreparation({'original_call': context.tool_call_id})
    async def execute(context, reference):
        c.state.executed += 1
        c.state.effects.append(reference['original_call'])
        if unknown:
            raise TimeoutError('original effect outcome requires its actual observer')
        return ManagedToolResult('Original effect completed', reference)
    async def observe(context, reference):
        c.state.lookups += 1
        if not c.state.resolved:
            return None
        if reference is None:
            return ManagedToolResult('Original effect was not dispatched', {'original_job': context.job_id}, 'not_dispatched')
        return ManagedToolResult('Original effect completed', reference)
    c.adapter = FunctionManagedTool(ManagedToolContract('example.original-owned-pending', 'f'*64), prepare, execute, observe)
    registered = c.loop.tool_registry.get('touch_once')
    c.loop.tool_registry.register(registered.definition, c.adapter.bind(c.runtime, registered.definition), overwrite=True)
    append = c.store.append_message
    async def raw_failure(sid, **kwargs):
        if kwargs.get('role') == 'tool' and sid != c.state.root:
            c.state.raw_writes += 1
            if c.state.raw_writes == raw_fail_at and c.state.raw_failures == 0:
                c.state.raw_failures += 1
                raise OSError('original owned raw tool receipt unavailable')
        return await append(sid, **kwargs)
    c.store.append_message = raw_failure
    if claim_failure is not None:
        claim = c.runtime.claim
        async def fail_claim(actor, registration, **kwargs):
            if registration.job.kind == 'background_tool':
                if claim_failure == 'after':
                    await claim(actor, registration, **kwargs)
                raise OSError('original managed claim interrupted')
            return await claim(actor, registration, **kwargs)
        c.runtime.claim = fail_claim
    if configure is not None:
        configure(c)
    initial = set(delegation._TASKS)
    try:
        await c.loop.send('Launch original work once.', c.state.root)
        if owner_exit_gate is None:
            await wait_owned(c, kind, initial)
            if flush_checkpoints:
                await c.runtime.flush_control_checkpoints()
        else:
            import asyncio
            await asyncio.wait_for(owner_exit_gate.wait(), 5)
        effects = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
        if len(effects) == 1:
            [c.effect] = effects
        else:
            pending = (await c.store.get_state(c.state.source_binding.actor.session_id)).pending
            call_id = next(call['id'] for call in pending['tool_calls'] if call['function']['name'] == 'touch_once')
            operation = await c.manager.db.fetchone(f"SELECT job_id FROM {c.manager.events} WHERE action='create' AND tool_call_id=?", (call_id,))
            c.effect = next(effect for effect in effects if effect['job_id'] == operation['job_id'])
        return c
    except BaseException:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.runtime.flush_control_checkpoints()
        await c.loop.aclose(drain_timeout_s=.1)
        raise
