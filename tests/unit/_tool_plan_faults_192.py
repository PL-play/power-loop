"""Fault the original batch transaction, before any callback can receive it."""
from contextlib import asynccontextmanager
from types import SimpleNamespace


def fail_plan_once(manager, stage):
    original = manager._write
    state = SimpleNamespace(fired=False, registrations=0)
    selected = {'first_registration': 'managed_effect_registered', 'second_registration': 'managed_effect_registered',
                'plan': 'tool_call_plan', 'seal': 'tool_call_plan_seal',
                'commit': 'tool_call_plan_seal', 'commit_ack': 'tool_call_plan_seal'}[stage]

    class Transaction:
        def __init__(self, target, flags):
            self.target, self.flags = target, flags

        def __getattr__(self, key):
            return getattr(self.target, key)

        async def execute(self, statement, params=()):
            event = params[2] if statement.startswith(f'INSERT INTO {manager.events} ') and len(params) > 2 else None
            if not state.fired and event == 'managed_effect_registered':
                state.registrations += 1
            if (not state.fired and event == selected
                    and (stage != 'second_registration' or state.registrations == 2)):
                self.flags['target'] = True
                if stage not in ('commit', 'commit_ack'):
                    state.fired = True
                    raise OSError('test_plan_' + stage)
            return await self.target.execute(statement, params)

    @asynccontextmanager
    async def intercepted():
        flags = {'target': False}
        async with original() as tx:
            yield Transaction(tx, flags)
            if flags['target'] and not state.fired and stage == 'commit':
                state.fired = True
                raise OSError('test_plan_' + stage)
        if flags['target'] and not state.fired and stage == 'commit_ack':
            state.fired = True
            raise OSError('test_plan_' + stage)
    manager._write = intercepted
    return state
