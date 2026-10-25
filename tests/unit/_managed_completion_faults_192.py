"""Storage fault injection for the actual SDK completion publication transactions."""
from contextlib import asynccontextmanager
from types import SimpleNamespace


def fail_completion_once(manager, stage):
    original = manager._write
    state = SimpleNamespace(fired=False)

    class Transaction:
        def __init__(self, target, flags):
            self.target, self.flags = target, flags

        def __getattr__(self, key):
            return getattr(self.target, key)

        async def execute(self, statement, params=()):
            event = None
            if (statement.startswith(f'INSERT INTO {manager.events} ')
                    and len(params) > 2 and str(params[1]).startswith('effect_')):
                event = params[2]
            selected = {'exit': 'managed_effect_resource_exit', 'anchor': 'managed_effect_completion',
                        'result': 'result', 'publication': 'managed_effect_result_recovery',
                        'exit_ack': 'managed_effect_completion', 'result_ack': 'managed_effect_result_recovery'}[stage]
            if event == selected and not state.fired:
                self.flags['target'] = True
                if stage not in ('exit_ack', 'result_ack'):
                    state.fired = True
                    raise OSError('test_completion_' + stage)
            return await self.target.execute(statement, params)

    @asynccontextmanager
    async def intercepted():
        flags = {'target': False}
        async with original() as tx:
            yield Transaction(tx, flags)
        if flags['target'] and not state.fired:
            state.fired = True
            raise OSError('test_completion_' + stage)

    manager._write = intercepted
    return state
