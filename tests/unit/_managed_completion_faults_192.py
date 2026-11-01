"""Storage fault injection for the actual SDK completion publication transactions."""
from contextlib import asynccontextmanager
from types import SimpleNamespace

from tests.control_faults import control_mutations


def fail_completion_once(manager, stage):
    original = manager._write
    state = SimpleNamespace(fired=False)

    @asynccontextmanager
    async def intercepted():
        flags = {'target': False}
        joined = manager._joined_write.get() is not None

        async def intercept(execute, statement, params):
            event = None
            if (statement.startswith(f'INSERT INTO {manager.events} ')
                    and len(params) > 2 and str(params[1]).startswith('effect_')):
                event = params[2]
            selected = {'exit': 'managed_effect_resource_exit', 'anchor': 'managed_effect_completion',
                        'result': 'result', 'publication': 'managed_effect_result_recovery',
                        'exit_ack': 'managed_effect_completion', 'result_ack': 'managed_effect_result_recovery'}[stage]
            if event == selected and not state.fired:
                flags['target'] = True
                if stage not in ('exit_ack', 'result_ack'):
                    state.fired = True
                    raise OSError('test_completion_' + stage)
            return await execute(statement, params)

        async with original() as tx:
            async with control_mutations(tx, intercept):
                yield tx
        if not joined and flags['target'] and not state.fired:
            state.fired = True
            raise OSError('test_completion_' + stage)

    manager._write = intercepted
    return state
