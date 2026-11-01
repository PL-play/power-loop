"""Fault the independent pre-admission checkpoint transaction."""
from contextlib import asynccontextmanager
from types import SimpleNamespace

from tests.control_faults import control_mutations


def fail_abort_once(manager, stage):
    original = manager._write
    state = SimpleNamespace(fired=False)
    selected = 'tool_call_abort' if stage == 'checkpoint' else 'tool_call_abort_anchor'
    class Transaction:
        def __init__(self, target, flags):
            self.target, self.flags = target, flags
        def __getattr__(self, name):
            return getattr(self.target, name)
        async def intercept(self, execute, statement, params):
            event = params[2] if statement.startswith(f'INSERT INTO {manager.events} ') and len(params) > 2 else None
            if event == selected and not state.fired:
                self.flags['target'] = True
                if stage in ('checkpoint', 'anchor'):
                    state.fired = True
                    raise OSError('test_abort_' + stage)
            return await execute(statement, params)
    @asynccontextmanager
    async def intercepted():
        flags = {'target': False}
        joined = manager._joined_write.get() is not None
        async with original() as tx:
            proxy = Transaction(tx, flags)
            async with control_mutations(tx, proxy.intercept):
                yield tx
            if not joined and flags['target'] and not state.fired and stage == 'commit':
                state.fired = True
                raise OSError('test_abort_' + stage)
        if not joined and flags['target'] and not state.fired and stage == 'commit_ack':
            state.fired = True
            raise OSError('test_abort_' + stage)
    manager._write = intercepted
    return state
