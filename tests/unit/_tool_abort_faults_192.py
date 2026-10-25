"""Fault the independent pre-admission checkpoint transaction."""
from contextlib import asynccontextmanager
from types import SimpleNamespace


def fail_abort_once(manager, stage):
    original = manager._write
    state = SimpleNamespace(fired=False)
    selected = 'tool_call_abort' if stage == 'checkpoint' else 'tool_call_abort_anchor'
    class Transaction:
        def __init__(self, target, flags):
            self.target, self.flags = target, flags
        def __getattr__(self, name):
            return getattr(self.target, name)
        async def execute(self, statement, params=()):
            event = params[2] if statement.startswith(f'INSERT INTO {manager.events} ') and len(params) > 2 else None
            if event == selected and not state.fired:
                self.flags['target'] = True
                if stage in ('checkpoint', 'anchor'):
                    state.fired = True
                    raise OSError('test_abort_' + stage)
            return await self.target.execute(statement, params)
    @asynccontextmanager
    async def intercepted():
        flags = {'target': False}
        async with original() as tx:
            yield Transaction(tx, flags)
            if flags['target'] and not state.fired and stage == 'commit':
                state.fired = True
                raise OSError('test_abort_' + stage)
        if flags['target'] and not state.fired and stage == 'commit_ack':
            state.fired = True
            raise OSError('test_abort_' + stage)
    manager._write = intercepted
    return state
