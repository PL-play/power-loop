"""Faults at the real claim transaction boundary, not a fabricated lease."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from tests.control_faults import control_mutations


def fail_claim_transaction(manager, stage, *, read_failure=None):
    original = manager._write
    state = SimpleNamespace(fired=False, job_id=None, resolve_failed=False)

    class Transaction:
        def __init__(self, target, flags):
            self.read, self.flags = target.fetchone, flags

        async def intercept(self, execute, sql, params):
            if (sql.startswith(f"UPDATE {manager.jobs} SET status='running',version=version+1,generation=")
                    and str(params[-1]).startswith('effect_') and not state.fired):
                self.flags['claimed'] = True
                state.job_id = params[-1]
            return await execute(sql, params)

        async def fetchone(self, sql, params=()):
            if (read_failure and state.fired and not state.resolve_failed
                    and sql == f'SELECT * FROM {manager.jobs} WHERE job_id=?' and params == (state.job_id,)):
                state.resolve_failed = True
                if read_failure == 'cancel':
                    raise asyncio.CancelledError('cancelled while reading original claim commit')
                raise OSError('cannot read original claim commit')
            return await self.read(sql, params)

    @asynccontextmanager
    async def intercepted():
        flags = {'claimed': False}
        joined = manager._joined_write.get() is not None
        async with original() as tx:
            proxy = Transaction(tx, flags)
            read = tx.fetchone
            tx.fetchone = proxy.fetchone
            try:
                async with control_mutations(tx, proxy.intercept):
                    yield tx
            finally:
                tx.fetchone = read
            if not joined and flags['claimed'] and stage == 'rollback':
                state.fired = True
                raise OSError('original claim transaction rolled back')
        if not joined and flags['claimed'] and not state.fired:
            state.fired = True
            if stage == 'cancel_ack':
                raise asyncio.CancelledError('original claim commit lost cancellation ACK')
            raise OSError('original claim commit lost ACK')

    manager._write = intercepted
    return state
