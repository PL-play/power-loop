"""Inject faults at declared control mutations without bypassing their intents."""
from contextlib import asynccontextmanager


@asynccontextmanager
async def control_mutations(transaction, intercept):
    execute = transaction._execute

    async def wrapped(statement, params=()):
        return await intercept(execute, statement, params)

    transaction._execute = wrapped
    try:
        yield transaction
    finally:
        transaction._execute = execute
