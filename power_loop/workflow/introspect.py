"""Read saved workflow progress without a live driver."""

from . import journal


async def list_workflows(store, parent_sid):
    return [await journal.read(store, parent_sid, run_id) for run_id in await journal.list_run_ids(store, parent_sid)]


async def get_workflow(store, parent_sid, run_id):
    return await journal.read(store, parent_sid, run_id)
