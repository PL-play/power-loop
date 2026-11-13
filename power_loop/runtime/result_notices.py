"""Ordinary result notifications; no job-event delivery ledger."""
from __future__ import annotations

from power_loop.agent.runtime_types import InputRef, RuntimeRequestError


async def accept(*_args, **_kwargs):
    raise RuntimeRequestError("capability_unavailable", "job result notices are removed; use result_route")


async def notify(client, target, key):
    ref = InputRef(target.store_id, target.session_id, key)
    return await client._notify(target, key, ref, ["work", "activation"])
