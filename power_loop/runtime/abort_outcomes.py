"""Unknown original-call outcomes are ordinary session inputs, not task proofs."""
from __future__ import annotations

from power_loop.agent.runtime_types import RuntimeRequestError, canonical, sha256_canonical

MODE = "session_pending_repair"
FIELD = "aborted_call_batch"
CONSTRAINTS = (
    "The listed original calls were interrupted or not dispatched. Do not replay unknown actions. "
    "Inspect the referenced session input and message coordinates for any later known result."
)


def notice_id(store_id, session_id, send_index):
    return "abort_" + sha256_canonical([store_id, session_id, send_index])[:52]


def render(public):
    return CONSTRAINTS + "\n\nOriginal tool outcome:\n" + canonical(public)


async def review_view(*_args, **_kwargs):
    raise RuntimeRequestError("capability_unavailable", "job abort review is removed")


async def handled_references(*_args, **_kwargs):
    return []


async def reconcile(*_args, **_kwargs):
    raise RuntimeRequestError("capability_unavailable", "use session recovery for interrupted inputs")


__all__ = ["CONSTRAINTS", "FIELD", "MODE", "handled_references", "notice_id", "reconcile", "render", "review_view"]
