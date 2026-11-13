"""Original tool-call abort facts; unknown repair belongs to session recovery."""
from __future__ import annotations

from power_loop.agent.runtime_types import RuntimeRequestError, sha256_canonical


def normalize(entries):
    return [{**entry, "job_id": None, "operation_id": None} for entry in entries]


def identity(manager, source, ref):
    return "abort_" + sha256_canonical([
        manager.store.store_id, source["session_id"], source["job_id"], source["generation"], ref.to_dict(),
    ])[:52]


def _freeze(pipeline, binding, job, ref, entries):
    return {
        "version": 1,
        "session_id": binding.actor.session_id,
        "send_index": ref.send_index,
        "call_ref": ref.to_dict(),
        "entries": normalize(entries),
    }


async def read_abort(*_args, **_kwargs):
    raise RuntimeRequestError("capability_unavailable", "job abort evidence is removed")


__all__ = ["identity", "normalize", "read_abort"]
