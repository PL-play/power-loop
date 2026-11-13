"""Tool registration snapshots and ordinary session dispatch."""
from __future__ import annotations

import json
from dataclasses import asdict

from power_loop.agent.runtime_types import RuntimeRequestError, canonical, sha256_canonical


def identity(manager, source, ref):
    return "tcp_" + sha256_canonical([
        manager.store.store_id, source["session_id"], source["job_id"], source["generation"], ref.to_dict(),
    ])[:48]


def start_id(plan, call_id):
    return "tcs_" + sha256_canonical([plan, call_id])[:48]


def registration_value(registered):
    if registered is None:
        return None
    from power_loop.tools.registry import validate_pricing

    from .tool_contracts import resolve

    implementation = resolve(registered)
    pricing = validate_pricing(registered)
    definition = registered.definition
    return json.loads(canonical({
        "schema": definition.to_openai_tool(),
        "required": list(definition.required_params),
        "interrupt": definition.interrupt,
        "async_capable": definition.async_capable if type(definition.async_capable) is bool else sorted(definition.async_capable),
        "usage_hint": definition.usage_hint,
        "background_hint": definition.background_hint,
        "pricing": {"rule_id": pricing.rule_id, "version": registered.pricing_version},
        "runtime_contract": asdict(implementation.runtime_contract) if implementation is not None else None,
    }))


async def prepare(pipeline):
    from .execution_tools import prepare as prepare_owned
    await prepare_owned(pipeline, {})


async def dispatch(pipeline, name, arguments, registered, *, attempt=None):
    from .execution_tools import dispatch as dispatch_owned
    return await dispatch_owned(pipeline, name, arguments, registered, attempt=attempt)


async def finish_receipt(*_args, **_kwargs):
    raise RuntimeRequestError("capability_unavailable", "job receipt reconciliation is removed")


async def reconcile_pending(*_args, **_kwargs):
    raise RuntimeRequestError("capability_unavailable", "job pending reconciliation is removed")


__all__ = ["dispatch", "finish_receipt", "reconcile_pending", "registration_value"]
