"""Unmigrated pre-P2 Job test helpers. Never imported by session-native tests."""
from __future__ import annotations

import pytest

from power_loop.runtime.jobs import JobActor, JobManager


def drop_index_sql(backend, index, table):
    return f"DROP INDEX {index}" + (f" ON {table}" if backend == "mysql" else "")


@pytest.fixture
async def runtime(registry_store):
    store, connect = registry_store
    now = [1_000_000]
    manager = JobManager(store, now_ms=lambda: now[0])
    sid = await store.create_session()
    actor = JobActor("tenant", "host", sid)
    root = await manager.register_job(actor, kind="send", session_id=sid, system_key="durable-root-key")
    lease = await manager.claim_job(
        actor, root.job.job_id, executor_id="executor", boot_id="boot", expected_version=1
    )
    return store, manager, actor, root, lease, connect, now


async def call(store, actor, call_id="call-1"):
    seq = await store.append_message(actor.session_id, role="assistant", tool_calls=[{
        "id": call_id, "type": "function", "function": {"name": "task", "arguments": "{}"}
    }])
    return await store.message_ref(actor.session_id, seq)


async def child(runtime, *, key="child", kind="subagent", independent=False, session_id=None):
    _, manager, actor, _, lease, _, _ = runtime
    return await manager.register_job(
        actor, kind=kind, parent=lease, system_key=key, independent=independent,
        session_id=session_id or actor.session_id,
    )


async def claimed(m, actor, kind, parent, key, **kwargs):
    reg = await m.register_job(
        actor, kind=kind, parent=parent, system_key=key, session_id=actor.session_id, **kwargs
    )
    proof = await m.claim_job(
        actor, reg.job.job_id, executor_id=key, boot_id="boot", expected_version=1
    )
    return reg, proof


async def node_tree(runtime):
    _, m, actor, _, root_proof, _, _ = runtime
    workflow, wp = await claimed(m, actor, "workflow", root_proof, "workflow", independent=True)
    node, np = await claimed(m, actor, "workflow_node", wp, "node")
    send, sp = await claimed(m, actor, "send", np, "node-send")
    return workflow, wp, node, np, send, sp
