"""Unmigrated pre-P2 Job test helpers. Never imported by session-native tests."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest

from power_loop import SessionStore
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.session_controls import ControlTransaction
from power_loop.runtime.task_inspection import InspectionPolicy

POLICY = InspectionPolicy(grace_ms=1000, runtime_ms=10000, result_ms=1000, escalation_ms=5000)


class _Gate(LLMService):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.gates: dict[int, asyncio.Event] = {}
        self.system_prompts = []

    def gate(self, n: int) -> asyncio.Event:
        self.gates[n] = asyncio.Event()
        return self.gates[n]

    async def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        n = len(self.calls)
        self.calls.append([dict(m) for m in request.messages])
        self.system_prompts.append(request.system_prompt)
        if n in self.gates:
            await self.gates[n].wait()
        return self.responses.pop(0) if self.responses else LLMResponse(raw_text="done")

    async def close(self) -> None:
        return None


async def _wait_calls(llm: _Gate, n: int) -> None:
    for _ in range(400):
        if len(llm.calls) >= n:
            return
        await asyncio.sleep(0.005)
    pytest.fail(f"expected ≥{n} LLM calls, got {len(llm.calls)}")


@pytest.fixture
async def store():
    session_store = await SessionStore.open(":memory:")
    yield session_store
    await session_store.close()


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


@pytest.fixture
async def inspection_runtime(registry_store):
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_types import (
        Command,
        RuntimeDescriptor,
        TargetRef,
        TrustedActor,
        canonical,
        plain,
    )

    store, connect = registry_store
    now = [1_000_000]
    manager = JobManager(store, now_ms=lambda: now[0])
    sid = await store.create_session()
    actor = JobActor("tenant", "host", sid)
    root = await manager.register_job(actor, kind="send", session_id=sid, system_key="durable-root-key")
    lease = await manager.claim_job(
        actor, root.job.job_id, executor_id="executor", boot_id="boot", expected_version=1
    )
    descriptor = RuntimeDescriptor(
        1, "agent", "inspection@1", None, "tools@1", "hooks@1", "resources@1",
        {"max_rounds": 3}, {"run_id": 101},
    )
    await manager.db.execute(
        f"UPDATE {store.t.sessions} SET metadata_json=? WHERE session_id=?",
        (canonical({
            "tenant_id": actor.tenant_id, "principal_id": actor.principal_id,
            "runtime_descriptor": plain(descriptor),
        }), actor.session_id),
    )
    port = RuntimeInputStore(store)
    async with manager._write() as tx:
        await port.accept_input(
            tx,
            Command("original", TargetRef(store.store_id, actor.session_id), "send", {"body": "original task"}),
            TrustedActor(actor.tenant_id, actor.principal_id, None),
        )
        await tx.execute(
            f"UPDATE {port.q} SET status='delivered',delivered_job_id=?,delivered_job_generation=1 "
            "WHERE session_id=? AND item_id='original'", (root.job.job_id, actor.session_id),
        )
    return store, manager, actor, root, lease, connect, now


async def troubled(runtime, *, state="unknown"):
    store, manager, actor, root, lease, connect, now = runtime
    reg = await child(runtime)
    proof = await manager.claim_job(
        actor, reg.job.job_id, executor_id="child", boot_id="child-boot", expected_version=1
    )
    if state == "unknown":
        await manager.db.execute(
            f"UPDATE {manager.jobs} SET resource_state='unknown' WHERE job_id=?", (proof.job_id,)
        )
    elif state == "cancel":
        await manager.request_cancel(actor, proof.job_id, system_key="cancel-once", reason="stop original")
    elif state == "runtime":
        now[0] += POLICY.runtime_ms
    elif state == "result":
        await manager.record_execution_result(
            actor, proof, outcome="failed", result={"text": "failed"}, resource_state="stopped"
        )
    await manager.close_model_execution(actor, lease, evidence={"exit": "completed"})
    await manager.db.execute(
        f"UPDATE {manager.jobs} SET result_status='handled' WHERE job_id=?", (root.job.job_id,)
    )
    now[0] += 2000
    return proof


@asynccontextmanager
async def observe(manager, monkeypatch):
    rows = await manager.db.fetchall(f"SELECT job_id,root_job_id FROM {manager.jobs}")
    roots = {row["job_id"]: row["root_job_id"] for row in rows}
    write, one, many = manager._write, ControlTransaction.fetchone, ControlTransaction.fetchall
    footprints = []

    def note(tx, rows):
        tx.__dict__.setdefault("_activation_reads", set()).update(
            row["job_id"] for row in rows if row is not None and "job_id" in row
        )

    async def fetchone(tx, sql, params=()):
        row = await one(tx, sql, params)
        note(tx, [row])
        return row

    async def fetchall(tx, sql, params=()):
        rows = await many(tx, sql, params)
        note(tx, rows)
        return rows

    @asynccontextmanager
    async def scoped():
        async with write() as tx:
            yield tx
        jobs = tx.touched_jobs | tx.__dict__.get("_activation_reads", set())
        footprints.append({roots[job] for job in jobs})

    with monkeypatch.context() as patch:
        patch.setattr(manager, "_write", scoped)
        patch.setattr(ControlTransaction, "fetchone", fetchone)
        patch.setattr(ControlTransaction, "fetchall", fetchall)
        yield footprints
