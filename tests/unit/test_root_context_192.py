"""Persist original business provenance before any provider request, without authority."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.runtime import root_context
from power_loop.runtime.jobs import JobActor, JobError, JobManager, _hash
from power_loop.runtime.task_runtime import _SEND, TaskRuntime

from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import finish, managed
from .test_workflow_task_runtime_192 import managed_workflow, settle

__all__ = ["registry_store", "managed", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
NAMESPACE = "deeptalk.agent-run.v1"


class Model(LLMService):
    def __init__(self):
        self.calls = []
        self.before = None
    async def complete(self, request, **kwargs):
        binding = _SEND.get()
        if self.before is not None:
            await self.before(binding)
        self.calls.append(binding.lease.job_id)
        return LLMResponse(raw_text="DONE", token_usage=LLMTokenUsage(
            prompt_tokens=7, completion_tokens=3, total_tokens=10))
    def stream(self, request):
        async def empty():
            if False:
                yield None
        return empty()
    async def close(self):
        pass


@pytest.fixture
async def root_case(registry_store):
    store, connect = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="root-context", principal_id="host",
        executor_id="executor", boot_id="boot", lease_ms=30000)
    model = Model()
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2))
    sid = await loop.new_session()
    try:
        yield SimpleNamespace(store=store, manager=manager, runtime=runtime, model=model,
            loop=loop, sid=sid, actor=runtime.actor(sid), connect=connect)
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def test_original_root_is_saved_before_provider_and_not_replaced_by_next_run(root_case):
    c = root_case
    source = {"origin_run_id": 101, "installation": {"instance_id": 7}}
    context = c.runtime.root_context(c.sid, namespace=NAMESPACE, payload=source)
    source["origin_run_id"] = 999
    source["installation"]["instance_id"] = 999
    observed = []
    async def before(binding):
        saved = await root_context.read(c.manager, c.actor, binding.lease.job_id, namespace=NAMESPACE)
        observed.append(saved)
    c.model.before = before
    with context:
        assert (await c.loop.send("first original work", c.sid)).status == "completed"
    first = c.model.calls[0]
    assert observed[0]["payload"] == {"origin_run_id": 101, "installation": {"instance_id": 7}}
    assert observed[0]["execution_authority"] is False
    with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 202}):
        assert (await c.loop.send("next independent root", c.sid)).status == "completed"
    assert observed[1]["payload"] == {"origin_run_id": 202}
    assert observed[1]["root_job_id"] != observed[0]["root_job_id"]
    reopened = await c.connect()
    try:
        reread = await root_context.read(JobManager(reopened), c.actor, first, namespace=NAMESPACE)
        assert reread == observed[0]
        reread["payload"]["origin_run_id"] = 777
        assert (await root_context.read(c.manager, c.actor, first))["payload"]["origin_run_id"] == 101
    finally:
        await reopened.close()
    assert len(c.model.calls) == 2  # historical reads never make another model request


async def test_default_root_request_is_unchanged_and_missing_context_is_explicit(root_case):
    c = root_case
    await c.loop.send("legacy-compatible root", c.sid)
    [row] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE action='create'")
    request = json.loads(row["payload_json"])["request"]
    assert set(request) == {"send_index", "model_attempt_protocol", "tool_call_plan_protocol"}
    assert request["model_attempt_protocol"] == request["tool_call_plan_protocol"] == 1
    assert await root_context.read(c.manager, c.actor, c.model.calls[0], required=False) is None
    with pytest.raises(JobError, match="root_context_missing"):
        await root_context.read(c.manager, c.actor, c.model.calls[0])


async def test_wrong_session_never_registers_or_spends(root_case):
    c = root_case
    other = await c.loop.new_session()
    with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 101}):
        with pytest.raises(JobError, match="root_context_binding_mismatch"):
            await c.loop.send("must not run", other)
    assert not c.model.calls
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs}")
    assert root_context.for_send(c.runtime, c.sid) is None


async def test_model_cannot_replace_host_context(root_case):
    c = root_case
    async def before(binding):
        with pytest.raises(JobError, match="root_context_requires_host"):
            with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 999}):
                pytest.fail("model scope cannot bind host attribution")
    c.model.before = before
    with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 101}):
        await c.loop.send("original work", c.sid)
    assert (await root_context.read(c.manager, c.actor, c.model.calls[0]))["payload"] == {"origin_run_id": 101}


async def test_context_reset_nested_bindings_and_foreign_runtime_are_safe(root_case):
    c = root_case
    foreign = TaskRuntime(c.manager, tenant_id="root-context", principal_id="other",
        executor_id="other", boot_id="other")
    with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 101}):
        with pytest.raises(JobError, match="root_context_already_bound"):
            with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 202}):
                pass
        with pytest.raises(JobError, match="root_context_binding_mismatch"):
            root_context.for_send(foreign, c.sid)
    with pytest.raises(RuntimeError, match="host aborted"):
        with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 303}):
            raise RuntimeError("host aborted")
    assert root_context.for_send(c.runtime, c.sid) is None
    assert not c.model.calls


@pytest.mark.parametrize("when", ["before-enter", "before-send"])
@pytest.mark.parametrize("field", ["principal_id", "manager"])
async def test_changed_runtime_authority_never_reuses_a_root_binding(root_case, when, field):
    c = root_case
    original = getattr(c.runtime, field)
    changed = "different-principal" if field == "principal_id" else JobManager(c.store)
    context = c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 101})
    try:
        if when == "before-enter":
            setattr(c.runtime, field, changed)
        # send first validates the runtime's original completion authority;
        # entering a pre-created host context validates its own binding.
        code = "managed_completion_runtime_changed" if when == "before-send" else "root_context_binding_mismatch"
        with pytest.raises(JobError, match=code):
            with context:
                if when == "before-send":
                    setattr(c.runtime, field, changed)
                await c.loop.send("must not change the payer or store", c.sid)
        assert not c.model.calls
        assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs}")
    finally:
        setattr(c.runtime, field, original)
    assert root_context.for_send(c.runtime, c.sid) is None


async def test_source_child_reads_original_root_without_duplicating_its_context(managed):
    store, manager, runtime, loop, model, sid, _ = managed
    runtime.lease_ms = 30000
    with runtime.root_context(sid, namespace=NAMESPACE, payload={"origin_run_id": 101}):
        await loop.send("delegate", sid)
    model.release.set()
    await finish(managed)
    [child] = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='subagent'")
    original = await root_context.read(manager, runtime.actor(sid), child["job_id"])
    assert original["payload"] == {"origin_run_id": 101}
    rows = await manager.db.fetchall(f"SELECT e.payload_json FROM {manager.events} e JOIN {manager.jobs} j ON j.job_id=e.job_id "
        "WHERE j.kind='send' AND j.parent_job_id IS NOT NULL AND e.action='create'")
    assert rows and all("host_context" not in json.loads(r["payload_json"])["request"] for r in rows)


async def test_workflow_nodes_keep_original_main_provenance(managed_workflow):
    _, manager, runtime, loop, _, sid, _, _ = managed_workflow
    runtime.lease_ms = 30000
    with runtime.root_context(sid, namespace=NAMESPACE, payload={"origin_run_id": 101}):
        await loop.send("run original workflow", sid)
    await settle(managed_workflow)
    nodes = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")
    assert len(nodes) == 2
    roots = set()
    for node in nodes:
        original = await root_context.read(manager, runtime.actor(sid), node["job_id"])
        assert original["payload"] == {"origin_run_id": 101} and original["execution_authority"] is False
        roots.add(original["root_job_id"])
    assert len(roots) == 1
    rows = await manager.db.fetchall(f"SELECT e.payload_json FROM {manager.events} e JOIN {manager.jobs} j ON j.job_id=e.job_id "
        "WHERE j.kind='send' AND j.parent_job_id IS NOT NULL AND e.action='create'")
    assert len(rows) == 2 and all("host_context" not in json.loads(r["payload_json"])["request"] for r in rows)


async def test_same_registration_cannot_replace_the_original_context(root_case):
    c = root_case
    with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 101}):
        await c.loop.send("original root", c.sid)
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='create'")
    data = json.loads(event["payload_json"])["request"]
    replay = await c.manager.register_job(c.actor, kind="send", session_id=c.sid,
        system_key=f"send:{data['send_index']}", request=data)
    assert replay.replayed and replay.job.job_id == c.model.calls[0]
    data["host_context"] = root_context._document(NAMESPACE, {"origin_run_id": 999})
    with pytest.raises(JobError, match="operation_conflict"):
        await c.manager.register_job(c.actor, kind="send", session_id=c.sid,
            system_key=f"send:{data['send_index']}", request=data)
    assert (await root_context.read(c.manager, c.actor, c.model.calls[0]))["payload"]["origin_run_id"] == 101
    assert len(c.model.calls) == 1


async def test_cancelled_original_send_resets_host_binding_and_keeps_saved_origin(root_case):
    c = root_case
    entered, release = asyncio.Event(), asyncio.Event()
    cleared, roots = [], []
    async def before(binding):
        roots.append(binding.lease.job_id)
        entered.set()
        await release.wait()
    c.model.before = before
    async def execute():
        try:
            with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 101}):
                await c.loop.send("original interrupted root", c.sid)
        finally:
            cleared.append(root_context.for_send(c.runtime, c.sid) is None)
    sending = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        sending.cancel()
        await asyncio.gather(sending, return_exceptions=True)
        assert cleared == [True]
        assert (await root_context.read(c.manager, c.actor, roots[0]))["payload"] == {"origin_run_id": 101}
    finally:
        release.set()
        await asyncio.gather(sending, return_exceptions=True)


@pytest.mark.parametrize("fault", ["payload", "hash", "missing", "namespace", "foreign", "version-alias"])
async def test_original_evidence_or_permission_mismatch_cannot_supply_an_origin(root_case, fault):
    c = root_case
    with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 101}):
        await c.loop.send("original work", c.sid)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE action='create'")
    if fault in ("payload", "hash"):
        data = json.loads(row["payload_json"])
        data["request"]["host_context"]["payload"]["origin_run_id"] = 999
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?",
            (json.dumps(data), "a" * 64 if fault == "hash" else row["request_hash"], row["event_id"]))
    elif fault == "missing":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (row["event_id"],))
    elif fault == "version-alias":
        data = json.loads(row["payload_json"])
        request = data["request"]
        request["host_context"]["version"] = True
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?",
            (json.dumps(data), _hash(["send", "", c.sid, None, False, request, None]), row["event_id"]))
    actor = JobActor("root-context", "foreign", c.sid) if fault == "foreign" else c.actor
    expected = "access_denied" if fault == "foreign" else "namespace_mismatch" if fault == "namespace" else "evidence_corrupt"
    with pytest.raises(JobError, match=expected):
        await root_context.read(c.manager, actor, c.model.calls[0], namespace="different" if fault == "namespace" else NAMESPACE)
    assert len(c.model.calls) == 1


@pytest.mark.parametrize("payload", [{}, {1: "ambiguous"}, {"n": float("nan")}, {"n": 1.0},
    {"n": b"not-json"}, {"n": {1}}, {"n": "\ud800"}, {"n": "x\x00y"},
    {"n": "x" * 16385}, {"n": 2**63}])
async def test_invalid_metadata_is_rejected_before_job_creation(payload):
    runtime = SimpleNamespace(actor=lambda sid: JobActor("tenant", "host", sid))
    with pytest.raises(JobError, match="invalid_root_context_payload"):
        root_context.bind(runtime, "main", namespace=NAMESPACE, payload=payload)


@pytest.mark.parametrize("namespace", ["", " leading", "BAD", "a/parent", "x" * 129, True, None])
async def test_namespace_is_an_exact_host_contract(namespace):
    runtime = SimpleNamespace(actor=lambda sid: JobActor("tenant", "host", sid))
    with pytest.raises(JobError, match="invalid_root_context_namespace"):
        root_context.bind(runtime, "main", namespace=namespace, payload={"run_id": 101})


async def test_recursive_context_never_enters_the_host_binding():
    runtime = SimpleNamespace(actor=lambda sid: JobActor("tenant", "host", sid))
    payload = {}
    payload["cycle"] = payload
    with pytest.raises(JobError, match="invalid_root_context_payload"):
        root_context.bind(runtime, "main", namespace=NAMESPACE, payload=payload)
    assert root_context.for_send(runtime, "main") is None


async def test_source_root_pointer_cannot_redirect_to_another_valid_root(root_case):
    c = root_case
    with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 101}):
        await c.loop.send("original work", c.sid)
    with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 202}):
        await c.loop.send("unrelated later work", c.sid)
    original, other = c.model.calls
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET root_job_id=? WHERE job_id=?", (other, original))
    with pytest.raises(JobError, match="root_context_lineage_mismatch"):
        await root_context.read(c.manager, c.actor, original, namespace=NAMESPACE)
    assert len(c.model.calls) == 2
