"""Real independent OS-process worker for Design 217 M2 5-leaf SQL ledger verification (B4).

Each process opens its own SessionStore, RuntimeInputStore, and in-memory driver waiter table.
Cross-process signals travel strictly over stdin/stdout JSON lines carrying ONLY:
- {"type": "wake", "session_id": ..., "reason": ...}
- {"type": "cancel", "session_id": ..., "cancel_by": ...}
Zero result payload is ever sent over the signal channel.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter

from power_loop import (
    AgentLoopConfig,
    DatabasePollingTransport,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    ProjectedRepresentation,
    RegistryRuntimeFactory,
    Runtime,
    ToolRegistry,
)
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import InputRef, RuntimeDescriptor, TargetRef, TrustedActor, result_message_id
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.store import SessionStore
from power_loop.workflow import register_workflow_tool
from tests.unit.test_send_sql_profile import Profiled, _category, _counts


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class _PipeWakeTransport:
    """Signal-only cross-process WakeTransport over stdin/stdout JSON lines (0 payload)."""

    def __init__(self) -> None:
        self._on_wake = None
        self._on_cancel = None

    async def subscribe(self, *, on_wake, on_cancel=None) -> None:
        self._on_wake = on_wake
        self._on_cancel = on_cancel

    async def publish_wake(self, session_id: str, *, reason: str = "accept") -> None:
        _emit({"type": "signal", "kind": "wake", "session_id": session_id, "reason": reason})

    async def publish_cancel(self, session_id: str, *, cancel_by: str = "user") -> None:
        _emit({"type": "signal", "kind": "cancel", "session_id": session_id, "cancel_by": cancel_by})

    async def deliver_inbound(self, msg: dict) -> None:
        if msg.get("kind") == "wake" and self._on_wake is not None:
            await self._on_wake(msg["session_id"], msg.get("reason", "accept"))
        elif msg.get("kind") == "cancel" and self._on_cancel is not None:
            await self._on_cancel(msg["session_id"], msg.get("cancel_by", "user"))

    async def poll_once(self, *, limit: int = 32) -> list[str]:
        return []

    async def aclose(self) -> None:
        return None


class _MainModel(LLMService):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, req, **kwargs):
        self.calls += 1
        return LLMResponse(raw_text=f"main-ack-{self.calls}")

    async def close(self) -> None:
        return None


class _LeafModel(LLMService):
    def __init__(self, scenario: str, release_all: asyncio.Event, release_l0: asyncio.Event) -> None:
        self.scenario = scenario
        self.release_all = release_all
        self.release_l0 = release_l0
        self.started_count = 0
        self.siblings_started: set[int] = set()
        self.siblings_ready = asyncio.Event()
        self.all_started = asyncio.Event()
        self.calls = 0

    async def complete(self, req, **kwargs):
        self.calls += 1
        idx = int(req.system_prompt.splitlines()[0].split("-")[1])
        self.started_count += 1
        _emit({"type": "event", "name": "leaf_started", "idx": idx})
        if self.scenario == "concentrated":
            if self.started_count >= 5:
                self.all_started.set()
            await self.all_started.wait()
            return LLMResponse(raw_text=f"leaf-{idx}-ok")
        if self.scenario == "dispersed":
            await asyncio.sleep(idx * 0.04)
            return LLMResponse(raw_text=f"leaf-{idx}-ok")
        if self.scenario == "mid_stop_1_leaf":
            if idx == 0:
                await self.release_l0.wait()
                return LLMResponse(raw_text="leaf-0-late")
            await self.release_all.wait()
            return LLMResponse(raw_text=f"leaf-{idx}-ok")
        if self.scenario == "stop_driver_subtree":
            await self.release_all.wait()
            return LLMResponse(raw_text=f"leaf-{idx}-late")
        if self.scenario == "real_halt":
            if idx == 0:
                await self.siblings_ready.wait()
                raise RuntimeError("L0 deliberate halt failure")
            self.siblings_started.add(idx)
            if len(self.siblings_started) >= 4:
                self.siblings_ready.set()
            await self.release_all.wait()
            return LLMResponse(raw_text=f"leaf-{idx}-late")
        return LLMResponse(raw_text=f"leaf-{idx}-ok")

    async def close(self) -> None:
        return None


async def _open_backend(backend: str, dsn_or_path: str, prefix: str):
    if backend == "sqlite":
        return await SessionStore.open(dsn_or_path, table_prefix=prefix)
    return await open_store(dsn_or_path, table_prefix=prefix, schema="tasks")


async def _run_sigkill_worker(backend: str, dsn_or_path: str, prefix: str, role: str) -> None:
    """Dedicated real OS-process worker for 2-leaf A->B Driver SIGKILL takeover verification."""
    store = await _open_backend(backend, dsn_or_path, prefix)
    release_b = asyncio.Event()
    calls: dict[str, int] = {"A": 0, "B": 0}

    class _SigkillModel(LLMService):
        async def complete(self, req, **kwargs):
            prompt = req.system_prompt or ""
            if prompt.startswith("LEAF-A"):
                calls["A"] += 1
                _emit({"type": "event", "name": "leaf_done", "node": "A", "calls": dict(calls)})
                return LLMResponse(raw_text="A-once-result")
            if prompt.startswith("LEAF-B"):
                calls["B"] += 1
                _emit({"type": "event", "name": "leaf_started", "node": "B", "calls": dict(calls)})
                await release_b.wait()
                user_text = ""
                for m in req.messages:
                    role_m = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                    if role_m == "user":
                        content_m = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                        user_text += str(content_m or "")
                return LLMResponse(raw_text=f"B-saw:{user_text}")
            return LLMResponse(raw_text="main-ack")

        async def close(self) -> None:
            return None

    model = _SigkillModel()
    reg = ToolRegistry()
    cfg = AgentLoopConfig(context_budget_tokens=10**9, representation=ProjectedRepresentation())
    fac = RegistryRuntimeFactory()

    def can_exec(desc: RuntimeDescriptor) -> bool:
        if role.startswith("sigkill_c"):
            return desc.kind == "workflow_driver"
        return desc.kind == "agent" and "workflow_node" in (desc.config or {})

    async def build(desc, ctx):
        if desc.kind == "workflow_driver":
            _emit({"type": "event", "name": "driver_claimed", "role": role, "send_index": ctx.send_index})
        return ExecutionEnvironment(model, reg, cfg)

    fac.register("m2.sigkill@215", build, can_execute=can_exec)
    rt = Runtime(
        store=store,
        factory=fac,
        node_id=role,
        boot_id=f"{role}-{os.getpid()}",
        capacity=2,
        transport=DatabasePollingTransport(RuntimeInputStore(store), interval_s=0.03),
        session_lease_ttl_s=1.0 if role == "sigkill_c1" else 10.0,
        recovery_interval_s=0.05,
        shared_execution=True,
        timers_enabled=False,
    )
    try:
        await rt.worker.start()
        _emit({"type": "ready", "role": role, "pid": os.getpid()})
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            cmd = json.loads(line)
            ctype = cmd.get("type")
            if ctype == "quit":
                break
            if ctype == "release_b":
                release_b.set()
            elif ctype == "report_calls":
                _emit({"type": "calls", "calls": dict(calls)})
    finally:
        await rt.aclose(timeout_s=2, cancel_running=True)
        await store.close()


async def main() -> None:
    backend, dsn_or_path, prefix, role, scenario = sys.argv[1:6]
    if scenario == "sigkill_ab":
        await _run_sigkill_worker(backend, dsn_or_path, prefix, role)
        return
    store = await _open_backend(backend, dsn_or_path, prefix)
    orig_db = store._db
    prof = Profiled(orig_db)
    store._db = prof

    wake_tp = _PipeWakeTransport()
    release_all = asyncio.Event()
    release_l0 = asyncio.Event()
    main_model = _MainModel()
    leaf_model = _LeafModel(scenario, release_all, release_l0)

    reg = ToolRegistry()
    if role == "a":
        register_workflow_tool(reg)
    cfg = AgentLoopConfig(context_budget_tokens=10**9, representation=ProjectedRepresentation())
    fac = RegistryRuntimeFactory()

    def can_exec(desc: RuntimeDescriptor) -> bool:
        if role == "a":
            if desc.kind == "workflow_driver":
                return True
            return desc.kind == "agent" and "workflow_node" not in (desc.config or {})
        return desc.kind == "agent" and "workflow_node" in (desc.config or {})

    async def build(desc, ctx):
        return ExecutionEnvironment(main_model if role == "a" else leaf_model, reg, cfg)

    fac.register("m2.xproc@215", build, can_execute=can_exec)
    tp = DatabasePollingTransport(RuntimeInputStore(store), interval_s=3600.0)
    rt = Runtime(
        store=store,
        factory=fac,
        node_id=f"proc-{role}",
        boot_id=f"boot-{role}-{os.getpid()}",
        capacity=2 if role == "a" else 5,
        transport=tp,
        wake_transport=wake_tp,
        session_lease_ttl_s=300.0,
        recovery_interval_s=3600.0,
        shared_execution=True,
        timers_enabled=False,
    )
    await rt.worker.start()

    actor = TrustedActor("t-xproc", "u-xproc", None)
    desc = RuntimeDescriptor(
        2, "agent", "m2.xproc@215", None, "m2.tools@215", "m2.hooks@215", "m2.res@215",
        {"context_budget_tokens": 10**9}, {},
    )
    main_target = TargetRef(store.store_id, "000_main")

    # Background stdin reader loop so signals & commands arrive asynchronously
    cmd_q: asyncio.Queue[dict] = asyncio.Queue()

    async def _stdin_pump() -> None:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                await cmd_q.put({"type": "quit"})
                return
            msg = json.loads(line)
            if msg.get("type") == "signal":
                await wake_tp.deliver_inbound(msg)
            else:
                await cmd_q.put(msg)

    pump_task = asyncio.create_task(_stdin_pump())

    try:
        if role == "a":
            # Pre-create main session before resetting profiler so setup SQL is excluded
            await rt.worker.inputs.create_session(desc, actor=actor, options={"session_id": "000_main"})
            prof.records.clear()
            prof.tx_sites.clear()
            prof.tx_categories.clear()
            prof.transactions = 0
        else:
            prof.records.clear()
            prof.tx_sites.clear()
            prof.tx_categories.clear()
            prof.transactions = 0

        _emit({"type": "ready", "role": role, "pid": os.getpid()})

        handle = None
        while True:
            cmd = await cmd_q.get()
            ctype = cmd.get("type")
            if ctype == "quit":
                break
            if ctype == "release_all":
                release_all.set()
            elif ctype == "release_l0":
                release_l0.set()
            elif ctype == "start_workflow" and role == "a":
                branches = [
                    {
                        "type": "agent", "id": f"L{i}", "input": f"work-{i}",
                        "spec": {"name": f"L{i}", "system_prompt": f"LEAF-{i}", "tools": []},
                    }
                    for i in range(5)
                ]
                on_err = "halt" if scenario == "real_halt" else "continue"
                spec = {
                    "name": f"xproc-{scenario}",
                    "root": {"type": "parallel", "on_error": on_err, "max_concurrency": 5, "branches": branches},
                }
                handle = await rt.client.workflow(
                    main_target, spec, message_id=f"wf-{scenario}", actor=actor, descriptor=desc,
                )
                _emit({
                    "type": "event",
                    "name": "workflow_submitted",
                    "driver_sid": handle.target.session_id,
                    "start_msg_id": handle.input_ref.message_id,
                })
            elif ctype == "stop_l0" and role == "a" and handle is not None:
                rows = await orig_db.fetchall(
                    f"SELECT session_id,descriptor_json FROM {store.t.sessions} WHERE parent_session_id=?",
                    (handle.target.session_id,),
                )
                l0_target = None
                for row in rows:
                    if row["descriptor_json"] and '"L0"' in row["descriptor_json"]:
                        l0_target = TargetRef(store.store_id, row["session_id"])
                        break
                assert l0_target is not None
                rcpt = await rt.client.cancel(l0_target, actor=actor, scope="self", reason="user stopped L0")
                _emit({"type": "event", "name": "l0_stopped", "state": rcpt.state})
            elif ctype == "stop_driver_subtree" and role == "a" and handle is not None:
                rcpt = await rt.client.cancel(
                    handle.target, actor=actor, scope="subtree", reason="abort workflow subtree",
                )
                _emit({"type": "event", "name": "driver_subtree_stopped", "state": rcpt.state})
            elif ctype == "wait_workflow_and_main" and role == "a" and handle is not None:
                wf_res = await handle.wait(deadline_ms=now_ms() + 15000)
                parent_ref = InputRef(
                    store.store_id, main_target.session_id,
                    result_message_id(handle.target, handle.input_ref, 1, "workflow_result"),
                )
                main_res = await rt.client.wait(parent_ref, actor=actor, deadline_ms=now_ms() + 15000)
                _emit({
                    "type": "event",
                    "name": "workflow_and_main_done",
                    "wf_status": wf_res.status,
                    "wf_cancel_by": wf_res.cancel_by,
                    "wf_result_statuses": {k: v.status for k, v in wf_res.results.items()},
                    "wf_result_cancel_bys": {k: v.cancel_by for k, v in wf_res.results.items()},
                    "main_status": main_res.status,
                })
            elif ctype == "wait_idle":
                async with asyncio.timeout(15):
                    while rt.worker._active:
                        await asyncio.sleep(0.01)
                _emit({"type": "event", "name": "idle", "role": role})
            elif ctype == "report":
                records = [r for r in prof.records if _category(r["chain"], r["sql"]) != "periodic"]
                counts = _counts(records, prof.tx_sites, prof.tx_categories)
                kv_sql = sum(1 for r in records if "session_runtime_state" in r["sql"].lower())
                by_func = Counter()
                for r in records:
                    cat = _category(r["chain"], r["sql"])
                    frames = [f for f in r["chain"] if not f.startswith(("store.backends", "store.db"))]
                    fn = " -> ".join(frames[-2:]) if frames else "other"
                    sql_head = " ".join(r["sql"].strip().split()[:4]).replace(prefix, "")
                    by_func[f"{cat}|{fn}|{sql_head}"] += 1
                _emit({
                    "type": "report",
                    "role": role,
                    "pid": os.getpid(),
                    "transactions": prof.transactions,
                    "session_control": counts["session_control"],
                    "message": counts["message"],
                    "legacy_job_control": counts["legacy_job_control"],
                    "billing": counts["billing"],
                    "total": len(records),
                    "kv_sql": kv_sql,
                    "main_llm_calls": main_model.calls,
                    "leaf_llm_calls": leaf_model.calls,
                    "by_func": dict(by_func),
                })
    finally:
        pump_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)
        await rt.aclose(timeout_s=2, cancel_running=True)
        store._db = orig_db
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
