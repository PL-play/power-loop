"""One real SDK worker per container. Only the model and fault gates are test code."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import resource
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web
from app.runtime_transport import KafkaMessageTransport
from redis.asyncio import Redis

import power_loop
from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    ProjectedRepresentation,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    ToolDefinition,
    TrustedActor,
    create_default_tool_registry,
)
from power_loop.agent.runtime_types import InputRef, TargetRef, plain
from power_loop.core.execution_context import get_execution_context
from power_loop.runtime.env import RuntimeEnv
from power_loop.runtime.pricing import Free
from power_loop.runtime.session_ownership import SessionOwnership
from power_loop.runtime.store.factory import open_store
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow import register_workflow_tool

ACTOR = TrustedActor("p8-test", "host", None)


class Node:
    def __init__(self):
        self.node_id = os.environ["NODE_ID"]
        self.boot_id = uuid.uuid4().hex
        self.namespace = os.environ["P8_NAMESPACE"]
        self.local_namespace = os.environ["STORE_NAMESPACE"]
        self.workspace = Path(os.environ["WORKSPACE"]) / self.local_namespace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.redis = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        self.enabled = False
        self.roles = ["main", "child", "workflow_driver", "background"]
        self.fault = None
        self.started = False
        self.drop_notifications = False
        self.published = []
        self.trace_records = []
        self.trace_sequence = 0
        self.trace_dropped = 0
        self.delay_second_spawn = False
        self.spawn_builds = 0

    def trace(self, event, **values):
        self.trace_sequence += 1
        if len(self.trace_records) >= 4096:
            self.trace_dropped += 1
            return
        self.trace_records.append({
            "event": event, "seq": self.trace_sequence, "node": self.node_id,
            "boot": self.boot_id, "utc": time.time(), "monotonic": time.monotonic(), **values,
        })

    def trace_dump(self):
        return {"events": self.trace_records, "dropped": self.trace_dropped,
                "complete": not self.trace_dropped}

    @staticmethod
    def batch_facts(batch):
        if batch is None:
            return {"claimed": False}
        return {
            "claimed": True, "session": batch.target.session_id, "send": batch.send_index,
            "epoch": batch.owner_epoch, "kind": batch.descriptor.kind,
            "parent": batch.parent_session_id,
            "inputs": [ref.message_id for ref in batch.input_refs],
        }

    @staticmethod
    def final_facts(batch):
        if batch is None:
            return {"final": False}
        return {
            "final": True, "execution_state": batch.execution_state,
            "result_ref": (batch.result_ref.message_id if batch.result_ref else None),
            "result_activation": (batch.result_activation.session_id if batch.result_activation else None),
            "results": [{"input": result.input_ref.message_id, "status": result.status,
                         "send": result.send_index, "delivery_error": result.result_delivery_error}
                        for result in batch.results],
        }

    def key(self, name):
        return self.namespace + ":" + name

    async def record(self, event, **values):
        await self.redis.rpush(self.key("events"), json.dumps({
            "event": event, "node": self.node_id, "boot": self.boot_id,
            "time": time.time(), **values,
        }))

    async def gate(self, key):
        self.trace("gate_enter", key=key)
        await self.record("gate", phase="enter", key=key)
        try:
            while not await self.redis.exists(self.key("release:" + key)):
                await asyncio.sleep(.025)
        except BaseException as exc:
            self.trace("gate_exception", key=key, error=type(exc).__name__)
            raise
        self.trace("gate_exit", key=key)
        await self.record("gate", phase="exit", key=key)

    def capable(self, descriptor):
        role = descriptor.kind
        if role == "agent":
            role = "child" if "child_spec" in descriptor.config or "workflow_node" in descriptor.config else "main"
        return self.enabled and role in self.roles

    async def start(self):
        dsn = os.environ["P8_PG_DSN"]
        if not urlsplit(dsn).path.endswith("_test") or not os.environ["P8_PREFIX"].startswith("p8_215_"):
            raise ValueError("isolated P8 store required")
        self.store = await open_store(dsn, table_prefix=os.environ["P8_PREFIX"], schema="tasks")
        self.install_commit_faults()
        factory = RegistryRuntimeFactory()
        factory.register("p8.agent@215", self.build, can_execute=self.capable)
        self.transport = KafkaMessageTransport(os.environ["KAFKA_SERVERS"], client_id=self.node_id + self.boot_id)
        publish = self.transport.publish

        async def observed_publish(packet):
            self.published.append({"channel": packet.channel, "session": packet.target.session_id})
            if self.drop_notifications:
                raise OSError("P8 injected notification loss")
            return await publish(packet)

        self.transport.publish = observed_publish
        # Real DB coordinator: claim/recover and renew use PostgreSQL server time.
        self.runtime = Runtime(
            store=self.store, factory=factory, node_id=self.node_id, boot_id=self.boot_id,
            transport=self.transport, namespace=self.namespace, shared_execution=True,
            capacity=8, session_lease_ttl_s=3, recovery_interval_s=.25, timers_enabled=False,
        )
        self.install_execution_trace()
        await self.redis.ping()
        await self.transport.provision(self.namespace, self.store.store_id)
        await self.runtime.worker.start()
        self.started = True
        pipeline = Path(power_loop.__file__).parent / "core/pipeline.py"
        await self.record(
            "started", pid=os.getpid(), hostname=os.environ.get("HOSTNAME"),
            import_path=power_loop.__file__,
            pipeline_sha256=hashlib.sha256(pipeline.read_bytes()).hexdigest(),
            node_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )

    def install_execution_trace(self):
        coordinator = self.runtime.coordinator
        original_claim = coordinator.claim

        async def claim(inputs, target, **options):
            self.trace("claim_enter", session=target.session_id)
            try:
                batch = await original_claim(inputs, target, **options)
            except BaseException as exc:
                self.trace("claim_exception", session=target.session_id, error=type(exc).__name__)
                raise
            self.trace("claim_return", **self.batch_facts(batch))
            return batch

        coordinator.claim = claim
        original_dispatch_claimed = self.runtime.worker.dispatch_claimed

        def dispatch_claimed(batch):
            self.trace("claim_return", **self.batch_facts(batch))
            return original_dispatch_claimed(batch)

        self.runtime.worker.dispatch_claimed = dispatch_claimed
        original_renew = coordinator.renew

        async def renew(lease, *, ttl_s):
            self.trace("renew_enter", session=lease.key[1], epoch=lease.epoch,
                       expires_at=lease.expires_at, lost=lease.lost, released=lease.released)
            try:
                renewed = await original_renew(lease, ttl_s=ttl_s)
            except BaseException as exc:
                self.trace("renew_exception", session=lease.key[1], epoch=lease.epoch,
                           error=type(exc).__name__)
                raise
            self.trace("renew_return", session=lease.key[1], epoch=lease.epoch,
                       renewed=renewed, expires_at=lease.expires_at, lost=lease.lost)
            return renewed

        coordinator.renew = renew
        original_recover = coordinator.recover

        async def recover(inputs, target, **options):
            self.trace("recover_enter", session=target.session_id)
            try:
                batch = await original_recover(inputs, target, **options)
            except BaseException as exc:
                self.trace("recover_exception", session=target.session_id, error=type(exc).__name__)
                raise
            self.trace("recover_return", session=target.session_id, **self.final_facts(batch))
            return batch

        coordinator.recover = recover
        inputs = self.runtime.worker.inputs
        original_follow_up = inputs.claim_follow_up

        async def claim_follow_up(batch):
            self.trace("follow_up_enter", session=batch.target.session_id,
                       send=batch.send_index, epoch=batch.owner_epoch)
            try:
                claimed = await original_follow_up(batch)
            except BaseException as exc:
                self.trace("follow_up_exception", session=batch.target.session_id,
                           send=batch.send_index, error=type(exc).__name__)
                raise
            self.trace("follow_up_return", session=batch.target.session_id,
                       send=batch.send_index, claimed=claimed is not None,
                       inputs=([ref.message_id for ref in claimed[0]] if claimed else []))
            return claimed

        inputs.claim_follow_up = claim_follow_up
        original_final = inputs.final

        async def final(batch, **options):
            consumed = options.get("consumed")
            self.trace("final_enter", session=batch.target.session_id, send=batch.send_index,
                       epoch=batch.owner_epoch, status=options.get("status", "completed"),
                       consumed=([ref.message_id for ref in consumed] if consumed is not None else None))
            try:
                final_batch = await original_final(batch, **options)
            except BaseException as exc:
                self.trace("final_exception", session=batch.target.session_id,
                           send=batch.send_index, epoch=batch.owner_epoch, error=type(exc).__name__)
                raise
            self.trace("final_return", session=batch.target.session_id, send=batch.send_index,
                       epoch=batch.owner_epoch, **self.final_facts(final_batch))
            return final_batch

        inputs.final = final
        client_inputs = self.runtime.client.inputs
        original_read_result_state = client_inputs.read_result_state

        async def read_result_state(ref, *, actor):
            self.trace("result_read_enter", session=ref.session_id, input=ref.message_id,
                       actor=actor.session_id)
            try:
                state = await original_read_result_state(ref, actor=actor)
            except BaseException as exc:
                self.trace("result_read_exception", session=ref.session_id, input=ref.message_id,
                           actor=actor.session_id, error=type(exc).__name__)
                raise
            result = state.result
            self.trace("result_read_return", session=ref.session_id, input=ref.message_id,
                       actor=actor.session_id, terminal=result is not None,
                       status=result.status if result else None,
                       send=result.send_index if result else None)
            return state

        client_inputs.read_result_state = read_result_state
        original_notify_results = self.runtime.worker._notify_results

        async def notify_results(final_batch):
            session = (final_batch.results[0].target.session_id if final_batch.results else
                       final_batch.result_ref.session_id if final_batch.result_ref else None)
            self.trace("results_ready", session=session, **self.final_facts(final_batch))
            try:
                await original_notify_results(final_batch)
            except BaseException as exc:
                self.trace("results_notify_exception", session=session, error=type(exc).__name__)
                raise

        self.runtime.worker._notify_results = notify_results
        original_lose = SessionOwnership.lose

        def lose(lease):
            self.trace("lose_enter", session=lease.key[1], epoch=lease.epoch,
                       expires_at=lease.expires_at, lost=lease.lost, released=lease.released)
            try:
                result = original_lose(lease)
            except BaseException as exc:
                self.trace("lose_exception", session=lease.key[1], epoch=lease.epoch,
                           error=type(exc).__name__)
                raise
            self.trace("lose_return", session=lease.key[1], epoch=lease.epoch,
                       lost=lease.lost, released=lease.released)
            return result

        SessionOwnership.lose = lose

    def install_commit_faults(self):
        original = type(self.store._db).transaction

        @asynccontextmanager
        async def transaction(db):
            fault = self.fault
            names = {frame.function for frame in inspect.stack()} if fault else set()
            trigger = bool(fault and fault["stage"] in names)
            async with original(db) as tx:
                yield tx
            if trigger and self.fault is fault:
                self.fault = None
                await self.record("commit_fault", stage=fault["stage"], action=fault["action"])
                if fault["action"] == "unknown":
                    raise ConnectionError("P8 injected lost COMMIT acknowledgement")
                await asyncio.Event().wait()  # Controller kills this container after the signal.

        type(self.store._db).transaction = transaction

    async def build(self, descriptor, context):
        if (self.delay_second_spawn and "child_spec" in descriptor.config
                and "workflow_node" not in descriptor.config):
            self.spawn_builds += 1
            if self.spawn_builds == 2:
                await self.gate(context.session_id + ":build")
        await self.record("build", session=context.session_id, send=context.send_index,
                          epoch=context.owner_epoch, kind=descriptor.kind)
        registry = create_default_tool_registry(workspace_dir=str(self.workspace))
        register_spawn_agent(registry)
        register_workflow_tool(registry)

        async def touch(block=False):
            execution = get_execution_context()
            count = await self.redis.incr(self.key("effect:" + execution.session_id))
            await self.record("tool", session=execution.session_id, count=count)
            if block:
                await self.gate(execution.session_id + ":tool")
            return "effect recorded"

        registry.register(ToolDefinition(
            "touch", "Record an external test effect", input_schema={"type": "object", "properties": {
                "block": {"type": "boolean"}}}, pricing=Free("test fixture"), async_capable=True,
        ), touch)
        return ExecutionEnvironment(
            Model(self, descriptor, context), registry,
            AgentLoopConfig(max_rounds=250, context_budget_tokens=10**9,
                            **({"representation": ProjectedRepresentation()}
                               if descriptor.config.get("projected") else {})),
            runtime_env=RuntimeEnv.from_env(workspace_dir=self.workspace, home_dir=self.workspace),
        )

    async def command(self, request):
        data = await request.json()
        op = data.pop("op")
        client = self.runtime.client
        try:
            if op == "configure":
                for name in ("enabled", "roles", "drop_notifications", "fault", "delay_second_spawn"):
                    if name in data:
                        setattr(self, name, data[name])
                await self.runtime.worker.recover()
                result = self.status()
            elif op == "create":
                config = {"max_rounds": 250, **data.pop("config", {})}
                descriptor = RuntimeDescriptor(2, "agent", "p8.agent@215", None,
                    "builtin.empty@215", "builtin.none@215", "builtin.none@215", config, {})
                parent = TargetRef(self.store.store_id, data["parent"]) if data.get("parent") else None
                result = await client.create_session(descriptor, actor=ACTOR, parent=parent,
                    cancel_propagation=data.get("cancel_propagation", "linked"))
            elif op in {"send", "cancel", "inspect", "wait", "messages"}:
                sid = data.pop("session")
                ref = TargetRef(self.store.store_id, sid)
                if op in {"wait", "inspect", "cancel"} and "message_id" in data:
                    ref = InputRef(self.store.store_id, sid, data.pop("message_id"))
                result = await getattr(client, op)(ref, actor=ACTOR, **data)
            elif op == "recover":
                await self.runtime.worker.recover()
                result = self.status()
            elif op == "release":
                self.trace("release_received", key=data["key"])
                await self.redis.set(self.key("release:" + data["key"]), "1")
                self.trace("release_committed", key=data["key"])
                result = {"released": True}
            elif op == "trace":
                result = self.trace_dump()
            elif op == "result_facts":
                result = [row for row in self.trace_records if row["event"] == "results_ready"]
            elif op == "facts":
                names = set(data["events"])
                result = [row for row in self.trace_records if row["event"] in names]
            elif op == "drop_projection":
                await self.store._db.execute(
                    f"DELETE FROM {self.store.t.project_messages} WHERE session_id=?", (data["session"],))
                result = {"deleted": True}
            else:
                raise ValueError(op)
            return web.json_response(plain(result))
        except Exception as exc:
            return web.json_response({"error": type(exc).__name__, "code": getattr(exc, "code", None)}, status=409)

    def status(self):
        return {"node_id": self.node_id, "boot_id": self.boot_id, "pid": os.getpid(),
                "hostname": os.environ.get("HOSTNAME"), "workspace": str(self.workspace),
                "local_namespace": self.local_namespace, "store_id": self.store.store_id,
                "store_prefix": self.store.table_prefix, "enabled": self.enabled,
                "roles": self.roles, "ready": self.started, "published": self.published,
                "runtime": self.runtime.status(), "cpu_s": time.process_time(),
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

    async def health(self, request):
        return web.json_response(self.status())


class Model(LLMService):
    def __init__(self, node, descriptor, context):
        self.node, self.descriptor, self.context = node, descriptor, context
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        node, sid = self.node, self.context.session_id
        config = self.descriptor.config
        await node.record("model", session=sid, send=self.context.send_index,
                          epoch=self.context.owner_epoch, kind=self.descriptor.kind,
                          call=self.calls, messages=request.messages)
        if self.calls in config.get("gates", []) and not request.system_prompt.startswith(("CHILD", "LEAF")):
            await node.gate(f"{sid}:model:{self.calls}")
        if request.system_prompt.startswith(("CHILD", "LEAF")):
            if "GATED" in request.system_prompt:
                await node.gate(sid + ":child")
            return LLMResponse(raw_text="child done " + node.node_id)
        latest = next((m.get("content", "") for m in reversed(request.messages) if m.get("role") == "user"), "")
        if any(kind in latest for kind in ("agent_result", "workflow_result", "background_result")):
            return LLMResponse(raw_text="handled results " + node.node_id)
        mode = config.get("mode", "echo")
        calls = []
        if self.calls == 1 and mode == "spawn":
            calls = [("spawn_agent", {"task": "test child", "system_prompt": "CHILD GATED", "tools": []})
                     for _ in range(config.get("children", 2))]
        elif self.calls == 1 and mode == "workflow":
            leaves = [{"type": "agent", "id": f"N{i}", "input": "leaf",
                       "spec": {"name": f"N{i}", "system_prompt": "LEAF GATED", "tools": []}}
                      for i in range(config.get("children", 2))]
            calls = [("workflow", {"action": "create", "spec": {"name": "p8", "root": {
                "type": "parallel", "branches": leaves}}})]
        elif self.calls == 1 and mode == "background":
            calls = [("background_run", {"action": "tool", "tool": "touch", "args": {"block": True}})]
        elif mode == "loop" and self.calls < config.get("rounds", 4):
            calls = [("touch", {"block": bool(config.get("tool_gate"))})]
        if calls:
            return LLMResponse(raw_text="", tool_calls=[{"id": f"call-{self.calls}-{i}",
                "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                for i, (name, args) in enumerate(calls)])
        return LLMResponse(raw_text="done " + node.node_id)

    async def close(self):
        pass


async def main():
    node = Node()
    await node.start()
    app = web.Application()
    app.router.add_get("/health", node.health)
    app.router.add_post("/command", node.command)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", 8080).start()
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
