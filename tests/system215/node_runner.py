"""One real SDK worker per container. Only the model and fault gates are test code."""
from __future__ import annotations

import asyncio
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

    def key(self, name):
        return self.namespace + ":" + name

    async def record(self, event, **values):
        await self.redis.rpush(self.key("events"), json.dumps({
            "event": event, "node": self.node_id, "boot": self.boot_id,
            "time": time.time(), **values,
        }))

    async def gate(self, key):
        while not await self.redis.exists(self.key("release:" + key)):
            await asyncio.sleep(.025)

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
        await self.redis.ping()
        await self.transport.provision(self.namespace, self.store.store_id)
        await self.runtime.worker.start()
        self.started = True
        await self.record("started", pid=os.getpid(), hostname=os.environ.get("HOSTNAME"))

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
                for name in ("enabled", "roles", "drop_notifications", "fault"):
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
                await self.redis.set(self.key("release:" + data["key"]), "1")
                result = {"released": True}
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
