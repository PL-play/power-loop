"""Host-side controller. Every model/worker executes in a separate Docker container."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from redis.asyncio import Redis

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "tests/system215/docker-compose.p8.yml"


class Cluster:
    def __init__(self):
        self.output = Path(os.environ["P8_OUTPUT"])
        self.manifest = json.loads((self.output / "cluster.json").read_text())
        self.http = httpx.AsyncClient(timeout=30)
        self.redis = Redis.from_url("redis://127.0.0.1:46379/0", decode_responses=True)
        self.offset = 0
        self.paused = set()
        self.controller = "b"
        self.commands = []
        self.command_sequence = 0
        self.command_dropped = 0
        self.test_name = ""

    def note(self, event, **values):
        if len(self.commands) >= 4096:
            self.command_dropped += 1
            return
        self.commands.append({"event": event, "seq": len(self.commands) + 1,
                              "utc": time.time(), "monotonic": time.monotonic(), **values})

    def save(self, label, payload):
        path = self.output / f"raw/{self.test_name}-{label}.json"
        path.write_text(json.dumps(payload, indent=2))

    async def call(self, node, op, **data):
        self.command_sequence += 1
        command_id = self.command_sequence
        identity = {key: data[key] for key in ("session", "message_id", "key") if key in data}
        self.note("command_send", command_id=command_id, node=node, op=op, **identity)
        response = await self.http.post(f"http://127.0.0.1:{48101 + 'abc'.index(node)}/command",
                                        json={"op": op, **data})
        self.note("command_response", command_id=command_id, node=node, op=op,
                  status=response.status_code, **identity)
        assert response.status_code == 200, (node, op, response.status_code, response.text)
        return response.json()

    async def health(self, node):
        response = await self.http.get(f"http://127.0.0.1:{48101 + 'abc'.index(node)}/health")
        response.raise_for_status()
        return response.json()

    async def until(self, check, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = await check()
            if value:
                return value
            await asyncio.sleep(.05)
        raise AssertionError("P8 condition timed out")

    async def events(self, event=None, **fields):
        rows = await self.redis.lrange(self.manifest["namespace"] + ":events", self.offset, -1)
        values = [json.loads(row) for row in rows]
        return [v for v in values if (event is None or v["event"] == event)
                and all(v.get(k) == x for k, x in fields.items())]

    async def event(self, name, **fields):
        return await self.until(lambda: self.events(name, **fields))

    async def configure(self, nodes, **options):
        return await asyncio.gather(*(self.call(n, "configure", **options) for n in nodes))

    async def create(self, **config):
        return (await self.call(self.controller, "create", config=config))["session_id"]

    async def send(self, sid, mid="start", node="b", body="start", **kwargs):
        return await self.call(node, "send", session=sid, message_id=mid, body=body, **kwargs)

    async def result(self, sid, mid="start"):
        return await self.call(self.controller, "wait", session=sid, message_id=mid,
                               deadline_ms=int(time.time() * 1000) + 25000)

    async def release(self, sid, suffix):
        await self.call(self.controller, "release", key=sid + ":" + suffix)

    async def messages(self, sid):
        return (await self.call(self.controller, "messages", session=sid, limit=100))["items"]

    async def docker(self, *args):
        self.note("docker_send", args=args)
        command = ["sudo", "-n", "--preserve-env=P8_ENV_FILE", "docker", "compose", "-f", str(COMPOSE), "-p", "p8-215", *args]
        process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ, "P8_ENV_FILE": str(self.output / "cluster.env")})
        output, _ = await process.communicate()
        self.note("docker_response", args=args, status=process.returncode)
        assert process.returncode == 0, output.decode()
        if args[0] == "pause":
            self.paused.update(args[1:])
        elif args[0] == "unpause":
            self.paused.difference_update(args[1:])

    async def ready(self):
        async def all_ready():
            try:
                return all(
                    h["ready"] for h in await asyncio.gather(*(self.health(n) for n in "abc")))
            except (httpx.HTTPError, KeyError):
                return False
        return await self.until(all_ready, timeout=60)


@pytest.fixture
async def cluster(request):
    if not os.environ.get("P8_OUTPUT"):
        pytest.skip("P8 real-container environment not configured (run prepare/up first)")
    c = Cluster()
    c.test_name = request.node.name
    await c.ready()
    await c.configure("abc", enabled=False, roles=["main", "child", "workflow_driver", "background"],
                      drop_notifications=False, fault=None, delay_second_spawn=False)
    c.offset = await c.redis.llen(c.manifest["namespace"] + ":events")
    started = time.monotonic()
    try:
        yield c
    finally:
        # Only our named nodes. Restart also clears deterministic gates/fault state.
        for node in tuple(c.paused):
            await c.docker("unpause", node)
        events = await c.events()
        stem = request.node.name
        with (c.output / "raw/three-node-observations.jsonl").open("a") as stream:
            stream.write(json.dumps({"test": stem, "elapsed_s": time.monotonic() - started,
                                     "events": events}) + "\n")
        for node in "abc":
            try:
                trace = await c.call(node, "trace")
                health = await c.health(node)
                (c.output / f"raw/{stem}-{node}-trace.json").write_text(json.dumps({
                    "trace": trace, "health": health,
                }, indent=2))
            except Exception as exc:
                (c.output / f"raw/{stem}-{node}-trace-error.json").write_text(json.dumps({
                    "error": type(exc).__name__, "detail": str(exc),
                }, indent=2))
        (c.output / f"raw/{stem}-commands.json").write_text(json.dumps({
            "events": c.commands, "dropped": c.command_dropped, "complete": not c.command_dropped,
        }, indent=2))
        await c.docker("restart", "a", "b", "c")
        await c.http.aclose()
        await c.redis.aclose()
