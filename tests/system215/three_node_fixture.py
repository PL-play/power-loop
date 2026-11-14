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

    async def call(self, node, op, **data):
        response = await self.http.post(f"http://127.0.0.1:{48101 + 'abc'.index(node)}/command",
                                        json={"op": op, **data})
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
        command = ["sudo", "-n", "--preserve-env=P8_ENV_FILE", "docker", "compose", "-f", str(COMPOSE), "-p", "p8-215", *args]
        process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ, "P8_ENV_FILE": str(self.output / "cluster.env")})
        output, _ = await process.communicate()
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
    await c.ready()
    await c.configure("abc", enabled=False, roles=["main", "child", "workflow_driver", "background"],
                      drop_notifications=False, fault=None)
    c.offset = await c.redis.llen(c.manifest["namespace"] + ":events")
    started = time.monotonic()
    try:
        yield c
    finally:
        # Only our named nodes. Restart also clears deterministic gates/fault state.
        for node in tuple(c.paused):
            await c.docker("unpause", node)
        events = await c.events()
        with (c.output / "raw/three-node-observations.jsonl").open("a") as stream:
            stream.write(json.dumps({"test": request.node.name, "elapsed_s": time.monotonic() - started,
                                     "events": events}) + "\n")
        await c.docker("restart", "a", "b", "c")
        await c.http.aclose()
        await c.redis.aclose()
