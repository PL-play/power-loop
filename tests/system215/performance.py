"""Measure actual container HTTP→accept→worker→result latency without SQL instrumentation."""
from __future__ import annotations

import asyncio
import json
import statistics
import time

from tests.system215.three_node_fixture import Cluster


def summary(values):
    ordered = sorted(values)
    return {"samples": len(values), "p50_ms": statistics.median(values) * 1000,
            "p95_ms": ordered[max(0, int(len(ordered) * .95) - 1)] * 1000,
            "max_ms": max(values) * 1000}


async def main():
    c = Cluster()
    try:
        await c.ready()
        await c.configure("abc", enabled=True, roles=["main"], drop_notifications=False, fault=None)
        before = {n: await c.health(n) for n in "abc"}
        results = []
        for concurrency in (1, 8):
            admission, completion, failures, sessions = [], [], [], []
            semaphore = asyncio.Semaphore(concurrency)

            async def send(index):
                async with semaphore:
                    sid = await c.create()
                    sessions.append(sid)
                    started = time.monotonic()
                    try:
                        await c.send(sid, body="x" * 128)
                        admission.append(time.monotonic() - started)
                        result = await c.result(sid)
                        if result["status"] != "completed":
                            failures.append(result["status"])
                    except Exception as exc:
                        failures.append(type(exc).__name__)
                    completion.append(time.monotonic() - started)

            started = time.monotonic()
            await asyncio.gather(*(send(i) for i in range(24)))
            elapsed = time.monotonic() - started
            unfinished = 0
            for sid in sessions:
                status = await c.call("b", "inspect", session=sid)
                unfinished += status["execution_state"] != "idle"
            results.append({"concurrency": concurrency, "input_bytes": 128, "model": "immediate fixture",
                "admission": summary(admission), "completion": summary(completion),
                "throughput_sessions_per_s": 24 / elapsed,
                "throughput_scope": "includes session creation, send and terminal wait",
                "elapsed_s": elapsed, "failures": failures, "unfinished_sessions": unfinished})
        after = {n: await c.health(n) for n in "abc"}
        payload = {"scope": "same-host three Docker workers, real PostgreSQL coordinator and Kafka transport",
            "baseline": None, "results": results, "nodes": {
                n: {"node_id": after[n]["node_id"], "boot_id": after[n]["boot_id"],
                    "cpu_seconds_delta": after[n]["cpu_s"] - before[n]["cpu_s"],
                    "max_rss_kib": after[n]["max_rss_kib"]} for n in "abc"},
            "control_latency": "admission measured; cancel latency not measured",
            "instrumentation": "no SQL/row meter; test-model trace writes to real Redis remain included"}
        (c.output / "raw/performance.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
    finally:
        await c.http.aclose()
        await c.redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
