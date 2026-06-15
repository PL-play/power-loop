"""Sweep definitions + the runnable suite for the SCALE-1 harness.

``python -m bench`` runs the full sweep; ``--smoke`` runs a tiny subset (what CI runs,
non-blocking). Output is a single JSON document on stdout — paste the table into
``docs/.../scaling.md`` (SCALE-5) and record the environment it was measured on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from bench.harness import run_big_history, run_fanout, run_throughput

# Full sweep — meaningful numbers on reference hardware. Latency 0 isolates loop/store
# overhead; the fan-out grid is the headline single-process-ceiling table.
FANOUT_SESSIONS = (1, 8, 32, 128, 512)
HISTORY_SIZES = (1_000, 10_000, 50_000)
THROUGHPUT_SENDS = 2_000

# Smoke — fast, deterministic, asserts the harness runs end-to-end (CI / pytest -m bench).
SMOKE_FANOUT_SESSIONS = (1, 8)
SMOKE_HISTORY_SIZES = (100, 1_000)
SMOKE_THROUGHPUT_SENDS = 50


async def run_suite(*, smoke: bool = False) -> dict[str, Any]:
    fanout_grid = SMOKE_FANOUT_SESSIONS if smoke else FANOUT_SESSIONS
    history_grid = SMOKE_HISTORY_SIZES if smoke else HISTORY_SIZES
    throughput_sends = SMOKE_THROUGHPUT_SENDS if smoke else THROUGHPUT_SENDS

    results: dict[str, Any] = {"mode": "smoke" if smoke else "full", "scenarios": {}}
    results["scenarios"]["fanout"] = [
        await run_fanout(sessions=n) for n in fanout_grid
    ]
    results["scenarios"]["big_history"] = [
        await run_big_history(history_size=h) for h in history_grid
    ]
    results["scenarios"]["throughput"] = await run_throughput(total_sends=throughput_sends)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench", description="power-loop SCALE-1 harness")
    parser.add_argument("--smoke", action="store_true", help="run the fast smoke subset")
    args = parser.parse_args(argv)
    results = asyncio.run(run_suite(smoke=args.smoke))
    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
