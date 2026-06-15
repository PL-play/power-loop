"""power-loop benchmark / soak harness (SCALE-1).

NOT shipped in the wheel (only ``power_loop*`` is packaged) and NOT a real-LLM client:
it drives a real ``StatefulAgentLoop`` over a real ``SessionStore`` with a deterministic
``FakeLLM`` (tunable artificial latency) so the numbers reflect *store/loop* cost, not a
provider. Use it to turn the documented single-process ceiling into measured numbers.

    python -m bench            # full sweep → JSON on stdout
    python -m bench --smoke    # tiny subset (also what CI runs, non-blocking)

See ``bench.harness`` for the scenario runners and ``bench.scenarios`` for the sweeps.
"""

from __future__ import annotations

from bench.harness import FakeLLM, run_big_history, run_fanout, run_throughput

__all__ = ["FakeLLM", "run_fanout", "run_big_history", "run_throughput"]
