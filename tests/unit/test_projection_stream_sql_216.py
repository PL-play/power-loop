"""Design 216 S01: real projection and streaming/wait SQL baseline.

This file measures the CURRENT code, not a target. Projection is the only SDK history
path: past sends reach the model through ``ProjectedRepresentation``. It drives
deterministic model/tool streams with normal notifications and emits one JSONL record per
scenario through ``P216_PROFILE_OUT``. Each record keeps the full per-statement chain so the
totals can be re-added by hand; category summaries alone are not evidence.

Scope notes recorded in the profile:
- the measured window starts at the first database operation of the request and ends
  after the worker task teardown; session creation/worker start are prepared before the
  counters reset and are reported separately as ``setup``.
- ``wait``/``subscribe_output`` are measured as the client actually uses them, including
  their own reads. The current 0.2s wait poll is measured as-is; this card must not make
  it disappear.
- model output is deterministic: the same total stream duration is used for 0/1/300
  deltas, so a longer run cannot be mistaken for a per-chunk cost.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

import power_loop
from power_loop import (
    AgentLoopConfig,
    InputRef,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
    LLMSummaryFold,
    LLMTokenUsage,
    ProjectedRepresentation,
    PublishReceipt,
    TargetRef,
    ToolDefinition,
    TransportPacket,
    WaitTimeout,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_transport import PROTOCOL_VERSION
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent
from tests.write_meter import WriteMeter

from .test_send_sql_profile import Profiled, _counts

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_ROOT = Path(__file__).resolve().parents[2]
_CODE_CACHE: dict | None = None


def _code_collection() -> dict:
    global _CODE_CACHE
    if _CODE_CACHE is None:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True)
        envelope = {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip(),
            "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=_ROOT, text=True).strip(),
            "dirty": bool(dirty),
            "diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD"], cwd=_ROOT)).hexdigest(),
            "import_path": power_loop.__file__,
            "version": getattr(power_loop, "__version__", None),
        }
        _CODE_CACHE = envelope
    return _CODE_CACHE


# ── deterministic models ──────────────────────────────────────────────────────


class StreamModel(LLMService):
    """A streaming-capable deterministic model.

    ``complete`` calls the same chunk callbacks the production provider uses, so the
    measured path is the real pipeline delta path. ``stream`` is implemented as well:
    the old SQL fixture raised here, which made it unusable for stream coverage.
    """

    def __init__(self, *, text="done", deltas=0, think_deltas=0, duration_s=0.0,
                 tool_calls=None, label=""):
        self.text = text
        self.deltas = int(deltas)
        self.think_deltas = int(think_deltas)
        self.duration_s = float(duration_s)
        self.tool_calls = list(tool_calls or [])
        self.label = label
        self.calls = 0
        self.requests = []
        self.closed = False

    async def complete(self, request, *, on_chunk_delta_text=None, on_chunk_think=None,
                       on_stream_end=None, **kwargs):
        self.calls += 1
        self.requests.append(str(request.messages))
        loop = asyncio.get_running_loop()
        started = loop.time()
        chunks: list[tuple[str, str]] = []
        chunks += [("text", "t")] * self.deltas
        chunks += [("think", "h")] * self.think_deltas
        for index, (kind, chunk) in enumerate(chunks, 1):
            due = started + self.duration_s * index / max(1, len(chunks))
            delay = due - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            callback = on_chunk_delta_text if kind == "text" else on_chunk_think
            if callback is not None:
                result = callback(chunk)
                if inspect.isawaitable(result):
                    await result
        if not chunks and self.duration_s:
            await asyncio.sleep(max(0.0, started + self.duration_s - loop.time()))
        response = LLMResponse(
            raw_text=self.text, tool_calls=list(self.tool_calls),
            token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )
        if on_stream_end is not None:
            result = on_stream_end(response)
            if inspect.isawaitable(result):
                await result
        return response

    def stream(self, request):
        async def chunks():
            for _ in range(self.deltas):
                yield LLMStreamChunk(delta_text="t")
            for _ in range(self.think_deltas):
                yield LLMStreamChunk(think="h")
            yield LLMStreamChunk(tool_calls=list(self.tool_calls), is_final=True,
                                 token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2,
                                                           total_tokens=5))
        return chunks()

    async def close(self):
        self.closed = True


class MatrixModel(LLMService):
    """models calls; the first n-1 issue ``tools_per_call`` echo calls, the last answers."""

    def __init__(self, models, tools_per_call):
        self.models = models
        self.tools_per_call = tools_per_call
        self.calls = 0

    async def complete(self, request, **kwargs):
        self.calls += 1
        calls = None
        if self.calls < self.models:
            calls = [{
                "id": f"call-{self.calls}-{index}", "type": "function",
                "function": {"name": "echo", "arguments": '{"text":"ok"}'},
            } for index in range(self.tools_per_call)]
        return LLMResponse(raw_text="done" if calls is None else "", tool_calls=calls or [],
                           token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2,
                                                     total_tokens=5))

    def stream(self, request):
        raise AssertionError("matrix model is non-streaming")

    async def close(self):
        pass


@dataclass
class FailingProjector(ProjectedRepresentation):
    """Repair-failure fixture: projection building raises for a selected send."""

    fail_sends: frozenset[int] = frozenset()

    def project_send(self, send_rows, *, send_index, tool_registry):
        if send_index in self.fail_sends:
            raise RuntimeError(f"projection repair failed for send {send_index}")
        return super().project_send(send_rows, send_index=send_index, tool_registry=tool_registry)


# ── harness ──────────────────────────────────────────────────────────────────


def _config(*, fold_keep=4, budget=1_000_000, max_rounds=4, representation=None):
    return AgentLoopConfig(
        representation=representation if representation is not None else ProjectedRepresentation(),
        fold_strategy=LLMSummaryFold(keep_last_sends=fold_keep, trigger_ratio=1.0,
                                     summary_max_tokens=200),
        context_budget_tokens=budget,
        max_rounds=max_rounds,
        max_tokens=64,
        temperature=0.0,
    )


def _echo_registry():
    registry = ToolRegistry()

    async def echo(text):
        return text

    registry.register(ToolDefinition("echo", "Return input",
                                     input_schema={"type": "object",
                                                   "properties": {"text": {"type": "string"}}},
                                     pricing=Free("fixture")), echo)
    return registry


def _reset(profiled: Profiled, model: StreamModel | MatrixModel):
    model.calls = 0
    profiled.records.clear()
    profiled.tx_sites.clear()
    profiled.tx_categories.clear()
    profiled.transactions = 0


def _phase(chain, sql):
    joined = "|".join(chain)
    low = sql.lower()
    # Session activity maintenance is time-driven background work (P), not run work (F_run).
    if re.match(r"update \S*sessions set updated_at", low) and (
            "append_message" in joined or "on_message_appended" in joined):
        return "P"
    if "agent.runtime_client:wait" in joined or "agent.runtime_client:subscribe_output" in joined \
            or "agent.runtime_client:OutputSubscription" in joined:
        return "W"
    if "agent.runtime_store:accept" in joined or "agent.runtime_store:_accept_tx" in joined:
        return "A"
    if "agent.runtime_store:claim_follow_up" in joined:
        return "L"
    if "agent.runtime_store:discover_unfinished" in joined:
        return "P"
    if "agent.runtime_client:create_session" in joined \
            or "agent.runtime_store:create_session" in joined \
            or "runtime.store.store:create_session" in joined:
        return "C"
    if "agent.runtime_store:boundary" in joined:
        return "F_run"
    if "agent.stateful_loop:_heal_send_projection" in joined:
        return "H_heal"
    if "agent.stateful_loop:_plan_and_run_projection_fold" in joined \
            or "runtime.store.store:commit_projection_fold" in joined:
        return "H_fold"
    if "agent.stateful_loop:_write_session_projection" in joined \
            or "runtime.store.projection_fold" in joined \
            or "runtime.store.store:prepare_send_projection_fold" in joined \
            or "runtime.store.store:write_send_projection_rows" in joined:
        return "F_projection"
    if "agent.runtime_store:claim" in joined or "agent.runtime_store:final" in joined \
            or "agent.runtime_store:_final_tx" in joined:
        return "F_control"
    if "agent.runtime_store:get_result" in joined or "agent.runtime_store:inspect" in joined \
            or "agent.runtime_store:read_result_state" in joined:
        return "W"
    if "agent.runtime_store:" in joined:
        return "F_control"
    if "runtime_worker" in joined or "stateful_loop" in joined or "core.pipeline" in joined \
            or "agent.sink" in joined or low.startswith(("select", "insert", "update", "delete")) \
            and "runtime.store.store" in joined:
        return "F_run"
    if "runtime_client" in joined:
        return "W"
    if "renew" in joined or "timers" in joined or "recover" in joined \
            or "poll_notifications" in joined or "runtime_transport" in joined:
        return "P"
    if "tools." in joined or "tool" in joined and "runtime.store" not in joined:
        return "X"
    return "other"


def _phase_counts(records):
    counts = {}
    for record in records:
        phase = _phase(record["chain"], record["sql"])
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def _result_reads(records):
    """Statements from the merged authorize+input+result read (one per actual read)."""
    return sum(
        1 for record in records
        if any("agent.runtime_store:read_result_state" in site for site in record["chain"])
    )


def _config_record(cfg: AgentLoopConfig, transport) -> dict:
    rep = cfg.representation
    return {
        "representation": {
            "kind": getattr(rep, "kind", None),
            "type": type(rep).__name__,
            "version": getattr(rep, "version", None),
            "max_chars": getattr(rep, "max_chars", None),
            "hot_window": getattr(rep, "hot_window", None),
            "render_config": asdict(rep.render_config) if hasattr(rep, "render_config") else None,
            "fold_statistics_policy": bool(rep.fold_statistics_policy())
            if getattr(rep, "fold_statistics_policy", None) else False,
        },
        "fold": {
            "type": type(cfg.fold_strategy).__name__,
            "keep_last_sends": getattr(cfg.fold_strategy, "keep_last_sends", None),
            "trigger_ratio": getattr(cfg.fold_strategy, "trigger_ratio", None),
            "summary_max_tokens": getattr(cfg.fold_strategy, "summary_max_tokens", None),
            "context_budget_tokens": cfg.context_budget_tokens,
            "effective_context_budget": cfg.effective_context_budget(),
        },
        "transport": {
            "type": type(transport).__name__,
            "capabilities": asdict(transport.capabilities)
            if hasattr(transport, "capabilities") else None,
        },
        "max_tokens": cfg.max_tokens,
        "max_rounds": cfg.max_rounds,
        "max_context_rows": cfg.max_context_rows,
    }


def _emit(scenario, *, db, cfg, transport, kind, history, models, tools, deltas,
          notifications, elapsed_s, records, transactions, tx_sites, tx_categories,
          writes=None, extra=None, setup=None, inputs=1, runs=1):
    phase_counts = _phase_counts(records)
    assert sum(phase_counts.values()) == len(records), (phase_counts, len(records))
    payload = {
        "design": "216-S01",
        "scenario": scenario,
        "db": db,
        "kind": kind,
        "code": _code_collection(),
        "config": _config_record(cfg, transport),
        "history_sends": history,
        "cold_hot": "cold" if history == 0 else "hot",
        "I": inputs,
        "R": runs,
        "M": models,
        "T": tools,
        "deltas": deltas,
        "notifications": notifications,
        "elapsed_s": elapsed_s,
        "setup": setup or {},
        "total_sql": len(records),
        "transactions": transactions,
        "counts": _counts(records, tx_sites, tx_categories),
        "phase_counts": phase_counts,
        "returned_rows": sum(int(record.get("rows") or 0) for record in records
                             if record.get("rows") is not None),
        "writes": writes,
        "records": records,
        "tx_sites": tx_sites,
        "tx_categories": tx_categories,
        "extra": extra or {},
    }
    if target := os.environ.get("P216_PROFILE_OUT"):
        with open(target, "a") as stream:
            stream.write(json.dumps(payload) + "\n")
    return payload


async def _projection_rows(store, sid, db=None):
    return await (db or store._db).fetchall(
        f"SELECT send_index,kind,projector_version FROM {store.t.project_messages} "
        "WHERE session_id=? ORDER BY send_index,kind", (sid,))


async def _message_rows(store, sid, db=None):
    return await (db or store._db).fetchall(
        f"SELECT seq,role,name,content,send_index FROM {store.t.messages} "
        "WHERE session_id=? AND state='active' ORDER BY seq", (sid,))


async def _compact_count(store, sid, db=None):
    row = await (db or store._db).fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.project_messages} "
        "WHERE session_id=? AND kind='compact'", (sid,))
    return int(row["n"])


async def _prime(agent, count, *, label="history"):
    for index in range(count):
        model = agent.model
        if isinstance(model, StreamModel):
            model.text = f"answer {index}"
        result = await agent.send(f"{label} {index}")
        assert result.status == "completed", (label, index, result)
        await agent.drain()


# ── setup: session creation + worker start, recorded separately ─────────────


async def test_session_setup_sql(registry_store):
    store, _ = registry_store
    model = StreamModel(text="done")
    cfg = _config()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    started = time.monotonic()
    try:
        agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    finally:
        elapsed = time.monotonic() - started
        records = list(profiled.records)
        transactions = profiled.transactions
        tx_sites = list(profiled.tx_sites)
        tx_categories = list(profiled.tx_categories)
        store._db = original
    _emit("session_setup", db=original.dialect.name, cfg=cfg,
          transport=agent.runtime.transport, kind="setup", history=0, models=0, tools=0,
          deltas={"text": 0, "think": 0}, notifications={}, elapsed_s=elapsed,
          records=records, transactions=transactions, tx_sites=tx_sites,
          tx_categories=tx_categories, inputs=0, runs=0,
          setup={"session_created": True, "worker_started": True})
    await agent.aclose()
    phases = _phase_counts(records)
    assert phases.get("C", 0) > 0 and phases.get("P", 0) > 0


# ── normal send: cold / hot / 20 history (projection is the only path) ──────


async def test_normal_send_cold_hot_history(registry_store):
    store, _ = registry_store
    model = StreamModel(text="done", label="projection")
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) \
        if os.environ.get("P216_WRITE_METER") else None
    profiled = Profiled(original, meter)
    store._db = profiled
    try:
        assert type(agent.runtime.transport).__name__ == "MemoryMessageTransport"
        # cold: first send in a fresh session.
        _reset(profiled, model)
        started = time.monotonic()
        cold = await agent.send("cold input")
        await agent.drain()
        cold_elapsed = time.monotonic() - started
        cold_records = list(profiled.records)
        cold_writes = await meter.report() if meter else None
        cold_record = _emit("normal_projection_cold", db=original.dialect.name, cfg=cfg,
              transport=agent.runtime.transport, kind="send_loop_wait", history=0,
              models=1, tools=0, deltas={"text": 0, "think": 0},
              notifications={"completed": "normal"}, elapsed_s=cold_elapsed,
              records=cold_records, transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              writes=cold_writes,
              extra={"result_status": cold.status, "projection_rows": len(
                  await _projection_rows(store, agent.target.session_id, original)),
                  "compact_rows": await _compact_count(store, agent.target.session_id, original)})
        assert cold.status == "completed"
        # The budget assertion comes after BOTH cold and hot are emitted, so a red cold
        # sample cannot prevent the hot raw record from being delivered.
        maximum = 24 if original.dialect.name == "mysql" else 22
        # The meter is cumulative from install; reset it after the cold request is
        # recorded so hot reports only the hot request's writes.
        if meter:
            await meter.reset()
        # hot: second send in the same session, one past send in context.
        _reset(profiled, model)
        model.text = "answer 0"
        started = time.monotonic()
        hot = await agent.send("hot input")
        await agent.drain()
        hot_elapsed = time.monotonic() - started
        hot_writes = await meter.report() if meter else None
        hot_record = _emit("normal_projection_hot", db=original.dialect.name, cfg=cfg,
              transport=agent.runtime.transport, kind="send_loop_wait", history=1,
              models=1, tools=0, deltas={"text": 0, "think": 0},
              notifications={"completed": "normal"}, elapsed_s=hot_elapsed,
              records=list(profiled.records), transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              writes=hot_writes,
              extra={"result_status": hot.status, "projection_rows": len(
                  await _projection_rows(store, agent.target.session_id, original)),
                  "compact_rows": await _compact_count(store, agent.target.session_id, original)})
        assert hot.status == "completed"
        assert cold_record["total_sql"] <= maximum, {
            "scenario": "normal_projection_cold", "total_sql": cold_record["total_sql"],
            "frozen_maximum": maximum}
        assert hot_record["total_sql"] <= maximum, {
            "scenario": "normal_projection_hot", "total_sql": hot_record["total_sql"],
            "frozen_maximum": maximum}
    finally:
        store._db = original
        await agent.aclose()
        if meter:
            await meter.close()


async def test_normal_worker_held_cold(registry_store):
    """Same scope as the old fixture: notifications held, one get_result."""
    store, _ = registry_store
    model = StreamModel(text="done")
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    real_publish = agent.runtime.transport.publish

    async def held(_packet):
        return PublishReceipt(False)

    try:
        _reset(profiled, model)
        agent.runtime.transport.publish = held
        started = time.monotonic()
        receipt = await agent.runtime.client.send(agent.target, "run", message_id="held-run",
                                                 actor=agent.actor)
        await agent.runtime.worker.recover()
        await agent.drain(after_input=receipt.input_ref)
        result = await agent.runtime.client.inputs.get_result(receipt.input_ref,
                                                              actor=agent.actor)
        elapsed = time.monotonic() - started
        _emit("normal_projection_worker_held_cold", db=original.dialect.name,
              cfg=cfg, transport=agent.runtime.transport, kind="worker_held_get_result",
              history=0, models=1, tools=0, deltas={"text": 0, "think": 0},
              notifications={"held": True}, elapsed_s=elapsed,
              records=list(profiled.records), transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              extra={"result_status": result.status})
        assert result.status == "completed"
    finally:
        agent.runtime.transport.publish = real_publish
        store._db = original
        await agent.aclose()


async def test_normal_worker_held_history_20(registry_store):
    """The old 43/43/45 fixture scope, now with the real projection representation."""
    store, _ = registry_store
    model = StreamModel(text="done")
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    real_publish = agent.runtime.transport.publish

    async def held(_packet):
        return PublishReceipt(False)

    try:
        await _prime(agent, 20)
        _reset(profiled, model)
        agent.runtime.transport.publish = held
        started = time.monotonic()
        receipt = await agent.runtime.client.send(agent.target, "measured input",
                                                 message_id="measured-20", actor=agent.actor)
        await agent.runtime.worker.recover()
        await agent.drain(after_input=receipt.input_ref)
        result = await agent.runtime.client.inputs.get_result(receipt.input_ref,
                                                              actor=agent.actor)
        elapsed = time.monotonic() - started
        records = list(profiled.records)
        rows = await _projection_rows(store, agent.target.session_id)
        _emit("normal_projection_worker_held_history_20", db=original.dialect.name,
              cfg=cfg, transport=agent.runtime.transport, kind="worker_held_get_result",
              history=20, models=1, tools=0, deltas={"text": 0, "think": 0},
              notifications={"held": True}, elapsed_s=elapsed, records=records,
              transactions=profiled.transactions, tx_sites=list(profiled.tx_sites),
              tx_categories=list(profiled.tx_categories),
              extra={"result_status": result.status, "projection_rows": len(rows),
                     "compact_rows": sum(1 for row in rows if row["kind"] == "compact"),
                     "projector_versions": sorted({int(row["projector_version"]) for row in rows})})
        assert result.status == "completed"
        assert rows and all(int(row["projector_version"]) == 1 for row in rows)
        assert not any(row["kind"] == "compact" for row in rows)
    finally:
        agent.runtime.transport.publish = real_publish
        store._db = original
        await agent.aclose()


# ── projection health / missing / repair failure ────────────────────────────


async def test_projection_health_and_repair_paths(registry_store, monkeypatch):
    from power_loop.runtime.store import projection_fold

    store, _ = registry_store
    windows, saved = [], []
    load, persist = store.load_projection_window, projection_fold.persist

    async def load_window(*args, **kwargs):
        window = await load(*args, **kwargs)
        windows.append(window)
        return window

    async def save(*args, **kwargs):
        saved.append((kwargs["send_index"], kwargs.get("window")))
        return await persist(*args, **kwargs)

    monkeypatch.setattr(store, "load_projection_window", load_window)
    monkeypatch.setattr(projection_fold, "persist", save)
    model = StreamModel(text="done")
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    sid = agent.target.session_id
    try:
        # Healthy: one tool call + final answer is projected from memory.
        first = await agent.send("question one")
        assert first.status == "completed"
        await agent.drain()
        assert saved == [(1, windows[0])]
        assert saved[0][1] is windows[0]
        project_rows = await _projection_rows(store, sid)
        assert {row["kind"] for row in project_rows} == {"user", "project"}
        assert all(int(row["projector_version"]) == 1 for row in project_rows)
        _reset(profiled, model)
        started = time.monotonic()
        second = await agent.send("question two")
        await agent.drain()
        _emit("projection_healthy_read", db=original.dialect.name, cfg=cfg,
              transport=agent.runtime.transport, kind="send_loop_wait", history=1,
              models=1, tools=0, deltas={"text": 0, "think": 0},
              notifications={"completed": "normal"}, elapsed_s=time.monotonic() - started,
              records=list(profiled.records), transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              extra={"heal": False, "fallback": False,
                     "compact_rows": await _compact_count(store, sid, original)})
        assert second.status == "completed"
        assert saved[-1][1] is windows[-1] and windows[-1] is not windows[0]
        assert "__pl_projection_fold" not in model.requests[-1]
        # Missing projection: delete send 1's rows; the next send must heal them.
        await store._db.execute(
            f"DELETE FROM {store.t.project_messages} WHERE session_id=? AND send_index=1", (sid,))
        _reset(profiled, model)
        started = time.monotonic()
        third = await agent.send("question three")
        await agent.drain()
        healed_rows = [row for row in await _projection_rows(store, sid, original)
                       if int(row["send_index"]) == 1]
        heal_records = [record for record in profiled.records
                        if any("_heal_send_projection" in c for c in record["chain"])]
        _emit("projection_missing_heal", db=original.dialect.name, cfg=cfg,
              transport=agent.runtime.transport, kind="send_loop_wait", history=2,
              models=1, tools=0, deltas={"text": 0, "think": 0},
              notifications={"completed": "normal"}, elapsed_s=time.monotonic() - started,
              records=list(profiled.records), transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              extra={"heal": True, "heal_sql": len(heal_records),
                     "healed_kinds": sorted(row["kind"] for row in healed_rows),
                     "compact_rows": await _compact_count(store, sid, original)})
        assert third.status == "completed"
        assert {row["kind"] for row in healed_rows} == {"user", "project"}
        assert heal_records, "expected the missing projection to be healed"
        assert saved[-1] == (3, None)  # Actual heal writes invalidate the save facts.
        assert "question one" in model.requests[-1]  # Repaired rows still reach this model call.
        # An empty historical index has nothing to heal and keeps the original window facts.
        await original.execute(
            f"UPDATE {store.t.sessions} SET send_index=4 WHERE session_id=?", (sid,))
        assert (await agent.send("after empty index")).status == "completed"
        await agent.drain()
        assert saved[-1][0] == 5 and saved[-1][1] is windows[-1]
        # Repair failure: the send stops with projection_unavailable before its model call.
        await original.execute(
            f"DELETE FROM {store.t.project_messages} WHERE session_id=? AND send_index=2", (sid,))
        failing = FailingProjector(fail_sends=frozenset({2}))
        failing_model = StreamModel(text="done")
        failing_cfg = _config(representation=failing)
        # Stop the first worker so only the failing-configured worker may claim the send.
        await agent.runtime.worker.shutdown(timeout_s=2)
        failing_agent = await RunningAgent(store, failing_model, _echo_registry(),
                                           failing_cfg).start(
            target=TargetRef(store.store_id, sid))
        try:
            _reset(profiled, failing_model)
            started = time.monotonic()
            fourth = await failing_agent.send("question four", message_id="failing-four")
            await failing_agent.drain()
            _emit("projection_repair_failed_stopped", db=original.dialect.name,
                  cfg=failing_cfg, transport=failing_agent.runtime.transport,
                  kind="send_loop_wait", history=5, models=0, tools=0,
                  deltas={"text": 0, "think": 0},
                  notifications={"completed": "normal"},
                  elapsed_s=time.monotonic() - started,
                  records=list(profiled.records), transactions=profiled.transactions,
                  tx_sites=list(profiled.tx_sites),
                  tx_categories=list(profiled.tx_categories),
                  extra={"heal": "failed", "fallback": False,
                         "failing_model_calls": failing_model.calls,
                         "error_code": fourth.error_code,
                         "compact_rows": await _compact_count(store, sid, original)})
            assert fourth.status == "failed", (fourth.status, fourth.error)
            assert fourth.error_code == "projection_unavailable"
            assert failing_model.calls == 0
        finally:
            await failing_agent.aclose()
    finally:
        store._db = original
        await agent.aclose()


async def test_projection_missing_tool_result_is_marked_unknown(registry_store):
    store, _ = registry_store
    cfg = _config()

    class ToolThenAnswer(StreamModel):
        async def complete(self, request, **kwargs):
            self.calls += 1
            self.requests.append(str(request.messages))
            if self.calls == 1:
                return LLMResponse(raw_text="", tool_calls=[{
                    "id": "call-1", "type": "function",
                    "function": {"name": "echo", "arguments": '{"text":"ok"}'}}])
            return LLMResponse(raw_text="after tool")

    model = ToolThenAnswer(text="after tool")
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    sid = agent.target.session_id
    try:
        assert (await agent.send("use the tool")).status == "completed"
        await agent.drain()
        # Remove the tool result row: the projection must keep the call and mark it missing.
        await store._db.execute(
            f"DELETE FROM {store.t.messages} WHERE session_id=? AND role='tool'", (sid,))
        await store._db.execute(
            f"DELETE FROM {store.t.project_messages} WHERE session_id=? AND send_index=1", (sid,))
        _reset(profiled, agent.model)
        started = time.monotonic()
        result = await agent.send("history now")
        await agent.drain()
        stored = await original.fetchall(
            f"SELECT content_json FROM {store.t.project_messages} WHERE session_id=? "
            "AND send_index=1 AND kind='project'", (sid,))
        _emit("projection_missing_tool_result", db=original.dialect.name, cfg=cfg,
              transport=agent.runtime.transport, kind="send_loop_wait", history=1,
              models=1, tools=1, deltas={"text": 0, "think": 0},
              notifications={"completed": "normal"}, elapsed_s=time.monotonic() - started,
              records=list(profiled.records), transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              extra={"stored_content": stored[0]["content_json"] if stored else None,
                     "compact_rows": await _compact_count(store, sid, original)})
        assert result.status == "completed"
        assert stored and "<missing>" in str(stored[0]["content_json"])
    finally:
        store._db = original
        await agent.aclose()


# ── model/tool matrix: separate a and b ─────────────────────────────────────


@pytest.mark.parametrize("models", [1, 2, 4])
@pytest.mark.parametrize("tools_per_call", [1, 4])
async def test_projection_model_tool_matrix(registry_store, models, tools_per_call):
    store, _ = registry_store
    model = MatrixModel(models, tools_per_call)
    cfg = _config(max_rounds=models + 2)
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    real_publish = agent.runtime.transport.publish

    async def held(_packet):
        return PublishReceipt(False)

    try:
        _reset(profiled, model)
        agent.runtime.transport.publish = held
        started = time.monotonic()
        receipt = await agent.runtime.client.send(agent.target, "matrix", message_id="matrix",
                                                 actor=agent.actor)
        await agent.runtime.worker.recover()
        await agent.drain(after_input=receipt.input_ref)
        result = await agent.runtime.client.inputs.get_result(receipt.input_ref,
                                                              actor=agent.actor)
        elapsed = time.monotonic() - started
        tools = tools_per_call * max(0, models - 1)
        _emit("matrix", db=original.dialect.name, cfg=cfg,
              transport=agent.runtime.transport, kind="worker_held_get_result", history=0,
              models=models, tools=tools, deltas={"text": 0, "think": 0},
              notifications={"held": True}, elapsed_s=elapsed,
              records=list(profiled.records), transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              extra={"models_arg": models, "tools_per_call": tools_per_call,
                     "result_status": result.status})
        assert result.status == "completed" and model.calls == models
    finally:
        agent.runtime.transport.publish = real_publish
        store._db = original
        await agent.aclose()


# ── history=False keeps its own input only and writes no projection ─────────


async def test_projection_history_false(registry_store):
    store, _ = registry_store
    model = StreamModel(text="done")
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    sid = agent.target.session_id
    try:
        assert (await agent.send("past secret")).status == "completed"
        await agent.drain()
        before = await _projection_rows(store, sid, original)
        # A new session is what the host does for history=False / LiveApp none; the SDK
        # also supports the per-request flag, reproduced here with a new session + options.
        other = await agent.runtime.open_session(
            replace(agent.descriptor, config={**agent.descriptor.config, "history": False}),
            actor=agent.actor)
        _reset(profiled, model)
        started = time.monotonic()
        receipt = await agent.runtime.client.send(other, "own input", message_id="own",
                                                 actor=agent.actor)
        await agent.runtime.worker.recover()
        await agent.drain(after_input=receipt.input_ref)
        result = await agent.runtime.client.inputs.get_result(receipt.input_ref,
                                                              actor=agent.actor)
        after = await _projection_rows(store, other.session_id, original)
        _emit("projection_history_false", db=original.dialect.name, cfg=cfg,
              transport=agent.runtime.transport, kind="worker_held_get_result", history=0,
              models=1, tools=0, deltas={"text": 0, "think": 0},
              notifications={"held": True}, elapsed_s=time.monotonic() - started,
              records=list(profiled.records), transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              extra={"result_status": result.status, "new_session_projection_rows": len(after),
                     "past_session_projection_rows": len(before)})
        assert result.status == "completed"
        assert after == []
    finally:
        store._db = original
        await agent.aclose()


# ── wait: 0/1/300 deltas, same completion time ─────────────────────────────


@pytest.mark.parametrize("channel", ["text", "think"])
@pytest.mark.parametrize("deltas", [0, 1, 300])
async def test_wait_delta_sql(registry_store, channel, deltas):
    store, _ = registry_store
    model = StreamModel(text="done", duration_s=0.5,
                        deltas=deltas if channel == "text" else 0,
                        think_deltas=deltas if channel == "think" else 0)
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        _reset(profiled, model)
        started = time.monotonic()
        receipt = await agent.runtime.client.send(agent.target, "stream",
                                                 message_id=f"stream-{channel}-{deltas}",
                                                 actor=agent.actor)
        result = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor,
                                                 deadline_ms=now_ms() + 10000)
        # The measured request ends only after the claimed worker released its
        # resources and ran the tail `_has_pending` check.
        await agent.drain()
        elapsed = time.monotonic() - started
        _emit("wait_delta", db=original.dialect.name, cfg=cfg,
              transport=agent.runtime.transport, kind="wait", history=0, models=1,
              tools=0, deltas={"text": deltas if channel == "text" else 0,
                               "think": deltas if channel == "think" else 0},
              notifications={"completed": "normal"}, elapsed_s=elapsed,
              records=list(profiled.records), transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              extra={"channel": channel, "requested_deltas": deltas,
                     "model_duration_s": 0.5,
                     "lease_ttl_s": agent.runtime.session_lease_ttl_s,
                     "result_reads": _result_reads(profiled.records),
                     "result_status": result.status})
        assert result.status == "completed" and result.final_text == "done"
        # Same 0.5s completion for 0/1/300 deltas: initial read + related completion read.
        assert _result_reads(profiled.records) <= 2
    finally:
        store._db = original
        await agent.aclose()


@pytest.mark.parametrize("duration", [1.0, 10.0])
async def test_wait_duration_sql(registry_store, duration):
    store, _ = registry_store
    model = StreamModel(text="done", duration_s=duration)
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        _reset(profiled, model)
        started = time.monotonic()
        receipt = await agent.runtime.client.send(agent.target, "slow",
                                                 message_id=f"slow-{duration}",
                                                 actor=agent.actor)
        result = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor,
                                                 deadline_ms=now_ms() + int(duration * 1000) + 5000)
        await agent.drain()
        elapsed = time.monotonic() - started
        _emit("wait_duration", db=original.dialect.name, cfg=cfg,
              transport=agent.runtime.transport, kind="wait", history=0, models=1,
              tools=0, deltas={"text": 0, "think": 0},
              notifications={"completed": "normal"}, elapsed_s=elapsed,
              records=list(profiled.records), transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              extra={"requested_duration_s": duration,
                     "lease_ttl_s": agent.runtime.session_lease_ttl_s,
                     "result_reads": _result_reads(profiled.records),
                     "result_status": result.status})
        assert result.status == "completed"
        # Waiting time never adds result reads (1s and 10s use the same two reads).
        assert _result_reads(profiled.records) <= 2
    finally:
        store._db = original
        await agent.aclose()


@pytest.mark.parametrize("waiters", [1, 3])
async def test_wait_multiple_waiters(registry_store, waiters):
    store, _ = registry_store
    model = StreamModel(text="done", duration_s=0.5)
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        _reset(profiled, model)
        started = time.monotonic()
        receipt = await agent.runtime.client.send(agent.target, "fanout",
                                                 message_id=f"fanout-{waiters}",
                                                 actor=agent.actor)
        results = await asyncio.gather(*[
            agent.runtime.client.wait(receipt.input_ref, actor=agent.actor,
                                      deadline_ms=now_ms() + 10000)
            for _ in range(waiters)
        ])
        await agent.drain()
        elapsed = time.monotonic() - started
        _emit("wait_waiters", db=original.dialect.name, cfg=cfg,
              transport=agent.runtime.transport, kind="wait", history=0, models=1,
              tools=0, deltas={"text": 0, "think": 0},
              notifications={"completed": "normal"}, elapsed_s=elapsed,
              records=list(profiled.records), transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              extra={"waiters": waiters,
                     "result_reads": _result_reads(profiled.records),
                     "statuses": [result.status for result in results]})
        assert all(result.status == "completed" for result in results)
        # Each waiter reads once at start and once on the completion event; no growth
        # from the number of waiters' shared notifications.
        assert _result_reads(profiled.records) <= 2 * waiters
    finally:
        store._db = original
        await agent.aclose()


# ── wait: unrelated InputRef / duplicate completion ─────────────────────────


@pytest.mark.parametrize("noise", ["unrelated_delta", "duplicate_completed", "none"])
async def test_wait_notification_variants(registry_store, noise):
    store, _ = registry_store
    model = StreamModel(text="done", duration_s=0.5)
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    transport = agent.runtime.transport
    real_publish = transport.publish
    publisher = None
    try:
        if noise == "duplicate_completed":
            async def duplicate(packet):
                receipt = await real_publish(packet)
                if packet.channel == "output" and (packet.payload or {}).get("kind") == "completed":
                    await real_publish(packet)
                return receipt
            transport.publish = duplicate
        _reset(profiled, model)
        started = time.monotonic()
        receipt = await agent.runtime.client.send(agent.target, "notify",
                                                 message_id=f"notify-{noise}",
                                                 actor=agent.actor)
        if noise == "unrelated_delta":
            other_ref = InputRef(store.store_id, agent.target.session_id, "unrelated")

            async def noise_task():
                for _ in range(10):
                    await real_publish(TransportPacket(
                        PROTOCOL_VERSION, agent.runtime.namespace, "unrelated",
                        TargetRef(store.store_id, agent.target.session_id), "output",
                        other_ref, {"kind": "delta", "text": "noise", "is_think": False}))
                    await asyncio.sleep(0.02)
            publisher = asyncio.create_task(noise_task())
        result = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor,
                                                 deadline_ms=now_ms() + 10000)
        if publisher is not None:
            await publisher
            publisher = None
        await agent.drain()
        elapsed = time.monotonic() - started
        _emit("wait_notification_variant", db=original.dialect.name, cfg=cfg,
              transport=transport, kind="wait", history=0, models=1, tools=0,
              deltas={"text": 0, "think": 0}, notifications={"variant": noise},
              elapsed_s=elapsed, records=list(profiled.records),
              transactions=profiled.transactions, tx_sites=list(profiled.tx_sites),
              tx_categories=list(profiled.tx_categories),
              extra={"noise": noise, "result_reads": _result_reads(profiled.records),
                     "result_status": result.status})
        assert result.status == "completed"
        # Unrelated deltas and duplicate completion packets never add a result read.
        assert _result_reads(profiled.records) <= 2
    finally:
        if publisher is not None:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
        transport.publish = real_publish
        store._db = original
        await agent.aclose()


# ── wait: completed notification lost, then quiet or still delta ───────────


@pytest.mark.parametrize("after", ["quiet", "still_delta"])
async def test_wait_lost_completion(registry_store, after):
    store, _ = registry_store
    # Completion is late enough that the waiter's one initial read always happens
    # while the send is still pending; otherwise this would test the initial read
    # racing the model instead of a lost notification.
    model = StreamModel(text="done", duration_s=1.0)
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    transport = agent.runtime.transport
    real_publish = transport.publish
    publisher = None

    async def drop_completed(packet):
        if packet.channel == "output" and (packet.payload or {}).get("kind") == "completed":
            return PublishReceipt(False)
        return await real_publish(packet)

    try:
        transport.publish = drop_completed
        _reset(profiled, model)
        started = time.monotonic()
        receipt = await agent.runtime.client.send(agent.target, "lost",
                                                 message_id=f"lost-{after}",
                                                 actor=agent.actor)
        if after == "still_delta":
            ref = receipt.input_ref

            async def noise_task():
                for _ in range(60):
                    await real_publish(TransportPacket(
                        PROTOCOL_VERSION, agent.runtime.namespace, "lost-noise",
                        TargetRef(store.store_id, agent.target.session_id), "output",
                        ref, {"kind": "delta", "text": "tick", "is_think": False}))
                    await asyncio.sleep(0.03)
            publisher = asyncio.create_task(noise_task())
        timed_out = None
        try:
            result = await agent.runtime.client.wait(
                receipt.input_ref, actor=agent.actor, deadline_ms=now_ms() + 2500)
        except WaitTimeout as exc:
            timed_out = str(exc)
            result = None
        if publisher is not None:
            await publisher
            publisher = None
        # The explicit read is the documented recovery entry; a missing notification must
        # not rewrite the durable result.
        if result is None:
            result = await agent.runtime.client.inputs.get_result(receipt.input_ref,
                                                                  actor=agent.actor)
        await agent.drain()
        elapsed = time.monotonic() - started
        _emit("wait_lost_completion", db=original.dialect.name, cfg=cfg,
              transport=transport, kind="wait", history=0, models=1, tools=0,
              deltas={"text": 0, "think": 0},
              notifications={"completed": "lost", "after": after},
              elapsed_s=elapsed, records=list(profiled.records),
              transactions=profiled.transactions, tx_sites=list(profiled.tx_sites),
              tx_categories=list(profiled.tx_categories),
              extra={"timed_out": timed_out is not None, "after": after,
                     "result_reads": _result_reads(profiled.records),
                     "result_status": result.status if result else None})
        # S03 removes the 0.2s result poll: a fully lost completion is not discovered
        # while waiting. At the original absolute deadline the waiter makes exactly one
        # final read, sees the durable terminal and returns the real result. Continuous
        # deltas neither shorten nor extend that point.
        assert timed_out is None, "the deadline read must return the durable result"
        assert result is not None and result.status == "completed"
        assert elapsed >= 2.4
        # Initial read + the one deadline read; no per-delta reads.
        assert _result_reads(profiled.records) == 2
    finally:
        if publisher is not None:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
        transport.publish = real_publish
        store._db = original
        await agent.aclose()


# ── subscribe_output: completed packet lost, quiet or still delta ──────────


@pytest.mark.parametrize("after", ["quiet", "still_delta"])
async def test_subscribe_output_lost_completion(registry_store, after):
    """Observation is ended by a fixed outer deadline; the durable read is explicit.

    This is the before-side of S03: OutputSubscription has no result poll, so a
    dropped completed packet leaves the observer waiting. Delta packets must not
    make it declare completion; only the explicit persistent read is authoritative.
    """
    store, _ = registry_store
    model = StreamModel(text="done", duration_s=0.8)
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    transport = agent.runtime.transport
    real_publish = transport.publish
    publisher = None
    stream = None

    async def drop_completed(packet):
        if packet.channel == "output" and (packet.payload or {}).get("kind") == "completed":
            return PublishReceipt(False)
        return await real_publish(packet)

    try:
        transport.publish = drop_completed
        _reset(profiled, model)
        sid = agent.target.session_id
        ref = InputRef(store.store_id, sid, f"sub-lost-{after}")
        started = time.monotonic()
        stream = await agent.runtime.client.subscribe_output(ref, actor=agent.actor,
                                                             before_send=True)
        receipt = await agent.runtime.client.send(agent.target, "subscribe-lost",
                                                 message_id=ref.message_id, actor=agent.actor)
        if after == "still_delta":
            async def noise_task():
                for _ in range(80):
                    await real_publish(TransportPacket(
                        PROTOCOL_VERSION, agent.runtime.namespace, "lost-sub-noise",
                        TargetRef(store.store_id, sid), "output", ref,
                        {"kind": "delta", "text": "tick", "is_think": False}))
                    await asyncio.sleep(0.03)
            publisher = asyncio.create_task(noise_task())

        items = []
        observation = "completed"
        try:
            # One fixed absolute deadline, not a per-delta timeout.
            async with asyncio.timeout(1.5):
                async for item in stream:
                    items.append(item)
        except TimeoutError:
            observation = "outer_deadline"
        finally:
            await stream.aclose()
        if publisher is not None:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
            publisher = None
        # Include worker teardown/tail check before freezing the request ledger.
        await agent.drain()
        read_start = len(profiled.records)
        result = await agent.runtime.client.inputs.get_result(ref, actor=agent.actor)
        explicit_read_sql = len(profiled.records) - read_start
        elapsed = time.monotonic() - started
        kinds = [item.kind for item in items]
        _emit("subscribe_lost_completion", db=original.dialect.name, cfg=cfg,
              transport=transport, kind="subscribe_output", history=0, models=1, tools=0,
              deltas={"text": 0, "think": 0},
              notifications={"completed": "lost", "after": after,
                             "observation_deadline_s": 1.5},
              elapsed_s=elapsed, records=list(profiled.records),
              transactions=profiled.transactions, tx_sites=list(profiled.tx_sites),
              tx_categories=list(profiled.tx_categories),
              extra={"receipt_state": receipt.state, "observation": observation,
                     "first_kind": kinds[0] if kinds else None, "kinds": kinds,
                     "delta_items": kinds.count("delta"),
                     "explicit_read_sql": explicit_read_sql,
                     "result_reads": _result_reads(profiled.records),
                     "result_status": result.status if result else None})
        assert receipt.state == "accepted"
        assert items and items[0].kind == "gap", kinds  # initial authorized read, no input yet
        assert observation == "outer_deadline", kinds
        assert result is not None and result.status == "completed"
        # The subscription read once before send and once only at the explicit read;
        # delta packets never issue get_result/inspect.
        assert _result_reads(profiled.records) == 2
        if after == "quiet":
            assert "delta" not in kinds
        else:
            assert kinds.count("delta") >= 1
    finally:
        if publisher is not None:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
        if stream is not None:
            await stream.aclose()
        transport.publish = real_publish
        store._db = original
        await agent.aclose()


# ── subscribe_output: 0/1/300 deltas, unrelated InputRef ───────────────────


async def test_subscribe_output_skips_unrelated_input(registry_store):
    store, _ = registry_store
    model = StreamModel(text="done", duration_s=0.4)
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    transport = agent.runtime.transport
    real_publish = transport.publish
    publisher = None
    try:
        _reset(profiled, model)
        sid = agent.target.session_id
        ref = InputRef(store.store_id, sid, "sub-own")
        other = InputRef(store.store_id, sid, "sub-other")
        started = time.monotonic()
        stream = await agent.runtime.client.subscribe_output(ref, actor=agent.actor,
                                                             before_send=True)
        receipt = await agent.runtime.client.send(agent.target, "subscribe",
                                                 message_id=ref.message_id, actor=agent.actor)

        async def noise_task():
            for _ in range(10):
                await real_publish(TransportPacket(
                    PROTOCOL_VERSION, agent.runtime.namespace, "sub-other",
                    TargetRef(store.store_id, sid), "output", other,
                    {"kind": "delta", "text": "noise", "is_think": False}))
                await asyncio.sleep(0.02)
        publisher = asyncio.create_task(noise_task())
        items = []
        async for item in stream:
            items.append(item)
        await publisher
        publisher = None
        await agent.drain()
        elapsed = time.monotonic() - started
        kinds = [item.kind for item in items]
        _emit("subscribe_output_unrelated", db=original.dialect.name, cfg=cfg,
              transport=transport, kind="subscribe_output", history=0, models=1, tools=0,
              deltas={"text": 0, "think": 0},
              notifications={"unrelated_input_ref": 10, "completed": "normal"},
              elapsed_s=elapsed, records=list(profiled.records),
              transactions=profiled.transactions, tx_sites=list(profiled.tx_sites),
              tx_categories=list(profiled.tx_categories),
              extra={"receipt_state": receipt.state, "kinds": kinds,
                     "result_reads": _result_reads(profiled.records),
                     "noise_items": sum(1 for item in items
                                        if (item.payload or {}).get("text") == "noise")})
        assert receipt.state == "accepted"
        assert kinds[-1] == "completed" and "noise" not in str(
            [(item.kind, item.payload) for item in items])
        assert _result_reads(profiled.records) == 2
    finally:
        if publisher is not None:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
        store._db = original
        await agent.aclose()


@pytest.mark.parametrize("channel", ["text", "think"])
@pytest.mark.parametrize("deltas", [0, 1, 300])
async def test_subscribe_output_delta_sql(registry_store, channel, deltas):
    store, _ = registry_store
    model = StreamModel(text="done", duration_s=0.5,
                        deltas=deltas if channel == "text" else 0,
                        think_deltas=deltas if channel == "think" else 0)
    cfg = _config()
    agent = await RunningAgent(store, model, _echo_registry(), cfg).start()
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        _reset(profiled, model)
        sid = agent.target.session_id
        ref = InputRef(store.store_id, sid, f"sub-{channel}-{deltas}")
        started = time.monotonic()
        stream = await agent.runtime.client.subscribe_output(ref, actor=agent.actor,
                                                             before_send=True)
        receipt = await agent.runtime.client.send(agent.target, "subscribe",
                                                 message_id=ref.message_id, actor=agent.actor)
        items = []
        async for item in stream:
            items.append(item)
        await agent.drain()
        elapsed = time.monotonic() - started
        kinds = [item.kind for item in items]
        _emit("subscribe_output", db=original.dialect.name, cfg=cfg,
              transport=agent.runtime.transport, kind="subscribe_output", history=0,
              models=1, tools=0,
              deltas={"text": deltas if channel == "text" else 0,
                      "think": deltas if channel == "think" else 0},
              notifications={"completed": "normal"}, elapsed_s=elapsed,
              records=list(profiled.records), transactions=profiled.transactions,
              tx_sites=list(profiled.tx_sites), tx_categories=list(profiled.tx_categories),
              extra={"channel": channel, "requested_deltas": deltas,
                     "model_duration_s": 0.5,
                     "lease_ttl_s": agent.runtime.session_lease_ttl_s,
                     "delivered_deltas": kinds.count("delta"),
                     "result_reads": _result_reads(profiled.records),
                     "kinds": kinds, "first_kind": kinds[0] if kinds else None})
        assert receipt.state == "accepted"
        assert kinds[-1] == "completed"
        if deltas == 0:
            assert "delta" not in kinds
        else:
            assert kinds.count("delta") >= 1
        # Initial subscribe read + one related completion read, independent of chunks.
        assert _result_reads(profiled.records) == 2
    finally:
        store._db = original
        await agent.aclose()
