"""OBS-1 / OBS-6 / OBS-2: event envelope serialization, monotonic clock, durable JSONL.

- ``to_dict``/``from_dict`` round-trip every event type, preserving ts/seq/mono and NOT
  advancing the process-wide seq counter on deserialization.
- ``mono`` is a process-relative monotonic clock (survives a wall-clock rollback).
- the JSONL sink durably persists events and ``replay`` reconstructs them in seq order.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from power_loop import AgentEvent, AgentEventBus, AgentEventType
from power_loop.contrib.jsonl_sink import JsonlSink, attach_jsonl_sink, replay

pytestmark = pytest.mark.unit


def test_roundtrip_preserves_envelope_for_every_type() -> None:
    for t in AgentEventType:
        ev = AgentEvent(type=t, session_id="s", round_index=2, payload={"k": "v"})
        back = AgentEvent.from_dict(ev.to_dict())
        assert back.type is ev.type
        assert back.session_id == ev.session_id
        assert back.round_index == ev.round_index
        assert back.payload == ev.payload
        assert back.ts == ev.ts and back.seq == ev.seq and back.mono == ev.mono


def test_from_dict_does_not_advance_the_seq_counter() -> None:
    """Deserializing must reuse the serialized seq, not mint a fresh one — so it does not
    perturb the ordering of live events."""
    ev = AgentEvent(type=AgentEventType.SYSTEM_LOG)
    serialized = ev.to_dict()
    next_live = AgentEvent(type=AgentEventType.SYSTEM_LOG)
    # reconstruct the old event many times; the next live seq must be unaffected
    for _ in range(50):
        back = AgentEvent.from_dict(serialized)
        assert back.seq == ev.seq  # preserved, not re-stamped
    after = AgentEvent(type=AgentEventType.SYSTEM_LOG)
    assert after.seq == next_live.seq + 1  # only the two live events consumed seqs


def test_from_dict_without_timing_assigns_fresh() -> None:
    """A dict lacking ts/seq/mono (e.g. a hand-authored event) gets fresh values."""
    back = AgentEvent.from_dict({"type": "system_log", "payload": {}})
    assert back.seq > 0 and back.ts > 0 and back.mono > 0


def test_mono_is_monotonic_under_wallclock_rollback(monkeypatch) -> None:
    e1 = AgentEvent(type=AgentEventType.SYSTEM_LOG)
    # roll wall-clock backwards; mono (perf_counter) must still not decrease
    real = time.time()
    monkeypatch.setattr(time, "time", lambda: real - 3600)
    e2 = AgentEvent(type=AgentEventType.SYSTEM_LOG)
    assert e2.mono >= e1.mono
    assert e2.ts < e1.ts  # wall-clock did go backwards (the point of having mono)


# ── JSONL durable sink + replay ─────────────────────────────────────────────


def test_jsonl_sink_persists_and_replays_in_order(tmp_path: Path) -> None:
    bus = AgentEventBus(suppress_subscriber_errors=True)
    path = tmp_path / "events.jsonl"
    sink = attach_jsonl_sink(bus, path)
    published = []
    for i in range(5):
        ev = AgentEvent(type=AgentEventType.SYSTEM_LOG, session_id="s", payload={"i": i})
        published.append(ev)
        bus.publish(ev)
    sink.close()

    assert path.read_text().count("\n") == 5
    got = list(replay(path))
    assert [e.payload["i"] for e in got] == [0, 1, 2, 3, 4]
    assert [e.seq for e in got] == [e.seq for e in published]  # exact seqs preserved
    assert got == [e for e in published]  # equality ignores ts/seq (compare=False)
    # strictly increasing seqs
    assert all(b.seq > a.seq for a, b in zip(got, got[1:]))


def test_jsonl_sink_redacts_secrets(tmp_path: Path) -> None:
    bus = AgentEventBus(suppress_subscriber_errors=True)
    path = tmp_path / "ev.jsonl"
    sink = attach_jsonl_sink(bus, path)
    bus.publish(AgentEvent(type=AgentEventType.SYSTEM_LOG, payload={"api_key": "sk-secret", "n": 1}))
    sink.close()
    line = path.read_text()
    assert "sk-secret" not in line and "***" in line and '"n": 1' in line


def test_jsonl_sink_value_secret_scrubbing_is_opt_in(tmp_path: Path) -> None:
    # M-contrib-observability-6: a secret embedded in a VALUE under a benign key is NOT scrubbed by
    # the default (key-name) policy, but IS scrubbed with redact_value_secrets=True.
    bus = AgentEventBus(suppress_subscriber_errors=True)
    payload = {"command": "curl -H 'Authorization: Bearer sk-abcdEFGH1234567890zzz' https://x"}

    default_path = tmp_path / "default.jsonl"
    sink = attach_jsonl_sink(bus, default_path)
    bus.publish(AgentEvent(type=AgentEventType.SYSTEM_LOG, payload=dict(payload)))
    sink.close()
    assert "Bearer sk-abcdEFGH1234567890zzz" in default_path.read_text()  # default: leaks

    scrub_path = tmp_path / "scrub.jsonl"
    sink2 = attach_jsonl_sink(bus, scrub_path, redact_value_secrets=True)
    bus.publish(AgentEvent(type=AgentEventType.SYSTEM_LOG, payload=dict(payload)))
    sink2.close()
    line = scrub_path.read_text()
    assert "sk-abcdEFGH1234567890zzz" not in line and "***" in line  # opt-in: scrubbed
    assert "curl" in line  # surrounding text preserved


def test_jsonl_sink_backup_count_zero_does_not_discard(tmp_path: Path) -> None:
    # contrib-observability-5: max_bytes>0 + backup_count=0 used to truncate the file on rotation,
    # silently discarding ALL events. Now backup_count<=0 disables rotation (file grows).
    path = tmp_path / "nb.jsonl"
    sink = JsonlSink(path, max_bytes=200, backup_count=0)
    for i in range(50):
        sink.write_line(f'{{"i": {i}, "pad": "{"x" * 50}"}}')  # far exceeds max_bytes total
    sink.close()
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 50  # every event retained (no truncation)
    assert not (tmp_path / "nb.jsonl.1").exists()  # and no backup rotation happened


def test_jsonl_sink_rotates(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    sink = JsonlSink(path, max_bytes=200, backup_count=2)
    for i in range(50):
        sink.write_line(f'{{"n": {i}, "pad": "{"x" * 30}"}}')
    sink.close()
    # rotation produced at least one backup, and replay still reads everything in order
    assert path.with_name(path.name + ".1").exists()
