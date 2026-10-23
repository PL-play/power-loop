"""The acceptance runner preserves coverage selection and interruption evidence."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/test_task_registry.py"
spec = importlib.util.spec_from_file_location("task_acceptance_runner", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_unit_shards_cover_every_file_exactly_once_in_stable_order():
    files = [f"tests/unit/test_{number:02}.py" for number in range(27)]
    groups = [runner.unit_partition(list(reversed(files)), f"{index}/4") for index in range(1, 5)]
    assert sorted(item for group in groups for item in group) == files
    assert sum(map(len, groups)) == len(set(item for group in groups for item in group))
    assert groups == [runner.unit_partition(files, f"{index}/4") for index in range(1, 5)]


@pytest.mark.parametrize("selection", ["", "1", "1/2/3", "0/4", "5/4", "1/0", "1/65", "true/4", None])
def test_bad_shards_fail_before_any_test_process(selection):
    with pytest.raises(ValueError):
        runner.unit_partition(["test_a.py"], selection)


def test_empty_shard_does_not_count_as_acceptance():
    with pytest.raises(ValueError, match="empty"):
        runner.unit_partition(["test_a.py"], "2/4")


def test_streamed_failure_is_saved_and_secrets_are_removed(tmp_path):
    log = tmp_path / "run.log"
    code = runner.stream_run([sys.executable, "-c", "print('private-key: result'); raise SystemExit(7)"],
        repo=tmp_path, env=os.environ.copy(), log_path=log,
        sanitize=lambda value: value.replace("private-key", "[redacted]"))
    assert code == 7
    assert log.read_text() == "[redacted]: result\n"


def test_existing_log_cannot_be_overwritten_by_a_new_process(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("original evidence")
    with pytest.raises(FileExistsError):
        runner.stream_run([sys.executable, "-c", "import time; time.sleep(30)"], repo=tmp_path,
            env=os.environ.copy(), log_path=log, sanitize=lambda value: value)
    assert log.read_text() == "original evidence"


@pytest.mark.parametrize("forced", [False, True])
def test_interrupted_read_always_stops_the_owned_child(tmp_path, monkeypatch, forced):
    calls = []
    class Output:
        def __iter__(self):
            yield "partial evidence\n"
            raise runner.RunnerInterrupted(15)
        def close(self):
            calls.append("close")
    class Process:
        stdout = Output()
        def poll(self):
            return None
        def terminate(self):
            calls.append("terminate")
        def kill(self):
            calls.append("kill")
        def wait(self, timeout=None):
            calls.append("wait")
            if forced and calls.count("wait") == 1:
                raise subprocess.TimeoutExpired("test child", timeout)
            return -15
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    log = tmp_path / "partial.log"
    with pytest.raises(runner.RunnerInterrupted):
        runner.stream_run(["test-child"], repo=tmp_path, env={}, log_path=log, sanitize=lambda value: value)
    assert log.read_text() == "partial evidence\n"
    assert calls == (["terminate", "wait", "kill", "wait", "close"] if forced else ["terminate", "wait", "close"])


def test_source_fingerprint_covers_all_groups_and_configuration(tmp_path):
    for directory in ("power_loop", "tests/unit", "tests/real", "scripts"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    for file in ("power_loop/a.py", "tests/unit/test_a.py", "tests/real/test_a.py", "scripts/test_task_registry.py", "pyproject.toml"):
        (tmp_path / file).write_text("initial")
    original = runner.source_hash(tmp_path)
    (tmp_path / "tests/real/test_a.py").write_text("changed")
    assert runner.source_hash(tmp_path) != original
    (tmp_path / "tests/real/test_a.py").write_text("initial")
    assert runner.source_hash(tmp_path) == original


def test_streamed_success_returns_child_exit(tmp_path):
    log = tmp_path / "success.log"
    code = runner.stream_run([sys.executable, "-c", "print('done')"], repo=tmp_path,
        env=os.environ.copy(), log_path=log, sanitize=lambda value: value)
    assert code == 0 and log.read_text() == "done\n"
