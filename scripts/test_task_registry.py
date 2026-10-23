#!/usr/bin/env python3
"""Run B1 tests in SQLite and a disposable PostgreSQL *_test DB.

Uses the existing local Postgres container without restarting it. MySQL tests
are explicitly excluded at the user's request. Real LLM tests use tests/.env
configuration and must actually run, never silently skip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

REQUIRED_REAL_CASES = 90


class RunnerInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def shard_numbers(selection):
    try:
        index, total = (int(value) for value in selection.split("/"))
    except (ValueError, AttributeError) as exc:
        raise ValueError("unit shard must be INDEX/TOTAL") from exc
    if not 1 <= index <= total <= 64:
        raise ValueError("unit shard requires 1 <= INDEX <= TOTAL <= 64")
    return index, total


def unit_partition(files, selection):
    """Deterministic disjoint file shards; no test selection is silently lost."""
    index, total = shard_numbers(selection)
    selected = sorted(files)[index-1::total]
    if not selected:
        raise ValueError("unit shard is empty")
    return selected


def real_case_ids(collection):
    """Canonical XML identities from pytest collection, before any paid call."""
    result = []
    for line in collection.splitlines():
        if not line.startswith("tests/real/") or "::" not in line:
            continue
        parts = line.split("::")
        path = Path(parts[0])
        if ".." in path.parts or path.suffix != ".py":
            raise ValueError("invalid real test collection path")
        result.append((".".join([str(path.with_suffix("")).replace("/", "."), *parts[1:-1]]), parts[-1]))
    if not result or len(result) != len(set(result)):
        raise ValueError("empty or duplicate real test collection")
    return result


def selected_real_cases(cases, scopes):
    modules = {str(Path(scope).with_suffix("")).replace("/", ".") for scope in scopes}
    selected = [case for case in cases if any(case[0] == m or case[0].startswith(m+".") for m in modules)]
    if not selected or {m for m in modules if not any(c[0] == m or c[0].startswith(m+".") for c in selected)}:
        raise ValueError("real selection includes missing or empty modules")
    return selected


def source_hash(repo):
    files = [*repo.joinpath("power_loop").rglob("*.py"), *repo.joinpath("tests").rglob("*.py"),
        repo / "scripts/test_task_registry.py", repo / "pyproject.toml"]
    values = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def stream_run(command, *, repo, env, log_path, sanitize):
    """Save sanitized output as it arrives; interruption always stops our child."""
    process = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        with log_path.open("x") as target:
            for line in process.stdout:
                target.write(sanitize(line))
                target.flush()
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        process.stdout.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run all non-MySQL unit tests plus B1 real tests")
    parser.add_argument("--container", default="deeptalk-infra-postgres-1")
    parser.add_argument("--label", help="unique artifact label (letters, numbers, dashes only)")
    parser.add_argument("--unit-shard", help="run one disjoint full unit shard INDEX/TOTAL, without real tests")
    parser.add_argument("--real-only", action="store_true", help="run all configured real-provider acceptance cases")
    parser.add_argument("--real-shard", help="run one disjoint real-provider file shard INDEX/TOTAL")
    args = parser.parse_args()
    if args.unit_shard and args.real_only:
        parser.error("unit shards and real-only are separate acceptance groups")
    if args.real_shard and (args.unit_shard or args.real_only or args.full):
        parser.error("real shards cannot be combined with unit-shard, real-only or full")
    for selection in (args.unit_shard, args.real_shard):
        if selection is not None:
            try:
                shard_numbers(selection)
            except ValueError as exc:
                parser.error(str(exc))
    repo = Path(__file__).resolve().parents[1]
    out = repo.parent / "outputs/async-task-tests-20261022"
    out.mkdir(parents=True, exist_ok=True)
    inspected = json.loads(subprocess.check_output(["docker", "inspect", args.container], text=True))[0]
    settings = dict(item.split("=", 1) for item in inspected["Config"]["Env"] if "=" in item)
    user, password = settings["POSTGRES_USER"], settings["POSTGRES_PASSWORD"]
    database = "b1192_" + secrets.token_hex(6) + "_test"
    assert database.endswith("_test") and database.replace("_", "").isalnum()
    dsn = f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@127.0.0.1:5433/{database}"

    def psql(statement):
        subprocess.run(["docker", "exec", args.container, "psql", "-U", user, "-d", "postgres",
                        "-v", "ON_ERROR_STOP=1", "-c", statement], capture_output=True, text=True, check=True)

    def sanitized(value):
        for secret in (dsn, quote(password, safe=""), password):
            if secret:
                value = value.replace(secret, "[redacted]")
        return value

    label = args.label or ("b1-full-" if args.full else "b1-focused-") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in label):
        parser.error("invalid artifact label")
    if any((out / f"{label}{suffix}").exists() for suffix in (".xml", ".log", ".meta.json", ".collect.log")):
        parser.error("artifact label already exists; refusing overwrite")
    report = out / f"{label}.xml"
    scopes = ["tests/unit"] if args.full else [
        "tests/unit/test_task_message_identity_192.py", "tests/unit/test_job_registry_192.py",
        "tests/unit/test_execution_membership_192.py",
        "tests/unit/test_workflow_controller_192.py",
        "tests/unit/test_workflow_node_output_192.py",
        "tests/unit/test_workflow_result_delivery_192.py",
        "tests/unit/test_workflow_projection_atomic_192.py",
        "tests/unit/test_workflow_profile_192.py",
        "tests/unit/test_workflow_resources_192.py",
        "tests/unit/test_workflow_continuation_budget_192.py",
        "tests/unit/test_workflow_resource_profile_192.py",
        "tests/unit/test_workflow_restore_resources_192.py",
        "tests/unit/test_workflow_bound_restore_192.py",
        "tests/unit/test_completion_window_192.py",
        "tests/unit/test_finalization_192.py",
        "tests/unit/test_model_attempts_192.py",
        "tests/unit/test_budget_contracts_192.py",
        "tests/unit/test_model_budgets_192.py",
        "tests/unit/test_model_attempt_transport_192.py",
        "tests/unit/test_model_attempt_edges_192.py",
        "tests/unit/test_answer_freshness_192.py",
        "tests/unit/test_executor_registry_192.py",
        "tests/unit/test_executor_runtime_192.py",
        "tests/unit/test_executor_process_192.py",
        "tests/unit/test_executor_lifecycle_192.py",
        "tests/unit/test_executor_ownership_192.py",
        "tests/unit/test_restored_answer_192.py",
        "tests/unit/test_workflow_completion_scope_192.py",
        "tests/unit/test_workflow_restore_plan_192.py",
        "tests/unit/test_workflow_restore_history_192.py",
        "tests/unit/test_workflow_resume_admission_192.py",
        "tests/unit/test_workflow_resume_evidence_192.py",
        "tests/unit/test_workflow_restore_runtime_192.py",
        "tests/unit/test_task_query_192.py",
        "tests/unit/test_task_control_192.py",
        "tests/unit/test_jobs_tool_192.py",
        "tests/unit/test_workflow_facade_192.py",
        "tests/unit/test_workflow_control_192.py",
        "tests/unit/test_node_control_192.py",
        "tests/unit/test_task_sessions_192.py",
        "tests/unit/test_task_history_192.py",
        "tests/unit/test_task_events_192.py",
        "tests/unit/test_workflow_node_facade_192.py",
        "tests/unit/test_legacy_workflow_history_192.py",
        "tests/unit/test_tool_invocation_identity_192.py", "tests/unit/test_execution_roles_192.py",
        "tests/unit/test_task_runtime_192.py",
        "tests/unit/test_workflow_task_runtime_192.py",
        "tests/unit/test_processing_decisions_192.py",
        "tests/unit/test_nudge_runtime_192.py",
        "tests/unit/test_node_nudge_runtime_192.py",
        "tests/unit/test_node_binding_generation_192.py",
        "tests/unit/test_node_resume_192.py",
        "tests/unit/test_workflow_yaml.py", "tests/unit/test_workflow_v2_yaml.py",
        "tests/unit/test_workflow_output_file_inputs.py", "tests/unit/test_workflow_yaml_native.py",
        "tests/unit/test_subagent_resume_192.py",
        "tests/unit/test_subagent_output_192.py",
        "tests/unit/test_workflow_live_continuation.py",
        "tests/unit/test_workflow_continuation_plan.py"]
    scopes.append("tests/real/test_real_task_registry_192.py")
    scopes.append("tests/real/test_real_workflow_task_runtime_192.py")
    scopes.append("tests/real/test_real_nudge_runtime_192.py")
    scopes.append("tests/real/test_real_node_nudge_runtime_192.py")
    scopes.append("tests/real/test_real_subagent_resume_192.py")
    scopes.append("tests/real/test_real_workflow_continuation_plan.py")
    scopes.append("tests/real/test_real_node_resume_192.py")
    scopes.append("tests/real/test_real_workflow_delivery_192.py")
    scopes.append("tests/real/test_real_workflow_restore_192.py")
    scopes.append("tests/real/test_real_workflow_native_restore_192.py")
    scopes.append("tests/real/test_real_task_query_192.py")
    scopes.append("tests/real/test_real_jobs_tool_192.py")
    scopes.append("tests/real/test_real_workflow_facade_192.py")
    scopes.append("tests/real/test_real_workflow_control_192.py")
    scopes.append("tests/real/test_real_node_control_192.py")
    scopes.append("tests/real/test_real_task_sessions_192.py")
    scopes.append("tests/real/test_real_task_history_192.py")
    scopes.append("tests/real/test_real_legacy_workflow_history_192.py")
    scopes.append("tests/real/test_real_task_events_192.py")
    scopes.append("tests/real/test_real_workflow_node_facade_192.py")
    scopes.append("tests/real/test_real_workflow_resources_192.py")
    scopes.append("tests/real/test_real_workflow_resource_profile_192.py")
    scopes.append("tests/real/test_real_workflow_continuation_budget_192.py")
    scopes.append("tests/real/test_real_completion_window_192.py")
    scopes.append("tests/real/test_real_model_attempts_192.py")
    scopes.append("tests/real/test_real_answer_freshness_192.py")
    scopes.append("tests/real/test_real_executor_registry_192.py")
    scopes.append("tests/real/test_real_executor_lifecycle_192.py")
    scopes.append("tests/real/test_real_budget_tree_192.py")
    scopes.append("tests/real/test_real_timer_roles_192.py")
    scopes.append("tests/real/test_real_timer_waits_192.py")
    scopes.append("tests/real/test_real_wait_handoff_192.py")
    scopes.append("tests/real/test_real_wait_effects_192.py")
    all_real_scopes = [scope for scope in scopes if scope.startswith("tests/real/")]
    run_real = args.unit_shard is None
    if args.real_only:
        scopes = [scope for scope in scopes if scope.startswith("tests/real/")]
    if args.real_shard:
        try:
            scopes = unit_partition(all_real_scopes, args.real_shard)
        except ValueError as exc:
            parser.error(str(exc).replace("unit shard", "real shard"))
    if args.unit_shard:
        all_units = [str(path.relative_to(repo)) for path in repo.joinpath("tests/unit").rglob("test_*.py")
                     if path.name != "test_store_parity_mysql.py"]
        try:
            scopes = unit_partition(all_units, args.unit_shard)
        except ValueError as exc:
            parser.error(str(exc))
    command = [sys.executable, "-m", "pytest", "-v" if args.real_shard else "-q", "-p", "no:cacheprovider",
               "-o", "junit_family=legacy", "--tb=short", "-k", "not mysql",
               "--ignore=tests/unit/test_store_parity_mysql.py", *scopes, "--junitxml=" + str(report)]
    env = {**os.environ, "POWER_LOOP_TASK_PG_DSN": dsn,
           "POWER_LOOP_TEST_PG_DSN": dsn, "POSTGRES_PASSWORD": password}
    metadata_path = out / f"{label}.meta.json"
    metadata = {"label": label, "database": database, "container": args.container,
        "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
        "source_hash": source_hash(repo), "unit_shard": args.unit_shard,
        "real_shard": args.real_shard,
        "real_only": args.real_only, "real_acceptance_enabled": run_real,
        "selected_scopes": scopes, "status": "preparing", "database_cleanup_confirmed": False}
    with metadata_path.open("x") as target:
        json.dump(metadata, target, ensure_ascii=False, indent=2)
        target.flush()
        os.fsync(target.fileno())

    def save_metadata():
        temporary = metadata_path.with_suffix(".new")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(metadata_path)

    def interrupted(signum, _frame):
        raise RunnerInterrupted(signum)

    previous_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_signals:
        signal.signal(sig, interrupted)
    try:
        psql(f'CREATE DATABASE "{database}"')
        expected_real = []
        if run_real:
            metadata["status"] = "collecting"
            save_metadata()
            collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "--color=no",
                "-p", "no:cacheprovider", *all_real_scopes], cwd=repo, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
            with (out / f"{label}.collect.log").open("x") as target:
                target.write(sanitized(collected.stdout))
            try:
                all_real = real_case_ids(collected.stdout)
                if collected.returncode or len(all_real) != REQUIRED_REAL_CASES:
                    raise ValueError("all configured real-provider cases must collect")
                expected_real = selected_real_cases(all_real, scopes if args.real_shard else all_real_scopes)
            except ValueError as exc:
                metadata.update(status="collection_failed", returncode=1, error=str(exc))
                save_metadata()
                return 1
            metadata["expected_real_cases"] = expected_real
            if source_hash(repo) != metadata["source_hash"]:
                metadata["status"] = "source_changed"
                save_metadata()
                return 1
        metadata["status"] = "running"
        save_metadata()
        code = stream_run(command, repo=repo, env=env, log_path=out / f"{label}.log", sanitize=sanitized)
        log = (out / f"{label}.log").read_text()
        print("\n".join(log.splitlines()[-45:]))
        metadata.update(status="finished", returncode=code)
        save_metadata()
        if not report.exists():
            return code or 1
        report.write_text(sanitized(report.read_text()))
        if source_hash(repo) != metadata["source_hash"]:
            metadata["status"] = "source_changed"
            save_metadata()
            print("Task acceptance failed: source changed while this group ran.")
            return 1
        if not run_real:
            return code
        cases = [c for c in ET.parse(report).getroot().iter("testcase") if c.get("classname", "").startswith("tests.real.")]
        actual = [(c.get("classname"), c.get("name")) for c in cases]
        if len(actual) != len(set(actual)) or set(actual) != set(expected_real) or any(c.find("skipped") is not None for c in cases):
            metadata.update(status="acceptance_failed", returncode=1)
            save_metadata()
            print("Task acceptance failed: every selected real-provider case must actually run exactly once.")
            return 1
        return code
    except RunnerInterrupted as exc:
        metadata.update(status="interrupted", signal=exc.signum)
        save_metadata()
        return 128 + exc.signum
    finally:
        try:
            psql(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
            metadata["database_cleanup_confirmed"] = True
            save_metadata()
            print("Disposable PostgreSQL test database removed; MySQL tests excluded.")
        finally:
            for sig, handler in previous_signals.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
