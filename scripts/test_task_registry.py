#!/usr/bin/env python3
"""Run B1 tests in SQLite and a disposable PostgreSQL *_test DB.

Uses the existing local Postgres container without restarting it. MySQL tests
are explicitly excluded at the user's request. Real LLM tests use tests/.env
configuration and must actually run, never silently skip.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run all non-MySQL unit tests plus B1 real tests")
    parser.add_argument("--container", default="deeptalk-infra-postgres-1")
    parser.add_argument("--label", help="unique artifact label (letters, numbers, dashes only)")
    args = parser.parse_args()
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
    if (out / f"{label}.xml").exists() or (out / f"{label}.log").exists():
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
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "-o", "junit_family=legacy", "--tb=short", "-k", "not mysql",
               "--ignore=tests/unit/test_store_parity_mysql.py", *scopes, "--junitxml=" + str(report)]
    env = {**os.environ, "POWER_LOOP_TASK_PG_DSN": dsn,
           "POWER_LOOP_TEST_PG_DSN": dsn, "POSTGRES_PASSWORD": password}
    psql(f'CREATE DATABASE "{database}"')
    try:
        result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True)
        log = sanitized(result.stdout + result.stderr)
        (out / f"{label}.log").write_text(log)
        print("\n".join(log.splitlines()[-45:]))
        if not report.exists():
            return result.returncode or 1
        report.write_text(sanitized(report.read_text()))
        cases = [c for c in ET.parse(report).getroot().iter("testcase")
                 if any(name in c.get("classname", "") for name in ("test_real_task_registry_192", "test_real_workflow_task_runtime_192", "test_real_nudge_runtime_192", "test_real_node_nudge_runtime_192", "test_real_subagent_resume_192", "test_real_workflow_continuation_plan", "test_real_node_resume_192", "test_real_workflow_delivery_192", "test_real_workflow_restore_192", "test_real_workflow_native_restore_192", "test_real_task_query_192", "test_real_jobs_tool_192", "test_real_workflow_facade_192", "test_real_task_sessions_192", "test_real_workflow_control_192", "test_real_node_control_192", "test_real_task_history_192"))]
        cases.extend(c for c in ET.parse(report).getroot().iter("testcase")
                     if any(name in c.get("classname", "") for name in ("test_real_legacy_workflow_history_192", "test_real_task_events_192", "test_real_workflow_node_facade_192")))
        if len(cases) != 55 or any(c.find("skipped") is not None for c in cases):
            print("Task acceptance failed: all fifty-five configured real LLM cases must actually run.")
            return 1
        return result.returncode
    finally:
        psql(f'DROP DATABASE "{database}" WITH (FORCE)')
        print("Disposable PostgreSQL test database removed; MySQL tests excluded.")


if __name__ == "__main__":
    raise SystemExit(main())
