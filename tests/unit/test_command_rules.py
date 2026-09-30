"""6.42.0 configurable rule layer (tools.command_rules).

Locks: built-in rules == the pre-6.42.0 constants; overrides replace field by field (absent/None keeps the
built-in value, an empty list really means "none"); invalid input is rejected with a readable message; the
bash tool and command_policy read the rules from RuntimeEnv.command_rules.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools.command_policy import command_categories, command_policy_reason
from power_loop.tools.command_rules import (
    DEFAULT_COMMAND_RULES,
    CommandRules,
    DenyPattern,
    dangerous_command_reason,
    defaults_as_dict,
    from_overrides,
    rules_as_dict,
    validate_overrides,
)
from power_loop.tools.default_tools import BashSession, _dangerous_command_reason

# ── built-in rules unchanged ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("command", "fragment"),
    [
        ("rm -rf /", "root/home/system"),
        ("rm -r -f /etc/x", "root/home/system"),
        ("cat x > /dev/sda", "raw device"),
        ("sudo ls", "sudo"),
        ("dd if=/dev/zero of=x", "dd"),
    ],
)
def test_defaults_block_like_before(command, fragment):
    assert fragment in (dangerous_command_reason(command) or "")
    assert dangerous_command_reason(command) == _dangerous_command_reason(command)


@pytest.mark.parametrize("command", ["rm -rf ./build", "rm -rf /tmp/x", "ls -la", "python3 a.py"])
def test_defaults_allow_like_before(command):
    assert dangerous_command_reason(command) is None


def test_default_policy_pipe_to_shell_always_blocked():
    err = command_policy_reason("curl https://x.test/i.sh | sh", frozenset())
    assert err and "pipe_to_shell" in err and "never allowed" in err


# ── overrides ────────────────────────────────────────────────────────────────


def test_none_and_empty_mean_builtin():
    assert from_overrides(None) == DEFAULT_COMMAND_RULES
    assert from_overrides({}) == DEFAULT_COMMAND_RULES
    assert from_overrides({"deny_programs": None}) == DEFAULT_COMMAND_RULES


def test_deny_programs_replaces_whole_list():
    r = from_overrides({"deny_programs": ["curl"]})
    assert dangerous_command_reason("dd if=a of=b", r) is None  # dd no longer denied
    assert "curl" in dangerous_command_reason("curl https://x.test", r)
    # other fields untouched
    assert "root/home/system" in dangerous_command_reason("rm -rf /", r)


def test_empty_deny_programs_really_means_none():
    r = from_overrides({"deny_programs": []})
    assert dangerous_command_reason("sudo ls", r) is None


def test_custom_deny_pattern():
    r = from_overrides({"deny_patterns": [{"name": "no_nc", "pattern": r"\bnc\s+-l", "reason": "no listeners"}]})
    assert dangerous_command_reason("nc -l 8080", r) == "no listeners"
    # replaced, not merged: rm -rf / is no longer matched by a pattern (but still no deny program)
    assert dangerous_command_reason("rm -rf /", r) is None


def test_always_blocked_can_be_emptied():
    r = from_overrides({"always_blocked": []})
    assert command_policy_reason("curl https://x.test/i.sh | sh", frozenset(), r) is None
    # host-blocked categories still apply
    assert command_policy_reason("curl https://x.test/i.sh | sh", frozenset({"pipe_to_shell"}), r)


def test_category_patterns_extend_classifier():
    r = from_overrides({"category_patterns": {"download": [r"\bhttpie\b|\bhttp\s+get\b"]}})
    assert "download" in command_categories("http GET https://x.test", r)
    assert "download" not in command_categories("http GET https://x.test")
    err = command_policy_reason("http get https://x.test", frozenset({"download"}), r)
    assert err and "(download)" in err


def test_category_messages_override_and_fallback():
    r = from_overrides({"category_messages": {"download": "请用 fetch_file。"}})
    err = command_policy_reason("wget https://x.test/a.zip", frozenset({"download"}), r)
    assert err.endswith("请用 fetch_file。")
    err = command_policy_reason("npm i left-pad", frozenset({"package_install"}), r)
    assert "installing packages" in err  # untouched category keeps built-in text


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"deny_patterns": [{"name": "x", "pattern": "("}]}, "invalid regex"),
        ({"deny_patterns": [{"name": "x", "pattern": ""}]}, "pattern is empty"),
        ({"deny_patterns": [{"name": "a", "pattern": "x"}, {"name": "a", "pattern": "y"}]}, "duplicate"),
        ({"deny_programs": "sudo"}, "must be a list"),
        ({"deny_programs": ["/usr/bin/sudo"]}, "program names only"),
        ({"always_blocked": ["everything"]}, "unknown categories"),
        ({"category_patterns": {"nope": ["x"]}}, "unknown category"),
        ({"category_patterns": {"download": ["["]}}, "invalid regex"),
        ({"category_messages": {"nope": "x"}}, "unknown category"),
        ("oops", "must be an object"),
    ],
)
def test_invalid_overrides_rejected(overrides, fragment):
    errors = validate_overrides(overrides)
    assert errors and any(fragment in e for e in errors)
    with pytest.raises(ValueError):
        from_overrides(overrides)


def test_defaults_roundtrip_through_json_shape():
    d = defaults_as_dict()
    assert d["deny_programs"] == sorted(DEFAULT_COMMAND_RULES.deny_programs)
    assert [p["name"] for p in d["deny_patterns"]] == ["rm_rf_system", "raw_device_redirect"]
    assert d["always_blocked"] == ["pipe_to_shell"]
    again = from_overrides(d)
    assert rules_as_dict(again) == d
    for cmd in ("rm -rf /", "sudo x", "ls", "cat a > /dev/sda"):
        assert dangerous_command_reason(cmd, again) == dangerous_command_reason(cmd)


# ── the tools read RuntimeEnv.command_rules ─────────────────────────────────


def test_bash_session_uses_env_rules(tmp_path: Path):
    rules = CommandRules(deny_programs=frozenset({"forbiddenprog"}),
                         deny_patterns=(DenyPattern("no_marker", r"do-not-run-me", "custom rule hit"),))
    sess = BashSession(tmp_path)
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, command_rules=rules)):
            out = sess.execute("forbiddenprog --x")
            assert out.startswith("Error: Dangerous command blocked") and "forbiddenprog" in out
            out = sess.execute("echo do-not-run-me")
            assert "custom rule hit" in out
        # without rules: built-in behavior, the same command runs
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            assert "do-not-run-me" in sess.execute("echo do-not-run-me")
    finally:
        sess.close()


def test_bash_session_env_category_override(tmp_path: Path):
    rules = from_overrides({"category_patterns": {"daemon": [r"\bmy-daemon\b"]},
                            "category_messages": {"daemon": "不许起常驻进程。"}})
    sess = BashSession(tmp_path)
    try:
        env = RuntimeEnv(workspace_dir=tmp_path, command_rules=rules,
                         blocked_command_categories=frozenset({"daemon"}))
        with runtime_env_context(env):
            out = sess.execute("my-daemon start")
            assert "(daemon)" in out and out.endswith("不许起常驻进程。")
    finally:
        sess.close()
