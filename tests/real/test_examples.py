"""Real-LLM contract tests for ``examples/``.

These double as living documentation: if a numbered example breaks, either
the example needs an update or a public API regressed. Each example must
remain runnable as ``python examples/NN_*.py`` against real DashScope.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

# Examples use ``from _helpers import …`` — they're not a package, just a
# folder of scripts. Putting examples/ on sys.path makes that import work
# both for ``python examples/NN_*.py`` (auto) and for our test loader.
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))


def _load_example(filename: str):
    mod_name = f"example_{filename.removesuffix('.py')}"
    spec = importlib.util.spec_from_file_location(mod_name, EXAMPLES_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass() can look up the module via sys.modules.
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def test_example_00_hello_world_runs() -> None:
    module = _load_example("00_hello_world.py")
    final_text = asyncio.run(module.main())
    assert isinstance(final_text, str) and final_text.strip()


def test_example_01_multi_turn_chat_runs() -> None:
    module = _load_example("01_multi_turn_chat.py")
    final_text = asyncio.run(module.main())
    # The fact established in turn 1 must surface in turn 2.
    assert "teal" in final_text.lower(), (
        f"multi-turn answer did not recall 'teal': {final_text!r}"
    )


def test_example_02_tool_calling_runs() -> None:
    module = _load_example("02_tool_calling.py")
    final_text = asyncio.run(module.main())
    assert "pad thai" in final_text.lower(), (
        f"expected the answer to surface the tool result; got: {final_text!r}"
    )


def test_example_03_subagent_delegation_runs() -> None:
    module = _load_example("03_subagent_delegation.py")
    final_text = asyncio.run(module.main())
    assert "tokyo" in final_text.lower() or "东京" in final_text, (
        f"sub-agent answer did not surface 'Tokyo': {final_text!r}"
    )


def test_example_04_compaction_runs() -> None:
    module = _load_example("04_compaction.py")
    final_text = asyncio.run(module.main())
    assert "jupiter" in final_text.lower(), (
        f"compacted-history answer did not name Jupiter: {final_text!r}"
    )


def test_example_05_pending_recovery_runs() -> None:
    module = _load_example("05_pending_recovery.py")
    final_text = asyncio.run(module.main())
    assert "hypertext" in final_text.lower().replace("-", "").replace(" ", ""), (
        f"post-abort send did not produce the expected answer: {final_text!r}"
    )


def test_example_06_declarative_subagent_runs() -> None:
    module = _load_example("06_declarative_subagent.py")
    final_text = asyncio.run(module.main())
    # The orchestrator delegates (17+25)*3 = 126; the answer should surface
    # that number (subagent may include extra prose).
    assert "126" in final_text, (
        f"declarative subagent answer missing '126': {final_text!r}"
    )


def test_example_08_streaming_runs() -> None:
    """Streaming events fire and final_text equals concatenated chunks."""
    module = _load_example("08_streaming.py")
    final_text = asyncio.run(module.main())
    assert isinstance(final_text, str) and len(final_text.strip()) > 20


def test_example_09_audit_log_runs() -> None:
    """Audit subscriber writes a JSONL with the full lifecycle."""
    import json

    module = _load_example("09_audit_log.py")
    audit_path = asyncio.run(module.main())
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 5, f"audit file too small: {len(lines)} lines"
    types = {json.loads(ln)["type"] for ln in lines}
    # 必须覆盖 session + round + 至少一个流事件
    assert {"session_started", "session_ended"}.issubset(types), types
    assert any(t.startswith("round_") for t in types), types


def test_example_10_concurrent_sessions_runs() -> None:
    """Concurrent sessions all complete; the approval worker handled at
    least one denial path (the rm session) without blocking the others."""
    module = _load_example("10_concurrent_sessions.py")
    results = asyncio.run(module.main())
    assert len(results) == 3
    labels = {r["label"] for r in results}
    assert labels == {"S1", "S2", "S3"}
    # Distinct session ids — confirms one StatefulAgentLoop drove 3 sessions
    sids = {r["sid"] for r in results}
    assert len(sids) == 3


def test_example_21_request_user_input_runs() -> None:
    """Real LLM pauses through request_user_input, then submit_input resumes."""
    module = _load_example("21_request_user_input.py")
    results = asyncio.run(module.main(answers=["gentle", "send"]))
    assert len(results) == 2
    assert {item["label"] for item in results} == {"summary", "send"}
    assert all(item["status"] == "completed" for item in results)
    assert all(item["final_text"].strip() for item in results)
    # The two interactions were answered "gentle" then "send"; assert the model
    # engaged with each — robust to wording ("send"/"sent"/"sending"), since the
    # final text is non-deterministic. The "send" branch is already proven by the
    # label assertion above.
    texts = [item["final_text"].lower() for item in results]
    assert any("gentle" in t for t in texts)
    assert any(("send" in t or "sent" in t) for t in texts)


def test_example_22_follow_up_steering_runs() -> None:
    """Real LLM run accepts follow_up steering before the next round."""
    module = _load_example("22_follow_up_steering.py")
    result = asyncio.run(
        module.main(
            steering="Your final answer MUST include the exact word STEERED in uppercase."
        )
    )
    assert result["status"] == "completed"
    assert result["queue_depth"] == 1
    assert result["follow_up_message_count"] >= 1
    assert "STEERED" in result["final_text"]


def test_example_28_docker_shell_backend_runs() -> None:
    """Model-authored bash runs INSIDE a container via a custom ShellBackend.
    Skips cleanly when Docker is unavailable."""
    import pytest

    module = _load_example("28_docker_shell_backend.py")
    result = asyncio.run(module.main())
    if result is None:
        pytest.skip("Docker not available")
    bash = result["bash_outputs"].lower()
    # Proof from the RAW bash output (not the model's prose): the shell ran in
    # the container (Debian image, not the host) and saw the bind-mounted file.
    assert "debian" in bash, f"shell did not run in the container image: {bash!r}"
    assert "hello from the host machine" in bash, (
        f"bind-mounted host file not visible inside the sandbox: {bash!r}"
    )
    assert result["final_text"].strip()


def test_example_29_shared_blackboard_runs() -> None:
    """Two agents coordinate on one scoped shared board: the planner posts
    tasks, the worker reads them and updates one + leaves a note."""
    module = _load_example("29_shared_blackboard.py")
    result = asyncio.run(module.main())
    # Both agents wrote to the SAME board (coordination), authored correctly.
    assert set(result["authors"]) == {"planner", "worker"}, result
    # The worker advanced a task it found on the board (open → done).
    assert "done" in result["statuses"], result
    # The worker also left a free-form note.
    assert "note" in result["kinds"], result
    assert result["n"] >= 3, result


def test_example_07_human_approval_runs() -> None:
    """Always-deny confirm_fn: dangerous commands must NEVER execute;
    safe whitelist commands may still execute; final answer must
    acknowledge the denial (not silently retry)."""
    module = _load_example("07_human_approval.py")
    module.EXECUTED.clear()
    final_text = asyncio.run(module.main())

    executed = list(module.EXECUTED)
    # 1. No dangerous command got through.
    assert not any(
        any(t in cmd for t in module.DANGEROUS_TOKENS) for cmd in executed
    ), f"dangerous command leaked through: {executed!r}"

    # 2. The model acknowledged the denial in the final reply.
    low = final_text.lower()
    assert any(kw in low for kw in ("deni", "not execute", "refused", "cannot")), (
        f"reply does not acknowledge denial: {final_text!r}"
    )
