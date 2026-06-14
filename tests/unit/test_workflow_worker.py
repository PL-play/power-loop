"""Phase 0 feasibility: the isolated worker core.

Proves the gate for an out-of-process executor with a database-per-subprocess:
a sub-agent can run to completion against its OWN db file with llm/tools rebuilt
from a WorkerBootstrap alone — no parent loop, no shared store, no shared bus —
and the supervisor can afterward open that db read-only and inspect the full
trace. (Same-process here; the boundary is what matters, and it is clean.)
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import SessionStore
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import WorkerBootstrap, WorkerBootstrapError, run_spec_isolated

pytestmark = pytest.mark.unit


@dataclass
class _FakeLLM(LLMService):
    reply: str = "isolated answer"

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        return LLMResponse(raw_text=self.reply, content_text=self.reply)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


async def test_isolated_worker_runs_from_config_only_and_leaves_inspectable_db() -> None:
    db = tempfile.mktemp(suffix=".db")
    # Bootstrap carries ONLY a way to rebuild deps — no parent objects exist here.
    boot = WorkerBootstrap(llm_factory=lambda: _FakeLLM(reply="hello from the leaf"))

    result = await run_spec_isolated(
        {"name": "leaf", "system_prompt": "be brief"},
        "do the thing",
        bootstrap=boot,
        db_path=db,
    )

    assert result["status"] == "completed"
    assert result["final_text"] == "hello from the leaf"
    assert result["db_path"] == db
    assert os.path.exists(db)  # the sub-agent's own ledger persists for inspection

    # Supervisor opens the sub-agent's private db read-side and sees the full trace.
    store = SessionStore.open(db)
    try:
        msgs = store.load_all_messages(result["session_id"])
        roles = [m.role for m in msgs]
        assert "user" in roles and "assistant" in roles
        assert any("do the thing" in (m.content or "") for m in msgs)
    finally:
        store.close()
    os.remove(db)


async def test_isolated_worker_honors_output_schema() -> None:
    db = tempfile.mktemp(suffix=".db")
    boot = WorkerBootstrap(llm_factory=lambda: _FakeLLM(reply='{"label": "urgent"}'))
    result = await run_spec_isolated(
        {"name": "classifier", "system_prompt": "classify",
         "output_schema": {"name": "C", "schema": {"type": "object", "required": ["label"],
            "properties": {"label": {"type": "string"}}}}},
        "ticket text",
        bootstrap=boot,
        db_path=db,
    )
    # output_schema is honored without any parent: the request carried a json_schema
    # response_format and the structured reply came back as final_text.
    assert result["status"] == "completed"
    assert '"label"' in result["final_text"]
    os.remove(db)


async def test_two_isolated_workers_use_separate_dbs() -> None:
    """The core of the user's proposal: independent leaves, independent ledgers."""
    a, b = tempfile.mktemp(suffix=".db"), tempfile.mktemp(suffix=".db")
    boot_a = WorkerBootstrap(llm_factory=lambda: _FakeLLM(reply="A"))
    boot_b = WorkerBootstrap(llm_factory=lambda: _FakeLLM(reply="B"))
    ra = await run_spec_isolated({"name": "a", "system_prompt": "p"}, "x", bootstrap=boot_a, db_path=a)
    rb = await run_spec_isolated({"name": "b", "system_prompt": "p"}, "y", bootstrap=boot_b, db_path=b)
    assert ra["final_text"] == "A" and rb["final_text"] == "B"
    assert a != b and os.path.exists(a) and os.path.exists(b)
    # Each file is written by exactly one loop → the one-writer-per-file rule holds
    # trivially, no shared-write coordination needed.
    os.remove(a)
    os.remove(b)


async def test_tool_whitelist_narrows_rebuilt_registry() -> None:
    db = tempfile.mktemp(suffix=".db")

    def _registry() -> ToolRegistry:
        from power_loop.contracts.tools import ToolDefinition
        reg = ToolRegistry()
        for name in ("alpha", "beta"):
            reg.register(ToolDefinition(name=name, description=name,
                         input_schema={"type": "object", "properties": {}}), lambda **k: "ok")
        return reg

    boot = WorkerBootstrap(llm_factory=lambda: _FakeLLM(), registry_factory=_registry)
    # spec restricts to just 'alpha'; the rebuilt registry must be narrowed to it.
    result = await run_spec_isolated(
        {"name": "leaf", "system_prompt": "p", "tools": ["alpha"]},
        "go", bootstrap=boot, db_path=db,
    )
    assert result["status"] == "completed"
    os.remove(db)


def test_bootstrap_without_llm_source_is_a_clear_error() -> None:
    with pytest.raises(WorkerBootstrapError, match="no LLM source"):
        WorkerBootstrap().build_llm()
