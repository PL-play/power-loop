"""S4 (3.14): ``register_workflow_tools`` host injection points
(``executor_factory`` / ``budget_factory`` / ``spec_transform``)."""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.core.agent_context import reset_current_loop, reset_session_id, set_current_loop, set_session_id
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import register_workflow_tools
from power_loop.workflow.result import SharedBudget
from power_loop.workflow.spec import AgentNode, WorkflowSpec, WorkflowSpecError

pytestmark = pytest.mark.asyncio


@dataclass
class _FakeLLM(LLMService):
    replies: list[str] = field(default_factory=lambda: ["leaf out"])
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        reply = self.replies[min(self._idx, len(self.replies) - 1)]
        self._idx += 1
        return LLMResponse(raw_text=reply)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


def _loop() -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=_FakeLLM(),
        db_path=tempfile.mktemp(suffix=".db"),
        config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=3),
    )


SPEC = {
    "name": "wf",
    "root": {"type": "agent", "id": "a", "spec": {"name": "a", "system_prompt": "p"}},
}


def _registered_handler(**seams: Any):
    reg = ToolRegistry()
    register_workflow_tools(reg, **seams)
    rt = reg.get("create_workflow")
    assert rt is not None
    return rt.handler


async def _invoke(handler, loop: StatefulAgentLoop, **kwargs: Any) -> tuple[dict, dict | None]:
    from power_loop.workflow import journal
    from power_loop.workflow.runner import get_run_handle
    sid = await loop.new_session()
    token = set_current_loop(loop)
    stok = set_session_id(sid)
    try:
        out = await handler(**kwargs)
        receipt = json.loads(out)
        saved = None
        if receipt.get("status") == "accepted":
            handle = get_run_handle(receipt["run_id"])
            assert handle is not None
            await handle.task
            saved = await journal.read(await loop._ensure_store(), sid, receipt["run_id"])
        return receipt, saved
    finally:
        reset_session_id(stok)
        reset_current_loop(token)


async def test_no_seams_registers_default_handler():
    from power_loop.workflow.tool import _handle_create_workflow

    reg = ToolRegistry()
    register_workflow_tools(reg)
    assert reg.get("create_workflow").handler is _handle_create_workflow
    assert reg.has("workflow_status")


async def test_executor_factory_receives_context_and_is_used():
    calls: list[tuple[Any, str | None]] = []

    class _StubExec:
        async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
            return {"status": "completed", "final_text": "stubbed", "session_id": None,
                    "usage": {"total_tokens": 1}}

    def executor_factory(loop: Any, parent_sid: str | None) -> Any:
        calls.append((loop, parent_sid))
        return _StubExec()

    loop = _loop()
    handler = _registered_handler(executor_factory=executor_factory)
    out, saved = await _invoke(handler, loop, spec=SPEC)
    assert out["status"] == "accepted" and saved["status"] == "completed"
    assert saved["steps"][0]["text"] == "stubbed"
    # Evaluated per invocation, with the invoking loop.
    assert len(calls) == 1 and calls[0][0] is loop
    out2, saved2 = await _invoke(handler, loop, spec=SPEC)
    assert out2["status"] == "accepted" and saved2["status"] == "completed"
    assert len(calls) == 2


async def test_budget_factory_budget_is_enforced():
    """budget_factory + executor_factory compose: the stub leaf reports 60
    tokens of usage, exhausting the 60-token host budget after step one."""

    class _CostlyExec:
        async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
            return {"status": "completed", "final_text": "ok", "session_id": None,
                    "usage": {"total_tokens": 60}}

    def budget_factory(loop: Any, parent_sid: str | None) -> SharedBudget:
        return SharedBudget(60)

    handler = _registered_handler(
        budget_factory=budget_factory,
        executor_factory=lambda loop, sid: _CostlyExec(),
    )
    out, saved = await _invoke(handler, _loop(), spec={
        "name": "wf",
        "root": {"type": "sequence", "steps": [
            {"type": "agent", "id": "s1", "spec": {"name": "a", "system_prompt": "p"}},
            {"type": "agent", "id": "s2", "spec": {"name": "b", "system_prompt": "p"}},
        ]},
    })
    assert out["status"] == "accepted" and saved["status"] == "budget_exceeded"
    assert [step["status"] for step in saved["steps"]] == ["completed", "budget_exceeded"]


async def test_spec_transform_rewrites_spec():
    def spec_transform(spec: WorkflowSpec) -> WorkflowSpec:
        assert isinstance(spec.root, AgentNode)
        return replace(spec, name=f"clamped-{spec.name}")

    handler = _registered_handler(spec_transform=spec_transform)
    out, saved = await _invoke(handler, _loop(), spec=SPEC)
    assert out["workflow"] == saved["spec"]["name"] == "clamped-wf"


async def test_spec_transform_rejection_reads_like_validation_error():
    def spec_transform(spec: WorkflowSpec) -> WorkflowSpec:
        raise WorkflowSpecError(["policy: leaf 'a' requests a forbidden tool"])

    handler = _registered_handler(spec_transform=spec_transform)
    out, saved = await _invoke(handler, _loop(), spec=SPEC)
    assert out["error"]["code"] == "invalid_workflow_spec" and saved is None
    assert "forbidden tool" in out["error"]["problems"][0]


async def test_spec_transform_bad_return_type_reports_admission_failure_without_execution():
    handler = _registered_handler(spec_transform=lambda spec: {"nope": 1})
    out, saved = await _invoke(handler, _loop(), spec=SPEC)
    assert out["error"]["code"] == "workflow_admission_requires_reconciliation" and saved is None
