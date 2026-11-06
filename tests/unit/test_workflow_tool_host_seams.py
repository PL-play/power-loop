"""Tool registration uses the durable workflow admission contract."""
import pytest

from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import register_workflow_tools

pytestmark=pytest.mark.asyncio


async def test_no_seams_registers_default_handler():
    from power_loop.runtime.tool_contracts import policy, resolve

    reg = ToolRegistry()
    register_workflow_tools(reg)
    registered = reg.get("create_workflow")
    assert resolve(registered).runtime_contract.key == "sdk.workflow-create"
    assert resolve(registered).runtime_contract.allowed_roles == ("main",)
    declared = policy(registered, {})
    assert declared.native_admitted and not declared.control_only
    assert reg.has("workflow_status")
