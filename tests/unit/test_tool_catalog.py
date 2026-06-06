"""Tests for M1.10 tool catalog auto-injection into system prompt.

Covers:
* ``format_tool_catalog`` from a list of ToolDefinition
* ``format_tool_catalog`` from a ToolRegistry
* ``AgentPipeline.__init__`` auto-injection (on/off/custom header)
* ``section_tool_catalog`` builder integration
* Empty registry / no-registry cases (no crash, no phantom section)
* Compaction safety: injected catalog is on self.system_prompt (string),
  never in self.history — compactor does not touch it.

**Note**: parameter schema is intentionally omitted from the catalog — it
is already sent to the LLM via the structured ``tools=`` API parameter.
The catalog only lists tool names and descriptions.
"""
from __future__ import annotations

import pytest

from power_loop import (
    AgentLoopConfig,
    SystemPromptBuilder,
    SystemPromptContext,
    ToolDefinition,
    ToolRegistry,
)
from power_loop.agent.system_prompt import (
    format_tool_catalog,
    section_tool_catalog,
)
from power_loop.core.events import AgentEventBus
from power_loop.core.hooks import AgentHooks
from power_loop.core.pipeline import AgentPipeline
from power_loop.core.state import ContextManager


# ── fixtures ─────────────────────────────────────────────────────────────

def _make_defs() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="calc",
            description="Evaluate arithmetic expressions",
            input_schema={
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "Expression to evaluate"},
                },
                "required": ["expr"],
            },
            required_params=("expr",),
        ),
        ToolDefinition(
            name="greet",
            description="Say hello",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Person name"},
                    "formal": {"type": "boolean", "description": "Use formal greeting"},
                },
                "required": ["name"],
            },
            required_params=("name",),
        ),
        ToolDefinition(
            name="ping",
            description="Health check",
        ),
    ]


def _make_registry(defs: list[ToolDefinition] | None = None) -> ToolRegistry:
    reg = ToolRegistry()
    for d in (defs or _make_defs()):
        reg.register(d, lambda **kw: "")
    return reg


def _make_pipeline(
    *,
    system_prompt: str = "You are an assistant.",
    inject: bool = True,
    header: str = "# Available Tools",
    registry: ToolRegistry | None = None,
) -> AgentPipeline:
    cfg = AgentLoopConfig(
        system_prompt=system_prompt,
        inject_tool_descriptions=inject,
        tool_catalog_header=header,
    )
    return AgentPipeline(
        llm=None,
        config=cfg,
        tool_registry=registry,
        hooks=AgentHooks(),
        bus=AgentEventBus(),
        ctx=ContextManager(),
    )


# ── format_tool_catalog ──────────────────────────────────────────────────


class TestFormatToolCatalog:
    def test_from_list(self):
        defs = _make_defs()
        result = format_tool_catalog(defs)
        assert result.startswith("# Available Tools")
        assert "- **calc**: Evaluate arithmetic expressions" in result
        assert "- **greet**: Say hello" in result
        assert "- **ping**: Health check" in result

    def test_from_registry(self):
        reg = _make_registry()
        result = format_tool_catalog(reg)
        assert "- **calc**:" in result

    def test_custom_header(self):
        result = format_tool_catalog(_make_defs(), header="# My Tools")
        assert result.startswith("# My Tools")

    def test_empty_list(self):
        assert format_tool_catalog([]) == ""

    def test_empty_registry(self):
        assert format_tool_catalog(ToolRegistry()) == ""

    def test_no_param_details_in_output(self):
        """Parameter schema should NOT appear in the catalog — it's
        already sent via the tools= API parameter."""
        defs = _make_defs()
        result = format_tool_catalog(defs)
        # No parameter formatting patterns (required marker, type parens, separator)
        assert "*(" not in result          # required param marker
        assert "(string)" not in result    # type annotation
        assert "(boolean)" not in result
        assert " — " not in result         # param separator
        assert "Person name" not in result # param-level description

    def test_preserves_full_description(self):
        """Descriptions are passed through as-is, not truncated."""
        long_desc = "This is a very long description that should be preserved in full"
        defs = [ToolDefinition(name="x", description=long_desc)]
        result = format_tool_catalog(defs)
        assert long_desc in result


# ── AgentPipeline auto-injection ─────────────────────────────────────────


class TestPipelineAutoInjection:
    def test_default_injection_on(self):
        p = _make_pipeline(registry=_make_registry())
        assert "# Available Tools" in p.system_prompt
        assert "- **calc**:" in p.system_prompt
        assert p.system_prompt.startswith("You are an assistant.")

    def test_injection_disabled(self):
        p = _make_pipeline(inject=False, registry=_make_registry())
        assert p.system_prompt == "You are an assistant."
        assert "# Available Tools" not in p.system_prompt

    def test_custom_header(self):
        p = _make_pipeline(header="# Tool Reference", registry=_make_registry())
        assert "# Tool Reference" in p.system_prompt
        assert "# Available Tools" not in p.system_prompt

    def test_no_registry_no_injection(self):
        p = _make_pipeline(registry=None)
        assert p.system_prompt == "You are an assistant."

    def test_empty_registry_no_injection(self):
        p = _make_pipeline(registry=ToolRegistry())
        assert p.system_prompt == "You are an assistant."

    def test_default_system_prompt_gets_catalog(self):
        # When system_prompt is None, DEFAULT_AGENT_SYSTEM_PROMPT is used
        cfg = AgentLoopConfig(
            system_prompt=None,
            inject_tool_descriptions=True,
        )
        p = AgentPipeline(
            llm=None, config=cfg, tool_registry=_make_registry(),
            hooks=AgentHooks(), bus=AgentEventBus(), ctx=ContextManager(),
        )
        assert "# Available Tools" in p.system_prompt
        assert "- **calc**:" in p.system_prompt

    def test_catalog_is_on_system_prompt_not_history(self):
        """Compaction safety: catalog lives on self.system_prompt (string attr),
        not in self.history. The compactor only modifies self.history."""
        p = _make_pipeline(registry=_make_registry())
        # system_prompt has the catalog
        assert "# Available Tools" in p.system_prompt
        # history is empty (no messages yet)
        assert p.history == []


# ── section_tool_catalog (builder integration) ───────────────────────────


class TestSectionToolCatalog:
    def test_with_definitions(self):
        ctx = SystemPromptContext(
            tool_definitions=_make_defs(),
            tool_catalog_header="# Builder Tools",
        )
        result = section_tool_catalog(ctx)
        assert result is not None
        assert "# Builder Tools" in result
        assert "- **calc**:" in result

    def test_without_definitions_returns_none(self):
        ctx = SystemPromptContext()
        assert section_tool_catalog(ctx) is None

    def test_builder_includes_section(self):
        ctx = SystemPromptContext(
            model="test",
            tool_definitions=_make_registry(),
            tool_catalog_header="# Registered",
        )
        prompt = (
            SystemPromptBuilder(use_defaults=False)
            .add("identity", lambda c: f"Model: {c.model}")
            .add("tool_catalog", section_tool_catalog)
            .build(ctx)
        )
        assert "Model: test" in prompt
        assert "# Registered" in prompt
        assert "- **calc**:" in prompt
