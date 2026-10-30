from dataclasses import replace

import pytest

from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free, PricingError
from power_loop.tools.registry import ToolRegistry, validate_pricing


def test_missing_declaration_fails_before_registration_or_overwrite():
    registry = ToolRegistry()
    with pytest.raises(PricingError, match='tool_pricing_required'):
        registry.register(ToolDefinition('missing', ''), lambda: 'bad')
    assert registry.names() == []
    original = ToolDefinition('local', '', pricing=Free('local'))
    registry.register(original, lambda: 'original')
    with pytest.raises(PricingError):
        registry.register(replace(original, pricing=None), lambda: 'changed', overwrite=True)
    assert registry.invoke('local', {}) == 'original'


@pytest.mark.asyncio
async def test_registered_rule_replacement_is_rejected_before_handler_runs():
    registry = ToolRegistry()
    definition = ToolDefinition('local', '', pricing=Free('local'))
    seen = []
    registry.register(definition, lambda: seen.append(True))
    object.__setattr__(definition, 'pricing', Free('different'))
    with pytest.raises(PricingError, match='binding_changed'):
        await registry.invoke_async('local', {})
    assert seen == []


def test_subset_preserves_original_price_snapshot_and_schema_hides_it():
    registry = ToolRegistry()
    definition = ToolDefinition('local', '', pricing=Free('local'))
    registry.register(definition, lambda: 'done')
    selected = registry.subset(['local'])
    assert selected.get('local') is registry.get('local')
    assert validate_pricing(selected.get('local')) is definition.pricing
    assert 'pricing' not in str(selected.to_openai_tools())


def test_library_default_catalog_has_explicit_price_declarations():
    from power_loop.tools.default_manifest import DEFAULT_TOOL_DEFINITIONS_MAP
    assert DEFAULT_TOOL_DEFINITIONS_MAP
    for definition in DEFAULT_TOOL_DEFINITIONS_MAP.values():
        assert isinstance(definition.pricing, Free), definition.name


@pytest.mark.asyncio
async def test_mcp_requires_complete_explicit_prices_before_registering_any_tool():
    from types import MappingProxyType

    from power_loop.contrib.mcp import McpToolSpec, register_mcp_tools

    class Source:
        async def list_tools(self):
            return [McpToolSpec('first'), McpToolSpec('second')]

        async def call_tool(self, name, arguments):
            return name

    registry = ToolRegistry()
    with pytest.raises(PricingError, match='required:second'):
        await register_mcp_tools(registry, Source(), pricing={'first': Free('self_hosted')})
    assert registry.names() == []
    prices = MappingProxyType({'first': Free('self_hosted'), 'second': Free('self_hosted')})
    assert await register_mcp_tools(registry, Source(), pricing=prices) == ['first', 'second']
    assert await registry.invoke_async('second', {}) == 'second'


def test_pending_call_identity_includes_frozen_price_version():
    from power_loop.runtime.tool_call_plans import registration_value
    first, second = ToolRegistry(), ToolRegistry()
    first.register(ToolDefinition('same', '', pricing=Free('local')), lambda: 'a')
    second.register(ToolDefinition('same', '', pricing=Free('different_rule')), lambda: 'a')
    assert first.to_openai_tools() == second.to_openai_tools()
    assert registration_value(first.get('same')) != registration_value(second.get('same'))
