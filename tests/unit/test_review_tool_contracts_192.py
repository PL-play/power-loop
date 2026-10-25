"""Pure review views follow actual implementations, never catalog names."""
from copy import deepcopy
from functools import wraps

import pytest

from power_loop import ToolDefinition
from power_loop.runtime.tool_contracts import (
    FunctionRuntimeTool,
    ToolContractError,
    ToolDispatchPolicy,
    ToolRuntimeContract,
    policy,
    review_spec,
)
from power_loop.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('name', ['jobs', 'workflow', 'send_message', 'inspect_saved'])
def test_named_lookalike_reviewers_do_not_advertise_or_authorize_controls(name):
    def ordinary(**kw):
        return 'unused'
    ordinary.review_spec = lambda spec: spec
    ordinary.runtime_contract = ToolRuntimeContract('fake')
    registry = ToolRegistry()
    definition = ToolDefinition(name, '')
    registry.register(definition, ordinary)
    assert review_spec(registry.get(name), definition.to_openai_tool()) is None
    assert policy(registry.get(name), {}) == ToolDispatchPolicy()


def test_actual_aliased_wrapped_review_schema_is_a_copy_and_grants_no_execution():
    def narrow(spec):
        spec['function']['parameters']['required'] = ['original_id']
        return spec
    implementation = FunctionRuntimeTool(lambda **kw: 'unused', ToolRuntimeContract('example.view'), reviewer=narrow)
    @wraps(implementation)
    async def wrapper(**kw):
        return await implementation(**kw)
    registry = ToolRegistry()
    definition = ToolDefinition('read_retained', '', input_schema={'type':'object','properties':{'original_id':{'type':'string'}}})
    registry.register(definition, wrapper)
    spec = definition.to_openai_tool()
    before = deepcopy(spec)
    selected = registry.subset(['read_retained']).get('read_retained')
    view = review_spec(selected, spec)
    assert spec == before and view['function']['parameters']['required'] == ['original_id']
    assert policy(selected, {'original_id':'saved'}) == ToolDispatchPolicy()


@pytest.mark.parametrize('value', [True, {}, {'type':'function','function':{'name':'other','parameters':{}}},
    {'type':'function','function':{'name':'read_original','parameters':[]}}])
def test_invalid_implementation_views_fail_closed(value):
    registry = ToolRegistry()
    definition = ToolDefinition('read_original', '')
    registry.register(definition, FunctionRuntimeTool(lambda **kw: '', ToolRuntimeContract('example.bad-view'), reviewer=lambda spec:value))
    with pytest.raises(ToolContractError, match='invalid_tool_review_spec'):
        review_spec(registry.get('read_original'), definition.to_openai_tool())
