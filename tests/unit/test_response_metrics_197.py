"""The default, mapped and host-code paths share one standard output contract."""
import json
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from power_loop import LLMProviderConfig, LLMRequest, create_llm_service_from_config
from power_loop.llm_metrics import ResponseExtractionError, ResponseMetrics, enforce_extraction_policy, read_path


@pytest.mark.parametrize('raw,expected', [
    ({'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}, (10,2,12)),
    ({'x_usage': {'input_tokens': 10, 'output_tokens': 2}}, (10,2,12)),
    ({'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 2, 'totalTokenCount': 12}}, (10,2,12)),
    ({'usage': {'input_tokens': 10, 'output_tokens': 2, 'cache_read_input_tokens': 5}}, (15,2,17)),
    ({'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}, (0,0,0)),
])
def test_default_interface_recognizes_common_supplier_usage(raw,expected):
    value=ResponseMetrics().inspect(raw)
    assert not value['errors']
    assert tuple(value['usage'][key] for key in ('prompt_tokens','completion_tokens','total_tokens'))==expected


@pytest.mark.parametrize('path', ['a.b[0].c','missing.foo','a.b[9].c','a.b.c','a.b[0].c.x'])
def test_paths_with_nested_arrays_and_wrong_containers_never_throw(path):
    value=read_path({'a':{'b':[{'c':'0'}]}},path)
    assert value==('0' if path=='a.b[0].c' else None)


@pytest.mark.parametrize('cost', ['0','0.123456789012345678901234567891','0.000000000000000000000000000001'])
def test_mapped_cost_and_nested_usage_keep_all_decimal_places(cost):
    config={'usage':{'paths':{'prompt_tokens':'data.tokens.in','completion_tokens':'data.tokens.out',
        'prompt_cached_tokens':'data.tokens.cached'}},'cost':{'amount_path':'data.bill.amount','currency_path':'data.bill.currency'}}
    value=ResponseMetrics(config).inspect({'data':{'tokens':{'in':100,'out':2,'cached':50},'bill':{'amount':cost,'currency':'USD'}}})
    assert value['usage']['prompt_tokens']==100 and value['usage']['total_tokens']==102
    assert value['usage']['prompt_cached_tokens']==50
    assert value['cost']=={'amount':cost,'currency':'USD'} and value['errors']==[]


def test_missing_amount_is_not_a_reported_zero_or_a_mapping_error():
    parser=ResponseMetrics({'cost':{'amount_path':'missing.cost'}})
    assert parser.inspect({'data':{}})['cost'] is None
    assert parser.inspect({'data':{}})['errors']==[]
    assert ResponseMetrics().inspect({'usage':{'cost':'0'}})['cost']['amount']=='0'


@pytest.mark.parametrize('value', [False, 'NaN', 'Infinity', '-1', {}, []])
def test_invalid_supplier_money_is_a_structured_error_not_a_field_lookup_crash(value):
    result=ResponseMetrics().inspect({'usage':{'cost':value}})
    assert result['cost'] is None and result['errors'][0]['on_error']=='raise'


def test_invalid_error_policy_is_rejected_before_transport_configuration():
    with pytest.raises(ResponseExtractionError):
        ResponseMetrics({'cost':{'on_error':'typo'}})


@pytest.mark.parametrize('policy', ['raise','continue'])
def test_invalid_value_is_captured_before_configured_policy(policy):
    parser=ResponseMetrics({'cost':{'amount_path':'fee','on_error':policy}})
    value=parser.inspect({'fee':'not-money','usage':{'prompt_tokens':3,'completion_tokens':1}})
    assert value['cost'] is None and value['usage']['total_tokens']==4
    assert value['source']['fee']=='not-money' and value['errors'][0]['on_error']==policy
    response=SimpleNamespace(debug={'response_extraction':value})
    if policy=='raise':
        with pytest.raises(ResponseExtractionError):
            enforce_extraction_policy(response)
    else:
        enforce_extraction_policy(response)


def test_host_code_extension_returns_the_same_usage_and_cost_types():
    parser=ResponseMetrics({'usage':{'code':"def extract(response):\n return {'prompt_tokens': response['a'] + response['b'], 'completion_tokens': 2}"},
        'cost':{'code':"def extract(response):\n return {'amount': str(Decimal(response['fee']) / Decimal('2')), 'currency': 'USD'}"}})
    value=parser.inspect({'a':4,'b':5,'fee':'0.000000000000000000000000000002'})
    assert value['usage']['total_tokens']==11
    assert Decimal(value['cost']['amount'])==Decimal('1e-30')
    assert not value['errors']


@pytest.mark.asyncio
@pytest.mark.parametrize('numeric_literal', [False, True])
async def test_actual_openai_transport_applies_custom_mapping_without_extra_requests(numeric_literal):
    from power_loop._vendor.llm_client.llm_factory import ExactAsyncOpenAI
    calls=[]
    def transport(request):
        calls.append(request)
        raw={'id':'response1','object':'chat.completion.chunk','created':1,'model':'test',
            'choices':[{'index':0,'delta':{'content':'OK'},'finish_reason':None}],
            'metrics':{'input':20,'output':3,'paid':'0.123456789012345678901234567891'}}
        wire = json.dumps(raw)
        if numeric_literal:
            wire = wire.replace('"0.123456789012345678901234567891"', '0.123456789012345678901234567891')
        return httpx.Response(200,headers={'content-type':'text/event-stream'},
            content='data: '+wire+'\n\ndata: [DONE]\n\n')
    cfg=LLMProviderConfig(base_url='https://test.invalid/v1',api_key='test',model='test',max_retries=0,
        response_extraction={'usage':{'paths':{'prompt_tokens':'metrics.input','completion_tokens':'metrics.output'}},
        'cost':{'amount_path':'metrics.paid','currency':'USD'}})
    service=create_llm_service_from_config(cfg)
    service._client=ExactAsyncOpenAI(base_url=cfg.base_url,api_key='test',max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
    try:
        result=await service.complete(LLMRequest(messages=[{'role':'user','content':'ok'}]))
        assert result.raw_text=='OK' and result.token_usage.total_tokens==23
        assert result.provider_cost.amount=='0.123456789012345678901234567891'
        assert result.provider_cost.currency=='USD' and len(calls)==1
        assert result.debug['response_extraction']['sources']['cost']['metrics']['paid']==result.provider_cost.amount
    finally:
        await service.close()
