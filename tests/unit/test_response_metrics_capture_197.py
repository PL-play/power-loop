"""Parsing errors preserve original evidence and never buy another model call."""
from power_loop import LLMResponse, LLMTokenUsage
from power_loop.llm_metrics import ProviderCost
from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.response_capture import ResponseCaptureReceipt, record

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case, start

__all__ = ['registry_store','budget_case']


async def test_cost_currency_and_invalid_source_are_saved_before_failed_extraction(budget_case):
    c=budget_case
    saved=[]
    async def capture(packet):
        saved.append(packet)
        return ResponseCaptureReceipt(packet.capture_id, packet.fingerprint, 'durable-source')
    c.host.capture_response=capture
    attempt=await start(c)
    response=LLMResponse(raw_text='original answer',token_usage=LLMTokenUsage(prompt_tokens=10,completion_tokens=2,total_tokens=12))
    response.debug['response_extraction']={'errors':[{'interface':'cost','on_error':'raise','message':'invalid number'}],
        'sources':{'cost':{'usage':{'cost':'invalid-number'}}}}
    facts=attempts.response_observation(response.token_usage,response)
    await record(attempt,reported=facts,observation=facts,estimated=False)
    assert saved[0].response_metrics['sources']['cost']['usage']['cost']=='invalid-number'
    state=await budgets.get_budget(c.manager,c.actor,attempt.attempt_id)
    assert state['observation']['error_type']=='response_cost_extraction_failed'
    assert state['observation']['status']=='returned' and c.host.apply_calls==1


async def test_exact_supplier_cost_has_its_currency_in_original_observation(budget_case):
    c=budget_case
    attempt=await start(c)
    response=LLMResponse(raw_text='ok',token_usage=LLMTokenUsage(prompt_tokens=0,completion_tokens=0,total_tokens=0),
        provider_cost=ProviderCost('0.000000000000000000000000000001','USD'))
    facts=attempts.response_observation(response.token_usage,response)
    await record(attempt,reported=facts,observation=facts,estimated=False)
    state=await budgets.get_budget(c.manager,c.actor,attempt.attempt_id)
    assert state['observation']['provider_cost']=='0.000000000000000000000000000001'
    assert state['observation']['provider_cost_currency']=='USD'
