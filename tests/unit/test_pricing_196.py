from decimal import Decimal

import pytest

from power_loop.runtime.pricing import (
    BillingReceipt,
    Charge,
    Free,
    Money,
    PricedCall,
    PricingError,
    PricingSpec,
    canonical,
    rule_version,
)


@pytest.mark.parametrize('amount', [1.1, '1.1', True, Decimal('NaN'),
                                   Decimal('Infinity'), Decimal('-0.01')])
def test_money_rejects_float_nonfinite_and_negative(amount):
    with pytest.raises(PricingError):
        Money(amount, 'CNY')


@pytest.mark.parametrize('currency', ['cny', 'CN', 'CNYY', '', None])
def test_money_requires_currency_code(currency):
    with pytest.raises(PricingError):
        Money(Decimal('0'), currency)


def test_rule_hash_is_order_independent_and_price_sensitive():
    assert rule_version({'price': '0.01', 'unit': 'image'}) == rule_version(
        {'unit': 'image', 'price': '0.01'})
    assert rule_version({'price': '0.01'}) != rule_version({'price': '0.02'})


def test_receipt_is_a_deep_snapshot_and_enforces_utf8_byte_limit():
    source = {'nested': [{'cost': '0.1'}]}
    receipt = BillingReceipt(raw=source)
    source['nested'][0]['cost'] = '999'
    assert receipt.raw['nested'][0]['cost'] == '0.1'
    with pytest.raises(TypeError):
        receipt.raw['nested'][0]['cost'] = '999'
    with pytest.raises(PricingError, match='too_large'):
        BillingReceipt(raw={'text': '字' * 3000})


def test_non_json_usage_is_rejected_instead_of_lossily_serialized():
    with pytest.raises(PricingError):
        BillingReceipt(units={'images': float('nan')})
    with pytest.raises(PricingError):
        BillingReceipt(raw={1: 'integer key'})


def test_free_success_and_failure_have_stable_explicit_rule():
    pricing = Free('self_hosted')
    call = PricedCall('tts', {'text': 'hello'}, 'main')
    assert isinstance(pricing, PricingSpec)
    assert pricing.estimate(call).maximum == Money(Decimal('0'), 'CNY')
    success = pricing.settle(call, BillingReceipt(units={'calls': 1}), None)
    failure = pricing.settle(call, None, RuntimeError('failed'))
    assert success.quality == failure.quality == 'free'
    assert success.rule_version == failure.rule_version == pricing.version()
    assert canonical(success.units) == '{"calls":1}'


def test_paid_receipt_cannot_be_discarded_by_free_declaration():
    with pytest.raises(PricingError, match='paid_receipt'):
        Free('local').settle(PricedCall('tool', {}, 'main'),
                             BillingReceipt(provider_amount=Money(Decimal('1'), 'USD')), None)


def test_free_charge_cannot_hide_nonzero_amount():
    with pytest.raises(PricingError):
        Charge(Money(Decimal('1'), 'CNY'), 'free', {}, 'rule', 'version')


def test_receipt_roundtrip_retains_decimal_precision_and_detaches_json():
    receipt = BillingReceipt(provider_amount=Money(Decimal('12345678901234567890.123456789012'), 'CNY'),
                             provider_request_id='provider-123', units={'images': 2}, raw={'usage': [1]})
    encoded = receipt.to_dict()
    assert BillingReceipt.from_dict(encoded) == receipt
    encoded['raw']['usage'].append(2)
    assert receipt.raw['usage'] == (1,)


@pytest.mark.parametrize('amount', [0.1, None, 'NaN', '-1', 'not-money'])
def test_receipt_decode_rejects_invalid_amount(amount):
    encoded = BillingReceipt().to_dict()
    encoded['provider_amount'] = {'amount': amount, 'currency': 'USD'}
    with pytest.raises(PricingError):
        BillingReceipt.from_dict(encoded)
