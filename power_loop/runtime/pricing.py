"""Host-supplied pricing contracts, independent of providers and persistence.

Values are frozen snapshots: a receipt or rule cannot change after settlement.
Amounts use Decimal throughout; the host chooses currencies and conversion rules.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable


class PricingError(ValueError):
    pass


def _plain(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PricingError('pricing_json_key_invalid')
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def canonical(value: Any) -> str:
    try:
        return json.dumps(_plain(value), sort_keys=True, ensure_ascii=False,
                          separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PricingError('pricing_json_invalid') from exc


def rule_version(value: Mapping[str, Any]) -> str:
    """Hash all executable rule parameters, including prices and effective time."""
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def pricing_snapshot(value, *, max_bytes=None):
    if not isinstance(value, Mapping):
        raise PricingError('pricing_mapping_required')
    encoded = canonical(value)
    if max_bytes is not None and len(encoded.encode('utf-8')) > max_bytes:
        raise PricingError('billing_receipt_raw_too_large')
    return _freeze(json.loads(encoded))


@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str

    def __post_init__(self):
        if not isinstance(self.amount, Decimal) or not self.amount.is_finite() or self.amount < 0:
            raise PricingError('pricing_amount_invalid')
        if not isinstance(self.currency, str) or re.fullmatch('[A-Z]{3}', self.currency) is None:
            raise PricingError('pricing_currency_invalid')


@dataclass(frozen=True)
class PricedCall:
    name: str
    arguments: Mapping[str, Any]
    role: str
    model_binding: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or not self.role:
            raise PricingError('priced_call_identity_missing')
        object.__setattr__(self, 'arguments', pricing_snapshot(self.arguments))
        object.__setattr__(self, 'model_binding', pricing_snapshot(self.model_binding))


@dataclass(frozen=True)
class Estimate:
    maximum: Money | None
    bounds: Mapping[str, Any]

    def __post_init__(self):
        if self.maximum is not None and not isinstance(self.maximum, Money):
            raise PricingError('pricing_maximum_invalid')
        object.__setattr__(self, 'bounds', pricing_snapshot(self.bounds))


@dataclass(frozen=True)
class BillingReceipt:
    units: Mapping[str, Any] = field(default_factory=dict)
    provider_amount: Money | None = None
    provider_request_id: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.provider_amount is not None and not isinstance(self.provider_amount, Money):
            raise PricingError('billing_provider_amount_invalid')
        if self.provider_request_id is not None and (
            not isinstance(self.provider_request_id, str) or not self.provider_request_id
        ):
            raise PricingError('billing_provider_request_id_invalid')
        object.__setattr__(self, 'units', pricing_snapshot(self.units))
        object.__setattr__(self, 'raw', pricing_snapshot(self.raw, max_bytes=8192))

    def to_dict(self) -> dict:
        return _plain(dict(units=self.units, provider_amount=(
            {'amount': str(self.provider_amount.amount), 'currency': self.provider_amount.currency}
            if self.provider_amount is not None else None),
            provider_request_id=self.provider_request_id, raw=self.raw))

    @classmethod
    def from_dict(cls, value) -> BillingReceipt:
        if not isinstance(value, dict) or set(value) != {
            'units', 'provider_amount', 'provider_request_id', 'raw'
        }:
            raise PricingError('billing_receipt_invalid')
        amount = value['provider_amount']
        if amount is not None:
            if (not isinstance(amount, dict) or set(amount) != {'amount', 'currency'}
                    or not isinstance(amount['amount'], str)):
                raise PricingError('billing_receipt_amount_invalid')
            try:
                amount = Money(Decimal(amount['amount']), amount['currency'])
            except ArithmeticError as exc:
                raise PricingError('billing_receipt_amount_invalid') from exc
        return cls(value['units'], amount, value['provider_request_id'], value['raw'])


@dataclass(frozen=True)
class Charge:
    amount: Money
    quality: Literal['exact', 'provider_reported', 'rule_estimated', 'free']
    units: Mapping[str, Any]
    rule_id: str
    rule_version: str

    def __post_init__(self):
        if not isinstance(self.amount, Money):
            raise PricingError('billing_charge_amount_invalid')
        if self.quality not in ('exact', 'provider_reported', 'rule_estimated', 'free'):
            raise PricingError('billing_quality_invalid')
        if self.quality == 'free' and self.amount.amount != 0:
            raise PricingError('billing_free_amount_nonzero')
        if not self.rule_id or not self.rule_version:
            raise PricingError('billing_rule_identity_missing')
        object.__setattr__(self, 'units', pricing_snapshot(self.units))


@runtime_checkable
class PricingSpec(Protocol):
    rule_id: str

    def version(self) -> str: ...

    def estimate(self, call: PricedCall) -> Estimate: ...

    def settle(self, call: PricedCall, receipt: BillingReceipt | None,
               error: BaseException | None) -> Charge: ...


@dataclass(frozen=True)
class Free:
    """Explicit no-charge declaration for local/control/composite operations.

    Kept in the library so its built-in tools need no host package dependency.
    A paid receipt contradicts this declaration and must never be silently discarded.
    """
    reason: str
    currency: str = 'CNY'
    rule_id: str = field(default='free', init=False)

    def __post_init__(self):
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise PricingError('pricing_free_reason_missing')
        Money(Decimal(0), self.currency)

    def version(self) -> str:
        return rule_version(dict(preset='free', reason=self.reason, currency=self.currency))

    def estimate(self, call: PricedCall) -> Estimate:
        return Estimate(Money(Decimal(0), self.currency), {'calls': 1})

    def settle(self, call: PricedCall, receipt: BillingReceipt | None,
               error: BaseException | None) -> Charge:
        if receipt is not None and receipt.provider_amount is not None:
            if receipt.provider_amount.amount != 0:
                raise PricingError('pricing_free_received_paid_receipt')
        return Charge(Money(Decimal(0), self.currency), 'free',
                      receipt.units if receipt else {}, self.rule_id, self.version())
