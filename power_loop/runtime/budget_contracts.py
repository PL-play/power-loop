"""Zero-dependency trusted-host ports for durable per-request budgets.

Business wallets, prices and payer authorization belong to the host. No model
argument can create a policy, scope or provider. Opaque payloads contain budget
metadata only; hosts must never put prompts, credentials or response bodies here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .jobs import JobError, _hash, _json


def identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 255 and value == value.strip() and value.isprintable()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def uint(value):
    return type(value) is int and 0 <= value < 2**63


def payload(value):
    """Bounded JSON copy with exact numbers, detached from mutable host objects."""

    def check(item, depth=0):
        if depth > 20:
            raise JobError("invalid_budget_payload")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is list:
            if len(item) > 256:
                raise JobError("invalid_budget_payload")
            for child in item:
                check(child, depth + 1)
            return
        if type(item) is dict:
            forbidden = {
                "api_key",
                "authorization",
                "password",
                "access_token",
                "secret_key",
                "prompt",
                "messages",
                "system_prompt",
                "response_body",
            }
            if len(item) > 128 or any(type(k) is not str or k.lower() in forbidden for k in item):
                raise JobError("invalid_budget_payload")
            for child in item.values():
                check(child, depth + 1)
            return
        raise JobError("invalid_budget_payload")

    if type(value) is not dict:
        raise JobError("invalid_budget_payload")
    check(value)
    raw = _json(value)
    if len(raw.encode()) > 65536:
        raise JobError("invalid_budget_payload")
    import json

    return json.loads(raw)


@dataclass(frozen=True)
class LocalBudgetScope:
    key: str
    max_calls: int | None = None
    max_tokens: int | None = None
    max_inflight: int | None = None

    def __post_init__(self):
        values = (self.max_calls, self.max_tokens, self.max_inflight)
        if (
            not identifier(self.key)
            or all(v is None for v in values)
            or any(v is not None and not uint(v) for v in values)
        ):
            raise JobError("invalid_local_budget_scope")


@dataclass(frozen=True)
class BudgetPlan:
    token_bound: int
    request: dict

    def __post_init__(self):
        if not uint(self.token_bound):
            raise JobError("invalid_budget_token_bound")
        object.__setattr__(self, "request", payload(self.request))

    @property
    def request_hash(self):
        return _hash(self.request)


@dataclass(frozen=True)
class BudgetGrant:
    attempt_id: str
    request_hash: str
    version: int
    state: str
    receipt: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            not identifier(self.attempt_id)
            or not digest(self.request_hash)
            or not uint(self.version)
            or self.version == 0
            or self.state not in ("reserved", "settling", "settled", "released")
        ):
            raise JobError("invalid_budget_grant")
        object.__setattr__(self, "receipt", payload(self.receipt))


@dataclass(frozen=True)
class BudgetSubmission:
    attempt_id: str
    revision: int
    request: dict

    def __post_init__(self):
        if not identifier(self.attempt_id) or not uint(self.revision) or self.revision == 0:
            raise JobError("invalid_budget_submission")
        object.__setattr__(self, "request", payload(self.request))

    @property
    def request_hash(self):
        return _hash(self.request)


@dataclass(frozen=True)
class BudgetAck:
    attempt_id: str
    revision: int
    request_hash: str
    receipt: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            not identifier(self.attempt_id)
            or not uint(self.revision)
            or self.revision == 0
            or not digest(self.request_hash)
        ):
            raise JobError("invalid_budget_ack")
        object.__setattr__(self, "receipt", payload(self.receipt))


class BudgetProvider(Protocol):
    async def prepare(self, context: dict, logical_request, *, service) -> BudgetPlan: ...
    async def reserve(self, plan: BudgetPlan) -> BudgetGrant: ...
    async def lookup(self, plan: BudgetPlan) -> BudgetGrant | None: ...
    async def prepare_usage(self, context: dict, grant: BudgetGrant, observation: dict) -> BudgetSubmission: ...
    async def apply_usage(self, submission: BudgetSubmission) -> BudgetAck: ...
    async def prepare_release(self, context: dict, grant: BudgetGrant | None, proof: dict) -> BudgetSubmission: ...
    async def release(self, submission: BudgetSubmission) -> BudgetAck: ...


@dataclass(frozen=True)
class BudgetConfig:
    provider_key: str
    scopes: tuple[LocalBudgetScope, ...]
    provider: BudgetProvider = field(compare=False, repr=False)
    timeout_s: float = 10
    max_pending: int = 100

    def __post_init__(self):
        ports = ("prepare", "reserve", "lookup", "prepare_usage", "apply_usage", "prepare_release", "release")
        if (
            not identifier(self.provider_key)
            or type(self.scopes) is not tuple
            or not 1 <= len(self.scopes) <= 64
            or any(type(s) is not LocalBudgetScope for s in self.scopes)
            or len({s.key for s in self.scopes}) != len(self.scopes)
            or any(not callable(getattr(self.provider, p, None)) for p in ports)
            or type(self.timeout_s) not in (int, float)
            or not math.isfinite(self.timeout_s)
            or not 0.1 <= self.timeout_s <= 60
            or type(self.max_pending) is not int
            or not 1 <= self.max_pending <= 10000
        ):
            raise JobError("invalid_budget_config")
        object.__setattr__(self, "scopes", tuple(sorted(self.scopes, key=lambda s: s.key)))

    def manifest(self):
        return {
            "version": 1,
            "provider_key": self.provider_key,
            "scopes": [asdict(s) for s in self.scopes],
            "max_pending": self.max_pending,
        }
