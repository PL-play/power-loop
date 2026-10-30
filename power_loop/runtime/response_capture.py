"""Pre-observation response capture by a trusted host, never a model tool.

The host durably stores the packet before returning its receipt. A receipt is
neither a usage ACK nor an execution exit. No local transaction can make a
remote response and its first durable copy atomic.
"""
from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

from .budget_contracts import BudgetGrant, digest, identifier, payload, uint
from .jobs import JobError, _hash

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelResponseCapture:
    context: dict
    grant: BudgetGrant
    started_at_ms: int
    received_at_ms: int
    usage: dict
    cost_amount: str | None = None
    response_id: str | None = None
    cost_currency: str | None = None
    response_metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        from .model_attempts import _TOKENS, _cost, _response_id
        context = payload(self.context)
        if (type(self.grant) is not BudgetGrant
                or context.get('attempt_id') != self.grant.attempt_id
                or not identifier(context.get('store_id'))
                or self.grant.state != 'reserved'
                or not uint(self.started_at_ms) or not uint(self.received_at_ms)
                or type(self.usage) is not dict or set(self.usage) - set(_TOKENS)
                or any(not uint(v) for v in self.usage.values())
                or not _cost(self.cost_amount) or not _response_id(self.response_id)):
            raise JobError('invalid_model_response_capture')
        object.__setattr__(self, 'context', context)
        object.__setattr__(self, 'grant', BudgetGrant(**asdict(self.grant)))
        object.__setattr__(self, 'usage', payload(self.usage))
        object.__setattr__(self, 'response_metrics', payload(self.response_metrics))

    @property
    def capture_id(self):
        return 'response_' + _hash([self.context['store_id'], self.grant.attempt_id])

    @property
    def fingerprint(self):
        return _hash(asdict(self))


@dataclass(frozen=True)
class ResponseCaptureReceipt:
    capture_id: str
    fingerprint: str
    record_id: str

    def __post_init__(self):
        if not identifier(self.capture_id) or not digest(self.fingerprint) or not identifier(self.record_id):
            raise JobError('invalid_response_capture_receipt')


@runtime_checkable
class ResponseCaptureProvider(Protocol):
    """Optional capability on the original trusted budget provider.

    Persist atomically and idempotently by capture_id; reject a different
    fingerprint at that identity. Do not apply usage or infer a currency/final
    invoice. Copies contain no prompt, response body or credentials.
    """
    async def capture_response(self, capture: ModelResponseCapture) -> ResponseCaptureReceipt: ...


@dataclass(frozen=True)
class CaptureBinding:
    context: dict
    grant: BudgetGrant
    started_at_ms: int
    callback: object = field(repr=False, compare=False)


async def bind(attempt):
    """Freeze authority before dispatch, with no post-response SDK DB read."""
    from . import model_budgets as budgets
    runtime, manager = attempt.binding.runtime, attempt.binding.runtime.manager
    config = runtime._original_budget_config
    if config is None:
        return None
    callback = getattr(config.provider, 'capture_response', None)
    if callback is None:
        return None
    if not callable(callback):
        raise JobError('invalid_response_capture_provider')
    async with manager._write() as tx:
        await budgets._policy(runtime, tx)
        job = await manager._fence(tx, attempt.binding.actor, attempt.binding.lease)
        state = await budgets._read(manager, tx, attempt.attempt_id)
        grant = BudgetGrant(**state['grant'])
        if (state['phase'] != 'host_reserved' or state['context'] != budgets._context(attempt, job)
                or grant.request_hash != state['plan_hash'] or grant.state != 'reserved'):
            raise JobError('response_capture_authority_mismatch')
        return CaptureBinding(payload(state['context']), grant, state['started_at_ms'], callback)


async def record(attempt, *, reported, observation, estimated):
    """Capture first, then save the normal observation, retaining actual tasks.

    Capture failure cannot repeat a model call or discard a usable observation.
    Caller cancellation leaves the original capture/observation task reachable
    through TaskRuntime's normal shutdown lifecycle and budget_control_tasks.
    """
    from .model_attempts import observe
    from .model_budgets import _port
    binding = attempt.capture_binding
    if binding is None:
        return await observe(attempt, **observation, estimated=estimated)
    # Detach before the first await. No callback may mutate the normal record.
    observation = copy.deepcopy(observation)
    captured = None
    try:
        captured = ModelResponseCapture(binding.context, binding.grant, binding.started_at_ms,
            attempt.binding.runtime.manager.now(), **reported)
    except Exception as exc:
        log.warning('Response capture validation failed: %s', type(exc).__name__)

    async def persist():
        try:
            if captured is not None:
                expected = (captured.capture_id, captured.fingerprint)

                async def capture():
                    # Receipt validation stays inside the retained task, even
                    # when the original waiter times out or is cancelled.
                    receipt = await binding.callback(captured)
                    if (type(receipt) is not ResponseCaptureReceipt
                            or (receipt.capture_id, receipt.fingerprint) != expected
                            or (captured.capture_id, captured.fingerprint) != expected):
                        raise JobError('response_capture_receipt_mismatch')
                    return ResponseCaptureReceipt(**asdict(receipt))

                await _port(attempt, 'capture_response', capture)
        except (Exception, asyncio.CancelledError) as exc:
            # A callback cancelling itself is not a request to retry the model.
            # The ordinary observation remains independently necessary.
            log.warning('Response capture unconfirmed: %s', type(exc).__name__)
        finally:
            await observe(attempt, **observation, estimated=estimated)

    try:
        await _port(attempt, 'response_observation', persist)
    except Exception as exc:
        attempt.binding.resource_evidence['resource_state'] = 'unknown'
        log.warning('Response observation pending: %s', type(exc).__name__)
