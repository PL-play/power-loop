"""Async tool-context preparation with durable derived results and attempt accounting.

Rendering remains synchronous. The session runner owns serialization; no transaction spans
an LLM request. Applications may supply a single-attempt LLM transport and an idempotent
usage receiver. Pending deliveries survive restarts in the session runtime state.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from power_loop.contracts.context_policy import (
    LEGACY_FIELDS,
    ToolContextPolicy,
    Trigger,
    contains_preserve_marker,
)

#: Runtime-state row name. The stored name is kept as-is (renaming it would drop live state).
TOOL_CONTEXT_STATE_KEY = 'tool_context.v1'
#: Fields of the per-send capacity snapshot; an older snapshot's extra keys are ignored.
CAPACITY_FIELDS = ('capacity_enabled', 'trigger_tokens', 'target_tokens', 'window_tokens',
                   'reserve_tokens', 'margin_tokens', 'hot_tail', 'merge_consecutive')
active_engine: ContextVar[ToolContextEngine | None] = ContextVar('tool_context_engine', default=None)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass
class ToolContextConfig:
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Include provider/endpoint/config revision, not secrets. Changing it invalidates caches.
    model_identity: str = ''
    rule_version: str = '1'
    summary_budget_s: float = 5.0
    delivery_budget_s: float = 2.0
    cache_scope: str | Callable[[], str] | None = None
    cache_ttl_s: float = 604800
    cache_max_entries: int = 10000
    cache_max_bytes: int = 67108864
    usage_metadata: Callable[[], dict[str, Any]] | None = None
    summary_min_chars: int = 8000
    summary_llm: Any = None
    # None selects provider-aware non-thinking defaults; {} explicitly inherits transport.
    summary_request_extra: dict[str, Any] | None = None
    rule: Callable[[str, Mapping[str, Any], str, Any], Any] | None = None
    deliver_usage: Callable[[dict[str, Any]], Awaitable[None]] | None = None

    # In-send fuse: at trigger_tokens, distill older calls down to target_tokens. Level 1 uses each
    # tool's insend_fuse rule and protects the newest hot_tail calls; level 2 falls back to general.
    capacity_enabled: bool = True
    trigger_tokens: int = 500000
    target_tokens: int = 350000
    window_tokens: int = 1000000
    reserve_tokens: int = 40000
    margin_tokens: int = 32000
    hot_tail: int = 8
    # Between sends, consecutive calls of the same tool become one group entry (one recall id).
    merge_consecutive: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.hot_tail, bool) or not isinstance(self.hot_tail, int) or self.hot_tail < 0:
            raise ValueError('hot_tail must be a nonnegative integer')
        for key in ('capacity_enabled', 'merge_consecutive'):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f'{key} must be boolean')
        for key in ('trigger_tokens', 'target_tokens', 'window_tokens', 'reserve_tokens', 'margin_tokens'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not 0 < self.target_tokens < self.trigger_tokens <= self.window_tokens - self.reserve_tokens - self.margin_tokens:
            raise ValueError('require target < trigger <= window - reserve - margin')
        if isinstance(self.summary_min_chars, bool) or not isinstance(self.summary_min_chars, int) or self.summary_min_chars < 0:
            raise ValueError("summary_min_chars must be a nonnegative integer")
        if self.summary_budget_s < 0 or self.delivery_budget_s <= 0:
            raise ValueError("summary budget must be nonnegative and delivery budget positive")
        if self.cache_ttl_s <= 0 or self.cache_max_entries < 1 or self.cache_max_bytes < 1:
            raise ValueError("cache limits must be positive")
        for patch in self.overrides.values():
            ToolContextPolicy().override(patch)


def recall_note(recall: str) -> str:
    """The one line appended after distilled content. No prefix before the content: models have
    copied bookkeeping prefixes (e.g. "[distilled #…]", "archived") into what they said."""
    return f'（原文 {recall}）'


@dataclass(frozen=True)
class ToolSource:
    session_id: str
    send_index: int | None
    seq: int | None
    call_id: str
    name: str
    arguments: Mapping[str, Any]
    result: str

    @property
    def recall(self) -> str:
        return f'recall_send(send_index={self.send_index}, seq={self.seq})'


class ToolContextEngine:
    def __init__(self, *, config: ToolContextConfig, store: Any, owner_session_id: str,
                 owner_send_index: int | None, llm: Any, model: str | None,
                 registry: Any = None, rule: Any = None, lineage: dict[str, Any] | None = None):
        transport = config.summary_llm or llm
        transport_config = getattr(transport, '_cfg', None)
        summary_model = model or getattr(transport_config, 'model', None) or ''
        self.summary_request_extra = (dict(config.summary_request_extra)
            if config.summary_request_extra is not None else
            {'extra_body': {'thinking': {'type': 'disabled'}}}
            if 'deepseek' in summary_model.lower() else {})
        identity = _hash({'declared': config.model_identity,
                          'summary_request_extra': self.summary_request_extra,
                          'summary_min_chars': config.summary_min_chars,
                          'transport': type(transport).__module__ + '.' + type(transport).__qualname__,
                          'endpoint': getattr(transport_config, 'base_url', None),
                          'model': model or getattr(transport_config, 'model', None)})
        self.started_ms = int(time.time() * 1000)
        self.config, self.store = config, store
        self.model_identity = identity
        from power_loop.runtime.store.context_cache import ContextCache
        self.cache = ContextCache(store)
        scope = config.cache_scope() if callable(config.cache_scope) else config.cache_scope
        self.cache_scope = scope or ('session:' + owner_session_id)
        self.executor_key = _hash({'scope': self.cache_scope, 'model': identity})
        self.worker = None
        self.sid, self.send_index = owner_session_id, owner_send_index
        self.llm, self.registry = llm, registry
        self.model = model or getattr(getattr(config.summary_llm or llm, "_cfg", None), "model", None)
        self.rule = config.rule or rule
        self.lineage = dict(lineage or {})
        self.usage: dict[str, int] = {}
        self.state: dict[str, Any] = {'operations': {}, 'attempts': {}}

    async def start(self) -> None:
        self.state = await self.store.get_runtime_state(self.sid, TOOL_CONTEXT_STATE_KEY, self.state)
        from power_loop.runtime.tool_archive import adopt_legacy_state
        adopt_legacy_state(self.state)
        snapshots = self.state.setdefault('send_policies', {})
        send_key = str(self.send_index)
        effective = {}
        for definition in self.registry.definitions() if self.registry else []:
            effective[definition.name] = asdict(definition.context_policy or ToolContextPolicy())
        for name, patch in self.config.overrides.items():
            effective[name] = {**effective.get(name, {}), **patch}
        effective = {name: {k: v for k, v in patch.items() if k not in LEGACY_FIELDS}
                     for name, patch in effective.items()}
        snapshot = snapshots.get(send_key)
        if snapshot is None:
            snapshot = {'capacity': {k: getattr(self.config, k) for k in CAPACITY_FIELDS}, 'overrides': effective, 'rule_version': self.config.rule_version,
                        'model_identity': self.model_identity, 'model': self.model}
            snapshots[send_key] = snapshot
            await self._save()
        else:
            # A resumed send keeps the rules it started with. A snapshot from an earlier release
            # may lack capacity fields (or carry retired ones): missing fields use current config.
            effective = {name: {k: v for k, v in patch.items() if k not in LEGACY_FIELDS}
                         for name, patch in snapshot['overrides'].items()}
        self._implementation_matches = (snapshot['rule_version'] == self.config.rule_version
            and snapshot['model_identity'] == self.model_identity and snapshot['model'] == self.model)
        capacity = {k: v for k, v in (snapshot.get('capacity') or {}).items() if k in CAPACITY_FIELDS}
        self.config = replace(self.config, overrides=effective, **capacity)
        # Session ownership excludes another live runner. A started row here belongs to
        # an interrupted invocation; provider billing is unknowable without reconciliation.
        changed = False
        for attempt in self.state['attempts'].values():
            if attempt['outcome'] == 'started':
                attempt.update(outcome='interrupted', success=False, usage_unknown=True, finished_at=time.time())
                changed = True
        if changed:
            await self._save()
        for attempt in self.state['attempts'].values():
            await self._record_stats(attempt)

    def projection_version(self, original: int) -> int:
        return int(_hash({'original': original, 'merge': self.config.merge_consecutive, 'overrides': self.config.overrides,
                          'rule': self.config.rule_version, 'model': self.model,
                          'model_identity': self.model_identity})[:7], 16)

    async def _save(self) -> None:
        await self.store.set_runtime_state(self.sid, TOOL_CONTEXT_STATE_KEY, self.state)

    async def flush_usage(self) -> None:
        receiver = self.config.deliver_usage
        if receiver is None:
            return
        if not hasattr(self, '_delivery_deadline'):
            self._delivery_deadline = time.monotonic() + self.config.delivery_budget_s
        for attempt in self.state['attempts'].values():
            remaining = self._delivery_deadline - time.monotonic()
            if remaining <= 0:
                break
            if attempt.get('delivered') or attempt['outcome'] == 'started':
                continue
            try:
                await asyncio.wait_for(receiver(dict(attempt)), remaining)  # deduplicate on attempt_id
            except Exception:
                continue  # durable outbox, retried on next preparation/recovery
            attempt['delivered'] = True
            await self._save()

    def policy(self, source: ToolSource) -> ToolContextPolicy:
        registered = self.registry.get(source.name) if self.registry else None
        policy = getattr(getattr(registered, 'definition', None), 'context_policy', None)
        return (policy or ToolContextPolicy()).override(self.config.overrides.get(source.name, {}))

    def _rule(self, source: ToolSource) -> str | None:
        registered = self.registry.get(source.name) if self.registry else None
        project = getattr(getattr(registered, 'definition', None), 'project', None)
        try:
            if self.config.rule is not None:
                value = self.config.rule(source.name, source.arguments, source.result, self.registry)
            elif project is not None:
                value = project(source.arguments, source.result)
            elif self.rule is not None:
                value = self.rule(source.name, source.arguments, source.result, self.registry)
            else:
                return None  # no lossy generic character cutoff
            return value if isinstance(value, str) else _json(value) if value is not None else None
        except Exception:
            return None

    async def distill(self, source: ToolSource, trigger: Trigger, *, policy_override=None) -> str | None:
        """Return summary including engine-owned recall, or None to preserve arguments+result."""
        from .tool_contracts import format_result
        registered = self.registry.get(source.name) if self.registry else None
        view = format_result(registered, source.result, self.policy(source).result_max_chars or 600)
        if view is not None:
            return view
        if self.is_result_container(source):
            return await self._background(source, trigger)
        policy = policy_override or self.policy(source)
        strategy = policy.strategy(trigger)
        if policy.hard_preserve:
            return None
        if trigger != 'general' and source.name == 'bash' and source.arguments.get('context_policy') == 'keep':
            strategy = 'preserve'
        if strategy == 'llm_summary' and len(source.result) < self.config.summary_min_chars:
            return None  # short original: preserve both arguments and result, no rule fallback
        if strategy == 'preserve' or not getattr(self, '_implementation_matches', True):
            return None
        # A recall without durable coordinates would misrepresent recoverability.
        if source.seq is None or source.send_index is None:
            return None
        if strategy == 'archive':
            from power_loop.runtime.tool_archive import archive_text
            return archive_text(source)
        if strategy == 'truncate':
            from power_loop.runtime.tool_truncation import truncate_view
            pair = truncate_view(source.name, _json(dict(source.arguments)), dict(source.arguments),
                                 source.result, policy, source.recall)
            return _json({'arguments': pair[0], 'result': pair[1]}) if pair else None
        original = _json({'arguments': dict(source.arguments), 'result': source.result})
        operation = _hash({'source': asdict(source), 'trigger': trigger, 'policy': asdict(policy),
                           'model': self.model, 'model_identity': self.model_identity,
                           'rule_version': self.config.rule_version})
        cached = self.state['operations'].get(operation)
        if cached is not None:
            return None if contains_preserve_marker(cached['text']) else cached['text']
        # Interrupted attempts are not blindly reissued: rule fallback becomes stable.
        text = None
        if strategy == 'llm_summary':
            # Read-only on the latency-critical path. Missing/pending/failed => rule now.
            text = await self.cache.get(self.cache_key(source, policy))
        if not self._valid(text, original, source):
            text = self._rule(source)
        if not self._valid(text, original, source):
            text = None
        else:
            text = f'{text}\n{recall_note(source.recall)}'
        self.state['operations'][operation] = {'text': text, 'policy': asdict(policy),
                                               'source_hash': _hash(asdict(source))}
        await self._save()
        return text

    async def usage_snapshot(self) -> dict[str, int]:
        """Known auxiliary costs at return; later receipts remain attributed in the ledger."""
        rows = await self.cache.db.fetchall(
            f'SELECT record_json,outcome FROM {self.cache.attempts} WHERE owner_session_id=? AND created_at>=?',
            (self.sid, self.started_ms))
        self.usage = {}
        for row in rows:
            record = json.loads(row['record_json'])
            if record.get('owner_send_index') == self.send_index and row['outcome'] != 'started':
                self._add_usage(record)
        pending = await self.cache.db.fetchone(
            f"SELECT COUNT(*) AS n FROM {self.cache.cache} WHERE owner_session_id=? AND status IN ('pending','running')",
            (self.sid,))
        self.usage['auxiliary_pending'] = int(pending['n'])
        return dict(self.usage)

    def cache_key(self, source: ToolSource, policy: ToolContextPolicy) -> str:
        return _hash({'scope': self.cache_scope, 'tool': source.name,
                      'args': dict(source.arguments), 'result': source.result,
                      'prompt': policy.prompt, 'policy_version': policy.version,
                      'max_tokens': policy.max_tokens, 'model': self.model_identity})

    async def submit(self, source: ToolSource) -> bool:
        """Persist eligible work only. No generation, no usage delivery, no waiting."""
        if self.is_result_container(source):
            accepted = False
            for target in await self._background_sources(source):
                accepted = await self.submit(target) or accepted
            return accepted
        policy = self.policy(source)
        if len(source.result) < self.config.summary_min_chars:
            return False
        if (not self._implementation_matches or source.seq is None
                or source.send_index is None or not any(policy.strategy(t) == 'llm_summary'
                for t in ('age_retirement', 'insend_fuse', 'general'))):
            return False
        payload = {'owner_session_id': self.sid, 'owner_send_index': self.send_index,
                   'source_send_index': source.send_index, 'source_seq': source.seq,
                   'source_call_id': source.call_id, 'name': source.name,
                   'input': _json({'arguments': dict(source.arguments), 'result': source.result}),
                   'prompt': policy.prompt, 'max_tokens': policy.max_tokens,
                   'timeout_s': policy.timeout_s, 'model': self.model,
                   'request_extra': self.summary_request_extra,
                   'attribution': {**(self.config.usage_metadata() if self.config.usage_metadata else {}),
                                   **self.lineage}}
        accepted = await self.cache.enqueue(self.cache_key(source, policy), self.executor_key, payload,
                                            self.config.cache_ttl_s,
                                            max_entries=self.config.cache_max_entries,
                                            max_bytes=self.config.cache_max_bytes)
        if accepted and self.worker is not None:
            self.worker.wakeup.set()
        return accepted

    async def _background_sources(self, source: ToolSource) -> list[ToolSource]:
        from .tool_contracts import resolve

        registered = self.registry.get(source.name) if self.registry else None
        implementation = resolve(registered)
        if implementation is None or not implementation.runtime_contract.result_container:
            return []
        return await implementation.result_sources(source, store=self.store, registry=self.registry)

    def is_result_container(self, source):
        from .tool_contracts import result_container

        return result_container(self.registry.get(source.name) if self.registry else None)

    async def _background(self, source: ToolSource, trigger: Trigger) -> str | None:
        sources = await self._background_sources(source)
        for target in sources:
            if (self.policy(target).strategy(trigger) == 'preserve'
                    or (trigger != 'general' and target.name == 'bash'
                        and target.arguments.get('context_policy') == 'keep')):
                return None
        summaries = []
        for target in sources:
            text = await self.distill(target, trigger)
            if text is None:
                return None
            summaries.append(f"{target.name} task={target.call_id}: {text}")
        return '\n'.join(summaries) if summaries else None

    @staticmethod
    def _valid(text: Any, original: str, source: ToolSource) -> bool:
        return (isinstance(text, str) and bool(text.strip()) and not contains_preserve_marker(text)
                and len(text) + len(source.recall) + 4 < len(original))

    async def _record_stats(self, attempt: dict[str, Any]) -> None:
        if attempt['outcome'] == 'started':
            return
        usage = dict(attempt.get('usage') or {})
        if usage.get('total_tokens') is None:
            usage['total_tokens'] = (usage.get('prompt_tokens') or 0) + (usage.get('completion_tokens') or 0)
        await self.store.record_auxiliary_usage(self.sid, attempt['attempt_id'], usage)

    def _add_usage(self, attempt: dict[str, Any]) -> None:
        u = attempt.get('usage') or {}
        values = {'calls': 1, 'tool_distill_calls': 1,
                  'unknown_usage_calls': int(attempt['usage_unknown']),
                  'prompt_tokens': u.get('prompt_tokens') or 0,
                  'completion_tokens': u.get('completion_tokens') or 0,
                  'reasoning_tokens': u.get('completion_reasoning_tokens') or u.get('reasoning_tokens') or 0,
                  'cache_read_tokens': u.get('prompt_cached_tokens') or u.get('cached_tokens') or u.get('cache_hit_tokens') or 0,
                  'total_tokens': u.get('total_tokens') or (u.get('prompt_tokens') or 0) + (u.get('completion_tokens') or 0)}
        for key, value in values.items():
            self.usage[key] = self.usage.get(key, 0) + int(value)

    async def prepare_projection(self, projected: Any, rows: list[Any], send_index: int) -> Any:
        """Between sends: apply each tool's ``general`` rule to the projector's entries, then merge
        consecutive calls of one tool into a group entry."""
        from power_loop.runtime.tool_archive import prepare_projection
        return await prepare_projection(self, projected, rows, send_index)

    def manages(self, name: str) -> bool:
        registered = self.registry.get(name) if self.registry else None
        from .tool_contracts import result_container
        return (result_container(registered) or name in self.config.overrides
                or getattr(getattr(registered, 'definition', None), 'context_policy', None) is not None)

    async def prepare_history(self, pipeline: Any, trigger: Trigger, *, target_tokens: int | None = None,
                              level: int = 1) -> int:
        """In-send rewrite of the working history (age retirement, or one fuse level)."""
        from power_loop.runtime.tool_archive import prepare_history
        return await prepare_history(self, pipeline, trigger, target_tokens=target_tokens, level=level)
