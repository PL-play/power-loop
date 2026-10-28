"""Tool-context preparation: code distillation, truncation, archive views and the in-send fuse.

No model request is made here. Derived views are memoized in the session runtime state.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
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
    # Include provider/endpoint/config revision, not secrets. Changing it invalidates derived views.
    model_identity: str = ''
    rule_version: str = '1'
    rule: Callable[[str, Mapping[str, Any], str, Any], Any] | None = None

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
                 registry: Any = None, rule: Any = None):
        transport_config = getattr(llm, '_cfg', None)
        self.model = model or getattr(transport_config, 'model', None)
        # The retired background summaries contributed the request extra and the 8000-char
        # minimum to this hash. Their former default values are kept so the projection version
        # of every existing session stays the same (a change would re-project all history).
        identity = _hash({'declared': config.model_identity,
                          'summary_request_extra': {'extra_body': {'thinking': {'type': 'disabled'}}}
                          if 'deepseek' in (self.model or '').lower() else {},
                          'summary_min_chars': 8000,
                          'transport': type(llm).__module__ + '.' + type(llm).__qualname__,
                          'endpoint': getattr(transport_config, 'base_url', None),
                          'model': self.model})
        self.config, self.store = config, store
        self.model_identity = identity
        self.sid, self.send_index = owner_session_id, owner_send_index
        self.llm, self.registry = llm, registry
        self.rule = config.rule or rule
        self.state: dict[str, Any] = {'operations': {}}

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

    def projection_version(self, original: int) -> int:
        return int(_hash({'context_presentation_protocol': 1, 'original': original, 'merge': self.config.merge_consecutive, 'overrides': self.config.overrides,
                          'rule': self.config.rule_version, 'model': self.model,
                          'model_identity': self.model_identity})[:7], 16)

    async def _save(self) -> None:
        await self.store.set_runtime_state(self.sid, TOOL_CONTEXT_STATE_KEY, self.state)

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
        if self.preserves_context(source, trigger):
            strategy = 'preserve'
        if strategy == 'preserve' or not getattr(self, '_implementation_matches', True):
            return None
        # A recall without durable coordinates would misrepresent recoverability.
        if source.seq is None or source.send_index is None:
            return None
        if strategy == 'archive':
            from power_loop.runtime.tool_archive import archive_text
            return archive_text(source, registered)
        if strategy == 'truncate':
            from .tool_contracts import truncate_context
            pair = truncate_context(registered, source, policy, raw_arguments=_json(dict(source.arguments)))
            return _json({'arguments': pair[0], 'result': pair[1]}) if pair else None
        original = _json({'arguments': dict(source.arguments), 'result': source.result})
        operation = _hash({'source': asdict(source), 'trigger': trigger, 'policy': asdict(policy),
                           'model': self.model, 'model_identity': self.model_identity,
                           'rule_version': self.config.rule_version})
        cached = self.state['operations'].get(operation)
        if cached is not None:
            return None if contains_preserve_marker(cached['text']) else cached['text']
        # A stored retired 'llm_summary' strategy is distilled by the code rule like 'rule'.
        text = self._rule(source)
        if not self._valid(text, original, source):
            text = None
        else:
            text = f'{text}\n{recall_note(source.recall)}'
        self.state['operations'][operation] = {'text': text, 'policy': asdict(policy),
                                               'source_hash': _hash(asdict(source))}
        await self._save()
        return text

    async def _background_sources(self, source: ToolSource) -> list[ToolSource]:
        from .tool_contracts import resolve

        registered = self.registry.get(source.name) if self.registry else None
        implementation = resolve(registered)
        if implementation is None or not implementation.runtime_contract.result_container:
            return []
        return await implementation.result_sources(source, store=self.store, registry=self.registry)

    def preserves_context(self, source, trigger):
        from .tool_contracts import context_presentation
        registered = self.registry.get(source.name) if self.registry else None
        return trigger != 'general' and context_presentation(registered, source).keep_in_send

    def is_result_container(self, source):
        from .tool_contracts import result_container

        return result_container(self.registry.get(source.name) if self.registry else None)

    async def _background(self, source: ToolSource, trigger: Trigger) -> str | None:
        sources = await self._background_sources(source)
        for target in sources:
            if (self.policy(target).strategy(trigger) == 'preserve'
                    or self.preserves_context(target, trigger)):
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
