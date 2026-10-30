"""Provider-independent response usage/cost extraction, without network calls."""
from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import asdict, dataclass, fields
from decimal import Decimal, InvalidOperation, localcontext
from typing import Protocol

from ._vendor.llm_client.interface import LLMTokenUsage
from ._vendor.llm_client.usage_normalization import normalize_usage


class ResponseExtractionError(ValueError):
    """Raised only after a physical response has been retained; never retry it."""


@dataclass(frozen=True)
class ProviderCost:
    amount: str
    currency: str | None = None

    def __post_init__(self):
        amount = exact_amount(self.amount)
        if self.currency is not None and not re.fullmatch(r'[A-Z]{3}', self.currency):
            raise ResponseExtractionError('invalid cost currency')
        object.__setattr__(self, 'amount', amount)


class TokenUsageExtractor(Protocol):
    def extract(self, response: dict) -> LLMTokenUsage | None: ...


class CostExtractor(Protocol):
    def extract(self, response: dict) -> ProviderCost | None: ...


def exact_amount(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal, float)):
        raise ResponseExtractionError('cost is not a decimal number')
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise ResponseExtractionError('cost is not a decimal number') from None
    if not amount.is_finite() or amount < 0 or len(str(value)) > 128 or abs(amount.as_tuple().exponent) > 128:
        raise ResponseExtractionError('cost must be finite and nonnegative')
    return format(amount, 'f')


def response_dict(response):
    if isinstance(response, dict):
        return response
    if hasattr(response, 'model_dump'):
        try:
            value = response.model_dump()
            if isinstance(value, dict):
                return value
        except Exception:
            return {}
    if hasattr(response, '__dict__'):
        return {key: value for key, value in vars(response).items() if not key.startswith('_')}
    return {}


def read_path(response, path):
    """A missing/incorrect container is a normal miss, never KeyError/IndexError."""
    if not isinstance(path, str) or not path or len(path) > 512:
        return None
    parts = path.replace('[', '.').replace(']', '').split('.')
    value = response
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, (list, tuple)) and part.isdecimal() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
        if value is None:
            return None
    return value


def _token(value):
    number = Decimal(exact_amount(value))
    if number != number.to_integral_value() or number >= 2**63:
        raise ResponseExtractionError('token usage must be a nonnegative integer')
    return int(number)


def _usage(values):
    keys = {f.name for f in fields(LLMTokenUsage)} - {'cost'}
    result = {k: _token(v) for k, v in values.items() if k in keys and v is not None}
    if not result:
        return None
    if result.get('total_tokens') is None and result.get('prompt_tokens') is not None and result.get('completion_tokens') is not None:
        result['total_tokens'] = result['prompt_tokens'] + result['completion_tokens']
    return LLMTokenUsage(**result)


class DefaultTokenUsageExtractor:
    def extract(self, response):
        for path in ('usage', 'x_openai_usage', 'x_usage', 'usage_metadata', 'usageMetadata', 'message.usage'):
            raw = read_path(response, path)
            if not isinstance(raw, dict):
                continue
            if 'promptTokenCount' in raw or 'candidatesTokenCount' in raw:
                return _usage({'prompt_tokens': raw.get('promptTokenCount'),
                    'completion_tokens': raw.get('candidatesTokenCount'), 'total_tokens': raw.get('totalTokenCount'),
                    'prompt_cached_tokens': raw.get('cachedContentTokenCount')})
            if 'input_tokens' in raw and ('cache_read_input_tokens' in raw or 'cache_creation_input_tokens' in raw):
                copied = dict(raw)
                copied['prompt_tokens'] = _token(raw['input_tokens']) + _token(raw.get('cache_read_input_tokens', 0)) + _token(raw.get('cache_creation_input_tokens', 0))
                copied['prompt_cached_tokens'] = raw.get('cache_read_input_tokens')
                copied['prompt_cache_miss_tokens'] = _token(raw['input_tokens']) + _token(raw.get('cache_creation_input_tokens', 0))
                return _usage(normalize_usage(copied))
            result = _usage(normalize_usage(raw))
            if result is not None:
                return result
        return None


class DefaultCostExtractor:
    def extract(self, response):
        for root in ('usage', 'x_openai_usage', 'x_usage', 'data.usage'):
            amount = read_path(response, root + '.cost')
            if amount is not None:
                return ProviderCost(exact_amount(amount), read_path(response, root + '.currency'))
        return None


def compile_extractor(code):
    """Trusted host extension; deliberately unavailable in the admin editor.

    The snippet declares extract(response). Imports, loops, private attributes,
    arbitrary builtins and mutation of the original response are unavailable.
    """
    if not isinstance(code, str) or len(code) > 16000:
        raise ResponseExtractionError('invalid extractor code size')
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    if len(nodes) > 1000 or any(isinstance(n, (ast.Import, ast.ImportFrom, ast.While, ast.For,
        ast.AsyncFor, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.ClassDef)) for n in nodes):
        raise ResponseExtractionError('unsupported extractor code')
    for node in nodes:
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ResponseExtractionError('private attributes are unavailable')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ResponseExtractionError('private names are unavailable')
    env = {'__builtins__': {'dict': dict, 'list': list, 'str': str, 'int': int, 'len': len,
        'sum': sum, 'min': min, 'max': max, 'isinstance': isinstance}, 'Decimal': Decimal}
    exec(compile(tree, '<response-extractor>', 'exec'), env)
    function = env.get('extract')
    if not callable(function):
        raise ResponseExtractionError('code must declare extract(response)')
    return function


class FieldTokenUsageExtractor:
    def __init__(self, paths):
        allowed = {f.name for f in fields(LLMTokenUsage)} - {'cost'}
        if set(paths) - allowed:
            raise ResponseExtractionError('unknown usage field')
        self.paths = dict(paths)

    def extract(self, response):
        return _usage({k: read_path(response, p) for k, p in self.paths.items()})


class FieldCostExtractor:
    def __init__(self, config):
        self.config = dict(config)

    def extract(self, response):
        value = read_path(response, self.config.get('amount_path'))
        if value is None:
            return None
        currency = read_path(response, self.config.get('currency_path')) or self.config.get('currency')
        return ProviderCost(exact_amount(value), currency)


class ResponseMetrics:
    def __init__(self, config=None, *, usage_extractor=None, cost_extractor=None):
        self.config = copy.deepcopy(config or {})
        if not isinstance(self.config, dict) or set(self.config) - {'usage', 'cost'}:
            raise ResponseExtractionError('invalid extraction configuration')
        for rule in self.config.values():
            if not isinstance(rule, dict) or rule.get('on_error', 'raise') not in ('raise', 'continue'):
                raise ResponseExtractionError('invalid extraction error policy')
        self.usage_extractor = usage_extractor
        self.cost_extractor = cost_extractor

    def extract_usage(self, response):
        rule = self.config.get('usage') or {}
        if rule.get('code'):
            with localcontext() as context:
                context.prec = 80
                return _usage(compile_extractor(rule['code'])(copy.deepcopy(response)))
        extractor = self.usage_extractor or (FieldTokenUsageExtractor(rule['paths']) if rule.get('paths') else DefaultTokenUsageExtractor())
        return extractor.extract(response)

    def extract_cost(self, response):
        rule = self.config.get('cost') or {}
        if rule.get('code'):
            with localcontext() as context:
                context.prec = 80
                value = compile_extractor(rule['code'])(copy.deepcopy(response))
            return ProviderCost(**value) if value is not None else None
        extractor = self.cost_extractor or (FieldCostExtractor(rule) if rule.get('amount_path') else DefaultCostExtractor())
        value = extractor.extract(response)
        if value is not None and value.currency is None and rule.get('currency'):
            value = ProviderCost(value.amount, rule['currency'])
        return value

    def inspect(self, response):
        raw = response_dict(response)
        result = dict(usage=None, cost=None, errors=[], source=raw, config=self.config)
        for name, extract in (('usage', self.extract_usage), ('cost', self.extract_cost)):
            try:
                value = extract(raw)
                result[name] = asdict(value) if value is not None else None
            except Exception as exc:
                rule = self.config.get(name) or {}
                result['errors'].append(dict(interface=name, error=type(exc).__name__,
                    message=str(exc)[:300], on_error=rule.get('on_error', 'raise')))
        # Decimal values are strings in durable evidence, never binary floats.
        return json.loads(json.dumps(result, default=str), parse_constant=str)

    def apply(self, response):
        value = self.inspect(response.raw_completion)
        response.token_usage = LLMTokenUsage(**value['usage']) if value['usage'] is not None else None
        response.provider_cost = ProviderCost(**value['cost']) if value['cost'] is not None else None
        response.debug['response_extraction'] = value
        return response


class ResponseMetricAccumulator:
    """Merge partial stream usage; later actual values replace earlier values."""
    def __init__(self, config=None):
        self.parser = ResponseMetrics(config)
        self.usage = {}
        self.cost = None
        self.errors = []
        self.sources = {}

    def add(self, response):
        try:
            value = self.parser.inspect(response)
        except Exception as exc:
            for name in ('usage', 'cost'):
                self.errors.append(dict(interface=name, error=type(exc).__name__,
                    message='response metrics could not be normalized',
                    on_error=(self.parser.config.get(name) or {}).get('on_error', 'raise')))
            return
        if value['usage'] is not None:
            self.usage.update({k: v for k, v in value['usage'].items() if v is not None and k != 'cost'})
            self.sources['usage'] = value['source']
        if value['cost'] is not None:
            self.cost = value['cost']
            self.sources['cost'] = value['source']
        for error in value['errors']:
            if error not in self.errors:
                self.errors.append(error)
            self.sources[error['interface']] = value['source']

    def snapshot(self):
        try:
            usage = asdict(_usage(self.usage)) if self.usage else None
        except Exception as exc:
            usage = None
            error = dict(interface='usage', error=type(exc).__name__, message='invalid merged token usage',
                on_error=(self.parser.config.get('usage') or {}).get('on_error', 'raise'))
            if error not in self.errors:
                self.errors.append(error)
        return dict(usage=usage,
                    cost=self.cost, errors=self.errors, sources=self.sources, config=copy.deepcopy(self.parser.config))


def enforce_extraction_policy(response):
    value = getattr(response, 'debug', {}).get('response_extraction', {})
    if any(error['on_error'] == 'raise' for error in value.get('errors', [])):
        raise ResponseExtractionError('response_metrics_extraction_failed')
