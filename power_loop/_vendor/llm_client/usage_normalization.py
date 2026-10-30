"""Common token normalization used by the default extraction interface."""
from typing import Any


def normalize_usage(usage: Any) -> dict[str, Any]:
    """
        Normalize token usage from various provider shapes.
        Supports:
        - OpenAI usage: prompt_tokens/completion_tokens/total_tokens
        - Some providers: input_tokens/output_tokens/total_tokens
        - Dict payloads (from model_dump / raw json)
        """
    if usage is None:
        return {'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None}
    raw: dict[str, Any] = {}
    if isinstance(usage, dict):
        raw = dict(usage)
    else:
        if hasattr(usage, 'model_dump'):
            try:
                dumped = usage.model_dump()
                if isinstance(dumped, dict):
                    raw = dict(dumped)
            except Exception:
                raw = {}
        if not raw:
            try:
                raw = dict(usage.__dict__)
            except Exception:
                raw = {}

    def _safe_int_or_none(v: Any) -> int | None:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return int(float(s))
            except Exception:
                return None
        return None

    def _pick_int(payload: dict[str, Any], keys: list[str]) -> int | None:
        for key in keys:
            if key in payload:
                iv = _safe_int_or_none(payload.get(key))
                if iv is not None:
                    return iv
        return None

    def _as_dict(v: Any) -> dict[str, Any]:
        if isinstance(v, dict):
            return v
        if hasattr(v, 'model_dump'):
            try:
                dumped = v.model_dump()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                return {}
        return {}
    prompt = _pick_int(raw, ['prompt_tokens', 'input_tokens', 'prompt_token_count', 'promptTokens', 'inputTokenCount', 'prompt_count'])
    completion = _pick_int(raw, ['completion_tokens', 'output_tokens', 'completion_token_count', 'completionTokens', 'outputTokenCount', 'generated_tokens', 'candidates_token_count'])
    total = _pick_int(raw, ['total_tokens', 'total_token_count', 'totalTokens', 'token_count', 'usage_tokens'])
    if total is None and prompt is not None and (completion is not None):
        total = prompt + completion
    prompt_details = _as_dict(raw.get('prompt_tokens_details'))
    completion_details = _as_dict(raw.get('completion_tokens_details'))
    output_details = _as_dict(raw.get('output_tokens_details'))
    input_details = _as_dict(raw.get('input_tokens_details'))
    prompt_audio_tokens = _pick_int(prompt_details, ['audio_tokens', 'audioTokenCount'])
    if prompt_audio_tokens is None:
        prompt_audio_tokens = _pick_int(input_details, ['audio_tokens', 'audioTokenCount'])
    prompt_cached_tokens = _pick_int(raw, ['prompt_cache_hit_tokens', 'cached_tokens', 'cache_hit_tokens', 'prompt_cached_tokens'])
    if prompt_cached_tokens is None:
        prompt_cached_tokens = _pick_int(prompt_details, ['cached_tokens', 'cache_hit_tokens', 'cacheHitTokens'])
    prompt_cache_miss_tokens = _pick_int(raw, ['prompt_cache_miss_tokens', 'cache_miss_tokens', 'cacheMissTokens'])
    if prompt_cache_miss_tokens is None:
        prompt_cache_miss_tokens = _pick_int(prompt_details, ['cache_miss_tokens', 'cacheMissTokens'])
    prompt_text_tokens = _pick_int(prompt_details, ['text_tokens', 'textTokenCount'])
    if prompt_text_tokens is None:
        prompt_text_tokens = _pick_int(input_details, ['text_tokens', 'textTokenCount'])
    prompt_image_tokens = _pick_int(prompt_details, ['image_tokens', 'imageTokenCount'])
    if prompt_image_tokens is None:
        prompt_image_tokens = _pick_int(input_details, ['image_tokens', 'imageTokenCount'])
    completion_reasoning_tokens = _pick_int(completion_details, ['reasoning_tokens', 'reasoningTokenCount', 'reasoning_token', 'reasoningTokens', 'thinking_tokens'])
    if completion_reasoning_tokens is None:
        completion_reasoning_tokens = _pick_int(output_details, ['reasoning_tokens', 'reasoningTokenCount', 'reasoning_token', 'reasoningTokens', 'thinking_tokens'])
    completion_audio_tokens = _pick_int(completion_details, ['audio_tokens', 'audioTokenCount'])
    if completion_audio_tokens is None:
        completion_audio_tokens = _pick_int(output_details, ['audio_tokens', 'audioTokenCount'])
    completion_text_tokens = _pick_int(completion_details, ['text_tokens', 'textTokenCount'])
    if completion_text_tokens is None:
        completion_text_tokens = _pick_int(output_details, ['text_tokens', 'textTokenCount'])
    completion_image_tokens = _pick_int(completion_details, ['image_tokens', 'imageTokenCount'])
    if completion_image_tokens is None:
        completion_image_tokens = _pick_int(output_details, ['image_tokens', 'imageTokenCount'])
    accepted_prediction_tokens = _pick_int(completion_details, ['accepted_prediction_tokens', 'acceptedPredictionTokens'])
    if accepted_prediction_tokens is None:
        accepted_prediction_tokens = _pick_int(output_details, ['accepted_prediction_tokens', 'acceptedPredictionTokens'])
    rejected_prediction_tokens = _pick_int(completion_details, ['rejected_prediction_tokens', 'rejectedPredictionTokens'])
    if rejected_prediction_tokens is None:
        rejected_prediction_tokens = _pick_int(output_details, ['rejected_prediction_tokens', 'rejectedPredictionTokens'])
    cached_tokens = _pick_int(raw, ['cached_tokens', 'cache_hit_tokens', 'prompt_cache_hit_tokens'])
    if cached_tokens is None:
        cached_tokens = prompt_cached_tokens
    cache_hit_tokens = _pick_int(raw, ['cache_hit_tokens', 'prompt_cache_hit_tokens', 'cached_tokens'])
    if cache_hit_tokens is None:
        cache_hit_tokens = prompt_cached_tokens
    cache_miss_tokens = _pick_int(raw, ['cache_miss_tokens', 'prompt_cache_miss_tokens'])
    if cache_miss_tokens is None:
        cache_miss_tokens = prompt_cache_miss_tokens
    reasoning_tokens = _pick_int(raw, ['reasoning_tokens', 'reasoning_token', 'reasoningTokens', 'thinking_tokens'])
    if reasoning_tokens is None:
        reasoning_tokens = completion_reasoning_tokens
    accepted_tokens = _pick_int(raw, ['accepted_prediction_tokens'])
    if accepted_tokens is None:
        accepted_tokens = accepted_prediction_tokens
    rejected_tokens = _pick_int(raw, ['rejected_prediction_tokens'])
    if rejected_tokens is None:
        rejected_tokens = rejected_prediction_tokens
    raw['prompt_tokens'] = prompt
    raw['completion_tokens'] = completion
    raw['total_tokens'] = total
    raw['prompt_audio_tokens'] = prompt_audio_tokens
    raw['prompt_cached_tokens'] = prompt_cached_tokens
    raw['prompt_cache_miss_tokens'] = prompt_cache_miss_tokens
    raw['prompt_text_tokens'] = prompt_text_tokens
    raw['prompt_image_tokens'] = prompt_image_tokens
    raw['completion_reasoning_tokens'] = completion_reasoning_tokens
    raw['completion_audio_tokens'] = completion_audio_tokens
    raw['completion_text_tokens'] = completion_text_tokens
    raw['completion_image_tokens'] = completion_image_tokens
    raw['accepted_prediction_tokens'] = accepted_prediction_tokens
    raw['rejected_prediction_tokens'] = rejected_prediction_tokens
    raw['cached_tokens'] = cached_tokens
    raw['cache_hit_tokens'] = cache_hit_tokens
    raw['cache_miss_tokens'] = cache_miss_tokens
    raw['reasoning_tokens'] = reasoning_tokens
    raw['accepted_tokens'] = accepted_tokens
    raw['rejected_tokens'] = rejected_tokens
    return raw
