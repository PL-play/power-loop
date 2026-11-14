"""Extract per-call usage for ordinary model telemetry."""
_TOKENS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "prompt_cached_tokens",
    "prompt_cache_miss_tokens",
    "prompt_image_tokens",
    "prompt_audio_tokens",
    "prompt_text_tokens",
    "completion_reasoning_tokens",
    "completion_audio_tokens",
    "completion_text_tokens",
    "completion_image_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
    "cached_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "accepted_tokens",
    "rejected_tokens",
)
def response_observation(usage, response):
    raw = getattr(response, "raw_completion", None)
    response_id = raw.get("id") if isinstance(raw, dict) else getattr(raw, "id", None)
    cost = getattr(response, "provider_cost", None)
    return {
        "usage": {key: getattr(usage, key) for key in _TOKENS if getattr(usage, key, None) is not None},
        "cost_amount": cost.amount if cost is not None else None,
        "cost_currency": cost.currency if cost is not None else None,
        "response_metrics": getattr(response, 'debug', {}).get('response_extraction', {}),
        "response_id": response_id,
    }
