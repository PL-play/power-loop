"""Shared payload sanitization for contrib event sinks (logging, JSONL).

Truncates long strings and redacts secret-looking keys so event payloads can be logged
or persisted without leaking credentials or blowing up volume. Used by both
``logging_sink`` and ``jsonl_sink`` so the redaction policy is defined once.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Keys whose VALUE is replaced with "***" anywhere in a payload (case-insensitive
# substring match on the key name). Tool inputs and request messages can carry secrets.
# NB: bare "token" is intentionally NOT here — it would redact the non-secret usage
# counts (prompt_tokens / completion_tokens / total_tokens). Specific token names are.
DEFAULT_REDACT_KEYS: tuple[str, ...] = (
    "api_key", "api-key", "apikey",
    "authorization", "bearer",
    "password", "passwd",
    "secret", "secret_key",
    "access_key", "private_key",
    "access_token", "refresh_token", "auth_token", "id_token",
)
REDACTED = "***"


def resolve_redact(redact_keys: Iterable[str] | None) -> tuple[str, ...]:
    """Lower-cased redaction key substrings. ``None`` → the default denylist; ``()``
    disables redaction; any iterable overrides."""
    keys = tuple(redact_keys if redact_keys is not None else DEFAULT_REDACT_KEYS)
    return tuple(k.lower() for k in keys)


def sanitize(value: Any, limit: int, redact_lower: tuple[str, ...], *, max_list: int = 50) -> Any:
    """Recursively truncate long strings to ``limit`` and redact values under keys whose
    (lower-cased) name contains any ``redact_lower`` substring. Lists are capped at
    ``max_list`` items."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"…(+{len(value) - limit})"
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            kl = str(k).lower()
            out[k] = REDACTED if any(r in kl for r in redact_lower) else sanitize(
                v, limit, redact_lower, max_list=max_list
            )
        return out
    if isinstance(value, list):
        return [sanitize(v, limit, redact_lower, max_list=max_list) for v in value[:max_list]]
    return value
