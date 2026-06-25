"""Shared payload sanitization for contrib event sinks (logging, JSONL).

Truncates long strings and redacts secret-looking keys so event payloads can be logged
or persisted without leaking credentials or blowing up volume. Used by both
``logging_sink`` and ``jsonl_sink`` so the redaction policy is defined once.

REDACTION SCOPE (important): by default redaction is **key-name based** — a value is
replaced only when its *key* matches the denylist. Secrets embedded in string VALUES under
benign keys (a ``Bearer …`` header inside a bash command string, an ``sk-…`` key pasted into
a tool argument) are NOT scrubbed by the default policy. Opt into value-content scrubbing with
``redact_value_secrets=True`` on the sink, which additionally regex-redacts common secret shapes
(see :data:`DEFAULT_VALUE_PATTERNS`) inside string values.
"""

from __future__ import annotations

import re
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

#: Regexes for secret-shaped substrings scrubbed from string VALUES when value-content redaction
#: is enabled (opt-in). Conservative shapes only, to avoid mangling ordinary text:
DEFAULT_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/\-]{12,}=*", re.IGNORECASE),  # Authorization: Bearer …
    re.compile(r"\b(?:sk|rk|pk|xoxb|xoxp|ghp|gho|github_pat)[-_][A-Za-z0-9_\-]{16,}"),  # provider keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                                 # AWS access key id
    re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),  # JWT
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),                           # Google API key
)


def resolve_redact(redact_keys: Iterable[str] | None) -> tuple[str, ...]:
    """Lower-cased redaction key substrings. ``None`` → the default denylist; ``()``
    disables redaction; any iterable overrides."""
    keys = tuple(redact_keys if redact_keys is not None else DEFAULT_REDACT_KEYS)
    return tuple(k.lower() for k in keys)


def scrub_value_secrets(text: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    """Replace every secret-shaped substring matched by ``patterns`` with :data:`REDACTED`."""
    for pat in patterns:
        text = pat.sub(REDACTED, text)
    return text


def sanitize(
    value: Any, limit: int, redact_lower: tuple[str, ...], *, max_list: int = 50,
    value_patterns: tuple[re.Pattern[str], ...] | None = None,
) -> Any:
    """Recursively truncate long strings to ``limit`` and redact values under keys whose
    (lower-cased) name contains any ``redact_lower`` substring. Lists are capped at ``max_list``
    items. When ``value_patterns`` is given, secret-shaped substrings inside string VALUES are
    scrubbed too (key-name redaction alone misses secrets embedded in values — M-observability-6)."""
    if isinstance(value, str):
        text = value if len(value) <= limit else value[:limit] + f"…(+{len(value) - limit})"
        return scrub_value_secrets(text, value_patterns) if value_patterns else text
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            kl = str(k).lower()
            out[k] = REDACTED if any(r in kl for r in redact_lower) else sanitize(
                v, limit, redact_lower, max_list=max_list, value_patterns=value_patterns
            )
        return out
    if isinstance(value, list):
        return [
            sanitize(v, limit, redact_lower, max_list=max_list, value_patterns=value_patterns)
            for v in value[:max_list]
        ]
    return value
