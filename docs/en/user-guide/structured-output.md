# Structured Output

[中文](../../zh/user-guide/structured-output.md) | [User Guide](../index.md)

Force LLMs to return valid JSON that matches a schema. `StructuredOutputSpec` tells the LLM what shape you need; `parse_structured()` repairs common JSON defects in the response.

## Quick Start

```python
from power_loop import LLMRequest, StructuredOutputSpec, parse_structured

spec = StructuredOutputSpec(
    name="UserCard",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "favorite_number": {"type": "integer"},
            "city": {"type": "string"},
        },
        "required": ["name", "favorite_number"],
        "additionalProperties": False,
    },
    description="Extract user profile from the message.",
)

req = LLMRequest(
    messages=[{"role": "user", "content": "我叫阿岚，住上海，最喜欢的数字是 37。"}],
    system_prompt="Extract the user profile into JSON matching the schema.",
    response_format=spec.to_openai_response_format(),
    max_tokens=200, temperature=0.0,
)

resp = await llm.complete(req)
card = parse_structured(resp, schema=spec.schema)
# → {"name": "阿岚", "favorite_number": 37, "city": "上海"}
```

## StructuredOutputSpec

```python
StructuredOutputSpec(
    name="UserCard",               # required: must match the schema title
    schema={...},                  # JSON Schema dict
    strict=True,                   # server-side strict mode (most providers)
    description="A user card",     # optional: helps the LLM
)
```

`.to_openai_response_format()` renders:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "UserCard",
    "schema": {...},
    "strict": true,
    "description": "A user card"
  }
}
```

## parse_structured — 4-Level Repair

The parser tries four strategies in order:

| Level | Strategy | Handles |
|---|---|---|
| 1 | Direct `json.loads` | Clean JSON |
| 2 | Strip markdown fences | ` ```json ... ``` ` |
| 3 | Extract first balanced `{...}` | Prose + embedded JSON |
| 4 | Repair trailing commas | `{"a": 1,}` → `{"a": 1}` |

If all fail, `StructuredOutputError` is raised with a machine-readable `reason`:

```python
try:
    card = parse_structured(resp, schema=spec.schema)
except StructuredOutputError as exc:
    print(exc.reason)     # "no_json" | "invalid_json" | "not_object" | "missing_required:name"
    print(exc.raw_text)   # LLM output (truncated to 1000 chars)
```

## Schema Validation

Local validation is **minimal by design** — only checks:

1. Top-level `type == "object"`
2. All `required` keys are present

Deeper validation (per-field types, enum, pattern) is left to the provider's strict mode. This avoids silently disagreeing with the provider in edge cases.

## Full Example

See [`examples/14_structured_card.py`](../../../examples/14_structured_card.py) for a runnable example with real LLM extraction + repair + schema-failure demonstration.

## Next

- [Tools](tools.md) — the other way agents produce structured data (tool calls)
- [Memory](memory.md) — cross-session recall
