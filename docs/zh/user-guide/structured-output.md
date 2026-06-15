# 结构化输出

[English](../../en/user-guide/structured-output.md) | [用户手册](../index.md)

强制 LLM 返回匹配 schema 的有效 JSON。`StructuredOutputSpec` 告诉 LLM 你需要什么形状；`parse_structured()` 修复回复中的常见 JSON 缺陷。

## 快速开始

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
)

req = LLMRequest(
    messages=[{"role": "user", "content": "我叫阿岚，住上海，最喜欢的数字是 37。"}],
    system_prompt="提取用户信息。",
    response_format=spec.to_openai_response_format(),
    max_tokens=200, temperature=0.0,
)

resp = await llm.complete(req)
card = parse_structured(resp, schema=spec.schema)
# → {"name": "阿岚", "favorite_number": 37, "city": "上海"}
```

## parse_structured — 四级修复

| 级 | 策略 | 处理 |
|---|---|---|
| 1 | 直接 `json.loads` | 干净 JSON |
| 2 | 剥离 markdown 围栏 | ` ```json ... ``` ` |
| 3 | 提取第一个平衡的 `{...}` | 夹杂 JSON 的散文 |
| 4 | 修复尾逗号 | `{"a": 1,}` → `{"a": 1}` |

全部失败时抛出 `StructuredOutputError`，带机器可读的 `reason`：

```python
try:
    card = parse_structured(resp, schema=spec.schema)
except StructuredOutputError as exc:
    print(exc.reason)     # "no_json" | "invalid_json" | "not_object" | "missing_required:name"
```

## Schema 校验

本地校验故意**最简化**——只检查：
1. 顶层 `type == "object"`
2. 所有 `required` 键存在

深层校验留给 provider 的 strict mode。避免在边缘情况与 provider 静默分歧。

## 下一步

- [工具](tools.md) — Agent 产出结构化数据的另一种方式
- [记忆](memory.md) — 跨会话召回