# Providers

[English](../../en/user-guide/providers.md) | [用户手册](../index.md)

power-loop 通过 `LLMProviderConfig` 和 `LLMService` 与 LLM 对话。OpenAI 兼容 `chat/completions` 端点可以直接使用；Anthropic 兼容 Messages API 端点会走原生 Anthropic transport。

模型返回的思考内容随 assistant 消息保存在现有元数据中。重载历史及原样投影会保留它，
请求出口只向来源模型回传；切换模型后不携带。模型身份来自 loop 配置或服务声明。
这修复了 [DeepSeek 思考模式](https://api-docs.deepseek.com/guides/thinking_mode/#tool-calls)
携带 tools 时要求回传所有先前 assistant 思考（包括非工具轮）的情况；不调用真实模型探测。

## 快速开始

```python
from power_loop import create_llm_service_from_env

llm = create_llm_service_from_env()  # 读取 POWER_LOOP_* 环境变量
```

或编程式：

```python
from power_loop import LLMProviderConfig, create_llm_service_from_config

cfg = LLMProviderConfig(
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="sk-…",
    model="gpt-4o-mini",
)
llm = create_llm_service_from_config(cfg)
```

## 环境变量

`POWER_LOOP_BASE_URL` / `POWER_LOOP_API_KEY` / `POWER_LOOP_MODEL` 必填。旧 `OPENAI_COMPAT_*` 名称作为回退仍可用。

## Provider 片段

### OpenAI
```bash
export POWER_LOOP_BASE_URL=https://api.openai.com/v1
export POWER_LOOP_API_KEY=sk-…
export POWER_LOOP_MODEL=gpt-4o-mini
```

### DashScope（阿里云 Qwen）
```bash
export POWER_LOOP_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export POWER_LOOP_API_KEY=sk-…
export POWER_LOOP_MODEL=qwen-plus
```

### DashScope Anthropic 兼容端点
```bash
export POWER_LOOP_PROVIDER=anthropic
export POWER_LOOP_BASE_URL=https://dashscope.aliyuncs.com/apps/anthropic
export POWER_LOOP_API_KEY=sk-…
export POWER_LOOP_MODEL=deepseek-v4-flash
```

### DeepSeek
```bash
export POWER_LOOP_BASE_URL=https://api.deepseek.com
export POWER_LOOP_API_KEY=sk-…
export POWER_LOOP_MODEL=deepseek-chat
```

### 本地（Ollama / vLLM）
```bash
export POWER_LOOP_BASE_URL=http://localhost:11434/v1
export POWER_LOOP_API_KEY=anything
export POWER_LOOP_MODEL=llama3.1
```

## 自定义前缀

```python
# 读取 MY_APP_BASE_URL, MY_APP_API_KEY, MY_APP_MODEL
llm = create_llm_service_from_env(prefix="MY_APP")
```

## 下一步

- [配置](configuration.md) — 所有 `AgentLoopConfig` 字段
- [架构设计](../../architecture.md) — LLM 调用如何融入 pipeline
