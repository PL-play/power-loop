# 安装

[English](../../en/user-guide/installation.md) | [用户手册](../index.md)

## 环境要求

- **Python 3.10+** — 需要 `str | None` 语法和 `asyncio` 改进。
- **SQLite 3.35+** — 所有主流平台 Python 自带。
- 一个 **OpenAI 兼容的 LLM 端点** — 任何暴露 `/chat/completions` API 的 provider。

## 安装

```bash
# 从 PyPI 安装
pip install power-loop

# 从 GitHub 安装（开发版）
pip install git+https://github.com/deep-talk0/power-loop.git

# 可编辑安装（DeepTalk 风格多仓库）
git clone https://github.com/deep-talk0/power-loop.git
cd power-loop
pip install -e .

# 带开发依赖（lint、测试）
pip install -e ".[dev]"
```

## 验证

```python
from power_loop import StatefulAgentLoop, AgentLoopConfig, __version__
print(__version__)  # → "0.2.0"
```

## 下一步

[配置](configuration.md) — 设置 LLM 凭证和调节循环参数。
