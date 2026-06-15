# 安装

[English](../../en/user-guide/installation.md) | [用户手册](../index.md)

## 环境要求

- **Python 3.10+** — 需要 `str | None` 语法和 `asyncio` 改进。
- **SQLite 3.35+** — 所有主流平台 Python 自带。
- 一个 **OpenAI 兼容的 LLM 端点**（或 Anthropic Messages API 端点）—— 以及对应的 transport
  extra（`[openai]` 或 `[anthropic]`）；核心本身 SDK-free。

## 安装

核心只依赖 `certifi`；OpenAI/Anthropic SDK 仅由你安装的 extra 拉入。不装对应 extra 就构造
provider 会抛出带安装提示的清晰 `ImportError`。

```bash
# 选一个 transport extra：
pip install 'power-loop[openai]'      # OpenAI 兼容 /chat/completions（DashScope、DeepSeek、Ollama/vLLM 等）
pip install 'power-loop[anthropic]'   # Anthropic Messages API 端点
pip install 'power-loop[skills]'      # load_skill 的 YAML frontmatter
pip install 'power-loop[pdf]'         # multimodal helper 的 PDF 输入
pip install 'power-loop[all]'         # 两家 transport + skills + pdf

# 从 GitHub（开发版）
pip install "power-loop[openai] @ git+https://github.com/PL-play/power-loop.git"

# 可编辑安装（DeepTalk 风格多仓库）
git clone https://github.com/PL-play/power-loop.git
cd power-loop
pip install -e '.[openai]'            # 带上 transport extra，才能构造 provider

# 带开发依赖（lint、测试 —— 含两家 transport）
pip install -e ".[dev]"
```

## 验证

```python
from power_loop import StatefulAgentLoop, AgentLoopConfig, __version__
print(__version__)  # → "0.7.2"
```

## 下一步

[配置](configuration.md) — 设置 LLM 凭证和调节循环参数。
