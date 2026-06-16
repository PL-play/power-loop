# 教程：构建聊天机器人

[English](../../en/tutorials/build-a-chatbot.md) | [教程](../index.md)

**目标**：构建一个带持久化历史的多轮 CLI 聊天机器人——50 行 Python。

**你会学到**：`StatefulAgentLoop`、`new_session()`、`send()`、`session_id` 多轮、会话持久化、`get_messages()`。

## 1. 环境

```bash
pip install 'power-loop[openai]' python-dotenv
```

配置 `.env`：

```bash
POWER_LOOP_BASE_URL=https://api.openai.com/v1
POWER_LOOP_API_KEY=sk-…
POWER_LOOP_MODEL=gpt-4o-mini
```

## 2. 最小聊天机器人

```python
import asyncio
from power_loop import (
    StatefulAgentLoop, AgentLoopConfig, create_llm_service_from_env,
)

async def main():
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm,
        config=AgentLoopConfig(
            system_prompt="你是一个友好的助手。回复简洁。",
            max_rounds=1,
        ),
    )
    session_id = await loop.new_session()

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ("exit", "quit"):
            break
        result = await loop.send(user_input, session_id=session_id)
        print(f"Bot: {result.final_text}")

    loop.close()

asyncio.run(main())
```

## 3. 多轮对话——记住上下文

上面启动时显式创建一个会话。输入 `new` 时再创建新的 `session_id`：

```python
session_id = await loop.new_session()

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "new":
        session_id = await loop.new_session()
        print("[新会话]")
        continue

    result = await loop.send(user_input, session_id=session_id)
    print(f"Bot: {result.final_text}")
```

现在机器人记住了：

```
You: 我叫阿岚。
Bot: 你好阿岚！
You: 我叫什么？
Bot: 你叫阿岚。
```

## 4. 持久化——重启不丢失

`db_path="./chatbot.db"` 让会话活在磁盘上：

```python
result = await loop.send("还记得我吗？", session_id="之前保存的 session_id")
print(result.final_text)  # → "你是阿岚！"
```

循环不持有权威状态，所以重启后的进程凭相同的 `dsn`/`db_path` + `session_id` 即可恢复会话。默认存储是
零基础设施的单 SQLite 文件；需要共享的多写入者服务时，把 `dsn=` 换成 PostgreSQL / MySQL URL 即可——
见 [存储后端](../user-guide/storage-backends.md)。

## 完整代码

```python
import asyncio
from power_loop import (
    StatefulAgentLoop, AgentLoopConfig, create_llm_service_from_env,
)

async def main():
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm, db_path="./chatbot.db",
        config=AgentLoopConfig(
            system_prompt="你是一个友好的助手。回复简洁。",
            max_rounds=1,
        ),
    )
    session_id = await loop.new_session()
    try:
        print("Chatbot (输入 'new' 新会话, 'exit' 退出)")
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input: continue
            if user_input.lower() == "exit": break
            if user_input.lower() == "new":
                session_id = await loop.new_session(); print("[新会话]"); continue
            result = await loop.send(user_input, session_id=session_id)
            print(f"Bot: {result.final_text}")
    finally:
        loop.close()

asyncio.run(main())
```

## 下一步

- [工具调用](tool-calling.md) — 给 Agent 加能力
- [人在回路](human-in-the-loop.md) — 执行危险操作前请求确认
