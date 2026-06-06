# Tutorial: Build a Chatbot

[中文](../../zh/tutorials/build-a-chatbot.md) | [Tutorials](../index.md)

**Goal**: Build a multi-turn CLI chatbot with persistent session history — 50 lines of Python.

**You'll learn**: `StatefulAgentLoop`, `new_session()`, `send()`, multi-turn with `session_id`, session persistence, `get_messages()`.

## 1. Setup

```bash
pip install power-loop python-dotenv
```

Create `.env`:

```bash
POWER_LOOP_BASE_URL=https://api.openai.com/v1
POWER_LOOP_API_KEY=sk-…
POWER_LOOP_MODEL=gpt-4o-mini
```

## 2. Minimal Chatbot

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
    session_id = loop.new_session()

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ("exit", "quit"):
            break
        result = await loop.send(user_input, session_id=session_id)
        print(f"Bot: {result.final_text}")

    loop.close()

asyncio.run(main())
```

Run it:

```
You: 你好！
Bot: 你好！有什么可以帮你的？
You: exit
```

## 3. Multi-Turn — Remember Context

The bot above creates one explicit session and reuses it. To start fresh on demand, create a new `session_id`:

```python
async def main():
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm,
        db_path="./chatbot.db",       # real file — survives restarts
        config=AgentLoopConfig(
            system_prompt="你是一个友好的助手。回复简洁。",
            max_rounds=1,
        ),
    )

    session_id = loop.new_session()

    print("Chatbot (type 'new' for fresh session, 'exit' to quit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "new":
            session_id = loop.new_session()
            print("[New session started]")
            continue

        result = await loop.send(user_input, session_id=session_id)
        print(f"Bot: {result.final_text}")

    loop.close()
```

Now the bot remembers:

```
You: 我叫阿岚。
Bot: 你好阿岚！

You: 我叫什么？
Bot: 你叫阿岚。
```

## 4. Persistence — Survive Restarts

`db_path="./chatbot.db"` means the session lives on disk. Restart the script:

```python
# Pass the session_id from the previous run
result = await loop.send("还记得我吗？", session_id="sess_saved_from_previous_run")
print(result.final_text)  # → "你是阿岚！"
```

## 5. View the History

```python
messages = loop.get_messages(session_id)
for m in messages:
    print(f"[{m['role']}] {m.get('content', '')[:80]}")
```

## Complete Code

```python
# chatbot.py — full version with session persistence
import asyncio
from power_loop import (
    StatefulAgentLoop, AgentLoopConfig, create_llm_service_from_env,
)

async def main():
    llm = create_llm_service_from_env()
    loop = StatefulAgentLoop(
        llm=llm,
        db_path="./chatbot.db",
        config=AgentLoopConfig(
            system_prompt="你是一个友好的助手。回复简洁。",
            max_rounds=1,
        ),
    )

    session_id = loop.new_session()
    print("Chatbot (type 'new', 'history', or 'exit')")
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                break
            if user_input.lower() == "new":
                session_id = loop.new_session()
                print("[New session]")
                continue
            if user_input.lower() == "history":
                for m in loop.get_messages(session_id):
                    print(f"  [{m['role']}] {m.get('content', '')[:100]}")
                continue

            result = await loop.send(user_input, session_id=session_id)
            print(f"Bot: {result.final_text}")
    finally:
        loop.close()

asyncio.run(main())
```

## Next

- [Tool Calling](tool-calling.md) — add abilities to your agent
- [Human-in-the-Loop](human-in-the-loop.md) — ask before dangerous actions
