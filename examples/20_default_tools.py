"""20 · 默认工具 / Default tools: exercise every built-in tool

What you learn / 你将学到
--------------------------
- ``create_default_tool_registry(preset="full")`` 一次性注册默认工具
  / registers the full default tool set
- ``read_file`` / ``write_file`` / ``edit_file`` / ``apply_patch`` 的安全写入流程
  / safe read-before-write editing flow
- ``glob`` / ``grep`` 用于代码定位，``bash`` 用于测试和构建命令
  / search with dedicated tools, use bash for CLI-only work
- ``background_run`` / ``check_background``、``todo``、``load_skill`` 的基础调用
  / background tasks, todo updates, and skill loading

Run / 运行
----------
    python examples/20_default_tools.py
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from power_loop import create_default_tool_registry


def show(title: str, value: object) -> None:
    print(f"\n## {title}")
    print(value)


def main() -> str:
    with TemporaryDirectory(prefix="power_loop_default_tools_") as tmp:
        workspace = Path(tmp)
        skills_dir = workspace / "skills"
        skills_dir.mkdir()
        registry = create_default_tool_registry(
            preset="full",
            workspace_dir=workspace,
            skills_dir=skills_dir,
        )
        root = workspace / ".tmp-default-tools-example" / uuid.uuid4().hex[:8]
        root.mkdir(parents=True, exist_ok=True)
        rel_root = root.relative_to(workspace).as_posix()
        target = f"{rel_root}/notes.txt"

        show("write_file", registry.invoke("write_file", {"path": target, "content": "alpha\nbeta\ngamma\n"}))
        show("read_file", registry.invoke("read_file", {"path": target, "limit": 20}))

        show(
            "edit_file",
            registry.invoke(
                "edit_file",
                {"path": target, "old_text": "beta", "new_text": "BETA"},
            ),
        )

        show(
            "apply_patch",
            registry.invoke(
                "apply_patch",
                {
                    "path": target,
                    "patch": """@@ -1,3 +1,4 @@
 alpha
 BETA
-gamma
+gamma!
+delta""",
                },
            ),
        )

        (root / "code.py").write_text("VALUE = 'delta'\nprint(VALUE)\n", encoding="utf-8")
        show("glob", registry.invoke("glob", {"path": rel_root, "pattern": "*.py"}))
        show("grep", registry.invoke("grep", {"path": rel_root, "pattern": "VALUE", "include": "*.py"}))
        show("bash", registry.invoke("bash", {"command": f"python -m py_compile {rel_root}/code.py", "timeout": 10}))

        show(
            "todo",
            registry.invoke(
                "todo",
                {
                    "items": [
                        {"id": "inspect", "text": "Inspect files", "status": "completed"},
                        {"id": "verify", "text": "Verify default tools", "status": "in_progress"},
                    ]
                },
            ),
        )

        background = str(registry.invoke("background_run", {"command": "printf 'background finished\\n'"}))
        show("background_run", background)
        task_id = background.split()[2]
        time.sleep(0.1)
        show("check_background", registry.invoke("check_background", {"task_id": task_id}))

        show("load_skill", registry.invoke("load_skill", {"name": "example-missing-skill"}))
        return str(Path(target))


if __name__ == "__main__":
    main()
