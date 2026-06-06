from __future__ import annotations

import uuid
from pathlib import Path

from power_loop import ToolDefinition, create_default_tool_registry


def test_default_tool_registry_smoke(tmp_path: Path) -> None:
    registry = create_default_tool_registry(workspace_dir=tmp_path, skills_dir=tmp_path / "skills")

    # Validate default set
    for name in [
        "write_file",
        "read_file",
        "edit_file",
        "apply_patch",
        "bash",
        "glob",
        "grep",
        "load_skill",
        "request_user_input",
    ]:
        assert registry.has(name), f"missing default tool: {name}"

    # Validate required params check
    err = registry.validate("write_file", {"path": "a.txt"})
    assert err is not None and "missing required parameter" in err.lower()

    # Default handler smoke: write + read
    temp_name = f"tmp-smoke-{uuid.uuid4().hex[:8]}.txt"
    rel = Path(temp_name).as_posix()

    write_res = registry.invoke("write_file", {"path": rel, "content": "hello"})
    assert "Wrote" in str(write_res)

    read_res = registry.invoke("read_file", {"path": rel})
    assert "hello" in str(read_res)

    # Dynamic register
    def _echo(**kw):
        return f"echo:{kw.get('text', '')}"

    registry.register(
        ToolDefinition(
            name="echo_tool",
            description="Echo text",
            required_params=("text",),
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        _echo,
    )

    assert registry.has("echo_tool")
    assert registry.invoke("echo_tool", {"text": "ok"}) == "echo:ok"

    # Cleanup temp file
    (tmp_path / temp_name).unlink(missing_ok=True)
