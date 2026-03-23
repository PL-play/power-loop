from __future__ import annotations

from power_loop.contracts.tools import ToolDefinition

# Tool definitions copied from zero-code BASE_TOOLS entries for the default core tool set.
DEFAULT_TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="write_file",
        description=(
            "Create or overwrite a file with the given content. Creates parent directories automatically. "
            "IMPORTANT: Both 'path' and 'content' parameters are REQUIRED and must be provided in a single valid JSON object. "
            "For large files, write the complete content in one call — do not split across multiple calls."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        required_params=("path", "content"),
    ),
    ToolDefinition(
        name="read_file",
        description="Read file contents with line numbers, or list directory entries.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
        required_params=("path",),
    ),
    ToolDefinition(
        name="edit_file",
        description="Replace exact text in a file (old_text->new_text).",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_text", "new_text"],
        },
        required_params=("path", "old_text", "new_text"),
    ),
    ToolDefinition(
        name="apply_patch",
        description="Apply a patch to a file using @@ context lines for positioning and +/- for line changes.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "patch": {"type": "string"},
            },
            "required": ["path", "patch"],
        },
        required_params=("path", "patch"),
    ),
    ToolDefinition(
        name="bash",
        description="Run a shell command in a persistent bash session rooted at workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "restart": {"type": "boolean"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
        required_params=("command",),
    ),
    ToolDefinition(
        name="glob",
        description="Find files by glob pattern, sorted by modification time (newest first).",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
        },
        required_params=("pattern",),
    ),
    ToolDefinition(
        name="grep",
        description="Search file contents by regex pattern.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "include": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["pattern"],
        },
        required_params=("pattern",),
    ),
    ToolDefinition(
        name="load_skill",
        description="Load specialized knowledge by name.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        },
        required_params=("name",),
    ),
    ToolDefinition(
        name="todo",
        description="Update the current task list (todo manager). Only one item can be in_progress at a time.",
        input_schema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                        "required": ["id", "text", "status"],
                    }
                }
            },
            "required": ["items"],
        },
        required_params=("items",),
    ),
    ToolDefinition(
        name="background_run",
        description="Run a shell command in a private background worker (non-interactive).",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        },
        required_params=("command",),
    ),
    ToolDefinition(
        name="check_background",
        description="Check your private background tasks. If task_id is omitted, list all tasks.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
            },
        },
        required_params=(),
    ),
]
