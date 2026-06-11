from __future__ import annotations

from collections.abc import Sequence

from power_loop.contracts.tools import ToolDefinition

DEFAULT_TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="write_file",
        description=(
            "Create a new UTF-8 text file or overwrite an existing one with the complete provided content. "
            "Parent directories are created automatically. For safety, overwriting an existing file requires that you "
            "have read it with read_file in this process and that it has not changed since that read; use edit_file or "
            "apply_patch for smaller edits to existing files. Send the entire desired file content in one call, not chunks."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path, absolute workspace path, or @workspace/..."},
                "content": {"type": "string", "description": "Complete UTF-8 file content to write."},
                "overwrite": {
                    "type": "boolean",
                    "description": "Whether an existing file may be overwritten. Defaults to true, still requiring a prior read.",
                },
            },
            "required": ["path", "content"],
        },
        required_params=("path", "content"),
    ),
    ToolDefinition(
        name="read_file",
        description=(
            "Read a UTF-8 text file with stable line numbers, or list a directory. Use this before edit_file, apply_patch, "
            "or overwriting an existing file. Large files are paged by line range; use offset and limit to continue. "
            "Binary-looking files are refused instead of decoded blindly."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path, absolute workspace path, or @workspace/..."},
                "offset": {"type": "integer", "description": "1-based starting line. Defaults to 1."},
                "limit": {"type": "integer", "description": "Maximum number of lines to return."},
            },
            "required": ["path"],
        },
        required_params=("path",),
    ),
    ToolDefinition(
        name="edit_file",
        description=(
            "Replace text in an existing UTF-8 file using an exact, unique old_text snippet. Read the file first, then "
            "include enough surrounding context that old_text matches exactly once. If multiple identical occurrences "
            "should all change, set replace_all=true. The tool preserves BOM and dominant line endings and returns a diff."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to edit."},
                "old_text": {"type": "string", "description": "Exact current text to replace. Must not be empty."},
                "new_text": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every exact occurrence instead of requiring uniqueness. Defaults to false.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
        required_params=("path", "old_text", "new_text"),
    ),
    ToolDefinition(
        name="apply_patch",
        description=(
            "Apply one or more unified-diff style hunks to an existing file. Read the file first. Prefer this for "
            "multi-line or multi-hunk edits where exact search/replace would be awkward. Each hunk should include context "
            "lines beginning with a space, deleted lines beginning with '-', and added lines beginning with '+'. Ambiguous "
            "or stale hunks are rejected rather than guessed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to patch."},
                "patch": {
                    "type": "string",
                    "description": "Unified diff hunks, e.g. @@ -1,2 +1,2 @@ followed by context/delete/add lines.",
                },
            },
            "required": ["path", "patch"],
        },
        required_params=("path", "patch"),
    ),
    ToolDefinition(
        name="bash",
        description=(
            "Run a shell command in a persistent bash session rooted at the workspace. Use dedicated tools for file reads, "
            "writes, search, and patches whenever possible; use bash for tests, builds, package managers, git inspection, "
            "and other CLI-only operations. Output is truncated, timeouts restart the shell to prevent leftover commands, "
            "and obviously dangerous privileged/device-level commands are blocked. Set restart=true to reset the session."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "restart": {"type": "boolean", "description": "Restart the persistent bash session instead of running command."},
                "timeout": {"type": "integer", "description": "Seconds to wait before timing out. Range: 1-600."},
            },
            "required": ["command"],
        },
        required_params=("command",),
    ),
    ToolDefinition(
        name="glob",
        description=(
            "Find files or directories by glob pattern, sorted by modification time newest first. Prefer glob over bash/find "
            "when locating paths. A bare name like '*.py' searches recursively as '**/*.py'. Common bulky directories such "
            "as .git, node_modules, build, and dist are skipped. Hidden paths are skipped unless include_hidden=true or the "
            "pattern explicitly mentions a hidden segment."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern. Bare filenames are searched recursively."},
                "path": {"type": "string", "description": "Directory to search. Defaults to workspace root."},
                "max_results": {"type": "integer", "description": "Maximum paths to return. Defaults to 100, max 500."},
                "include_hidden": {"type": "boolean", "description": "Include hidden files and directories. Defaults to false."},
            },
            "required": ["pattern"],
        },
        required_params=("pattern",),
    ),
    ToolDefinition(
        name="grep",
        description=(
            "Search UTF-8 text file contents by regex, returning file:line:content matches. Prefer grep over bash/rg/grep "
            "commands for code search because it has workspace scoping, result caps, and binary/bulky directory filtering. "
            "Use include for a glob filter such as '*.py' or 'src/**/*.ts'. Set literal=true when searching for exact text "
            "rather than regex syntax."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern unless literal=true."},
                "path": {"type": "string", "description": "File or directory to search. Defaults to workspace root."},
                "include": {"type": "string", "description": "Optional glob filter, e.g. '*.py' or 'src/**/*.ts'."},
                "max_results": {"type": "integer", "description": "Maximum matches to return. Defaults to 50, max 500."},
                "literal": {"type": "boolean", "description": "Treat pattern as plain text instead of regex. Defaults to false."},
                "case_sensitive": {"type": "boolean", "description": "Case-sensitive search. Defaults to true."},
                "include_hidden": {"type": "boolean", "description": "Include hidden files and directories. Defaults to false."},
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
        name="note_add",
        description=(
            "Save a short persistent note to your own memory. Notes survive context compaction and are "
            "shown back to you at the start of every turn. Use for durable facts, preferences, ongoing "
            "task state — not for transcripts or long content (put those in files). When notes are full "
            "you must delete or merge old ones first. Set pinned=true for notes that must never be "
            "hidden or auto-evicted."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The note text. Keep it short and self-contained."},
                "pinned": {"type": "boolean", "description": "Pinned notes are always visible and never auto-evicted. Default false."},
            },
            "required": ["content"],
        },
        required_params=("content",),
    ),
    ToolDefinition(
        name="note_update",
        description=(
            "Rewrite or (un)pin one of your existing notes by its #id. Use to keep memory current "
            "instead of piling up near-duplicates, and to merge related notes into one."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "note_id": {"type": "integer", "description": "The #id shown in YOUR NOTES."},
                "content": {"type": "string", "description": "New text replacing the old content."},
                "pinned": {"type": "boolean", "description": "Set or clear the pinned flag."},
            },
            "required": ["note_id"],
        },
        required_params=("note_id",),
    ),
    ToolDefinition(
        name="note_delete",
        description="Delete one of your notes by its #id when it is stale or no longer worth remembering.",
        input_schema={
            "type": "object",
            "properties": {
                "note_id": {"type": "integer", "description": "The #id shown in YOUR NOTES."},
            },
            "required": ["note_id"],
        },
        required_params=("note_id",),
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
    ToolDefinition(
        name="request_user_input",
        description=(
            "Ask the caller or product UI for external input, then pause the agent loop. "
            "The loop returns status='waiting_for_input' with pending_interactions; resume by calling "
            "StatefulAgentLoop.submit_input(session_id, interaction_id, value). Use this for confirmations, choices, "
            "or text that must come from outside the LLM process."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": "Input kind such as text, confirm, or choice. Defaults to text.",
                },
                "prompt": {"type": "string", "description": "User-facing prompt or confirmation request."},
                "options": {
                    "type": "array",
                    "description": "Optional choices for choice/confirm prompts.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "metadata": {
                    "type": "object",
                    "description": "Opaque caller metadata returned unchanged in pending_interactions.",
                },
            },
            "required": ["prompt"],
        },
        required_params=("prompt",),
    ),
]

# Indexed lookup for selective registration.
DEFAULT_TOOL_DEFINITIONS_MAP: dict[str, ToolDefinition] = {d.name: d for d in DEFAULT_TOOL_DEFINITIONS}

# ---------------------------------------------------------------------------
# Tool presets (matching zero-code's BASE_TOOLS / EXPLORE_TOOLS categories)
# ---------------------------------------------------------------------------

# Core coding tools — the minimal set for an agent that reads, writes, and runs code.
CORE_TOOL_NAMES: tuple[str, ...] = (
    "bash",
    "read_file",
    "write_file",
    "edit_file",
    "apply_patch",
    "glob",
    "grep",
    "load_skill",
    "request_user_input",
)

# Read-only / exploration tools — no file mutation.  Matches zero-code EXPLORE_TOOLS.
EXPLORE_TOOL_NAMES: tuple[str, ...] = (
    "bash",
    "read_file",
    "glob",
    "grep",
    "load_skill",
    "request_user_input",
)

# Full set — everything including todo and background tasks.
FULL_TOOL_NAMES: tuple[str, ...] = tuple(d.name for d in DEFAULT_TOOL_DEFINITIONS)

TOOL_PRESETS: dict[str, tuple[str, ...]] = {
    "core": CORE_TOOL_NAMES,
    "explore": EXPLORE_TOOL_NAMES,
    "full": FULL_TOOL_NAMES,
}


def get_tool_definitions(
    *,
    preset: str | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[ToolDefinition]:
    """Return a filtered list of default tool definitions.

    Priority: *include* > *preset* > all.
    *exclude* is applied last regardless.

    Args:
        preset: One of "core", "explore", "full".
        include: Explicit tool names to include (ignores preset).
        exclude: Tool names to drop from the result.
    """
    if include is not None:
        names = list(include)
    elif preset is not None:
        names_tuple = TOOL_PRESETS.get(preset)
        if names_tuple is None:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {', '.join(TOOL_PRESETS)}")
        names = list(names_tuple)
    else:
        names = [d.name for d in DEFAULT_TOOL_DEFINITIONS]

    if exclude:
        exclude_set = set(exclude)
        names = [n for n in names if n not in exclude_set]

    result: list[ToolDefinition] = []
    for name in names:
        defn = DEFAULT_TOOL_DEFINITIONS_MAP.get(name)
        if defn is None:
            raise ValueError(f"Unknown default tool: '{name}'")
        result.append(defn)
    return result
