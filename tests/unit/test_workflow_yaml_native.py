"""Native YAML stays optional and carries executable capability limits."""
from __future__ import annotations

import subprocess
import sys

import pytest

from power_loop.workflow import WorkflowSpec
from power_loop.workflow.yaml import WorkflowYamlError, compile_workflow_yaml

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("tools", ["[]", "[read_file]", "[read_file, read_file]"])
def test_native_inline_tools_are_real_ir_limits_including_empty_list(tools):
    spec = WorkflowSpec.from_yaml(f"name: native\njobs:\n  author:\n    prompt: Write nothing.\n    tools: {tools}\n")
    expected = [] if tools == "[]" else ["read_file"]
    assert spec.root.spec.tools == expected
    assert spec.root.spec.metadata["_action_tools"] == expected


def test_native_inputs_require_explicit_values_and_resolve_before_runtime():
    document = "name: native\ninputs:\n  requirement:\njobs:\n  a:\n    prompt: ${{ inputs.requirement }}\n"
    with pytest.raises(WorkflowYamlError, match="缺必填参数"):
        WorkflowSpec.from_yaml(document)
    spec = WorkflowSpec.from_yaml(document, args={"requirement": "exact task"})
    assert spec.root.spec.system_prompt == "exact task"


def test_native_action_empty_tools_are_preserved():
    files = {"a/action.yaml": "charter: role.md\ntools: []", "a/role.md": "Do the task."}
    spec = compile_workflow_yaml("name: native\njobs:\n  x:\n    uses: skill:sample/a\n", load_file=lambda _, path: files.get(path))
    assert spec.root.spec.tools == [] and spec.root.spec.metadata["_action_tools"] == []


def test_text_and_legacy_file_delivery_are_explicit_and_do_not_mix_instructions():
    document = "name: native\njobs:\n  a:\n    prompt: Answer the question.\n"
    text = WorkflowSpec.from_yaml(document)
    assert "output_file" not in text.root.spec.metadata and "write_file" not in text.root.input
    files = WorkflowSpec.from_yaml(document, output_mode="file")
    assert files.root.spec.metadata["output_file"] == "outputs/a.md"
    assert "write_file" in files.root.input


@pytest.mark.parametrize("declaration", ["output-file: outputs/custom.md", "with:\n      upstream: ${{ needs.a.output }}"])
def test_text_mode_rejects_file_delivery_references(declaration):
    document = f"name: native\njobs:\n  a:\n    prompt: First.\n  b:\n    needs: [a]\n    prompt: Second.\n    {declaration}\n"
    with pytest.raises(WorkflowYamlError, match="text output mode"):
        WorkflowSpec.from_yaml(document)


def test_unknown_delivery_mode_is_a_compile_error():
    with pytest.raises(WorkflowYamlError, match="output_mode"):
        WorkflowSpec.from_yaml("name: native\njobs:\n  a:\n    prompt: task\n", output_mode="unknown")


@pytest.mark.parametrize("document", [
    "[]", "name: x\njobs: []", "name: x\njobs: {}", "jobs:\n  a:\n    prompt: x",
    "name: x\nunknown: x\njobs:\n  a:\n    prompt: x",
    "name: x\nname: y\njobs:\n  a:\n    prompt: x",
    "name: x\ntimeout-minutes: false\njobs:\n  a:\n    prompt: x",
    "name: x\njobs:\n  a:\n    prompt: x\n    needs: [missing]",
    "name: x\njobs:\n  a:\n    prompt: x\n    needs: [a]",
    "name: x\njobs:\n  a:\n    prompt: x\n    tools: 42",
    "name: x\njobs:\n  a:\n    uses: skill:missing/a",
    "name: x\njobs:\n  a:\n    uses: file:missing.md",
    "!!python/object/apply:os.system ['should-never-execute']",
])
def test_invalid_native_documents_are_rejected_before_any_execution(document):
    with pytest.raises(WorkflowYamlError):
        WorkflowSpec.from_yaml(document)


def test_core_import_does_not_require_yaml_and_compilation_reports_extra():
    script = """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    level = kwargs.get('level', args[3] if len(args) > 3 else 0)
    if not level and (name == 'yaml' or name.startswith('yaml.')):
        raise ModuleNotFoundError('intentionally unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from power_loop.workflow import WorkflowSpec
assert WorkflowSpec.from_json({'name':'plain','root':{'type':'agent','id':'x','input':'x','spec':{'name':'x','system_prompt':'x'}}})
try:
    WorkflowSpec.from_yaml('name: test')
except ImportError as exc:
    assert 'power-loop[workflow]' in str(exc)
else:
    raise AssertionError('YAML extra silently available')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
