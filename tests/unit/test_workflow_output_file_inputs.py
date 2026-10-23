"""output-file 认 ${{ inputs.* }}：同一张图同时开几份 run 时产出文件带编号，不互相覆盖（一个包一个 workflow）。"""

from __future__ import annotations

import json

import pytest

from power_loop.workflow.yaml import WorkflowYamlError, compile_workflow_document

DOC = """name: t
inputs:
  slot:
jobs:
  author:
    prompt: 你是作者。
    output-file: outputs/pack-${{ inputs.slot }}-author.md
  review:
    needs: [author]
    prompt: 你是核对者。
    output-file: outputs/pack-${{ inputs.slot }}.md
    with:
      author: ${{ needs.author.output }}
"""


def _compile(doc: str, args: dict) -> str:
    spec = compile_workflow_document(doc, args, load_file=lambda _s, _p: None, read_workspace=None, where="t.yaml")
    return json.dumps(spec.to_dict(), ensure_ascii=False)


def test_output_file_takes_inputs_and_flows_downstream():
    text = _compile(DOC, {"slot": "g1"})
    assert "outputs/pack-g1-author.md" in text and "outputs/pack-g1.md" in text
    assert "@@FILEREF:outputs/pack-g1-author.md@@" in text
    assert "outputs/author.md" not in text and "outputs/review.md" not in text
    assert "outputs/pack-g2" in _compile(DOC, {"slot": "g2"})


def test_output_file_rejects_non_input_expressions():
    bad = DOC.replace("outputs/pack-${{ inputs.slot }}.md", "outputs/${{ needs.author.output }}.md")
    with pytest.raises(WorkflowYamlError):
        _compile(bad, {"slot": "g1"})
