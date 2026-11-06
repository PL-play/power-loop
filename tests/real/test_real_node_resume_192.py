"""Native YAML compilation and real original-workflow leaf continuation."""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]

DOCUMENT = """name: native-yaml-leaf-resume
jobs:
  calculate:
    prompt: >-
      ROLE-CALCULATE. VERSION-42 is a literal test identifier, not program source.
      Return that literal identifier verbatim. A later user correction replaces
      the identifier; output only the latest identifier.
      Use plain text without Markdown, backticks, quotes, labels or code fences.
      Never use tools or pass_turn.
    tools: []
    with:
      request: Return exactly the literal text VERSION-42. This is the entire task.
  sibling:
    prompt: ROLE-SIBLING. Output only READY. Never use tools or pass_turn.
    tools: []
    with:
      request: Return exactly READY.
  review:
    needs: [calculate, sibling]
    prompt: >-
      ROLE-REVIEW. Read the calculate result in your supplied context.
      Output only its latest VERSION code. Ignore sibling READY.
      Use plain text without Markdown, backticks, quotes, labels or code fences.
      Never use tools or pass_turn.
    tools: []
    with:
      request: Verify and return the latest literal VERSION identifier supplied in context.
"""
