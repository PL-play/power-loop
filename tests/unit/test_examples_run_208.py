"""Execute current examples with their behavior assertions and actual local ports.

18 demonstrates live providers; remote format/summary/structured behavior is in
real/test_delivery_203.py. Docker and MCP are explicit required dependencies for
their examples; failures cannot turn into skips. The old per-leaf private-store
SubprocessExecutor example is retired; current workflow process execution is
covered by real/test_real_subprocess.py and unit runtime/workflow tests.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = [p for p in sorted((ROOT/'examples').glob('[0-9]*.py')) if not p.name.startswith('18_')]
SCRIPTS += sorted((ROOT/'examples/advanced_runtime').glob('[0-9]*.py'))
SCRIPTS.append(ROOT/'examples/design208_timers.py')

@pytest.mark.unit
@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: str(p.relative_to(ROOT/'examples')))
def test_example_executes_current_runtime(script):
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip(), script
