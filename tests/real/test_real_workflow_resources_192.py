"""Real provider behind a trusted registry; not a native restoration test yet."""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]
