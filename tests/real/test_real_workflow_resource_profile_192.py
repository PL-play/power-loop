"""Real model creates a v2-bound native workflow through the library facade."""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]
