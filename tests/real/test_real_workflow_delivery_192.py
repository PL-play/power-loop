"""Real provider evidence survives failed workflow projection without reexecution."""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]
