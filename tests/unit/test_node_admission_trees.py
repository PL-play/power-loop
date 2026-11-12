"""Original node continuation must not inspect another tree under its writer."""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
