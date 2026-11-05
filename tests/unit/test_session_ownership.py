"""The bundled single-process implementation passes the application contract."""
import unittest

import pytest

from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator
from power_loop.testing.session_ownership import SessionOwnershipContractTests

pytestmark = pytest.mark.unit


class TestMemorySessionOwnership(SessionOwnershipContractTests, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.shared = MemorySessionOwnershipCoordinator()

    async def coordinator(self):
        return self.shared

    async def asyncTearDown(self):
        self.assertFalse(self.shared._leases)
