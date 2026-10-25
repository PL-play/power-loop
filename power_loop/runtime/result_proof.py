"""Trusted, read-only proof for publishing an original execution result.

This is not tool/model authority. An implementation must bind the exact request,
original generation and immutable actual-exit evidence in the supplied transaction.
It cannot grant dispatch, a new lease, downstream completion or billing settlement.
"""
from abc import ABC, abstractmethod


class ResultPublicationProof(ABC):
    @abstractmethod
    async def validate(self, manager, transaction, execution_row, requested_result):
        """Return exactly True only for this original, already exited execution."""
        raise NotImplementedError()
