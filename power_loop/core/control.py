"""Host-controlled stops at lifecycle boundaries, separate from business errors."""


class ControlledLoopStop(Exception):
    """A host hook requests a terminal result without another model request."""

    def __init__(self, reason: str, *, message: str = "", status: str = "budget_exceeded"):
        if not reason or status not in ("budget_exceeded", "cancelled", "degraded"):
            raise ValueError("invalid controlled loop stop")
        self.reason = reason
        self.message = message
        self.status = status
        super().__init__(reason)

