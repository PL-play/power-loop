"""Versioned raw message references; message IDs are not session seq coordinates."""
from __future__ import annotations

from dataclasses import dataclass


def integer_id(value: int | str, name: str = "message_id") -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} must be an integer or decimal string")
    if isinstance(value, str) and (not value.isascii() or not value.isdecimal()):
        raise ValueError(f"{name} must be a decimal string")
    number = int(value)
    if not 1 <= number <= 2**63 - 1:
        raise ValueError(f"{name} out of range")
    return number


@dataclass(frozen=True)
class MessageRef:
    store_id: str | None
    session_id: str
    id: int | None
    seq: int
    role: str
    tool_call_id: str | None = None
    send_index: int | None = None
    round_index: int | None = None

    def __post_init__(self):
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("invalid session_id")
        if not isinstance(self.seq, int) or isinstance(self.seq, bool):
            raise ValueError("seq must be an integer")
        integer_id(self.seq, "seq")
        if self.id is not None:
            if not isinstance(self.id, int) or isinstance(self.id, bool):
                raise ValueError("message_id must be an integer")
            integer_id(self.id)
        # Zero can occur in imported legacy sends; preserve it instead of
        # claiming a reconstructed positive send identity.
        for name, value, minimum in (("send_index", self.send_index, 0), ("round_index", self.round_index, 0)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
                raise ValueError(f"invalid {name}")

    def to_dict(self) -> dict:
        return {"version": 1, "store_id": self.store_id, "session_id": self.session_id,
                "id": str(self.id) if self.id is not None else None, "seq": str(self.seq),
                "role": self.role, "tool_call_id": self.tool_call_id,
                "send_index": str(self.send_index) if self.send_index is not None else None,
                "round_index": self.round_index,
                "id_status": "available" if self.id is not None else "unsupported"}
