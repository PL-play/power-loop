"""Disable hidden SDK retries when the caller maintains a per-attempt ledger."""
from contextvars import ContextVar

single_attempt: ContextVar[bool] = ContextVar('llm_single_attempt', default=False)
