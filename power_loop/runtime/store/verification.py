"""Read verification boundaries for the current session owner."""
from contextvars import ContextVar
from functools import wraps

TRUST_OWN_WRITES = ContextVar("session_trust_own_writes", default=False)


def verify_stored(fn):
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        token = TRUST_OWN_WRITES.set(False)
        try:
            return await fn(*args, **kwargs)
        finally:
            TRUST_OWN_WRITES.reset(token)
    return wrapper
