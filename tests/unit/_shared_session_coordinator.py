"""A file-backed coordinator only for same-host process contract tests."""
import fcntl
import json
import time
from contextlib import contextmanager

from power_loop.runtime.session_ownership import SessionOwnership, SessionOwnershipCoordinator


class SharedSessionCoordinator(SessionOwnershipCoordinator):
    def __init__(self, path):
        self.path = path

    @contextmanager
    def records(self):
        with open(self.path, 'a+') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.seek(0)
            data = json.loads(file.read() or '{}')
            yield data
            file.seek(0)
            file.truncate()
            json.dump(data, file)
            file.flush()

    async def acquire(self, key, *, owner_id, ttl_s, on_lost):
        now = time.monotonic()
        with self.records() as rows:
            old = rows.get(key)
            if old is not None and old['expires_at'] > now:
                return None
            lease = SessionOwnership(key, owner_id, now + ttl_s, on_lost)
            rows[key] = dict(token=lease.token, expires_at=lease.expires_at)
            return lease

    async def renew(self, lease, *, ttl_s):
        now = time.monotonic()
        with self.records() as rows:
            row = rows.get(lease.key)
            if row is None or row['token'] != lease.token or row['expires_at'] <= now:
                lease.lose()
                return False
            lease.expires_at = now + ttl_s
            row['expires_at'] = lease.expires_at
            return True

    async def release(self, lease):
        with self.records() as rows:
            if rows.get(lease.key, {}).get('token') == lease.token:
                del rows[lease.key]
        lease.released = True
