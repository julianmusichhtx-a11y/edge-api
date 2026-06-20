"""
Simple in-memory TTL cache.
For production on Railway, consider Redis (Upstash or Railway Redis) for persistence across deploys.
"""

import time
from typing import Any, Optional

class TTLCache:
    def __init__(self):
        self._store: Dict[str, tuple] = {}  # key -> (value, expiry)

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        value, expiry = self._store[key]
        if time.time() > expiry:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 3600):
        self._store[key] = (value, time.time() + ttl)

    def clear(self):
        self._store.clear()

cache = TTLCache()
