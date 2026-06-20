"""
Async rate limiter for external APIs.
Simple token bucket style. For high volume, use aiolimiter or Redis-based.
"""

import asyncio
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = defaultdict(float)

    async def wait(self, key: str = "default"):
        now = time.time()
        elapsed = now - self.last_call[key]
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_call[key] = time.time()

# Global instances
sportradar_limiter = RateLimiter(0.9)  # ~1 per sec
theodds_limiter = RateLimiter(0.15)
