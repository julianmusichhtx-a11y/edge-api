"""
Async rate limiter that enforces per-API request limits.
Sportradar trial = 1 req/sec. MLB Stats API = unlimited.
"""
import asyncio
import time
from collections import defaultdict


class RateLimiter:
    """Simple per-API rate limiter using asyncio locks and timestamps."""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_request: dict[str, float] = defaultdict(float)
        self._daily_counts: dict[str, int] = defaultdict(int)
        self._daily_reset: dict[str, float] = defaultdict(float)

    async def acquire(self, api_name: str, min_interval: float = 1.1, daily_limit: int = 99999):
        """
        Wait until it's safe to make a request to this API.
        min_interval: minimum seconds between requests (e.g., 1.1 for Sportradar)
        daily_limit: max requests per day
        """
        async with self._locks[api_name]:
            now = time.time()

            # Reset daily counter if it's a new day
            if now - self._daily_reset.get(api_name, 0) > 86400:
                self._daily_counts[api_name] = 0
                self._daily_reset[api_name] = now

            # Check daily limit
            if self._daily_counts[api_name] >= daily_limit:
                raise QuotaExceededError(f"{api_name}: daily limit of {daily_limit} reached")

            # Wait for rate limit
            elapsed = now - self._last_request.get(api_name, 0)
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            self._last_request[api_name] = time.time()
            self._daily_counts[api_name] += 1

    def get_usage(self) -> dict:
        """Return current usage stats for all APIs."""
        return {
            api: {"used_today": count, "last_request": self._last_request.get(api, 0)}
            for api, count in self._daily_counts.items()
        }


class QuotaExceededError(Exception):
    pass


# Global singleton — shared across all requests
rate_limiter = RateLimiter()
