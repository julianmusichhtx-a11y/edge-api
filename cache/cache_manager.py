"""
In-memory cache with TTL.
Prevents redundant API calls — completed game box scores never change,
player season stats only update after games, schedules refresh hourly.
"""
from cachetools import TTLCache
from config import CACHE_TTL


# Each cache type has its own TTL and max size
_caches = {
    "schedule":      TTLCache(maxsize=200, ttl=CACHE_TTL["schedule"]),
    "team_roster":   TTLCache(maxsize=100, ttl=CACHE_TTL["team_roster"]),
    "player_season": TTLCache(maxsize=500, ttl=CACHE_TTL["player_season"]),
    "game_boxscore": TTLCache(maxsize=500, ttl=CACHE_TTL["game_boxscore"]),
    "player_gamelog": TTLCache(maxsize=500, ttl=CACHE_TTL["player_gamelog"]),
}


def get(cache_type: str, key: str):
    """Get a value from cache. Returns None if not found or expired."""
    cache = _caches.get(cache_type)
    if cache is None:
        return None
    return cache.get(key)


def put(cache_type: str, key: str, value):
    """Store a value in cache with the configured TTL."""
    cache = _caches.get(cache_type)
    if cache is None:
        return
    cache[key] = value


def stats() -> dict:
    """Return cache hit/miss stats for monitoring."""
    return {
        name: {"size": len(cache), "maxsize": cache.maxsize, "ttl": cache.ttl}
        for name, cache in _caches.items()
    }
