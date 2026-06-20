"""
Sport-specific adapters.
For sports requiring keys (Sportradar, SportsDataIO), they check for key presence.
Stubs return useful defaults so the system works end-to-end even without all keys.
Full implementations follow the same pattern as MLBAdapter.
"""

from typing import Dict, Any, Tuple, Optional
import httpx
from .base_adapter import BaseAdapter
from cache.cache_manager import cache
from config import settings
import asyncio

class NBAAdapter(BaseAdapter):
    async def get_player_stats(self, player_name: str, stat_type: str, opponent: Optional[str] = None, game_date: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        key = f"nba:{player_name.lower()}:{stat_type}"
        if cached := cache.get(key):
            return cached, 0

        # TODO: Implement with Sportradar or SportsDataIO when key present
        # For now: return smart defaults based on typical NBA usage
        result = {
            "season_avg": 18.5,  # placeholder - real would query
            "last_5_avg": 20.2,
            "last_10_avg": 19.1,
            "vs_opponent_avg": 17.8 if opponent else None,
            "usage_or_minutes": 32.4,
            "recent_trend": "hot" if "Points" in stat_type else "neutral",
            "matchup_note": f"vs {opponent} (top-10 defense)" if opponent else "",
            "rest_days": 1,
            "injury_status": None,
            "source": "stub (add Sportradar key for live data)",
            "note": "Replace this stub with real API call in production"
        }
        cache.set(key, result, ttl=1800)
        return result, 0

class WNBAAdapter(NBAAdapter):
    # WNBA is very similar to NBA structurally
    pass

class NFLAdapter(BaseAdapter):
    async def get_player_stats(self, player_name: str, stat_type: str, opponent: Optional[str] = None, game_date: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        key = f"nfl:{player_name.lower()}:{stat_type}"
        if cached := cache.get(key):
            return cached, 0
        result = {
            "season_avg": 65.0,  # e.g. pass yards or rush yards typical
            "last_5_avg": 72.0,
            "last_10_avg": 68.5,
            "vs_opponent_avg": None,
            "usage_or_minutes": None,
            "recent_trend": "neutral",
            "matchup_note": "",
            "rest_days": None,
            "injury_status": None,
            "source": "stub",
            "note": "NFL adapter ready for Sportradar or SportsDataIO integration"
        }
        cache.set(key, result, ttl=3600)
        return result, 0

class NHLAdapter(BaseAdapter):
    async def get_player_stats(self, player_name: str, stat_type: str, opponent: Optional[str] = None, game_date: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        key = f"nhl:{player_name.lower()}:{stat_type}"
        if cached := cache.get(key):
            return cached, 0
        result = {
            "season_avg": 2.8,
            "last_5_avg": 3.1,
            "last_10_avg": 2.9,
            "vs_opponent_avg": None,
            "usage_or_minutes": 18.5,  # TOI
            "recent_trend": "neutral",
            "matchup_note": "",
            "rest_days": None,
            "injury_status": None,
            "source": "stub",
            "note": "NHL adapter - high variance, rest/back-to-back critical"
        }
        cache.set(key, result, ttl=3600)
        return result, 0

class SoccerAdapter(BaseAdapter):
    async def get_player_stats(self, player_name: str, stat_type: str, opponent: Optional[str] = None, game_date: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        key = f"soccer:{player_name.lower()}:{stat_type}"
        if cached := cache.get(key):
            return cached, 0
        result = {
            "season_avg": 0.45,  # goals or shots typical
            "last_5_avg": 0.6,
            "last_10_avg": 0.5,
            "vs_opponent_avg": None,
            "usage_or_minutes": None,
            "recent_trend": "neutral",
            "matchup_note": f"vs {opponent}",
            "rest_days": None,
            "injury_status": None,
            "source": "stub",
            "note": "Soccer - form and H2H very important"
        }
        cache.set(key, result, ttl=3600)
        return result, 0
