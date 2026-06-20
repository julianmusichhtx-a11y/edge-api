"""
Abstract base for all sport adapters.
Every adapter must return a standardized player_stats dict.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import asyncio

class BaseAdapter(ABC):
    @abstractmethod
    async def get_player_stats(
        self,
        player_name: str,
        stat_type: str,
        opponent: Optional[str] = None,
        game_date: Optional[str] = None
    ) -> Tuple[Dict[str, Any], int]:
        """
        Returns (player_stats_dict, api_calls_made)
        player_stats_dict should contain at minimum:
        {
            "season_avg": float,
            "last_5_avg": float,
            "last_10_avg": float,
            "vs_opponent_avg": float or None,
            "usage_or_minutes": float or None,
            "recent_trend": str,  # "hot", "cold", "neutral"
            "matchup_note": str,
            "rest_days": int or None,
            "injury_status": str or None,
            # Sport specific extras
        }
        """
        pass

    async def _rate_limited_get(self, client, url, headers=None):
        """Helper with basic rate limit awareness (override in subclasses)."""
        await asyncio.sleep(0.8)  # Conservative for trial keys
        resp = await client.get(url, headers=headers or {})
        return resp