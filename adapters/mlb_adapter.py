"""
MLB Adapter using the free MLB Stats API (statsapi.mlb.com)
No API key required. Great for testing.
"""

import httpx
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import asyncio
from .base_adapter import BaseAdapter
from cache.cache_manager import cache

MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"

class MLBAdapter(BaseAdapter):
    async def get_player_stats(
        self,
        player_name: str,
        stat_type: str,
        opponent: Optional[str] = None,
        game_date: Optional[str] = None
    ) -> Tuple[Dict[str, Any], int]:
        cache_key = f"mlb:{player_name.lower()}:{stat_type}:{opponent or 'any'}"
        cached = cache.get(cache_key)
        if cached:
            return cached, 0

        async with httpx.AsyncClient(timeout=15.0) as client:
            # 1. Find player ID
            search_url = f"{MLB_STATS_BASE}/people/search?names={player_name}&sportId=1"
            try:
                r = await client.get(search_url)
                r.raise_for_status()
                people = r.json().get("people", [])
                if not people:
                    return {"error": "Player not found"}, 1
                player_id = people[0]["id"]
            except Exception as e:
                return {"error": f"Player search failed: {str(e)}"}, 1

            # 2. Get season stats (2025 or current)
            season = "2025" if datetime.now().year >= 2025 else "2024"
            stats_url = f"{MLB_STATS_BASE}/people/{player_id}/stats?stats=season&season={season}&sportId=1"
            try:
                r = await client.get(stats_url)
                r.raise_for_status()
                stats_data = r.json()
                season_stats = stats_data.get("stats", [{}])[0].get("splits", [{}])[0].get("stat", {})
            except:
                season_stats = {}

            # Map stat_type to MLB stat
            stat_map = {
                "Hits": "hits", "Home Runs": "homeRuns", "RBIs": "rbi",
                "Strikeouts": "strikeOuts", "Total Bases": "totalBases",
                "Runs": "runs", "Walks": "baseOnBalls"
            }
            mlb_stat = stat_map.get(stat_type, "hits")

            season_avg = float(season_stats.get(mlb_stat, 0) or 0)

            # For simplicity in MVP: use season avg as proxy for recent (real impl would fetch game logs)
            last_5 = round(season_avg * 0.95, 1)  # Slight regression to mean for demo
            last_10 = round(season_avg * 0.98, 1)

            result = {
                "season_avg": season_avg,
                "last_5_avg": last_5,
                "last_10_avg": last_10,
                "vs_opponent_avg": None,  # Would require game log filtering
                "usage_or_minutes": None,
                "recent_trend": "neutral",
                "matchup_note": f"vs {opponent}" if opponent else "",
                "rest_days": None,
                "injury_status": None,
                "source": "MLB Stats API (public)",
                "note": "MVP version - full game log analysis coming in next iteration"
            }

            cache.set(cache_key, result, ttl=3600 * 6)  # 6 hours
            return result, 2  # 2 calls made
