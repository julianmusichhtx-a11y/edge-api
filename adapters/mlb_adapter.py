import os
import time
import logging
from typing import Dict, Any, List, Optional
import httpx
from datetime import datetime
from .base_adapter import BaseSportAdapter

logger = logging.getLogger(__name__)

LINEUP_STATUS_URL = "https://edgelab.julianmusichhtx.workers.dev/lineup-status"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

# Caching
_cache: Dict[str, Any] = {}
_cache_ttl: Dict[str, float] = {}

def _get_cached(key: str, ttl_seconds: int = 600):
    if key in _cache and (time.time() - _cache_ttl.get(key, 0)) < ttl_seconds:
        return _cache[key]
    return None

def _set_cache(key: str, value: Any):
    _cache[key] = value
    _cache_ttl[key] = time.time()


class MLBAdapter(BaseSportAdapter):
    sport_key = "mlb"
    sport_label = "MLB"

    def enrich_prop(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            enriched = dict(prop)
            player_name = (prop.get("player_name") or "").strip()
            stat_display = (prop.get("stat_display") or "").lower()

            enriched["sport"] = "mlb"
            enriched["player_name_clean"] = player_name.lower()
            enriched["is_pitcher"] = self._is_pitcher_prop(stat_display)

            # Cached Lineup
            lineup_key = f"lineup:{player_name.lower()}"
            lineup = _get_cached(lineup_key)
            if lineup is None:
                lineup = self._fetch_lineup_status(player_name)
                _set_cache(lineup_key, lineup or {})
            if lineup:
                enriched["lineup_status"] = lineup
                enriched["is_confirmed"] = lineup.get("status") == "CONFIRMED"

            # Cached Game Info
            game_key = f"game:{datetime.now().strftime('%Y-%m-%d')}"
            game_info = _get_cached(game_key)
            if game_info is None:
                game_info = self._fetch_mlb_game_info(player_name)
                _set_cache(game_key, game_info or {})
            if game_info:
                enriched.update(game_info)

            enriched["verdict"] = "PICK" if enriched.get("is_confirmed") else "SKIP"
            enriched["reason"] = "Confirmed in lineup" if enriched.get("is_confirmed") else "Lineup not confirmed"

            return enriched
        except Exception as e:
            logger.warning(f"Enrich error for {player_name}: {e}")
            return prop

    def enrich_props(self, props: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for i in range(0, len(props), 400):
            batch = props[i:i+400]
            results.extend([self.enrich_prop(p) for p in batch])
        return results

    def get_player_stats(self, player_name: str) -> Dict[str, Any]:
        # Placeholder - can be expanded later with Sportradar or other sources
        return {}

    def get_todays_games(self) -> List[Dict[str, Any]]:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{MLB_SCHEDULE_URL}?sportId=1&date={today}"
            with httpx.Client(timeout=4) as client:
                data = client.get(url).json()
                return data.get("dates", [{}])[0].get("games", [])
        except:
            return []

    def _is_pitcher_prop(self, stat: str) -> bool:
        return any(k in stat for k in ["strikeout", "pitch", "era", "whip"])

    def _fetch_lineup_status(self, player: str) -> Optional[Dict]:
        try:
            with httpx.Client(timeout=2.5) as client:
                res = client.get(f"{LINEUP_STATUS_URL}?player={player}")
                return res.json() if res.status_code == 200 else None
        except:
            return None

    def _fetch_mlb_game_info(self, player: str) -> Optional[Dict]:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{MLB_SCHEDULE_URL}?sportId=1&date={today}&hydrate=team,venue"
            with httpx.Client(timeout=3) as client:
                data = client.get(url).json()
                games = data.get("dates", [{}])[0].get("games", [])
                for g in games:
                    home = g.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
                    away = g.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
                    if player.lower() in (home + away).lower():
                        return {
                            "game_time": g.get("gameDate"),
                            "home_team": home,
                            "away_team": away,
                        }
            return None
        except:
            return None