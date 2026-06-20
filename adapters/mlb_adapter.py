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
SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY")

# ====================== CACHING ======================
_cache: Dict[str, Any] = {}
_cache_ttl: Dict[str, float] = {}

def _get_cached(key: str, ttl: int = 600):
    if key in _cache and (time.time() - _cache_ttl.get(key, 0)) < ttl:
        return _cache[key]
    return None

def _set_cache(key: str, value: Any):
    _cache[key] = value
    _cache_ttl[key] = time.time()

# ====================== MLB ADAPTER ======================
class MLBAdapter(BaseSportAdapter):
    sport_key = "mlb"
    sport_label = "MLB"

    def enrich_prop(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            enriched = dict(prop)  # copy
            player_name = prop.get("player_name", "")
            stat_display = (prop.get("stat_display") or "").lower()

            enriched["sport"] = "mlb"
            enriched["player_name_clean"] = player_name.lower().strip()

            # Skip expensive Sportradar for very common/low-value props
            skip_sportradar = any(x in stat_display for x in ["hits", "total bases", "rbi", "runs"]) or not stat_display

            enriched["is_pitcher"] = self._is_pitcher_prop(stat_display)

            # Fast calls
            lineup = self._get_lineup_status(player_name)
            if lineup:
                enriched["lineup_status"] = lineup
                enriched["is_confirmed"] = lineup.get("status") == "CONFIRMED"

            game_info = self._get_mlb_game_info(player_name)
            if game_info:
                enriched.update(game_info)

            # Sportradar only when useful + cached
            if not skip_sportradar:
                cache_key = f"sr:{player_name.lower()}"
                cached = _get_cached(cache_key)
                if cached:
                    enriched.update(cached)
                else:
                    stats = self._get_sportradar_safe(player_name, enriched["is_pitcher"])
                    if stats:
                        _set_cache(cache_key, stats)
                        enriched.update(stats)

            return enriched

        except Exception as e:
            logger.warning(f"Enrich failed for {player_name}: {e}")
            return prop

    def enrich_props(self, props: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Process in smaller batches to avoid timeouts
        results = []
        for i in range(0, len(props), 200):
            batch = props[i:i+200]
            results.extend([self.enrich_prop(p) for p in batch])
        return results

    def get_player_stats(self, player_name: str) -> Dict[str, Any]:
        return self._get_sportradar_safe(player_name, False)

    def get_todays_games(self) -> List[Dict[str, Any]]:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{MLB_SCHEDULE_URL}?sportId=1&date={today}"
            with httpx.Client(timeout=5) as client:
                return client.get(url).json().get("dates", [{}])[0].get("games", [])
        except:
            return []

    def _is_pitcher_prop(self, stat: str) -> bool:
        return any(k in stat for k in ["strikeout", "pitch", "era", "whip", "earned run"])

    def _get_lineup_status(self, player: str) -> Optional[Dict]:
        try:
            with httpx.Client(timeout=3) as client:
                res = client.get(f"{LINEUP_STATUS_URL}?player={player}")
                return res.json() if res.status_code == 200 else None
        except:
            return None

    def _get_mlb_game_info(self, player: str) -> Optional[Dict]:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{MLB_SCHEDULE_URL}?sportId=1&date={today}&hydrate=team,venue"
            with httpx.Client(timeout=4) as client:
                games = client.get(url).json().get("dates", [{}])[0].get("games", [])
                for g in games:
                    teams = (g.get("teams", {}).get("home", {}).get("team", {}).get("name", "") +
                             g.get("teams", {}).get("away", {}).get("team", {}).get("name", ""))
                    if player.lower() in teams.lower():
                        return {
                            "game_time": g.get("gameDate"),
                            "home_team": g.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                            "away_team": g.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                            "venue": g.get("venue", {}).get("name"),
                        }
            return None
        except:
            return None

    def _get_sportradar_safe(self, player_name: str, is_pitcher: bool) -> Dict:
        if not SPORTRADAR_API_KEY:
            return {}
        try:
            search_url = f"https://api.sportradar.com/mlb/trial/v7/en/players/search.json?api_key={SPORTRADAR_API_KEY}&name={player_name}"
            with httpx.Client(timeout=5) as client:
                search = client.get(search_url)
                if search.status_code != 200:
                    return {}
                players = search.json().get("players", [])
                if not players:
                    return {}
                pid = players[0]["id"]

                profile_url = f"https://api.sportradar.com/mlb/trial/v7/en/players/{pid}/profile.json?api_key={SPORTRADAR_API_KEY}"
                profile = client.get(profile_url).json()

                stats = {"sportradar_player_id": pid}
                if is_pitcher:
                    s = profile.get("seasons", [{}])[-1].get("team", {}).get("statistics", {}).get("pitching", {})
                    stats.update({"season_era": s.get("era"), "season_whip": s.get("whip")})
                else:
                    s = profile.get("seasons", [{}])[-1].get("team", {}).get("statistics", {}).get("hitting", {})
                    stats.update({"season_avg": s.get("avg"), "season_ops": s.get("ops")})
                return stats
        except:
            return {}