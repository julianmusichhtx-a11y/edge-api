from typing import Dict, Any, Optional
import logging
import os
import requests
from datetime import datetime
from .base_adapter import BaseSportAdapter

logger = logging.getLogger(__name__)

LINEUP_STATUS_URL = "https://edgelab.julianmusichhtx.workers.dev/lineup-status"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY")  # Make sure this is set in Railway


class MLBAdapter(BaseSportAdapter):
    """
    Best version of MLBAdapter.
    Uses: Lineup Status + MLB Stats API + Sportradar
    """

    def enrich_prop(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            enriched = prop.copy()
            player_name = prop.get("player_name", "")
            stat_display = prop.get("stat_display", "").lower()

            enriched["sport"] = "mlb"
            enriched["player_name_clean"] = player_name.lower().strip()
            enriched["is_pitcher"] = self._is_pitcher_prop(stat_display)

            # 1. Lineup Status
            lineup = self._get_lineup_status(player_name)
            if lineup:
                enriched["lineup_status"] = lineup
                enriched["is_confirmed"] = lineup.get("status") == "CONFIRMED"
                enriched["batting_order"] = lineup.get("battingOrder")

            # 2. MLB Stats API (Game context)
            game_info = self._get_mlb_game_info(player_name)
            if game_info:
                enriched.update(game_info)

            # 3. Sportradar - Player Stats (Most Important)
            sportradar_stats = self._get_sportradar_player_stats(player_name, enriched["is_pitcher"])
            if sportradar_stats:
                enriched.update(sportradar_stats)

            return enriched

        except Exception as e:
            logger.warning(f"MLB enrichment failed for {player_name}: {e}")
            return None

    def _is_pitcher_prop(self, stat_display: str) -> bool:
        keywords = ["strikeout", "pitch", "earned run", "hit allowed", "walk allowed"]
        return any(kw in stat_display for kw in keywords)

    def _get_lineup_status(self, player_name: str) -> Optional[Dict]:
        try:
            url = f"{LINEUP_STATUS_URL}?player={player_name}"
            res = requests.get(url, timeout=4)
            return res.json() if res.status_code == 200 else None
        except:
            return None

    def _get_mlb_game_info(self, player_name: str) -> Optional[Dict]:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{MLB_SCHEDULE_URL}?sportId=1&date={today}&hydrate=team,venue"
            res = requests.get(url, timeout=5)
            if res.status_code != 200:
                return None

            games = res.json().get("dates", [{}])[0].get("games", [])
            for game in games:
                home = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
                away = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
                if player_name.lower() in (home + away).lower():
                    return {
                        "game_time": game.get("gameDate"),
                        "home_team": home,
                        "away_team": away,
                        "venue": game.get("venue", {}).get("name"),
                    }
            return None
        except:
            return None

    # ============================================================
    # SPORTSRADAR INTEGRATION (Real calls)
    # ============================================================

    def _get_sportradar_player_stats(self, player_name: str, is_pitcher: bool) -> Dict:
        """
        Fetches player stats from Sportradar.
        Returns recent form + season averages when available.
        """
        if not SPORTRADAR_API_KEY:
            logger.warning("SPORTRADAR_API_KEY not set. Skipping Sportradar calls.")
            return {}

        try:
            # Step 1: Search for player ID (you can cache this later)
            search_url = f"https://api.sportradar.com/mlb/trial/v7/en/players/search.json?api_key={SPORTRADAR_API_KEY}&name={player_name}"
            search_res = requests.get(search_url, timeout=6)

            if search_res.status_code != 200:
                return {}

            players = search_res.json().get("players", [])
            if not players:
                return {}

            player_id = players[0].get("id")

            # Step 2: Get player profile + stats
            profile_url = f"https://api.sportradar.com/mlb/trial/v7/en/players/{player_id}/profile.json?api_key={SPORTRADAR_API_KEY}"
            profile_res = requests.get(profile_url, timeout=8)

            if profile_res.status_code != 200:
                return {}

            profile = profile_res.json()

            stats = {}

            if is_pitcher:
                # Pitcher stats
                season_stats = profile.get("seasons", [{}])[-1].get("team", {}).get("statistics", {}).get("pitching", {})
                stats["season_era"] = season_stats.get("era")
                stats["season_whip"] = season_stats.get("whip")
                stats["season_k_per_9"] = season_stats.get("strikeouts_per_9")
            else:
                # Hitter stats
                season_stats = profile.get("seasons", [{}])[-1].get("team", {}).get("statistics", {}).get("hitting", {})
                stats["season_avg"] = season_stats.get("avg")
                stats["season_ops"] = season_stats.get("ops")
                stats["season_hr"] = season_stats.get("home_runs")

            # You can expand this with game logs for "recent form" later
            stats["sportradar_player_id"] = player_id

            return stats

        except Exception as e:
            logger.debug(f"Sportradar enrichment failed for {player_name}: {e}")
            return {}