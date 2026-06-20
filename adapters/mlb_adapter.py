from typing import Dict, Any, List, Optional
import logging
import httpx
from datetime import datetime
from .base_adapter import BaseSportAdapter

logger = logging.getLogger(__name__)

LINEUP_STATUS_URL = "https://edgelab.julianmusichhtx.workers.dev/lineup-status"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY")  # Make sure this is set


class MLBAdapter(BaseSportAdapter):
    sport_key = "mlb"
    sport_label = "MLB"

    def enrich_prop(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Your existing enrichment logic from before
        try:
            enriched = prop.copy()
            player_name = prop.get("player_name", "")
            stat_display = prop.get("stat_display", "").lower()

            enriched["sport"] = "mlb"
            enriched["player_name_clean"] = player_name.lower().strip()
            enriched["is_pitcher"] = self._is_pitcher_prop(stat_display)

            lineup = self._get_lineup_status(player_name)
            if lineup:
                enriched["lineup_status"] = lineup
                enriched["is_confirmed"] = lineup.get("status") == "CONFIRMED"

            game_info = self._get_mlb_game_info(player_name)
            if game_info:
                enriched.update(game_info)

            sportradar_stats = self._get_sportradar_player_stats(player_name, enriched["is_pitcher"])
            if sportradar_stats:
                enriched.update(sportradar_stats)

            return enriched
        except Exception as e:
            logger.warning(f"MLB enrich_prop failed for {player_name}: {e}")
            return None

    def enrich_props(self, props: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.enrich_prop(p) for p in props if p]

    def get_player_stats(self, player_name: str) -> Dict[str, Any]:
        return self._get_sportradar_player_stats(player_name, False)

    def get_todays_games(self) -> List[Dict[str, Any]]:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{MLB_SCHEDULE_URL}?sportId=1&date={today}"
            with httpx.Client(timeout=5.0) as client:
                res = client.get(url)
                return res.json().get("dates", [{}])[0].get("games", [])
        except:
            return []

    def _is_pitcher_prop(self, stat_display: str) -> bool:
        keywords = ["strikeout", "pitch", "earned run", "hit allowed", "walk allowed"]
        return any(kw in stat_display for kw in keywords)

    def _get_lineup_status(self, player_name: str) -> Optional[Dict]:
        try:
            with httpx.Client(timeout=4.0) as client:
                url = f"{LINEUP_STATUS_URL}?player={player_name}"
                res = client.get(url)
                return res.json() if res.status_code == 200 else None
        except:
            return None

    def _get_mlb_game_info(self, player_name: str) -> Optional[Dict]:
        # Same as before
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{MLB_SCHEDULE_URL}?sportId=1&date={today}&hydrate=team,venue"
            with httpx.Client(timeout=5.0) as client:
                res = client.get(url)
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

    def _get_sportradar_player_stats(self, player_name: str, is_pitcher: bool) -> Dict:
        if not SPORTRADAR_API_KEY:
            return {}
        # Same Sportradar logic as before...
        try:
            search_url = f"https://api.sportradar.com/mlb/trial/v7/en/players/search.json?api_key={SPORTRADAR_API_KEY}&name={player_name}"
            with httpx.Client(timeout=6.0) as client:
                search_res = client.get(search_url)
                if search_res.status_code != 200:
                    return {}
                players = search_res.json().get("players", [])
                if not players:
                    return {}
                player_id = players[0].get("id")

                profile_url = f"https://api.sportradar.com/mlb/trial/v7/en/players/{player_id}/profile.json?api_key={SPORTRADAR_API_KEY}"
                profile_res = client.get(profile_url)
                if profile_res.status_code != 200:
                    return {}

                profile = profile_res.json()
                stats = {"sportradar_player_id": player_id}

                if is_pitcher:
                    season_stats = profile.get("seasons", [{}])[-1].get("team", {}).get("statistics", {}).get("pitching", {})
                    stats["season_era"] = season_stats.get("era")
                    stats["season_whip"] = season_stats.get("whip")
                else:
                    season_stats = profile.get("seasons", [{}])[-1].get("team", {}).get("statistics", {}).get("hitting", {})
                    stats["season_avg"] = season_stats.get("avg")
                    stats["season_ops"] = season_stats.get("ops")

                return stats
        except Exception as e:
            logger.debug(f"Sportradar error: {e}")
            return {}