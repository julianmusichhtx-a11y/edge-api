"""
MLB data adapter using the free MLB Stats API (statsapi.mlb.com).
No API key needed. No rate limits. This is the gold standard adapter.
"""
import httpx
from datetime import datetime, timedelta
from typing import Optional

from adapters.base_adapter import BaseSportAdapter, PlayerStats, GameInfo
from cache import cache_manager as cache
from utils.rate_limiter import rate_limiter

MLB_BASE = "https://statsapi.mlb.com/api/v1"

# Maps canonical stat keys to MLB game log field extractors
PITCHING_STATS = {
    "strikeouts":     lambda s: int(s.get("strikeOuts", 0)),
    "earned_runs":    lambda s: int(s.get("earnedRuns", 0)),
    "hits_allowed":   lambda s: int(s.get("hits", 0)),
    "walks_allowed":  lambda s: int(s.get("baseOnBalls", 0)),
    "pitching_outs":  lambda s: round(float(s.get("inningsPitched", "0")) * 3),
    "innings_pitched": lambda s: float(s.get("inningsPitched", "0")),
}

BATTING_STATS = {
    "hits":            lambda s: int(s.get("hits", 0)),
    "total_bases":     lambda s: int(s.get("totalBases", 0)),
    "home_runs":       lambda s: int(s.get("homeRuns", 0)),
    "runs":            lambda s: int(s.get("runs", 0)),
    "rbi":             lambda s: int(s.get("rbi", 0)),
    "stolen_bases":    lambda s: int(s.get("stolenBases", 0)),
    "walks":           lambda s: int(s.get("baseOnBalls", 0)),
    "singles":         lambda s: int(s.get("hits", 0)) - int(s.get("doubles", 0)) - int(s.get("triples", 0)) - int(s.get("homeRuns", 0)),
    "doubles":         lambda s: int(s.get("doubles", 0)),
    "batter_strikeouts": lambda s: int(s.get("strikeOuts", 0)),
    "hits_runs_rbis":  lambda s: int(s.get("hits", 0)) + int(s.get("runs", 0)) + int(s.get("rbi", 0)),
}

# Stat display keywords that indicate a pitcher prop
PITCHER_KEYWORDS = [
    "strikeout", "earned run", "hits allowed", "walks allowed",
    "pitching out", "outs recorded", "innings pitched",
    "pitch count", "batters faced",
]


class MLBAdapter(BaseSportAdapter):
    sport_key = "mlb"
    sport_label = "MLB"

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=10.0)
        self._name_to_id: dict = {}
        self._roster_loaded = False

    async def _fetch(self, path: str) -> dict | None:
        """Fetch from MLB Stats API. No rate limiting needed (free API)."""
        try:
            url = f"{MLB_BASE}{path}"
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception as e:
            print(f"[MLB] fetch failed {path}: {e}")
            return None

    async def get_todays_games(self) -> list[GameInfo]:
        today = datetime.now().strftime("%Y-%m-%d")
        cached = cache.get("schedule", f"mlb:{today}")
        if cached:
            return cached

        data = await self._fetch(f"/schedule?sportId=1&date={today}&hydrate=probablePitcher,team")
        if not data or not data.get("dates"):
            return []

        games = []
        for game in data["dates"][0].get("games", []):
            games.append(GameInfo(
                game_id=str(game.get("gamePk", "")),
                status=game.get("status", {}).get("detailedState", "Unknown"),
                home_team=game.get("teams", {}).get("home", {}).get("team", {}).get("name", ""),
                away_team=game.get("teams", {}).get("away", {}).get("team", {}).get("name", ""),
                scheduled=game.get("gameDate", ""),
                venue=game.get("venue", {}).get("name"),
            ))

        cache.put("schedule", f"mlb:{today}", games)
        return games

    async def _load_rosters(self):
        """Load all active rosters for today's games. Build name→ID map."""
        if self._roster_loaded:
            return

        today = datetime.now().strftime("%Y-%m-%d")
        data = await self._fetch(f"/schedule?sportId=1&date={today}&hydrate=probablePitcher,team")
        if not data or not data.get("dates"):
            return

        team_ids = set()
        for game in data["dates"][0].get("games", []):
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            if home.get("team", {}).get("id"):
                team_ids.add(home["team"]["id"])
            if away.get("team", {}).get("id"):
                team_ids.add(away["team"]["id"])
            # Probable pitchers
            for side in [home, away]:
                pp = side.get("probablePitcher")
                if pp:
                    name = pp.get("fullName", "").lower()
                    self._name_to_id[name] = {"id": pp["id"], "type": "pitcher", "probable": True}

        # Fetch rosters (parallel is fine — MLB API has no rate limit)
        for tid in team_ids:
            roster_data = await self._fetch(f"/teams/{tid}/roster?rosterType=active")
            if not roster_data or not roster_data.get("roster"):
                continue
            for p in roster_data["roster"]:
                name = p.get("person", {}).get("fullName", "").lower()
                if not name:
                    continue
                is_pitcher = p.get("position", {}).get("abbreviation") == "P"
                if name not in self._name_to_id:
                    self._name_to_id[name] = {
                        "id": p["person"]["id"],
                        "type": "pitcher" if is_pitcher else "batter",
                    }

        self._roster_loaded = True
        print(f"[MLB] Loaded {len(self._name_to_id)} players from rosters")

    def _is_pitcher_stat(self, stat_display: str) -> bool:
        st = stat_display.lower()
        return any(kw in st for kw in PITCHER_KEYWORDS)

    async def get_player_stats(
        self, player_name: str, stat_key: str, line: float,
        home_team: str = "", away_team: str = ""
    ) -> Optional[PlayerStats]:
        await self._load_rosters()

        name_lower = player_name.lower().strip()
        match = self._name_to_id.get(name_lower)
        if not match:
            return None

        player_id = match["id"]
        is_pitcher = match["type"] == "pitcher"
        group = "pitching" if is_pitcher or stat_key in PITCHING_STATS else "hitting"

        # Check cache
        cache_key = f"mlb:{player_id}:{group}"
        cached = cache.get("player_gamelog", cache_key)
        if cached:
            season_stat, game_splits = cached
        else:
            # Fetch season stats + game log
            season_data = await self._fetch(f"/people/{player_id}/stats?stats=season&group={group}&season=2026")
            log_data = await self._fetch(f"/people/{player_id}/stats?stats=gameLog&group={group}&season=2026")

            season_stat = {}
            if season_data and season_data.get("stats"):
                splits = season_data["stats"][0].get("splits", [])
                if splits:
                    season_stat = splits[0].get("stat", {})

            game_splits = []
            if log_data and log_data.get("stats"):
                game_splits = log_data["stats"][0].get("splits", [])

            cache.put("player_gamelog", cache_key, (season_stat, game_splits))

        # Extract stat values from game log
        stat_map = PITCHING_STATS if group == "pitching" else BATTING_STATS
        extractor = stat_map.get(stat_key)
        if not extractor:
            return None

        last_games = game_splits[-10:] if game_splits else []
        last5_vals = [extractor(g.get("stat", {})) for g in last_games[-5:]]
        last10_vals = [extractor(g.get("stat", {})) for g in last_games]

        # Season average
        games_key = "gamesStarted" if group == "pitching" else "gamesPlayed"
        gp = int(season_stat.get(games_key, 0))
        season_avg = None
        if gp > 0:
            try:
                total_val = extractor(season_stat)
                season_avg = total_val / gp if group == "hitting" else total_val / max(gp, 1)
            except Exception:
                pass

        return PlayerStats(
            player_name=player_name,
            player_id=str(player_id),
            team=home_team or away_team,
            season_avg=season_avg,
            last5=last5_vals,
            last10=last10_vals,
            games_played=gp,
            is_starter=match.get("probable", False) if is_pitcher else None,
        )

    async def enrich_props(self, props: list[dict]) -> list[dict]:
        """Batch-enrich MLB props with player stats."""
        await self._load_rosters()

        for prop in props:
            player_name = prop.get("player_name", "")
            stat_display = prop.get("stat_display", prop.get("stat_type", ""))
            line = float(prop.get("line", 0))

            # Determine canonical stat key
            stat_key = self._resolve_stat_key(stat_display)
            if not stat_key:
                continue

            stats = await self.get_player_stats(
                player_name, stat_key, line,
                prop.get("home_team", ""), prop.get("away_team", "")
            )
            if stats and (stats.last5 or stats.season_avg is not None):
                prop["_playerStats"] = {
                    "seasonAvg": stats.season_avg,
                    "last5": stats.last5,
                    "last10": stats.last10,
                    "last15": [],
                }
                if stats.is_starter is not None:
                    prop["_isProbablePitcher"] = stats.is_starter

        return props

    def _resolve_stat_key(self, stat_display: str) -> str | None:
        """Map a stat display string to a canonical stat key."""
        from config import PROP_STAT_MAP
        sd = stat_display.lower().strip()

        # Direct match
        if sd in PROP_STAT_MAP:
            return PROP_STAT_MAP[sd]

        # Partial match (longest key first)
        for key in sorted(PROP_STAT_MAP.keys(), key=len, reverse=True):
            if key in sd or sd in key:
                return PROP_STAT_MAP[key]

        return None
