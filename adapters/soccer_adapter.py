"""
SoccerAdapter — FIFA World Cup 2026 via Sportradar Soccer v4 API.

Overrides SportradarAdapter's date-based schedule fetching with the
competition/season-based endpoints that Soccer v4 actually uses.

Confirmed IDs (June 2026):
  competition: sr:competition:16  (FIFA World Cup, men)
  season:      sr:season:101177   (World Cup 2026, Jun 11 – Jul 19 2026)

Player stat fields from summary endpoint:
  statistics.totals.competitors[].players[].statistics
  → goals_scored, assists, shots_on_target, shots_off_target,
    shots_blocked, corner_kicks, offsides, yellow_cards, red_cards
"""

import httpx
from datetime import datetime, timedelta
from typing import Optional

from adapters.base_adapter import BaseSportAdapter, PlayerStats, GameInfo
from cache import cache_manager as cache
from config import SPORTRADAR_API_KEY
from utils.rate_limiter import rate_limiter


SOCCER_BASE = "https://api.sportradar.com/soccer/trial/v4/en"
FIFA_WC_COMPETITION_ID = "sr:competition:16"
FIFA_WC_SEASON_ID = "sr:season:101177"


class SoccerAdapter(BaseSportAdapter):
    LOOKBACK_DAYS = 14

    @property
    def sport_key(self) -> str:
        return "soccer"

    @property
    def sport_label(self) -> str:
        return "Soccer"

    def get_player_stats(self, player_name: str):
        """Not used — SoccerAdapter uses batch enrich_props instead."""
        return None

    def get_todays_games(self):
        """Not used — SoccerAdapter uses season schedule instead."""
        return []

    STAT_EXTRACTORS = {
        "goals":            lambda s: int(s.get("goals_scored", 0)),
        "assists":          lambda s: int(s.get("assists", 0)),
        "goals_assists":    lambda s: int(s.get("goals_scored", 0)) + int(s.get("assists", 0)),
        "shots_on_target":  lambda s: int(s.get("shots_on_target", 0)),
        "shots":            lambda s: (
            int(s.get("shots_on_target", 0)) +
            int(s.get("shots_off_target", 0)) +
            int(s.get("shots_blocked", 0))
        ),
        "goalkeeper_saves":  lambda s: int(s.get("saves", s.get("goalkeeper_saves", 0))),
        "goals_allowed":    lambda s: int(s.get("goals_conceded", s.get("goals_against", 0))),
        "tackles":          lambda s: int(s.get("tackles", 0)),
        "passes":           lambda s: int(s.get("passes", 0)),
        "fouls_committed":   lambda s: int(s.get("fouls_committed", s.get("fouls", 0))),
        "fouls_drawn":       lambda s: int(s.get("fouls_drawn", s.get("fouls_suffered", 0))),
        "yellow_cards":     lambda s: int(s.get("yellow_cards", 0)),
        "corner_kicks":     lambda s: int(s.get("corner_kicks", 0)),
        "offsides":         lambda s: int(s.get("offsides", 0)),
    }

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=15.0)
        self._player_game_log: dict = {}   # normalized_name → [stats_dict, ...]
        self._name_to_id: dict = {}        # normalized_name → sr player id
        self._loaded = False

    def _url(self, path: str) -> str:
        return f"{SOCCER_BASE}/{path}?api_key={SPORTRADAR_API_KEY}"

    async def _fetch(self, path: str, cache_type: str = None, cache_key: str = None):
        import asyncio
        if cache_type and cache_key:
            cached = cache.get(cache_type, cache_key)
            if cached is not None:
                return cached

        for attempt in range(3):
            await rate_limiter.acquire("sportradar", min_interval=1.2, daily_limit=1000)
            try:
                url = self._url(path)
                resp = await self._client.get(url)
                if resp.status_code == 429:
                    wait = 2.0 * (attempt + 1)
                    print(f"[Soccer] 429 rate limit on attempt {attempt+1}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code != 200:
                    print(f"[Soccer] HTTP {resp.status_code} for {path}")
                    return None
                data = resp.json()
                if cache_type and cache_key:
                    cache.put(cache_type, cache_key, data)
                return data
            except Exception as e:
                print(f"[Soccer] fetch failed {path}: {e}")
                if attempt < 2:
                    await asyncio.sleep(1.5)
        return None

    @staticmethod
    def _normalize_name(raw: str) -> str:
        """
        Sportradar returns 'Last, First' format. Normalize to 'first last'
        to match Underdog/Apify prop player names.
        """
        raw = raw.strip()
        if "," in raw:
            parts = raw.split(",", 1)
            return f"{parts[1].strip()} {parts[0].strip()}".lower()
        return raw.lower()

    async def _load_game_logs(self):
        if self._loaded:
            return

        # Fetch the full season schedule to find completed games
        data = await self._fetch(
            f"seasons/{FIFA_WC_SEASON_ID}/schedules.json",
            cache_type="schedule",
            cache_key=f"soccer:wc2026:schedule"
        )
        if not data:
            print("[Soccer] Failed to fetch season schedule")
            return

        schedules = data.get("schedules", [])
        cutoff = datetime.utcnow() - timedelta(days=self.LOOKBACK_DAYS)

        completed_event_ids = []
        for entry in schedules:
            se = entry.get("sport_event", {})
            status = entry.get("sport_event_status", {}).get("status", "")
            start_time_str = se.get("start_time", "")

            if status not in ("closed", "ended", "complete", "finalizado"):
                continue

            try:
                start_dt = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                start_dt = start_dt.replace(tzinfo=None)
                if start_dt < cutoff:
                    continue
            except Exception:
                pass

            event_id = se.get("id")
            if event_id:
                completed_event_ids.append(event_id)

        print(f"[Soccer] Found {len(completed_event_ids)} completed WC games in last {self.LOOKBACK_DAYS} days")

        # Fetch summaries for each completed game
        for event_id in completed_event_ids[:20]:  # cap for rate limits
            cache_key = f"soccer:summary:{event_id}"
            summary = await self._fetch(
                f"sport_events/{event_id}/summary.json",
                cache_type="game_boxscore",
                cache_key=cache_key
            )
            if not summary:
                continue

            competitors = (
                summary.get("statistics", {})
                       .get("totals", {})
                       .get("competitors", [])
            )

            # Build opponent defensive stats: for each team, record how many
            # goals/shots_on_target they CONCEDED (= the other team's totals)
            team_totals = {}
            for comp in competitors:
                team_id = comp.get("id", "")
                team_stats = comp.get("statistics", {})
                goals_scored = int(team_stats.get("goals_scored", 0))
                shots_on_target = int(team_stats.get("shots_on_target", 0))
                team_totals[team_id] = {
                    "goals_scored": goals_scored,
                    "shots_on_target": shots_on_target,
                    "team_name": comp.get("name", ""),
                    "qualifier": comp.get("qualifier", ""),
                }

            # Map each team's conceded stats from the other team's scored stats
            team_ids = list(team_totals.keys())
            opponent_def_stats = {}
            if len(team_ids) == 2:
                opponent_def_stats[team_ids[0]] = {
                    "goals_conceded": team_totals[team_ids[1]]["goals_scored"],
                    "sot_conceded": team_totals[team_ids[1]]["shots_on_target"],
                    "opponent_name": team_totals[team_ids[1]]["team_name"],
                }
                opponent_def_stats[team_ids[1]] = {
                    "goals_conceded": team_totals[team_ids[0]]["goals_scored"],
                    "sot_conceded": team_totals[team_ids[0]]["shots_on_target"],
                    "opponent_name": team_totals[team_ids[0]]["team_name"],
                }

            for comp in competitors:
                team_id = comp.get("id", "")
                opp_stats = opponent_def_stats.get(team_id, {})
                for player in comp.get("players", []):
                    raw_name = player.get("name", "")
                    if not raw_name:
                        continue

                    norm = self._normalize_name(raw_name)
                    stats = player.get("statistics", {})

                    # Attach opponent defensive context to each game log entry
                    stats["_opp_goals_conceded"] = opp_stats.get("goals_conceded", 0)
                    stats["_opp_sot_conceded"] = opp_stats.get("sot_conceded", 0)
                    stats["_opponent_name"] = opp_stats.get("opponent_name", "")

                    if norm not in self._name_to_id:
                        self._name_to_id[norm] = player.get("id", "")
                    if norm not in self._player_game_log:
                        self._player_game_log[norm] = []
                    self._player_game_log[norm].append(stats)

        print(f"[Soccer] Built game logs for {len(self._player_game_log)} players")
        if self._player_game_log:  # Only mark loaded if we actually got data
            self._loaded = True

    def _resolve_stat_key(self, stat_display: str) -> Optional[str]:
        """Map Underdog stat display strings to our canonical keys."""
        sd = stat_display.lower().strip()

        # Strip half-time prefixes before matching
        for prefix in ["1h ", "2h ", "first half ", "second half "]:
            if sd.startswith(prefix):
                sd = sd[len(prefix):]
                break

        mapping = {
            "goals": "goals", "goal scored": "goals", "soccer goals": "goals",
            "assists": "assists", "soccer assists": "assists",
            "goals + assists": "goals_assists", "goals+assists": "goals_assists",
            "shots on target": "shots_on_target", "shots on goal": "shots_on_target",
            "shots attempted": "shots", "shots": "shots", "total shots": "shots",
            "saves": "goalkeeper_saves", "goalkeeper saves": "goalkeeper_saves",
            "tackles": "tackles",
            "passes": "passes", "passes attempted": "passes",
            "fouls": "fouls_committed", "fouls committed": "fouls_committed",
            "fouls drawn": "fouls_drawn", "fouls suffered": "fouls_drawn",
            "yellow cards": "yellow_cards",
            "corner kicks": "corner_kicks",
            "offsides": "offsides",
            "goals allowed": "goals_allowed",
        }
        if sd in mapping:
            return mapping[sd]
        for k, v in mapping.items():
            if k in sd or sd in k:
                return v
        return None

    def _unsupported_market_reason(self, stat_display: str, player_name: str) -> str:
        sd = (stat_display or "").lower()
        if not player_name:
            return "unsupported_team_market"
        if any(token in sd for token in ["match result", "moneyline", "double chance", "both teams", "btts"]):
            return "unsupported_binary_market"
        if any(token in sd for token in ["team goals", "corners", "total goals"]):
            return "unsupported_team_market"
        return "stat_not_supported"

    async def enrich_props(self, props: list[dict]) -> list[dict]:
        await self._load_game_logs()

        enriched = 0
        for prop in props:
            player_name = prop.get("player_name", "").strip()
            stat_display = prop.get("stat_display", prop.get("stat_type", ""))
            prop["_sport_key"] = "soccer"
            stat_key = self._resolve_stat_key(stat_display)
            if not stat_key:
                reason = self._unsupported_market_reason(stat_display, player_name)
                prop["_projectionUnavailableReason"] = reason
                prop["unavailableReason"] = reason
                prop["unavailableCode"] = reason
                continue

            extractor = self.STAT_EXTRACTORS.get(stat_key)
            if not extractor:
                prop["_projectionUnavailableReason"] = "stat_not_supported"
                prop["unavailableReason"] = "stat_not_supported"
                prop["unavailableCode"] = "stat_not_supported"
                continue

            # Try exact normalized match first, then fuzzy
            norm = player_name.lower().strip()
            game_log = self._player_game_log.get(norm)
            match_confidence = 0.98 if game_log else None

            if not game_log:
                # Try reversed order: "Raul Jimenez" → check "jimenez raul"
                parts = norm.split()
                if len(parts) >= 2:
                    reversed_norm = f"{parts[-1]} {' '.join(parts[:-1])}"
                    game_log = self._player_game_log.get(reversed_norm)
                    if game_log:
                        match_confidence = 0.9

            if not game_log:
                # Last-name fuzzy fallback
                last_name = norm.split()[-1] if norm else ""
                for key, log in self._player_game_log.items():
                    if last_name and last_name in key.split():
                        game_log = log
                        match_confidence = 0.76
                        break

            if not game_log or len(game_log) < 1:
                prop["_projectionUnavailableReason"] = "player_not_matched" if self._player_game_log else "provider_unavailable"
                prop["unavailableReason"] = prop["_projectionUnavailableReason"]
                prop["unavailableCode"] = prop["_projectionUnavailableReason"]
                continue

            last5 = [extractor(g) for g in game_log[:5]]
            last10 = [extractor(g) for g in game_log[:10]]
            season_avg = sum(last10) / len(last10) if last10 else None

            # World Cup: players may have only 1-3 games — allow with lower threshold
            # Compute opponent defensive averages across last 5 games
            opp_goals_conceded_l5 = [g.get("_opp_goals_conceded", 0) for g in game_log[:5]]
            opp_sot_conceded_l5 = [g.get("_opp_sot_conceded", 0) for g in game_log[:5]]
            avg_opp_goals_conceded = round(sum(opp_goals_conceded_l5) / max(len(opp_goals_conceded_l5), 1), 2)
            avg_opp_sot_conceded = round(sum(opp_sot_conceded_l5) / max(len(opp_sot_conceded_l5), 1), 2)
            last_opponent = game_log[0].get("_opponent_name", "") if game_log else ""

            prop["_playerStats"] = {
                "seasonAvg": round(season_avg, 2) if season_avg is not None else None,
                "last5": last5,
                "last10": last10,
                "last15": [],
                # Opponent defensive context — used by AI for narrative quality
                "oppGoalsConcededL5": avg_opp_goals_conceded,
                "oppShotsConcededL5": avg_opp_sot_conceded,
                "lastOpponent": last_opponent,
            }
            prop["_playerMatchConfidence"] = match_confidence
            enriched += 1

        print(f"[Soccer] Enriched {enriched} of {len(props)} props")
        return props
