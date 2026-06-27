"""
Base Sportradar adapter — shared logic for all sports that use the
Sportradar API. Each sport subclass just defines stat extractors
and sport-specific field mappings.

Rate-limited to 1 req/sec via the global rate limiter.
"""
from __future__ import annotations

import httpx
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Optional

from adapters.base_adapter import BaseSportAdapter, PlayerStats, GameInfo
from cache import cache_manager as cache
from config import SPORTRADAR_API_KEY, SPORTRADAR_SPORTS
from utils.rate_limiter import rate_limiter


class SportradarAdapter(BaseSportAdapter):
    """
    Generic Sportradar adapter. Subclass for each sport and set:
      - sport_key: "wnba", "nba", "nfl", etc.
      - sport_label: "WNBA", "NBA", "NFL", etc.
      - STAT_EXTRACTORS: dict mapping canonical stat keys to
        functions that extract from a game box score player entry
      - SEASON_AVG_FIELDS: dict mapping canonical stat keys to
        the field name in Sportradar's season averages
    """

    # Override in subclass
    STAT_EXTRACTORS: dict = {}
    SEASON_AVG_FIELDS: dict = {}
    STAT_ALIASES: dict = {}
    LOOKBACK_DAYS: int = 7   # How many days of game history to fetch

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=10.0)
        self._name_to_id: dict = {}
        self._name_aliases: dict = {}
        self._player_game_log: dict = {}  # name → [game_stats_dict, ...]
        self._player_season_stats: dict = {}

    def _normalize_player_name(self, name: str) -> str:
        name = unicodedata.normalize("NFKD", name or "")
        name = "".join(ch for ch in name if not unicodedata.combining(ch))
        name = name.lower().strip()
        name = re.sub(r"[^\w\s]", " ", name)
        parts = [p for p in name.split() if p not in {"jr", "sr", "ii", "iii", "iv"}]
        return " ".join(parts)

    def _register_player(self, player: dict):
        names = [
            player.get("full_name"),
            player.get("name"),
            player.get("display_name"),
            player.get("preferred_name"),
        ]
        if player.get("first_name") and player.get("last_name"):
            names.append(f"{player.get('first_name')} {player.get('last_name')}")

        normalized_names = [self._normalize_player_name(n) for n in names if n]
        normalized_names = [n for n in normalized_names if n]
        if not normalized_names:
            return None

        canonical = normalized_names[0]
        self._name_to_id.setdefault(canonical, {
            "id": player.get("id"),
            "position": player.get("position", ""),
        })
        for alias in normalized_names:
            self._name_aliases.setdefault(alias, canonical)
        return canonical

    def _find_player_log(self, player_name: str) -> tuple[str | None, list, float | None]:
        normalized = self._normalize_player_name(player_name)
        if not normalized:
            return None, [], None

        if normalized in self._player_game_log or normalized in self._player_season_stats:
            return normalized, self._player_game_log.get(normalized, []), 0.98

        alias = self._name_aliases.get(normalized)
        if alias and (alias in self._player_game_log or alias in self._player_season_stats):
            return alias, self._player_game_log.get(alias, []), 0.94

        parts = normalized.split()
        if len(parts) >= 2:
            first_initial = parts[0][0]
            last_name = parts[-1]
            candidates = []
            provider_names = set(self._player_game_log.keys()) | set(self._player_season_stats.keys())
            for provider_name in provider_names:
                provider_parts = provider_name.split()
                if (
                    len(provider_parts) >= 2
                    and provider_parts[0].startswith(first_initial)
                    and provider_parts[-1] == last_name
                ):
                    candidates.append(provider_name)
            if len(candidates) == 1:
                key = candidates[0]
                return key, self._player_game_log.get(key, []), 0.86

        return None, [], None

    def _mark_projection_unavailable(self, prop: dict, reason: str):
        prop["_projectionUnavailableReason"] = reason

    def _extract_stats_payload(self, player: dict) -> dict:
        stats = player.get("statistics") or player.get("stats") or {}
        if isinstance(stats, dict) and stats:
            return stats
        for key in ("season", "season_stats", "season_average", "average", "average_statistics"):
            value = player.get(key)
            if isinstance(value, dict) and value:
                return value
        return {}

    def _season_games_played(self, stats: dict) -> int:
        for key in ("games_played", "games", "played"):
            try:
                games = int(stats.get(key))
                if games > 0:
                    return games
            except (TypeError, ValueError):
                continue
        return 0

    def _build_url(self, path: str) -> str:
        """Build full Sportradar URL for this sport."""
        cfg = SPORTRADAR_SPORTS.get(self.sport_key, {})
        base = cfg.get("base", "")
        version = cfg.get("version", "v8")
        lang = cfg.get("lang", "en")
        return f"{base}/trial/{version}/{lang}{path}?api_key={SPORTRADAR_API_KEY}"

    async def _fetch(self, path: str, cache_type: str = None, cache_key: str = None) -> dict | None:
        """Fetch from Sportradar with rate limiting and optional caching."""
        if cache_type and cache_key:
            cached = cache.get(cache_type, cache_key)
            if cached is not None:
                return cached

        await rate_limiter.acquire("sportradar", min_interval=1.1, daily_limit=1000)

        try:
            url = self._build_url(path)
            resp = await self._client.get(url)
            if resp.status_code != 200:
                print(f"[{self.sport_label}] HTTP {resp.status_code} for {path}")
                return None
            data = resp.json()
            if cache_type and cache_key:
                cache.put(cache_type, cache_key, data)
            return data
        except Exception as e:
            print(f"[{self.sport_label}] fetch failed {path}: {e}")
            return None

    async def get_todays_games(self) -> list[GameInfo]:
        now = datetime.now()
        year, month, day = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")
        cache_key = f"{self.sport_key}:{year}-{month}-{day}"

        data = await self._fetch(
            f"/games/{year}/{month}/{day}/schedule.json",
            cache_type="schedule", cache_key=cache_key
        )
        if not data:
            return []

        games = []
        raw_games = data.get("games", data.get("league", {}).get("games", []))
        for g in raw_games:
            games.append(GameInfo(
                game_id=g.get("id", ""),
                status=g.get("status", "unknown"),
                home_team=g.get("home", {}).get("name", g.get("home", {}).get("market", "")),
                away_team=g.get("away", {}).get("name", g.get("away", {}).get("market", "")),
                scheduled=g.get("scheduled", ""),
                venue=g.get("venue", {}).get("name"),
            ))

        return games

    async def _load_recent_game_logs(self):
        """
        Fetch game summaries for the last N days to build per-player game logs.
        Also expands name_to_id from game rosters (covers teams not playing today).
        """
        if self._player_game_log:
            return  # Already loaded

        # Fetch schedules for the last N days
        recent_game_ids = []
        for days_ago in range(1, self.LOOKBACK_DAYS + 1):
            past = datetime.now() - timedelta(days=days_ago)
            year, month, day = past.strftime("%Y"), past.strftime("%m"), past.strftime("%d")
            cache_key = f"{self.sport_key}:{year}-{month}-{day}"

            data = await self._fetch(
                f"/games/{year}/{month}/{day}/schedule.json",
                cache_type="schedule", cache_key=cache_key
            )
            if not data:
                continue

            raw_games = data.get("games", data.get("league", {}).get("games", []))
            for g in raw_games:
                status = g.get("status", "")
                if status in ("closed", "complete", "final"):
                    recent_game_ids.append(g.get("id"))

        print(f"[{self.sport_label}] Found {len(recent_game_ids)} recent completed games")

        # Fetch game summaries (box scores)
        for game_id in recent_game_ids[:12]:  # Cap to manage rate limits
            cache_key = f"{self.sport_key}:box:{game_id}"
            data = await self._fetch(
                f"/games/{game_id}/summary.json",
                cache_type="game_boxscore", cache_key=cache_key
            )
            if not data:
                continue

            game_data = data.get("game", data)
            for side in ("home", "away"):
                team_data = game_data.get(side, {})
                players = team_data.get("players", [])
                for p in players:
                    name_key = self._register_player(p)
                    if not name_key:
                        continue

                    # Store game stats
                    stats = self._extract_stats_payload(p)
                    if not stats:
                        continue
                    if name_key not in self._player_game_log:
                        self._player_game_log[name_key] = []
                    self._player_game_log[name_key].append(stats)
                    self._player_season_stats.setdefault(name_key, stats)

        print(f"[{self.sport_label}] Built game logs for {len(self._player_game_log)} players, "
              f"name_to_id has {len(self._name_to_id)} entries")

    async def _load_today_rosters(self):
        """Fetch team profiles for today's games to get rosters."""
        games = await self.get_todays_games()
        team_ids = set()
        for g in games:
            # Try to get team IDs from the raw schedule data
            # We'll need to re-fetch with full data
            pass

        # Fetch team profiles for today's games
        today_key = datetime.now().strftime("%Y-%m-%d")
        data = await self._fetch(
            f"/games/{today_key.replace('-', '/')}/schedule.json",
            cache_type="schedule", cache_key=f"{self.sport_key}:full:{today_key}"
        )
        if not data:
            return

        raw_games = data.get("games", data.get("league", {}).get("games", []))
        for g in raw_games:
            for side in ("home", "away"):
                team = g.get(side, {})
                team_id = team.get("id")
                if team_id:
                    team_ids.add(team_id)

        for team_id in team_ids:
            cache_key = f"{self.sport_key}:team:{team_id}"
            team_data = await self._fetch(
                f"/teams/{team_id}/profile.json",
                cache_type="team_roster", cache_key=cache_key
            )
            if not team_data:
                continue

            players = team_data.get("players", team_data.get("roster", []))
            for p in players:
                name_key = self._register_player(p)
                stats = self._extract_stats_payload(p)
                if name_key and stats:
                    self._player_season_stats.setdefault(name_key, stats)

    async def get_player_stats(
        self, player_name: str, stat_key: str, line: float,
        home_team: str = "", away_team: str = ""
    ) -> Optional[PlayerStats]:
        await self._load_recent_game_logs()

        name_key, game_log, _confidence = self._find_player_log(player_name)

        extractor = self.STAT_EXTRACTORS.get(stat_key)
        if not extractor:
            return None

        # Detect period-specific props and apply scaling
        stat_display_lower = stat_key.lower()
        period_mult = 1.0
        # Period detection happens at the prop level, not here

        last5 = [extractor(g) for g in game_log[:5]]
        last10 = [extractor(g) for g in game_log[:10]]

        if len(last5) < 3:
            return None  # Not enough data

        season_avg_field = self.SEASON_AVG_FIELDS.get(stat_key)
        season_avg = None
        if last10:
            season_avg = sum(last10) / len(last10)

        return PlayerStats(
            player_name=player_name,
            player_id=self._name_to_id.get(name_key or "", {}).get("id"),
            team=home_team or away_team,
            season_avg=season_avg,
            last5=last5,
            last10=last10,
            games_played=len(game_log),
        )

    async def enrich_props(self, props: list[dict]) -> list[dict]:
        """Batch-enrich props with player stats from Sportradar game logs."""
        await self._load_recent_game_logs()
        await self._load_today_rosters()

        enriched = 0
        for prop in props:
            player_name = prop.get("player_name") or prop.get("player") or ""
            stat_display = prop.get("stat_display", prop.get("stat_type", ""))
            line = float(prop.get("line", 0))
            prop["_sport_key"] = self.sport_key

            if self.sport_key == "wnba" and self._has_period_prefix(stat_display):
                self._mark_projection_unavailable(prop, "stat_not_supported")
                continue

            stat_key = self._resolve_stat_key(stat_display)
            if not stat_key:
                self._mark_projection_unavailable(prop, "stat_not_supported")
                continue

            # Period scaling for 1Q/1H/2H props
            period_mult = self._get_period_multiplier(stat_display)

            name_key, game_log, match_confidence = self._find_player_log(player_name)
            if not name_key:
                reason = "no_game_logs" if not self._player_game_log else "player_not_matched"
                self._mark_projection_unavailable(prop, reason)
                continue

            extractor = self.STAT_EXTRACTORS.get(stat_key)
            if not extractor:
                self._mark_projection_unavailable(prop, "stat_not_supported")
                continue

            season_stats = self._player_season_stats.get(name_key, {})
            if len(game_log) < 3 and not season_stats:
                reason = "no_game_logs" if not game_log else "no_stat_history"
                self._mark_projection_unavailable(prop, reason)
                continue

            last5 = [round(extractor(g) * period_mult, 1) for g in game_log[:5]]
            last10 = [round(extractor(g) * period_mult, 1) for g in game_log[:10]]
            season_avg = sum(last10) / len(last10) if last10 else None
            if season_avg is None and season_stats:
                season_avg = extractor(season_stats)

            if len(last5) < 3 and season_avg is None:
                self._mark_projection_unavailable(prop, "no_stat_history")
                continue

            prop["_playerStats"] = {
                "seasonAvg": round(season_avg, 2) if season_avg else None,
                "last5": last5,
                "last10": last10 if len(last10) >= 3 else last5,
                "last15": [],
                "gamesPlayed": len(game_log) if game_log else self._season_games_played(season_stats),
            }
            prop["_projectionSource"] = "sportradar_recent_stats"
            prop["projectionMatchConfidence"] = match_confidence
            enriched += 1

        print(f"[{self.sport_label}] Enriched {enriched} of {len(props)} props")
        return props

    def _resolve_stat_key(self, stat_display: str) -> str | None:
        """Map stat display string to canonical stat key."""
        from config import PROP_STAT_MAP
        sd = re.sub(r"[_-]+", " ", (stat_display or "").lower().strip())
        compact_sd = sd.replace(" ", "_")
        # Remove period prefixes for matching
        for prefix in ["1q ", "2q ", "3q ", "4q ", "1h ", "2h ",
                        "first quarter ", "second quarter ",
                        "first half ", "second half ",
                        "1st quarter ", "2nd quarter ",
                        "1st half ", "2nd half ",
                        "1st inn. ", "1st inning "]:
            if sd.startswith(prefix):
                sd = sd[len(prefix):]
                compact_sd = sd.replace(" ", "_")
                break

        aliases = getattr(self, "STAT_ALIASES", {}) or {}
        if compact_sd in aliases:
            return aliases[compact_sd]
        if sd in aliases:
            return aliases[sd]
        if sd in PROP_STAT_MAP:
            return PROP_STAT_MAP[sd]
        if compact_sd in PROP_STAT_MAP:
            return PROP_STAT_MAP[compact_sd]
        for key in sorted(PROP_STAT_MAP.keys(), key=len, reverse=True):
            if key in sd or sd in key:
                return PROP_STAT_MAP[key]
        return None

    def _has_period_prefix(self, stat_display: str) -> bool:
        sd = re.sub(r"[_-]+", " ", (stat_display or "").lower().strip())
        return any(sd.startswith(prefix) for prefix in [
            "1q ", "2q ", "3q ", "4q ", "1h ", "2h ",
            "first quarter ", "second quarter ", "third quarter ", "fourth quarter ",
            "first half ", "second half ",
            "1st quarter ", "2nd quarter ", "3rd quarter ", "4th quarter ",
            "1st half ", "2nd half ",
        ])

    def _get_period_multiplier(self, stat_display: str) -> float:
        """Scale full-game stats for period-specific props."""
        sd = stat_display.lower()
        if any(x in sd for x in ["1q", "first quarter", "1st quarter"]):
            return 0.28
        if any(x in sd for x in ["2q", "second quarter", "2nd quarter"]):
            return 0.25
        if any(x in sd for x in ["3q", "third quarter", "3rd quarter"]):
            return 0.25
        if any(x in sd for x in ["4q", "fourth quarter", "4th quarter"]):
            return 0.22
        if any(x in sd for x in ["1h", "first half", "1st half"]):
            return 0.53
        if any(x in sd for x in ["2h", "second half", "2nd half"]):
            return 0.47
        return 1.0
