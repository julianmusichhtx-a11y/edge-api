"""
MLB Adapter — pulls real per-game stat logs from the free MLB Stats API
(statsapi.mlb.com) and writes prop["_playerStats"] = {seasonAvg, last5, last10}
in the exact shape scoring/scorer.py requires.

MLB Stats API is free and effectively unlimited, so unlike Sportradar there's
no hard rate limit — but we still cache aggressively because a single slate
of 700+ props usually maps to only ~150-250 unique players, and each player
needs 2 network calls (ID lookup + game log) the first time we see them.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from cache import cache_manager as cache
from config import MLB_STATS_BASE, PROP_STAT_MAP
from .base_adapter import BaseSportAdapter

logger = logging.getLogger(__name__)

LINEUP_STATUS_URL = "https://edgelab.julianmusichhtx.workers.dev/lineup-status"

# Be polite to the free public API even though it has no published hard limit.
_MIN_INTERVAL = 0.08  # ~12 req/sec ceiling
_last_call_at = 0.0


def _throttle():
    global _last_call_at
    now = time.time()
    wait = _MIN_INTERVAL - (now - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.time()


def _http_get(client: httpx.Client, url: str) -> Optional[dict]:
    try:
        _throttle()
        res = client.get(url)
        if res.status_code != 200:
            return None
        return res.json()
    except Exception as e:
        logger.debug(f"MLB Stats API GET failed for {url}: {e}")
        return None


# ── Stat extractors: pull a single game's value for a canonical stat key ──
# MLB Stats API gameLog "stat" blocks use these field names for hitting/pitching.
_HITTING_EXTRACTORS = {
    "hits": lambda s: s.get("hits", 0),
    "total_bases": lambda s: s.get("totalBases", 0),
    "home_runs": lambda s: s.get("homeRuns", 0),
    "runs": lambda s: s.get("runs", 0),
    "rbi": lambda s: s.get("rbi", 0),
    "stolen_bases": lambda s: s.get("stolenBases", 0),
    "walks": lambda s: s.get("baseOnBalls", 0),
    "singles": lambda s: max(
        s.get("hits", 0) - s.get("doubles", 0) - s.get("triples", 0) - s.get("homeRuns", 0), 0
    ),
    "doubles": lambda s: s.get("doubles", 0),
    "batter_strikeouts": lambda s: s.get("strikeOuts", 0),
    "hits_runs_rbis": lambda s: s.get("hits", 0) + s.get("runs", 0) + s.get("rbi", 0),
}

_PITCHING_EXTRACTORS = {
    "strikeouts": lambda s: s.get("strikeOuts", 0),
    "earned_runs": lambda s: s.get("earnedRuns", 0),
    "hits_allowed": lambda s: s.get("hits", 0),
    "walks_allowed": lambda s: s.get("baseOnBalls", 0),
    "pitching_outs": lambda s: round(float(s.get("inningsPitched", 0) or 0) * 3),
    "innings_pitched": lambda s: float(s.get("inningsPitched", 0) or 0),
}

_ALL_STAT_KEYS = set(_HITTING_EXTRACTORS) | set(_PITCHING_EXTRACTORS)


class MLBAdapter(BaseSportAdapter):
    sport_key = "mlb"
    sport_label = "MLB"

    # ────────────────────────── public interface ──────────────────────────

    def enrich_prop(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=6.0) as client:
                return self._enrich_one(prop, client)
        except Exception as e:
            logger.warning(f"MLB enrich_prop failed for {prop.get('player_name')}: {e}")
            return prop

    def enrich_props(self, props: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        with httpx.Client(timeout=6.0) as client:
            for prop in props:
                results.append(self._enrich_one(prop, client))
        enriched_count = sum(1 for p in results if p.get("_playerStats"))
        logger.info(f"[MLB] Enriched {enriched_count} of {len(results)} props with real stats")
        return results

    def get_player_stats(self, player_name: str) -> Dict[str, Any]:
        with httpx.Client(timeout=6.0) as client:
            stats = self._get_game_log_stats(player_name, client, is_pitcher=False)
            if not stats:
                stats = self._get_game_log_stats(player_name, client, is_pitcher=True)
            return stats or {}

    def get_todays_games(self) -> List[Dict[str, Any]]:
        with httpx.Client(timeout=5.0) as client:
            data = _http_get(client, self._schedule_url())
            if not data:
                return []
            return data.get("dates", [{}])[0].get("games", [])

    # ────────────────────────── core enrichment ──────────────────────────

    def _enrich_one(self, prop: Dict[str, Any], client: httpx.Client) -> Dict[str, Any]:
        enriched = dict(prop)
        player_name = (prop.get("player_name") or "").strip()
        stat_display = (prop.get("stat_display") or prop.get("stat_type") or "").lower()
        stat_key = self._resolve_stat_key(stat_display)
        is_pitcher = stat_key in _PITCHING_EXTRACTORS if stat_key else self._looks_like_pitcher_stat(stat_display)

        enriched["sport"] = "mlb"
        enriched["player_name_clean"] = player_name.lower()
        enriched["is_pitcher"] = is_pitcher

        # Lineup confirmation (cheap, cached, useful context even though
        # the scorer doesn't require it)
        lineup = self._cached_lineup_status(player_name, client)
        if lineup:
            enriched["lineup_status"] = lineup
            enriched["is_confirmed"] = lineup.get("status") == "CONFIRMED"
        else:
            enriched["is_confirmed"] = False

        # Today's game context
        game_info = self._cached_game_info(player_name, client)
        if game_info:
            enriched.update(game_info)

        # The actual data the scorer needs: real per-game stat history
        if stat_key:
            player_stats = self._get_game_log_stats(player_name, client, is_pitcher, stat_key)
            if player_stats:
                enriched["_playerStats"] = player_stats

        return enriched

    def _resolve_stat_key(self, stat_display: str) -> Optional[str]:
        sd = stat_display.strip()
        for prefix in ("1st inn. ", "1st inning ", "1q ", "2q ", "1h ", "2h "):
            if sd.startswith(prefix):
                sd = sd[len(prefix):]
                break
        if sd in PROP_STAT_MAP:
            key = PROP_STAT_MAP[sd]
            return key if key in _ALL_STAT_KEYS else None
        for k in sorted(PROP_STAT_MAP.keys(), key=len, reverse=True):
            if k in sd or sd in k:
                key = PROP_STAT_MAP[k]
                if key in _ALL_STAT_KEYS:
                    return key
        return None

    def _looks_like_pitcher_stat(self, stat_display: str) -> bool:
        return any(k in stat_display for k in ("strikeout", "pitch", "earned run", "hit allowed", "walk allowed"))

    # ────────────────────────── MLB Stats API calls ──────────────────────────

    def _player_id_lookup(self, player_name: str, client: httpx.Client) -> Optional[int]:
        cache_key = player_name.lower().strip()
        cached = cache.get("player_season", f"id:{cache_key}")
        if cached is not None:
            return cached or None  # cached 0/None means "not found", don't re-fetch

        search_url = str(httpx.URL(f"{MLB_STATS_BASE}/people/search", params={"names": player_name}))
        data = _http_get(client, search_url)
        player_id = None
        if data:
            people = data.get("people", [])
            if people:
                player_id = people[0].get("id")

        cache.put("player_season", f"id:{cache_key}", player_id or 0)
        return player_id

    def _get_game_log_stats(
        self,
        player_name: str,
        client: httpx.Client,
        is_pitcher: bool,
        stat_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        player_id = self._player_id_lookup(player_name, client)
        if not player_id:
            return None

        group = "pitching" if is_pitcher else "hitting"
        season = datetime.now().year
        cache_key = f"gamelog:{player_id}:{group}:{season}"
        game_log = cache.get("player_gamelog", cache_key)

        if game_log is None:
            data = _http_get(
                client,
                f"{MLB_STATS_BASE}/people/{player_id}/stats"
                f"?stats=gameLog&group={group}&season={season}",
            )
            game_log = []
            if data:
                for stat_block in data.get("stats", []):
                    for split in stat_block.get("splits", []):
                        s = split.get("stat", {})
                        if s:
                            game_log.append(s)
                # Most recent games first
                game_log = list(reversed(game_log))
            cache.put("player_gamelog", cache_key, game_log)

        if len(game_log) < 3:
            return None

        extractors = _PITCHING_EXTRACTORS if is_pitcher else _HITTING_EXTRACTORS
        extractor = extractors.get(stat_key) if stat_key else None

        if extractor:
            last5 = [round(extractor(g), 1) for g in game_log[:5]]
            last10 = [round(extractor(g), 1) for g in game_log[:10]]
        else:
            # No specific stat key resolved — can't build meaningful series
            return None

        if len(last5) < 3:
            return None

        season_avg = round(sum(last10) / len(last10), 2) if last10 else None

        return {
            "seasonAvg": season_avg,
            "last5": last5,
            "last10": last10 if len(last10) >= 3 else last5,
            "last15": [],
        }

    # ────────────────────────── lineup + schedule (existing, working) ──────────────────────────

    def _cached_lineup_status(self, player: str, client: httpx.Client) -> Optional[Dict]:
        cache_key = player.lower()
        cached = cache.get("player_gamelog", f"lineup:{cache_key}")
        if cached is not None:
            return cached or None
        try:
            res = client.get(f"{LINEUP_STATUS_URL}?player={player}", timeout=2.5)
            result = res.json() if res.status_code == 200 else None
        except Exception:
            result = None
        cache.put("player_gamelog", f"lineup:{cache_key}", result or {})
        return result

    def _schedule_url(self) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        return f"{MLB_STATS_BASE}/schedule?sportId=1&date={today}&hydrate=team,venue"

    def _cached_game_info(self, player: str, client: httpx.Client) -> Optional[Dict]:
        """
        Best-effort "is there a game today" context. NOTE: matching a player
        to a specific game by name alone isn't reliable without a roster
        lookup (which costs another API call per player) — this only tells
        you a game is happening today for *some* team, sourced from the
        lineup_status data when available. Kept deliberately lightweight
        since the scorer doesn't depend on this; it's just extra context
        for the response payload.
        """
        today_key = datetime.now().strftime("%Y-%m-%d")
        games = cache.get("schedule", f"mlb:{today_key}")
        if games is None:
            data = _http_get(client, self._schedule_url())
            games = data.get("dates", [{}])[0].get("games", []) if data else []
            cache.put("schedule", f"mlb:{today_key}", games)
        return {"games_today": len(games)} if games else None