"""
Sport-specific Sportradar adapters.
Each one is a thin subclass that defines stat extractors for that sport's
box score format. All the heavy lifting (rate limiting, caching, game log
fetching) is handled by SportradarAdapter.
"""
from adapters.sportradar_base import SportradarAdapter


def _stat_int(stats: dict, *keys: str) -> int:
    for key in keys:
        value = stats.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


class WNBAAdapter(SportradarAdapter):
    sport_key = "wnba"
    sport_label = "WNBA"
    LOOKBACK_DAYS = 7

    STAT_ALIASES = {
        "points": "points",
        "pts": "points",
        "rebounds": "rebounds",
        "rebs": "rebounds",
        "total_rebounds": "rebounds",
        "assists": "assists",
        "asts": "assists",
        "three_pointers_made": "three_pointers_made",
        "three_points_made": "three_pointers_made",
        "three_pointers": "three_pointers_made",
        "3pt_made": "three_pointers_made",
        "3_pointers_made": "three_pointers_made",
        "threes": "three_pointers_made",
        "made_threes": "three_pointers_made",
        "points_rebounds_assists": "points_rebounds_assists",
        "pts_rebs_asts": "points_rebounds_assists",
        "pra": "points_rebounds_assists",
        "points_rebounds": "points_rebounds",
        "pts_rebs": "points_rebounds",
        "points_assists": "points_assists",
        "pts_asts": "points_assists",
        "rebounds_assists": "rebounds_assists",
        "rebs_asts": "rebounds_assists",
    }

    STAT_EXTRACTORS = {
        "points":                    lambda s: _stat_int(s, "points"),
        "rebounds":                  lambda s: _stat_int(s, "rebounds", "total_rebounds"),
        "assists":                   lambda s: _stat_int(s, "assists"),
        "three_pointers":            lambda s: _stat_int(s, "three_points_made", "three_pointers_made"),
        "three_pointers_made":       lambda s: _stat_int(s, "three_points_made", "three_pointers_made"),
        "steals":                    lambda s: _stat_int(s, "steals"),
        "blocks":                    lambda s: _stat_int(s, "blocks"),
        "turnovers":                 lambda s: _stat_int(s, "turnovers"),
        "offensive_rebounds":        lambda s: _stat_int(s, "offensive_rebounds", "off_rebounds"),
        "pra":                       lambda s: _stat_int(s, "points") + _stat_int(s, "rebounds", "total_rebounds") + _stat_int(s, "assists"),
        "points_rebounds_assists":   lambda s: _stat_int(s, "points") + _stat_int(s, "rebounds", "total_rebounds") + _stat_int(s, "assists"),
        "points_rebounds":           lambda s: _stat_int(s, "points") + _stat_int(s, "rebounds", "total_rebounds"),
        "points_assists":            lambda s: _stat_int(s, "points") + _stat_int(s, "assists"),
        "rebounds_assists":          lambda s: _stat_int(s, "rebounds", "total_rebounds") + _stat_int(s, "assists"),
        "blocks_steals":             lambda s: _stat_int(s, "blocks") + _stat_int(s, "steals"),
    }


class NBAAdapter(SportradarAdapter):
    sport_key = "nba"
    sport_label = "NBA"
    LOOKBACK_DAYS = 7

    # NBA uses the same stat fields as WNBA
    STAT_EXTRACTORS = WNBAAdapter.STAT_EXTRACTORS.copy()


class NFLAdapter(SportradarAdapter):
    sport_key = "nfl"
    sport_label = "NFL"
    LOOKBACK_DAYS = 14  # NFL plays weekly, need more lookback

    STAT_EXTRACTORS = {
        "passing_yards":   lambda s: int(s.get("passing", {}).get("yards", s.get("pass_yards", 0))),
        "rushing_yards":   lambda s: int(s.get("rushing", {}).get("yards", s.get("rush_yards", 0))),
        "receiving_yards": lambda s: int(s.get("receiving", {}).get("yards", s.get("rec_yards", 0))),
        "receptions":      lambda s: int(s.get("receiving", {}).get("receptions", s.get("receptions", 0))),
        "touchdowns":      lambda s: int(s.get("touchdowns", {}).get("total", s.get("total_touchdowns", 0))),
        "passing_tds":     lambda s: int(s.get("passing", {}).get("touchdowns", s.get("pass_td", 0))),
        "rush_attempts":   lambda s: int(s.get("rushing", {}).get("attempts", s.get("rush_att", 0))),
        "interceptions":   lambda s: int(s.get("passing", {}).get("interceptions", s.get("interceptions", 0))),
        "completions":     lambda s: int(s.get("passing", {}).get("completions", s.get("completions", 0))),
        "points":          lambda s: int(s.get("points", 0)),  # For kickers/team props
    }


class NHLAdapter(SportradarAdapter):
    sport_key = "nhl"
    sport_label = "NHL"
    LOOKBACK_DAYS = 7

    STAT_EXTRACTORS = {
        "goals":          lambda s: int(s.get("goals", 0)),
        "assists":        lambda s: int(s.get("assists", 0)),
        "points":         lambda s: int(s.get("goals", 0)) + int(s.get("assists", 0)),
        "shots_on_goal":  lambda s: int(s.get("shots", s.get("shots_on_goal", 0))),
        "saves":          lambda s: int(s.get("saves", 0)),
        "pp_points":      lambda s: int(s.get("powerplay_goals", 0)) + int(s.get("powerplay_assists", 0)),
        "blocks":         lambda s: int(s.get("blocked_shots", s.get("blocks", 0))),
        "hits":           lambda s: int(s.get("hits", 0)),
    }


class MMAAdapter(SportradarAdapter):
    sport_key = "mma"
    sport_label = "MMA"
    LOOKBACK_DAYS = 30  # MMA fighters fight rarely

    STAT_EXTRACTORS = {
        "total_rounds":    lambda s: int(s.get("total_rounds", s.get("rounds", 0))),
        "sig_strikes":     lambda s: int(s.get("significant_strikes", s.get("sig_strikes_landed", 0))),
    }


# Registry of all available adapters
ADAPTER_REGISTRY = {
    "mlb": None,       # MLB uses its own adapter (MLBAdapter), not Sportradar
    "wnba": WNBAAdapter,
    "nba": NBAAdapter,
    "nfl": NFLAdapter,
    "nhl": NHLAdapter,
    "soccer": None,    # SoccerAdapter is in soccer_adapter.py (FIFA WC 2026 build)
    "mma": MMAAdapter,
}


class EsportsAdapter:
    """
    Passthrough adapter for esports (CS2, LoL, Valorant, Dota 2, Rocket League, CoD).
    No external enrichment — esports game logs are not available via Sportradar trial.
    Props are scored math-only using market-implied probability from the line odds.
    The scorer will handle these gracefully when game_logs is empty.
    """
    sport_key = "esports"
    sport_label = "Esports"

    # Stat extractors are defined for the scorer's reference even if logs are empty.
    # If we ever add an esports data source, these are the canonical keys.
    STAT_EXTRACTORS = {
        "kills":         lambda s: float(s.get("kills", 0)),
        "deaths":        lambda s: float(s.get("deaths", 0)),
        "assists":       lambda s: float(s.get("assists", 0)),
        "headshots":     lambda s: float(s.get("headshots", 0)),
        "maps_played":   lambda s: float(s.get("maps_played", s.get("maps", 0))),
        "adr":           lambda s: float(s.get("adr", s.get("avg_damage_round", 0))),
        "rating":        lambda s: float(s.get("rating", 0)),
        "goals":         lambda s: float(s.get("goals", 0)),    # Rocket League
        "saves":         lambda s: float(s.get("saves", 0)),    # Rocket League
        "score":         lambda s: float(s.get("score", 0)),    # Rocket League
        "eliminations":  lambda s: float(s.get("eliminations", s.get("kills", 0))),  # CoD
        "kills_assists": lambda s: float(s.get("kills", 0)) + float(s.get("assists", 0)),
    }

    def enrich_props(self, props: list) -> list:
        """
        No enrichment — mark each prop so the scorer knows logs are unavailable.
        Math-only scoring will use market odds as the probability anchor.
        """
        for p in props:
            p.setdefault("game_logs", [])
            p.setdefault("season_avg", None)
            p["_enrichment_source"] = "none"
            p["_sport_key"] = "esports"
        return props
