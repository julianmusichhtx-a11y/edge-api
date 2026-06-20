"""
Central configuration for Edge API.
All API keys from environment variables.
Rate limits and sport adapter registry.
"""

import os
from typing import Dict, Type

class Settings:
    # API Keys (set in Railway Variables or .env)
    SPORTRADAR_API_KEY: str = os.getenv("SPORTRADAR_API_KEY", "")
    THE_ODDS_API_KEY: str = os.getenv("THE_ODDS_API_KEY", "")
    SPORTSDATAIO_API_KEY: str = os.getenv("SPORTSDATAIO_API_KEY", "")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    # MLB Stats API is public, no key needed

    # Rate limiting (per source)
    SPORTRADAR_RPM: int = 60  # Trial is often 1 req/sec = 60/min
    THE_ODDS_RPM: int = 10    # Free tier careful

    # Caching TTLs (seconds)
    BOXSCORE_TTL: int = 7 * 24 * 3600      # 7 days for completed games (immutable)
    LIVE_GAME_TTL: int = 300               # 5 min for in-progress
    PLAYER_LOG_TTL: int = 3600             # 1 hour for recent form
    ODDS_CONSENSUS_TTL: int = 1800         # 30 min for line comparison

    # Supported sports -> adapter class name (imported dynamically)
    SPORT_ADAPTERS: Dict[str, str] = {
        "mlb": "MLBAdapter",
        "nba": "NBAAdapter",
        "wnba": "WNBAAdapter",
        "nfl": "NFLAdapter",
        "nhl": "NHLAdapter",
        "soccer": "SoccerAdapter",
        # Add more as implemented
    }

    # Default stat type mappings (for normalization)
    STAT_MAPS: Dict[str, Dict[str, str]] = {
        "nba": {
            "Points": "pts", "Rebounds": "reb", "Assists": "ast",
            "PRA": "pts_reb_ast", "3PTM": "fg3m"
        },
        "mlb": {
            "Hits": "hits", "Home Runs": "hr", "RBIs": "rbi",
            "Strikeouts": "so", "Total Bases": "tb"
        },
        # Extend for others
    }

settings = Settings()

def get_adapter_class(sport: str) -> Optional[Type]:
    from adapters import sport_adapters  # lazy import to avoid circular
    adapter_name = settings.SPORT_ADAPTERS.get(sport.lower())
    if adapter_name and hasattr(sport_adapters, adapter_name):
        return getattr(sport_adapters, adapter_name)
    return None