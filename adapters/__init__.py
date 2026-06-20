from .base_adapter import BaseSportAdapter as BaseAdapter
from .mlb_adapter import MLBAdapter
from .sport_adapters import (
    NBAAdapter,
    WNBAAdapter,
    NFLAdapter,
    NHLAdapter,
    SoccerAdapter,
    MMAAdapter,
    EsportsAdapter,
)

def get_adapter(sport: str):
    sport = sport.lower()
    if sport == "mlb":
        return MLBAdapter()
    elif sport == "nba":
        return NBAAdapter()
    elif sport == "wnba":
        return WNBAAdapter()
    elif sport == "nfl":
        return NFLAdapter()
    elif sport == "nhl":
        return NHLAdapter()
    elif sport == "soccer":
        return SoccerAdapter()
    elif sport == "mma":
        return MMAAdapter()
    elif sport in ("esports", "cs2", "lol", "league of legends", "valorant",
                   "dota 2", "dota2", "rocket league", "call of duty", "cod"):
        return EsportsAdapter()
    else:
        return None
