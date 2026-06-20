from .base_adapter import BaseSportAdapter as BaseAdapter
from .mlb_adapter import MLBAdapter
from .sport_adapters import (
    NBAAdapter,
    WNBAAdapter,
    NFLAdapter,
    NHLAdapter,
    SoccerAdapter,
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
    else:
        return None