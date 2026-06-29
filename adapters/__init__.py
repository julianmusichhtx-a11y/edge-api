from .base_adapter import BaseSportAdapter as BaseAdapter
from .mlb_adapter import MLBAdapter
from .soccer_adapter import SoccerAdapter
from .sport_adapters import (
    NBAAdapter,
    WNBAAdapter,
    NFLAdapter,
    NHLAdapter,
    MMAAdapter,
    EsportsAdapter,
    TennisAdapter,
)

def get_adapter(sport: str):
    sport = (sport or "").lower()
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
    elif sport in ("soccer", "world_cup", "worldcup", "fifa_world_cup"):
        return SoccerAdapter()
    elif sport in ("tennis", "tennis_atp", "tennis_wta", "atp", "wta"):
        return TennisAdapter()
    elif sport == "mma":
        return MMAAdapter()
    elif sport in ("esports", "cs2", "lol", "league of legends", "valorant",
                   "dota 2", "dota2", "rocket league", "call of duty", "cod"):
        return EsportsAdapter()
    else:
        return None
