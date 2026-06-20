from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# === These were missing - adding them so imports work ===
@dataclass
class PlayerStats:
    player_name: str = ""
    season_avg: Optional[float] = None
    recent_form: Optional[float] = None
    season_ops: Optional[float] = None
    season_era: Optional[float] = None
    season_whip: Optional[float] = None

@dataclass
class GameInfo:
    game_time: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    venue: Optional[str] = None

class BaseSportAdapter(ABC):
    """Base class for all sport adapters."""

    @abstractmethod
    def enrich_prop(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def enrich_props(self, props: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_player_stats(self, player_name: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_todays_games(self) -> List[Dict[str, Any]]:
        pass

    @property
    @abstractmethod
    def sport_key(self) -> str:
        pass

    @property
    @abstractmethod
    def sport_label(self) -> str:
        pass