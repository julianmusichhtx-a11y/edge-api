from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseSportAdapter(ABC):
    """Base class for all sport adapters."""

    @abstractmethod
    def enrich_prop(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pass

    # Add these to satisfy the abstract class
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