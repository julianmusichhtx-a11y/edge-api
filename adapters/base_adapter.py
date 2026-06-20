from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class PlayerStats:
    """
    Matches the fields SportradarAdapter.get_player_stats() actually
    constructs (see adapters/sportradar_base.py). Kept here mainly for
    type-hint/documentation purposes — the hot enrichment path
    (enrich_props) writes plain dicts into prop["_playerStats"] directly
    rather than building this dataclass, since that's what scoring/scorer.py
    consumes.
    """
    player_name: str = ""
    player_id: Optional[str] = None
    team: str = ""
    season_avg: Optional[float] = None
    last5: List[float] = field(default_factory=list)
    last10: List[float] = field(default_factory=list)
    games_played: int = 0


@dataclass
class GameInfo:
    game_id: str = ""
    status: str = "unknown"
    game_time: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    venue: Optional[str] = None
    # Some call sites use `scheduled` instead of `game_time` — keep both
    # so either naming convention works without a KeyError.
    scheduled: Optional[str] = None


class BaseSportAdapter(ABC):
    """
    Base class for all sport adapters.

    Note: there is intentionally no abstract `enrich_prop` (singular) method.
    main.py calls `enrich_props` (plural, batch) exactly once per request —
    that's what lets Sportradar-backed adapters load rosters/game-logs once
    and reuse them across the whole slate instead of refetching per prop.
    `enrich_props` may be sync (MLBAdapter) or async (SportradarAdapter
    subclasses) — main.py checks via inspect.iscoroutinefunction and awaits
    accordingly.
    """

    @abstractmethod
    def enrich_props(self, props: List[Dict[str, Any]]):
        """Returns List[Dict] if sync, or an awaitable resolving to one if async."""
        pass

    @abstractmethod
    def get_player_stats(self, player_name: str):
        pass

    @abstractmethod
    def get_todays_games(self):
        pass

    @property
    @abstractmethod
    def sport_key(self) -> str:
        pass

    @property
    @abstractmethod
    def sport_label(self) -> str:
        pass