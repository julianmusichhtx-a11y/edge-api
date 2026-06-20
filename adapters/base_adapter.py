"""
Base adapter interface.
Every sport adapter produces the same PlayerStats shape so the scoring
engine doesn't need to know which sport it's scoring.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlayerStats:
    """Standardized player stats — same shape regardless of sport."""
    player_name: str
    player_id: Optional[str] = None
    team: str = ""
    opponent: str = ""
    season_avg: Optional[float] = None        # Per-game average for the relevant stat
    last5: list[float] = field(default_factory=list)   # Last 5 game values
    last10: list[float] = field(default_factory=list)  # Last 10 game values
    games_played: int = 0
    is_starter: Optional[bool] = None         # For pitchers, QBs, etc.
    projected_minutes: Optional[float] = None  # For basketball
    injury_status: Optional[str] = None        # Active, Questionable, Out, etc.
    matchup_data: Optional[dict] = None        # Opponent-specific context


@dataclass
class GameInfo:
    """Standardized game info."""
    game_id: str
    status: str  # scheduled, in_progress, closed
    home_team: str
    away_team: str
    scheduled: str  # ISO datetime
    venue: Optional[str] = None


class BaseSportAdapter(ABC):
    """
    Abstract base class for sport-specific data adapters.
    Each sport implements these methods to fetch data from its API(s)
    and return standardized PlayerStats objects.
    """

    @property
    @abstractmethod
    def sport_key(self) -> str:
        """Short key like 'mlb', 'nba', 'nfl'."""
        ...

    @property
    @abstractmethod
    def sport_label(self) -> str:
        """Display name like 'MLB', 'NBA', 'NFL'."""
        ...

    @abstractmethod
    async def get_todays_games(self) -> list[GameInfo]:
        """Fetch today's schedule. Returns list of GameInfo."""
        ...

    @abstractmethod
    async def get_player_stats(
        self, player_name: str, stat_key: str, line: float,
        home_team: str = "", away_team: str = ""
    ) -> Optional[PlayerStats]:
        """
        Fetch player stats relevant to a specific prop.
        Returns PlayerStats with season_avg, last5, last10 populated
        for the given stat_key, or None if player not found.
        """
        ...

    @abstractmethod
    async def enrich_props(self, props: list[dict]) -> list[dict]:
        """
        Batch-enrich a list of props with player stats.
        This is the main entry point called by the scoring engine.
        Returns the same props list with _playerStats attached.
        """
        ...

    def get_stat_extractor(self, stat_key: str):
        """
        Returns a function that extracts the relevant stat value
        from a game log entry. Override per sport.
        """
        raise NotImplementedError(f"No stat extractor for {stat_key}")
