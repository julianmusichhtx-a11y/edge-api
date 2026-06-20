"""
Odds conversion and math utilities.
"""
import math


def american_to_implied(odds: int | float | None) -> float | None:
    """Convert American odds to implied probability (0-1)."""
    if odds is None or odds == 0:
        return None
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def implied_to_american(prob: float) -> int:
    """Convert implied probability (0-1) to American odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    return round(100 * (1 - prob) / prob)


def calculate_edge(model_prob: float, market_prob: float) -> float:
    """Calculate edge as model probability minus market probability."""
    return model_prob - market_prob


def calculate_ev(model_prob: float, odds: int) -> float:
    """Calculate expected value of a bet at given odds."""
    if odds > 0:
        profit = odds / 100
    else:
        profit = 100 / abs(odds)
    return (model_prob * profit) - (1 - model_prob)


def kelly_fraction(model_prob: float, odds: int, fraction: float = 0.25) -> float:
    """
    Calculate Kelly Criterion fraction for bankroll sizing.
    Uses fractional Kelly (default 25%) for safety.
    """
    if odds > 0:
        decimal_odds = 1 + odds / 100
    else:
        decimal_odds = 1 + 100 / abs(odds)

    edge = model_prob * decimal_odds - 1
    if edge <= 0:
        return 0.0

    kelly = edge / (decimal_odds - 1)
    return max(0, kelly * fraction)


def classify_volatility(stat_key: str, line: float) -> str:
    """Classify a prop's volatility based on stat type and line."""
    high_vol = {"home_runs", "stolen_bases", "rbi", "batter_strikeouts",
                "touchdowns", "interceptions", "total_rounds"}
    low_vol = {"hits", "hits_runs_rbis", "strikeouts", "points", "rebounds",
               "assists", "passing_yards", "rushing_yards", "receiving_yards"}

    if stat_key in high_vol:
        return "high"
    if stat_key in low_vol and line <= 1.5:
        return "low"
    if stat_key in low_vol:
        return "medium"
    return "medium"
