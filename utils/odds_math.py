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


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1: Sharp Line De-Vigging Engine
# ─────────────────────────────────────────────────────────────────────────────

def devig_sharp_line(over_odds: float | None, under_odds: float | None) -> tuple[float, float] | None:
    """
    Remove the bookmaker's vig from a two-sided market to get true probabilities.

    Uses the standard additive de-vig method:
        true_p = implied_p / (implied_over + implied_under)

    Args:
        over_odds:  American odds for the over/higher side (e.g. -115, +110)
        under_odds: American odds for the under/lower side

    Returns:
        (true_over_prob, true_under_prob) as decimals, or None if odds missing.

    Example:
        devig_sharp_line(-115, -105) -> (0.511, 0.489)
        devig_sharp_line(-110, -110) -> (0.500, 0.500)  # balanced line
        devig_sharp_line(+120, -150) -> (0.455, 0.600) raw → devigged (0.431, 0.569)
    """
    if over_odds is None or under_odds is None:
        return None

    implied_over  = american_to_implied(over_odds)
    implied_under = american_to_implied(under_odds)

    if implied_over is None or implied_under is None:
        return None

    total = implied_over + implied_under
    if total <= 0:
        return None

    true_over  = implied_over  / total
    true_under = implied_under / total
    return round(true_over, 4), round(true_under, 4)


def calculate_true_edge(model_prob: float,
                        over_odds: float | None,
                        under_odds: float | None,
                        side: str) -> dict:
    """
    Calculate edge against the de-vigged (true) market probability.

    Returns a dict with:
        true_market_prob  — de-vigged probability for the selected side
        raw_market_prob   — vigged probability (what we used to use)
        true_edge         — model_prob minus de-vigged market
        raw_edge          — model_prob minus vigged market (legacy)
        vig               — the bookmaker's margin (total implied - 1)
        devigged          — bool, whether de-vigging succeeded
    """
    raw_market = american_to_implied(
        over_odds if side == "Higher" else under_odds
    ) or 0.50

    devigged = devig_sharp_line(over_odds, under_odds)

    if devigged:
        true_over, true_under = devigged
        true_market = true_over if side == "Higher" else true_under
        implied_over  = american_to_implied(over_odds)  or 0.5
        implied_under = american_to_implied(under_odds) or 0.5
        vig = round((implied_over + implied_under - 1.0), 4)
    else:
        true_market = raw_market
        vig = 0.0

    return {
        "true_market_prob": round(true_market, 4),
        "raw_market_prob":  round(raw_market,  4),
        "true_edge":        round(model_prob - true_market, 4),
        "raw_edge":         round(model_prob - raw_market,  4),
        "vig":              vig,
        "devigged":         devigged is not None,
    }


def kelly_criterion(model_prob: float,
                    payout_multiplier: float,
                    fraction: float = 0.5) -> dict:
    """
    Kelly Criterion sizing for fixed-multiplier DFS platforms.

    On Underdog/PrizePicks, you don't get traditional odds — you get a fixed
    payout multiplier. Kelly formula for fixed-multiplier:

        f* = (b*p - q) / b
        where b = payout_multiplier - 1, p = win_prob, q = 1 - p

    Args:
        model_prob:        True probability of winning this leg (0-1)
        payout_multiplier: e.g. 3.0 for 2-pick power play, 10.0 for 5-pick flex
        fraction:          Kelly fraction — 0.5 = Half Kelly (recommended)

    Returns dict with full_kelly, fractional_kelly, ev_per_dollar, is_positive_ev
    """
    b = payout_multiplier - 1.0  # net profit per dollar
    p = model_prob
    q = 1.0 - p

    ev_per_dollar = (p * payout_multiplier) - 1.0

    if b <= 0 or ev_per_dollar <= 0:
        return {
            "full_kelly": 0.0,
            "fractional_kelly": 0.0,
            "ev_per_dollar": round(ev_per_dollar, 4),
            "is_positive_ev": False,
        }

    full_kelly = (b * p - q) / b
    full_kelly = max(0.0, full_kelly)

    return {
        "full_kelly":        round(full_kelly, 4),
        "fractional_kelly":  round(full_kelly * fraction, 4),
        "ev_per_dollar":     round(ev_per_dollar, 4),
        "is_positive_ev":    ev_per_dollar > 0,
    }