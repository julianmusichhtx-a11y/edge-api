"""
Odds conversion and EV calculations.
"""

def american_to_implied_prob(american_odds: int) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def calculate_ev(model_prob: float, american_odds: int = -110) -> float:
    """Expected value per $100 bet."""
    implied = american_to_implied_prob(american_odds)
    decimal = 1.909 if american_odds == -110 else (100 / abs(american_odds) + 1 if american_odds > 0 else 1 + 100 / abs(american_odds))
    return (model_prob * (decimal - 1) - (1 - model_prob)) * 100

def kelly_criterion(model_prob: float, american_odds: int = -110, fraction: float = 0.25) -> float:
    """Recommended bet size as fraction of bankroll."""
    b = (100 / abs(american_odds)) if american_odds < 0 else (american_odds / 100)
    q = 1 - model_prob
    kelly = (model_prob * (b + 1) - 1) / b
    return max(0, kelly * fraction)
