"""
Scoring engine — calculates model probabilities, market probabilities,
edges, and verdicts from standardized player stats.

This replaces propScoring.js. The math is sport-agnostic — it just needs
season_avg, last5, last10, line, and odds.
"""
from utils.odds_math import american_to_implied, classify_volatility, calculate_edge


def score_prop(prop: dict) -> dict | None:
    """
    Score a single prop. Returns scoring data or None if insufficient data.

    Input prop must have:
      - _playerStats: { seasonAvg, last5, last10 }
      - line: float
      - higher_american_odds / lower_american_odds (optional)
      - stat_display or stat_type: string

    Returns dict with: modelProb, marketProb, edge, verdict, tier, volatility, signals, selectedSide
    """
    stats = prop.get("_playerStats")
    if not stats:
        return None

    season_avg = stats.get("seasonAvg")
    last5 = stats.get("last5", [])
    last10 = stats.get("last10", [])
    line = float(prop.get("line", 0))

    if len(last5) < 3 and season_avg is None:
        return None

    stat_display = prop.get("stat_display", prop.get("stat_type", "")).lower()

    # ── Calculate hit rates ──
    higher_l5 = sum(1 for v in last5 if v > line) / max(len(last5), 1) if last5 else 0.5
    higher_l10 = sum(1 for v in last10 if v > line) / max(len(last10), 1) if last10 else 0.5
    lower_l5 = sum(1 for v in last5 if v < line) / max(len(last5), 1) if last5 else 0.5
    lower_l10 = sum(1 for v in last10 if v < line) / max(len(last10), 1) if last10 else 0.5

    # ── Season average signal ──
    avg_signal_higher = 0.0
    avg_signal_lower = 0.0
    if season_avg is not None and line > 0:
        gap = (season_avg - line) / max(line, 0.5)
        avg_signal_higher = min(max(gap * 0.3, -0.15), 0.15)
        avg_signal_lower = -avg_signal_higher

    # ── Smoothed hit rates (blend L5 and L10) ──
    if last5 and last10:
        smoothed_higher = higher_l5 * 0.6 + higher_l10 * 0.3 + (0.5 + avg_signal_higher) * 0.1
        smoothed_lower = lower_l5 * 0.6 + lower_l10 * 0.3 + (0.5 + avg_signal_lower) * 0.1
    elif last5:
        smoothed_higher = higher_l5 * 0.7 + (0.5 + avg_signal_higher) * 0.3
        smoothed_lower = lower_l5 * 0.7 + (0.5 + avg_signal_lower) * 0.3
    else:
        smoothed_higher = 0.5 + avg_signal_higher
        smoothed_lower = 0.5 + avg_signal_lower

    # ── Clamp to reasonable range (no model should say 95%+) ──
    smoothed_higher = max(0.20, min(0.85, smoothed_higher))
    smoothed_lower = max(0.20, min(0.85, smoothed_lower))

    # ── Market probabilities from odds ──
    higher_odds = prop.get("higher_american_odds") or prop.get("american_odds")
    lower_odds = prop.get("lower_american_odds") or prop.get("lower_odds")

    higher_market = american_to_implied(higher_odds) if higher_odds else 0.5
    lower_market = american_to_implied(lower_odds) if lower_odds else 0.5

    # ── Calculate edges ──
    higher_edge = smoothed_higher - (higher_market or 0.5)
    lower_edge = smoothed_lower - (lower_market or 0.5)

    # ── Select best side ──
    if higher_edge >= lower_edge:
        selected_side = "Higher"
        model_prob = smoothed_higher
        market_prob = higher_market or 0.5
        edge = higher_edge
    else:
        selected_side = "Lower"
        model_prob = smoothed_lower
        market_prob = lower_market or 0.5
        edge = lower_edge

    # ── Resolve canonical stat key for volatility ──
    from config import PROP_STAT_MAP
    stat_key = None
    sd = stat_display.strip()
    for prefix in ["1q ", "2q ", "1h ", "2h ", "first quarter ", "first half ", "second half "]:
        if sd.startswith(prefix):
            sd = sd[len(prefix):]
            break
    for k in sorted(PROP_STAT_MAP.keys(), key=len, reverse=True):
        if k in sd or sd in k:
            stat_key = PROP_STAT_MAP[k]
            break

    volatility = classify_volatility(stat_key or "", line)

    # ── Count signals ──
    signals = 0
    if last5 and higher_l5 >= 0.6: signals += 1
    if last10 and higher_l10 >= 0.6: signals += 1
    if season_avg is not None and season_avg > line * 1.1: signals += 1
    if edge > 0.05: signals += 1
    if edge > 0.15: signals += 1

    # ── Classify verdict and tier ──
    verdict, tier = classify_pick(model_prob, edge, signals, volatility)

    return {
        "modelProb": round(model_prob * 100, 1),
        "marketProb": round(market_prob * 100, 1),
        "edge": round(edge * 100, 1),
        "verdict": verdict,
        "tier": tier,
        "volatility": volatility,
        "signals": signals,
        "selectedSide": selected_side,
        "hitRates": {
            "l5Higher": round(higher_l5 * 100),
            "l10Higher": round(higher_l10 * 100),
            "l5Lower": round(lower_l5 * 100),
            "l10Lower": round(lower_l10 * 100),
        },
        "seasonAvg": round(season_avg, 2) if season_avg else None,
        "line": line,
    }


def classify_pick(model_prob: float, edge: float, signals: int, volatility: str) -> tuple[str, str]:
    """Classify a pick into verdict and tier."""
    if edge <= 0:
        return "SKIP", "Pass"

    # Tier classification
    if model_prob >= 0.65 and signals >= 3 and volatility in ("low", "medium"):
        tier = "A"
    elif model_prob >= 0.58 and signals >= 2:
        tier = "B"
    elif model_prob >= 0.53:
        tier = "C"
    else:
        tier = "Pass"

    # Verdict classification
    if tier == "Pass" or edge < 0.02:
        verdict = "SKIP"
    elif tier == "A" and edge >= 0.10:
        verdict = "STRONG PLAY"
    elif tier in ("A", "B") and edge >= 0.05:
        verdict = "PLAY"
    elif edge >= 0.03:
        verdict = "LEAN"
    else:
        verdict = "SKIP"

    return verdict, tier


def filter_prop(prop: dict) -> dict:
    """
    Filter a prop before scoring.
    Returns { status: 'pass' | 'hard_reject' | 'warn', reason: str }
    """
    stats = prop.get("_playerStats")
    line = float(prop.get("line", 0))

    # Fantasy props not supported
    stat_display = prop.get("stat_display", prop.get("stat_type", "")).lower()
    if "fantasy" in stat_display:
        return {"status": "hard_reject", "reason": "Fantasy score props not supported"}

    # Need player stats for multi-event props
    if not stats and line > 0.5:
        return {"status": "hard_reject", "reason": f"Multi-event prop (line {line}) requires player stats — none found"}

    # Need minimum game data
    if stats:
        last5 = stats.get("last5", [])
        if len(last5) < 3 and stats.get("seasonAvg") is None:
            return {"status": "hard_reject", "reason": f"Insufficient player data (need 3+ recent games, found {len(last5)})"}

    return {"status": "pass", "reason": ""}
