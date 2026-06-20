"""
Core scoring engine.
Takes enriched props and produces calibrated probabilities + edge.
MVP uses weighted recent form + simple adjustments.
Future: Replace with trained XGBoost / LightGBM per sport.
"""

from typing import List, Dict, Any
from utils.odds_math import american_to_implied_prob, calculate_ev
import numpy as np

def estimate_probability(prop: Dict[str, Any], platform: str) -> float:
    """
    Estimate P(Over hits) using available stats.
    This is the heart of the 'predictive model'.
    """
    stats = prop.get("player_stats", {})
    line = prop["line"]
    stat_type = prop["stat"]

    season_avg = stats.get("season_avg", line)
    last5 = stats.get("last_5_avg", season_avg)
    last10 = stats.get("last_10_avg", season_avg)
    usage = stats.get("usage_or_minutes", 30)  # proxy for opportunity

    # Weighted projection (more weight to recent)
    projected = (last5 * 0.5 + last10 * 0.3 + season_avg * 0.2)

    # Simple adjustments (extend this heavily)
    adjustments = 0.0
    if stats.get("recent_trend") == "hot":
        adjustments += 0.08
    if stats.get("recent_trend") == "cold":
        adjustments -= 0.06

    # Usage / opportunity boost
    if usage and usage > 32:
        adjustments += 0.04

    # Matchup (very basic for MVP)
    if "top-10 defense" in (stats.get("matchup_note") or "").lower():
        adjustments -= 0.05

    final_proj = projected + (adjustments * projected * 0.15)  # dampen adjustment

    # Convert projection to probability of beating the line
    # Use a simple logistic-like function (std ~ 8-12 for most props)
    std = 9.5 if "Points" in stat_type or "Yards" in stat_type else 6.0
    z = (final_proj - line) / std
    prob = 1 / (1 + np.exp(-z * 1.2))  # calibrated-ish

    # Clamp
    return max(0.15, min(0.92, round(prob, 4)))

def score_props(enriched_props: List[Dict], platform: str = "PrizePicks", min_edge: float = 0.05, persona: str = "hybrid") -> List[Dict]:
    scored = []
    for p in enriched_props:
        model_prob = estimate_probability(p, platform)

        # Market prob - assume typical -110 for most DFS unless provided
        # In real version, we would pull from The Odds API for sharp consensus
        market_prob = 0.5238  # ~ -110 implied (break even for -110)

        edge = round(model_prob - market_prob, 4)

        if edge >= min_edge + 0.03:
            verdict = "PLAY"
            tier = "A"
        elif edge >= min_edge:
            verdict = "LEAN"
            tier = "B"
        else:
            verdict = "PASS"
            tier = "C"

        # Build rich context for the AI narrative generator (this is your moat)
        stats = p.get("player_stats", {})
        context_parts = [
            f"Season avg {stats.get('season_avg', 'N/A')} vs line {p['line']}.",
            f"Recent form (L5/L10): {stats.get('last_5_avg', 'N/A')}/{stats.get('last_10_avg', 'N/A')}.",
            f"Trend: {stats.get('recent_trend', 'neutral')}.",
            f"Matchup: {stats.get('matchup_note', 'standard')}.",
            f"Usage/opportunity: {stats.get('usage_or_minutes', 'N/A')}.",
        ]
        if stats.get("note"):
            context_parts.append(stats["note"])

        analysis_context = " | ".join(context_parts)

        scored.append({
            "player": p["player"],
            "stat": p["stat"],
            "line": p["line"],
            "sport": p["sport"],
            "model_prob": model_prob,
            "market_prob": market_prob,
            "edge": edge,
            "verdict": verdict,
            "tier": tier,
            "player_stats": stats,
            "key_factors": [k for k in ["recent_form", "matchup", "usage", "trend"] if stats.get(k)],
            "analysis_context": analysis_context,
            "confidence": round(min(0.95, 0.6 + abs(edge) * 4), 2)  # Higher edge = higher confidence in our model
        })
    return scored
