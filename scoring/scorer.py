"""
Scoring engine — calculates model probabilities, market probabilities,
edges, and verdicts from standardized player stats.

This replaces propScoring.js. The math is sport-agnostic — it just needs
season_avg, last5, last10, line, and odds.

Calibration notes (June 2026):
  - Pure L5/L10 hit rates led to systematic overconfidence on low-line props.
    A player going 4/5 on "scored a run" gets hit_rate=80%, but the Poisson
    true probability for a 0.6 runs/game player is only ~45%.
  - For lines <= 1.5 (binary-ish thresholds), we now anchor to a Poisson
    base probability derived from the season average, then apply a modest
    recency adjustment from L5/L10. This brings model probs in line with
    what's actually achievable.
  - For higher lines (4.5+), we use a blended approach that still respects
    the season average more than recent streaks.
  - Signal-scaled ceiling keeps max prob at 80% even for 5-signal A-tier picks.
  - Market-anchoring dampener: if model deviates from market by > 28pp in
    either direction, we pull it 35% back toward market. This is a safety
    valve for cases where our model over- or under-reacts to small samples.
"""
from __future__ import annotations

import math
from utils.odds_math import american_to_implied, classify_volatility, calculate_edge, devig_sharp_line, kelly_criterion, calculate_true_edge, kelly_criterion


def _safe_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return None
        return num
    except (TypeError, ValueError):
        return None


def _clean_numbers(values) -> list[float]:
    if not isinstance(values, list):
        return []
    return [num for num in (_safe_float(v) for v in values) if num is not None]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std_dev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = _mean(values)
    variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _default_projection_volatility(sport: str, stat: str) -> float:
    sd = (stat or "").lower()
    sport_key = (sport or "").lower()
    if sport_key in ("nba", "wnba"):
        if any(x in sd for x in ("points", "rebounds", "assists", "pra")):
            return 6.0
        return 3.5
    if sport_key == "mlb":
        if any(x in sd for x in ("home run", "stolen base")):
            return 1.2
        if any(x in sd for x in ("strikeout", "pitching outs")):
            return 2.5
        return 1.8
    if sport_key == "soccer":
        if any(x in sd for x in ("goal", "assist")):
            return 1.0
        return 3.0
    return 4.0


def estimate_probability_from_projection(
    projection: float,
    line: float,
    std_dev: float | None,
    sample_size: int,
    sport: str,
    stat: str,
) -> float:
    """
    Conservative projection-vs-line probability estimate.

    This is exposed for projection metadata only; the existing scorer's
    calibrated modelProb remains the canonical score when score_prop has data.
    """
    projection = float(projection)
    line = float(line)
    sample_size = max(0, int(sample_size or 0))
    sigma = std_dev if std_dev and std_dev > 0 else _default_projection_volatility(sport, stat)
    sigma = max(float(sigma), 0.75)

    z = (projection - line) / sigma
    higher_prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    if sample_size >= 8 and std_dev and std_dev > 0:
        cap = 0.68
    elif sample_size >= 3:
        cap = 0.62
    else:
        cap = 0.58

    lower_bound = max(0.32, 1.0 - cap)
    return max(lower_bound, min(cap, higher_prob))


def _projection_source_for_prop(prop: dict) -> str:
    if prop.get("_projectionSource"):
        return prop["_projectionSource"]
    sport_key = (prop.get("_sport_key") or prop.get("sport") or "").lower()
    if sport_key in ("wnba", "nba", "nfl", "nhl", "soccer", "mma"):
        return "sportradar_recent_stats"
    if sport_key == "mlb":
        return "mlb_stats_recent_stats"
    return "recent_stats"


def build_projection_metadata(prop: dict, scored: dict | None = None) -> dict:
    stats = prop.get("_playerStats") or {}
    season_avg = _safe_float(stats.get("seasonAvg"))
    last5 = _clean_numbers(stats.get("last5", []))
    last10 = _clean_numbers(stats.get("last10", []))
    games_played = int(_safe_float(stats.get("gamesPlayed") or stats.get("games_played")) or 0)
    line = _safe_float(prop.get("line"))
    sport_key = prop.get("_sport_key") or prop.get("sport") or ""
    stat_display = prop.get("stat_display", prop.get("stat_type", ""))

    buckets: list[tuple[float, float]] = []
    last5_avg = _mean(last5)
    last10_avg = _mean(last10)
    if last5_avg is not None:
        buckets.append((0.45, last5_avg))
    if last10_avg is not None:
        buckets.append((0.35, last10_avg))
    if season_avg is not None:
        buckets.append((0.20, season_avg))

    if line is None or not buckets:
        reason = prop.get("_projectionUnavailableReason")
        return {
            "projection": None,
            "projectionEdge": None,
            "projectionSource": None,
            "probabilitySource": "fallback_model" if scored and scored.get("modelProb") is not None else None,
            "confidence": None,
            "sampleSize": max(len(last10), len(last5), 0),
            "recentAverage": None,
            "recentStdDev": None,
            "seasonAverage": season_avg,
            "hitRateL5": None,
            "hitRateL10": None,
            "playerMatchConfidence": prop.get("_playerMatchConfidence"),
            "projectionAvailable": False,
            "projectionUnavailableReason": reason,
            "unavailableReason": reason,
        }

    total_weight = sum(weight for weight, _ in buckets)
    projection = sum(weight * value for weight, value in buckets) / total_weight
    recent_values = last10 or last5
    sample_size = len(recent_values) or games_played
    recent_average = _mean(recent_values)
    recent_std_dev = _std_dev(recent_values)
    projected_higher_prob = estimate_probability_from_projection(
        projection, line, recent_std_dev, sample_size, sport_key, stat_display
    )

    if not recent_values and season_avg is not None:
        confidence = 0.45
    elif sample_size >= 10 and recent_std_dev is not None:
        confidence = 0.62
    elif sample_size >= 5:
        confidence = 0.56
    elif sample_size >= 3:
        confidence = 0.50
    else:
        confidence = 0.42

    selected_side = (scored or {}).get("selectedSide") or prop.get("side") or ""
    selected_side = str(selected_side).lower()
    projection_probability = (
        1.0 - projected_higher_prob if selected_side == "lower" else projected_higher_prob
    )

    return {
        "projection": round(projection, 2),
        "projectionEdge": round(projection - line, 2),
        "projectionSource": _projection_source_for_prop(prop),
        "probabilitySource": "projection_estimator",
        "projectionProbability": round(projection_probability * 100, 1),
        "confidence": round(confidence, 2),
        "sampleSize": sample_size,
        "recentAverage": round(recent_average, 2) if recent_average is not None else None,
        "recentStdDev": round(recent_std_dev, 2) if recent_std_dev is not None else None,
        "seasonAverage": round(season_avg, 2) if season_avg is not None else None,
        "hitRateL5": prop.get("hitRateL5"),
        "hitRateL10": prop.get("hitRateL10"),
        "playerMatchConfidence": prop.get("_playerMatchConfidence"),
        "unavailableReason": None,
        "projectionAvailable": True,
    }


def _projection_failure_sample(prop: dict, reason: str) -> dict:
    return {
        "player": prop.get("player_name") or prop.get("player") or "",
        "stat": prop.get("stat_display") or prop.get("stat_type") or "",
        "line": prop.get("line"),
        "side": prop.get("side") or prop.get("selectedSide") or "",
        "reason": reason,
    }


def _record_projection_health(projection_health: dict, prop: dict, projection: dict):
    if projection.get("projectionAvailable"):
        projection_health["projectionMatched"] += 1
        projection_health["provider"] = projection_health["provider"] or "railway_prediction_api"
        return

    projection_health["projectionUnavailable"] += 1
    reason = projection.get("projectionUnavailableReason") or prop.get("_projectionUnavailableReason") or "no_stat_history"
    reason_counts = projection_health["unavailableReasons"]
    if reason not in reason_counts:
        reason = "projection_exception"
    reason_counts[reason] += 1
    if len(projection_health["sampleFailures"]) < 10:
        projection_health["sampleFailures"].append(_projection_failure_sample(prop, reason))


def _poisson_prob_higher(lambda_val: float, line: float) -> float:
    """
    P(X >= line + epsilon) for a Poisson distributed stat with mean lambda_val.
    Used for low-line binary props (runs 0.5, hits 0.5, hits 1.5, etc.)
    where line is 0.5 or 1.5 — i.e. we need at least ceil(line) events.
    """
    k = int(line + 0.6)  # 0.5 -> need >= 1, 1.5 -> need >= 2
    k = max(1, k)
    # P(X >= k) = 1 - sum P(X=i) for i in 0..k-1
    cumulative = 0.0
    for i in range(k):
        cumulative += (lambda_val ** i * math.exp(-lambda_val)) / math.factorial(i)
    return max(0.05, min(0.95, 1.0 - cumulative))


def _poisson_prob_lower(lambda_val: float, line: float) -> float:
    """
    P(X < line) = P(X <= floor(line)) for a Poisson stat.
    For line=0.5: P(X=0) = e^(-lambda)
    For line=1.5: P(X <= 1) = e^(-lambda) * (1 + lambda)
    """
    k = int(line)  # 0.5 -> floor=0, 1.5 -> floor=1
    cumulative = 0.0
    for i in range(k + 1):
        cumulative += (lambda_val ** i * math.exp(-lambda_val)) / math.factorial(i)
    return max(0.05, min(0.95, cumulative))



def _bayesian_hit_rate(
    prior_rate: float,
    recent_hits: int,
    recent_games: int,
    confidence: float,
) -> float:
    """
    Bayesian update of hit rate given a prior (season average) and recent evidence.

    Formula: posterior = (prior * confidence + recent_hits) / (confidence + recent_games)

    The confidence parameter is the "equivalent prior sample size" — how many
    games of prior data does the season average represent relative to recent form.

    Higher confidence = model stays closer to season average (more stable)
    Lower confidence  = model updates faster toward recent form

    Calibrated values by sport/stat:
      - MLB hits/runs (high variance, ~162 game season): confidence = 8
        → 5 recent games moves estimate ~38% toward recent, 10 games ~56%
      - MLB strikeouts (lower variance, pitcher controlled): confidence = 6
        → 5 recent games moves estimate ~46% toward recent
      - Soccer goals (very rare, high variance): confidence = 12
        → even 5 WC games only move estimate ~29% toward recent
      - Soccer shots (moderate frequency): confidence = 8

    This directly replaces the fixed 40/60 Poisson-to-recency blend with a
    mathematically principled update that respects sample size.
    """
    if recent_games <= 0:
        return prior_rate
    posterior = (prior_rate * confidence + recent_hits) / (confidence + recent_games)
    return max(0.05, min(0.95, posterior))


def _get_bayesian_confidence(stat_display: str, sport_key: str) -> float:
    """
    Return the Bayesian prior confidence (equivalent prior sample size) for a stat.

    Higher = trust season average more, slower to update on recent streaks.
    Lower  = update faster toward recent form.
    """
    sd = stat_display.lower()

    if sport_key == "soccer":
        if any(x in sd for x in ["goals", "goal scored", "goals + assists", "goals+assists"]):
            return 12.0   # Goals are rare — don't over-update on 1-3 WC games
        if any(x in sd for x in ["shots on target", "shots attempted"]):
            return 8.0    # Shot volume is more stable
        return 10.0       # Default soccer

    if sport_key == "mlb":
        if any(x in sd for x in ["strikeout", "pitching outs", "earned run", "hits allowed"]):
            return 5.0    # Pitcher stats: skill-driven, update faster on recent starts
        if any(x in sd for x in ["home run", "stolen base"]):
            return 12.0   # Very rare events — trust season rate heavily
        if any(x in sd for x in ["runs", "rbi"]):
            return 10.0   # Sequencing-dependent — high variance, trust prior
        if any(x in sd for x in ["hits + runs", "hits+runs"]):
            return 8.0    # Combo stat
        return 6.0        # Default MLB batter hits/total bases — update faster

    # Other sports
    if sport_key in ("nba", "wnba"):
        return 5.0        # NBA: frequent games, stats are stable
    if sport_key == "nhl":
        return 8.0        # NHL: goals rare, shots more stable
    return 8.0            # Safe default


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
        # Esports / no-enrichment path: score purely from market odds
        sport_key = prop.get("_sport_key", "")
        is_esports = (sport_key == "esports" or prop.get("_enrichment_source") == "none")
        if not is_esports:
            return None

        # Use market-implied probability as model probability (no independent edge signal)
        higher_odds = prop.get("higher_american_odds")
        lower_odds = prop.get("lower_american_odds")
        market_higher = american_to_implied(higher_odds) if higher_odds else 0.52
        market_lower = american_to_implied(lower_odds) if lower_odds else 0.48

        # No game log data → model prob = market prob (zero edge by default)
        # Only flag as a pick if the market is significantly skewed (>60% one side)
        model_prob = market_higher
        market_prob = market_higher
        edge = 0.0  # No alpha without data
        selected_side = "HIGHER" if market_higher > market_lower else "LOWER"

        return {
            "modelProb": round(model_prob * 100, 1),
            "marketProb": round(market_prob * 100, 1),
            "edge": round(edge * 100, 1),
            "verdict": "LEAN" if market_higher > 0.60 or market_lower > 0.60 else "SKIP",
            "tier": "C" if market_higher > 0.60 or market_lower > 0.60 else "Pass",
            "volatility": "high",
            "signals": 0,
            "selectedSide": selected_side,
            "seasonAvg": None,
            "_mathOnly": True,
            "_esports": True,
        }


    season_avg = stats.get("seasonAvg")
    last5 = stats.get("last5", [])
    last10 = stats.get("last10", [])
    line = float(prop.get("line", 0))
    sport_key_inner = prop.get("_sport_key", "")

    # Soccer needs at least 2 games to avoid binary 0%/100% hit rate artifacts
    min_required = 1 if sport_key_inner == "soccer" else 3  # WC 2026: players may have 1-3 games
    if len(last5) < min_required and season_avg is None:
        return None

    stat_display = prop.get("stat_display", prop.get("stat_type", "")).lower()

    # ── Calculate raw hit rates from game logs ──
    higher_l5 = sum(1 for v in last5 if v > line) / max(len(last5), 1) if last5 else 0.5
    higher_l10 = sum(1 for v in last10 if v > line) / max(len(last10), 1) if last10 else 0.5
    lower_l5 = sum(1 for v in last5 if v < line) / max(len(last5), 1) if last5 else 0.5
    lower_l10 = sum(1 for v in last10 if v < line) / max(len(last10), 1) if last10 else 0.5

    # ── Blended recent form (L5 weighted more than L10) ──
    recent_higher = higher_l5 * 0.6 + higher_l10 * 0.4
    recent_lower = lower_l5 * 0.6 + lower_l10 * 0.4

    # ── Choose probability model based on line type ──
    # For low-line binary props (0.5 or 1.5 threshold), Poisson anchoring is
    # much more calibrated than raw hit rates from a small L5/L10 sample.
    # For higher lines (4.5+), we blend season avg with recent form.
    use_lambda = season_avg is not None and season_avg > 0

    # ── Bayesian prior confidence (sample-size-aware blending) ──
    # Instead of fixed weights (40/60 Poisson blend), we use Bayesian updating:
    # posterior = (prior × confidence + recent_hits) / (confidence + recent_games)
    # This naturally weights recent form MORE when we have more games, and stays
    # closer to the season average when samples are small (e.g. 1-3 WC games).
    bayes_confidence = _get_bayesian_confidence(stat_display, sport_key_inner)
    n5  = len(last5)
    n10 = len(last10)

    if line <= 1.5 and use_lambda:
        # Poisson base from season average — the mathematically correct prior
        # for a counting stat with a known rate parameter.
        poisson_higher = _poisson_prob_higher(season_avg, line)
        poisson_lower  = _poisson_prob_lower(season_avg, line)

        # Count actual hits/misses in recent games (not just rates)
        hits_higher_l5  = sum(1 for v in last5  if v > line)
        hits_higher_l10 = sum(1 for v in last10 if v > line)
        hits_lower_l5   = sum(1 for v in last5  if v < line)
        hits_lower_l10  = sum(1 for v in last10 if v < line)

        # Bayesian update: combine L5 and L10 evidence with confidence-weighted prior
        # L5 gets slightly more weight (more recent), L10 adds stability
        # We use total hits across both windows, deduped by taking the larger window
        total_games   = max(n10, n5)
        total_hits_h  = hits_higher_l10 if n10 >= n5 else hits_higher_l5
        total_hits_l  = hits_lower_l10  if n10 >= n5 else hits_lower_l5

        # Bayesian posterior hit rate (Poisson prior → bernoulli hit rate)
        bayes_higher = _bayesian_hit_rate(poisson_higher, total_hits_h, total_games, bayes_confidence)
        bayes_lower  = _bayesian_hit_rate(poisson_lower,  total_hits_l, total_games, bayes_confidence)

        # For soccer: extra Poisson anchor on Lower to suppress balanced-line noise
        # (actual suppression happens later in the balanced_line block)
        if sport_key_inner == "soccer":
            smoothed_higher = bayes_higher
            smoothed_lower  = poisson_lower * 0.70 + bayes_lower * 0.30
        else:
            smoothed_higher = bayes_higher
            smoothed_lower  = bayes_lower

    else:
        # For higher lines (4.5+): Bayesian update on raw hit rates
        # Prior is derived from season_avg vs line gap
        prior_higher = 0.5 + min(max((season_avg - line) / max(line, 0.5) * 0.25, -0.15), 0.15) if use_lambda else 0.5
        prior_lower  = 1.0 - prior_higher

        hits_higher = sum(1 for v in last10 if v > line)
        hits_lower  = sum(1 for v in last10 if v < line)
        total_games = max(n10, 1)

        smoothed_higher = _bayesian_hit_rate(prior_higher, hits_higher, total_games, bayes_confidence)
        smoothed_lower  = _bayesian_hit_rate(prior_lower,  hits_lower,  total_games, bayes_confidence)

    # ── Feature 4: Opponent Defensive Rating adjustment ──
    # For MLB batter props, adjust model probability based on opposing pitcher quality.
    # A pitcher with ERA 2.00 suppresses hits more than one with ERA 5.00.
    # Adjustment is small (max ±6pp) to avoid overriding the Bayesian model.
    opp_pitcher = prop.get("_oppPitcher")
    if opp_pitcher and sport_key_inner == "mlb" and not prop.get("is_pitcher"):
        era  = opp_pitcher.get("era")
        whip = opp_pitcher.get("whip")
        k9   = opp_pitcher.get("k_per_9")

        if era is not None and whip is not None:
            # ERA-based adjustment: league average ERA ~4.20
            # Elite pitcher (ERA 2.50): -4pp on Higher hits
            # Poor pitcher (ERA 5.50): +4pp on Higher hits
            era_adj = max(-0.06, min(0.06, (float(era) - 4.20) * 0.015))

            # WHIP-based adjustment: league average WHIP ~1.30
            # Low WHIP (1.00): extra -2pp on Higher hits
            whip_adj = max(-0.03, min(0.03, (float(whip) - 1.30) * 0.08))

            # K/9 adjustment for strikeout props specifically
            k9_adj = 0.0
            if k9 is not None and any(x in stat_display for x in ["strikeout", "batter strikeout"]):
                # High K/9 pitcher → Higher batter strikeouts more likely
                k9_adj = max(-0.04, min(0.04, (float(k9) - 8.5) * 0.008))

            total_adj = era_adj + whip_adj + k9_adj

            # Apply: positive total_adj means pitcher is hittable → boost Higher, cut Lower
            smoothed_higher = max(0.15, min(0.80, smoothed_higher + total_adj))
            smoothed_lower  = max(0.15, min(0.80, smoothed_lower  - total_adj))

    # ── Pre-selection ceiling ──
    # Cap at 0.74 before side selection; direction-aware ceiling applied below.
    # Lower than before (was 0.80) to prevent inflated inputs to the dampener.
    smoothed_higher = max(0.20, min(0.74, smoothed_higher))
    smoothed_lower = max(0.20, min(0.74, smoothed_lower))

    # ── Market probabilities from odds (de-vigged) ──
    # De-vigging removes the bookmaker's margin before computing edge.
    # A -110/-110 line has 4.76% vig; raw implied = 52.4% each side.
    # De-vigged true probability = 50.0% each side — a meaningful difference
    # when computing whether a pick is genuinely +EV vs the true market consensus.
    higher_odds = prop.get("higher_american_odds") or prop.get("american_odds")
    lower_odds = prop.get("lower_american_odds") or prop.get("lower_odds")

    higher_market_raw = american_to_implied(higher_odds) if higher_odds else 0.5
    lower_market_raw  = american_to_implied(lower_odds)  if lower_odds else 0.5

    # Apply de-vigging if both sides are available
    devigged_result = devig_sharp_line(higher_odds, lower_odds)
    if devigged_result:
        higher_market_true, lower_market_true = devigged_result
        vig_magnitude = round((higher_market_raw + lower_market_raw - 1.0) * 100, 2)
    else:
        higher_market_true = higher_market_raw
        lower_market_true  = lower_market_raw
        vig_magnitude = 0.0

    # ── Market-anchoring dampener ──
    # Fires when model deviates >20pp from de-vigged market.
    # Pull strength scales with raw signal count:
    #   0-1 signals: pull 50% toward market (model has weak support — trust market)
    #   2 signals:   pull 38% toward market
    #   3+ signals:  pull 28% toward market (model has strong support — trust it more)
    # This prevents the dampener from crushing well-supported picks.
    DAMPENER_THRESHOLD = 0.20
    _raw_sig_count = sum([
        1 for v in last5 if v > line] + [1 for v in last10 if v > line]) // 2  # rough estimate
    _raw_sig_count = min(3, max(0, len([v for v in last5 if v > line or v < line])))

    # Simpler: use number of games as proxy for signal strength
    n_games = max(len(last5), 1)
    if n_games >= 8:
        DAMPENER_PULL = 0.28   # Lots of data — trust model
    elif n_games >= 5:
        DAMPENER_PULL = 0.38   # Moderate data
    else:
        DAMPENER_PULL = 0.50   # Small sample — trust market more

    # Use de-vigged for edge calculation, raw for dampener reference
    mkt_h = higher_market_true
    mkt_l = lower_market_true

    if abs(smoothed_higher - mkt_h) > DAMPENER_THRESHOLD:
        smoothed_higher = smoothed_higher + (mkt_h - smoothed_higher) * DAMPENER_PULL

    if abs(smoothed_lower - mkt_l) > DAMPENER_THRESHOLD:
        smoothed_lower = smoothed_lower + (mkt_l - smoothed_lower) * DAMPENER_PULL

    # ── Calculate edges ──
    higher_edge = smoothed_higher - mkt_h
    lower_edge = smoothed_lower - mkt_l

    # ── Select best side ──
    if higher_edge >= lower_edge:
        selected_side = "Higher"
        model_prob = smoothed_higher
        market_prob = mkt_h
        edge = higher_edge
    else:
        selected_side = "Lower"
        model_prob = smoothed_lower
        market_prob = mkt_l
        edge = lower_edge

    # ── Direction-aware signal ceiling ──
    # Now that we know which side we picked, compute how many signals actually
    # support that direction and tighten the model_prob ceiling accordingly.
    raw_signals_estimate = 0
    if selected_side == "Higher":
        if last5 and higher_l5 >= 0.6: raw_signals_estimate += 1
        if last10 and higher_l10 >= 0.6: raw_signals_estimate += 1
        if use_lambda and season_avg > line * 1.1: raw_signals_estimate += 1
    else:
        if last5 and lower_l5 >= 0.6: raw_signals_estimate += 1
        if last10 and lower_l10 >= 0.6: raw_signals_estimate += 1
        if use_lambda and season_avg < line * 0.9: raw_signals_estimate += 1
    # Signal ceiling — deliberately conservative.
    # Max 3 raw signals (L5, L10, season_avg). big_avg_gap no longer gives bonus
    # signals because season_avg vs line is already captured by L5/L10 hit rates.
    # Ceiling lowered from 0.77 to 0.72 max to prevent systematic overconfidence.
    #
    # Stat-specific adjustments:
    #   - runs/rbi: sequencing-dependent, reduce ceiling by 0.03 (need hit + baserunner)
    #   - goals (soccer): rare events, reduce ceiling by 0.02
    #   - 1st inning props: near-coinflip, hard cap at 0.66
    is_sequencing_stat = any(x in stat_display for x in ["runs", "rbi", "goals allowed"])
    is_goals_only = stat_display in ("goals", "soccer goals", "goal scored")
    is_inning_prop = prop.get("_is_inning_prop", False)

    base_ceiling = {0: 0.60, 1: 0.62, 2: 0.65, 3: 0.68, 4: 0.70, 5: 0.72}.get(
        min(raw_signals_estimate, 5), 0.65
    )
    if is_inning_prop:
        signal_ceiling = min(base_ceiling, 0.66)
    elif is_sequencing_stat:
        signal_ceiling = base_ceiling - 0.03
    elif is_goals_only and sport_key_inner == "soccer":
        # Only penalize goals ceiling when selecting Lower (reducing noise)
        # For Higher goals picks, full ceiling applies since market already priced it high
        if selected_side == "Lower":
            signal_ceiling = base_ceiling - 0.02
        else:
            signal_ceiling = base_ceiling
    else:
        signal_ceiling = base_ceiling

    model_prob = max(0.20, min(signal_ceiling, model_prob))

    # ── CRITICAL FIX: Recalculate edge after ceiling clamp ──
    # Previously, edge was set from the pre-ceiling model_prob and never updated.
    # After the signal ceiling clamps model_prob down, the displayed edge was
    # still the pre-ceiling gap, making it systematically inflated.
    # Example: model starts 0.75, ceiling caps to 0.68, market=0.50 →
    #   old code: edge=0.25 (pre-ceiling), displayed +25%
    #   fixed:    edge=0.18 (post-ceiling), displayed +18%
    edge = model_prob - market_prob

    # Also compute raw (non-de-vigged) market probability for the selected side.
    # De-vigging removes the bookmaker's margin, making market_prob lower and edge
    # appear larger. The raw implied prob from the selected side's exact odds is
    # the conservative/transparent edge measure users expect to see.
    if selected_side == "Higher":
        raw_selected_market = higher_market_raw
    else:
        raw_selected_market = lower_market_raw

    raw_edge = model_prob - raw_selected_market

    # ── Suppress no-odds noise picks ──
    # When the market defaults to exactly 50% (no real odds data), the "edge"
    # is entirely model-driven with no market anchor. For soccer props with
    # small samples (1-2 games), this produces hundreds of spurious 65%/50%
    # picks. Require real market odds OR a higher model confidence threshold.
    no_real_odds = (not prop.get("higher_american_odds") and not prop.get("lower_american_odds")
                    and not prop.get("american_odds") and not prop.get("lower_odds"))
    sport_key = prop.get("_sport_key", "")

    # Detect "balanced" lines — Underdog posts -110/-110 on many soccer props
    # which gives market ~47.6%/47.6%. Treat these like no-odds for suppression.
    higher_odds_val = prop.get("higher_american_odds") or prop.get("american_odds")
    lower_odds_val = prop.get("lower_american_odds") or prop.get("lower_odds")
    try:
        h_abs = abs(float(str(higher_odds_val).replace("+", ""))) if higher_odds_val else None
        l_abs = abs(float(str(lower_odds_val).replace("+", ""))) if lower_odds_val else None
        balanced_line = (
            h_abs is not None and l_abs is not None and
            abs(h_abs - l_abs) <= 8 and          # within 8 points of each other
            h_abs <= 120                           # neither side heavily juiced
        )
    except (ValueError, TypeError):
        balanced_line = False
    weak_market = no_real_odds or balanced_line

    if weak_market and sport_key == "soccer":
        # Lower props on weak/balanced markets are trivially satisfied for low-output players.
        # Require strong signal support: signals >= 3 means at least one game-log signal
        # actually supports the direction, not just Poisson math on a small sample.
        if selected_side == "Lower" and raw_signals_estimate < 2:
            return None
        threshold = 0.82 if selected_side == "Lower" else 0.68
        if model_prob < threshold:
            return None

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
    # Signals must align with the selected side. A Lower pick on a player who
    # rarely scores doesn't get credit for "high Lower hit rate" — that's not
    # edge, it's just a low-output player on a low line. We require the season
    # average to actually support the direction too.
    signals = 0
    if selected_side == "Higher":
        if last5 and higher_l5 >= 0.6: signals += 1
        if last10 and higher_l10 >= 0.6: signals += 1
        if use_lambda and season_avg > line * 1.1: signals += 1
    else:  # Lower
        if last5 and lower_l5 >= 0.6: signals += 1
        if last10 and lower_l10 >= 0.6: signals += 1
        if use_lambda and season_avg < line * 0.9: signals += 1
    if edge > 0.05: signals += 1
    if edge > 0.15: signals += 1

    # ── Classify verdict and tier ──
    verdict, tier = classify_pick(model_prob, edge, signals, volatility)

    # ── Kelly sizing for fixed-multiplier DFS ──
    kelly_2pick = kelly_criterion(model_prob, 3.0,  fraction=0.5)
    kelly_5pick = kelly_criterion(model_prob, 10.0, fraction=0.5)

    return {
        "modelProb":      round(model_prob * 100, 1),
        "marketProb":     round(raw_selected_market * 100, 1),  # raw implied from selected side's exact odds
        "trueMarketProb": round(market_prob * 100, 1),          # de-vigged market prob (for reference)
        "edge":           round(raw_edge * 100, 1),             # edge vs raw implied (conservative, what users expect)
        "trueEdge":       round(edge * 100, 1),                 # edge vs de-vigged market (for internal reference)
        "rawMarketProb":  round(raw_selected_market * 100, 1),  # explicit raw implied for frontend validation
        "rawEdge":        round(raw_edge * 100, 1),             # explicit raw edge for frontend validation
        "vig":            vig_magnitude,
        "devigged":       devigged_result is not None,
        "verdict":        verdict,
        "tier":           tier,
        "volatility":     volatility,
        "signals":        signals,
        "selectedSide":   selected_side,
        "hitRates": {
            "l5Higher":  round(higher_l5 * 100),
            "l10Higher": round(higher_l10 * 100),
            "l5Lower":   round(lower_l5 * 100),
            "l10Lower":  round(lower_l10 * 100),
        },
        "seasonAvg": round(season_avg, 2) if season_avg else None,
        "line": line,
        "kelly": {
            "powerPlay2Pick": kelly_2pick,
            "flex5Pick":      kelly_5pick,
        },
    }


def classify_pick(model_prob: float, edge: float, signals: int, volatility: str) -> tuple[str, str]:
    """
    Classify a pick into verdict and tier.

    Thresholds calibrated for the Poisson-anchored model where 54-65% is
    a realistic well-supported pick and 70%+ is genuinely high-confidence.

    Signal requirements (direction-aware counting):
      signals 1-2 = only edge signals fired (no game-log or avg support) → C/Pass
      signals 3   = at least one game-log or avg signal supports the direction → B eligible
      signals 4-5 = strong multi-signal agreement → A eligible
    """
    if edge <= 0:
        return "SKIP", "Pass"

    # Tier classification
    # Require signals >= 3 for Tier B now — this means at least one of L5, L10, or
    # season_avg must genuinely support the direction (not just edge arithmetic).
    if model_prob >= 0.63 and signals >= 4 and edge >= 0.08 and volatility in ("low", "medium"):
        tier = "A"
    elif model_prob >= 0.58 and signals >= 3 and edge >= 0.04:
        tier = "B"
    elif model_prob >= 0.54 and signals >= 3:
        tier = "C"
    else:
        tier = "Pass"

    # Verdict classification
    if tier == "Pass" or edge < 0.02:
        verdict = "SKIP"
    elif tier == "A" and edge >= 0.15:
        verdict = "STRONG PLAY"
    elif tier in ("A", "B") and edge >= 0.05:
        verdict = "PLAY"
    elif edge >= 0.03:
        verdict = "LEAN"
    else:
        verdict = "SKIP"

    return verdict, tier



# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 5: Correlation Matrix for Flex Builds
# ─────────────────────────────────────────────────────────────────────────────

_BATTER_COUNTING = {"hits", "runs", "rbi", "total_bases", "hits_runs_rbis", "home_runs", "stolen_bases", "walks"}
_BATTER_K        = {"batter_strikeouts"}
_PITCHER_STATS   = {"strikeouts", "earned_runs", "hits_allowed", "walks_allowed", "outs_pitched", "pitcher_outs"}
_GOAL_STATS      = {"goals", "goals_assists", "shots", "shots_on_goal", "shots_on_target"}
_SCORING_STATS   = {"points", "rebounds", "assists", "blocks", "steals", "threes"}

def _stat_category(stat_display: str) -> str:
    sd = stat_display.lower().strip()
    for s in _PITCHER_STATS:
        if s in sd: return "pitcher"
    for s in _BATTER_K:
        if s in sd: return "batter_k"
    for s in _BATTER_COUNTING:
        if s in sd: return "batter_counting"
    for s in _GOAL_STATS:
        if s in sd: return "goal"
    for s in _SCORING_STATS:
        if s in sd: return "scoring"
    return "other"

def _game_key(pick: dict) -> str | None:
    home = (pick.get("home_team") or "").strip().upper()
    away = (pick.get("away_team") or "").strip().upper()
    if not home and not away: return None
    teams = sorted([t for t in [home, away] if t])
    return "@".join(teams)

def _compute_correlation(pick_a: dict, pick_b: dict) -> str:
    cat_a = _stat_category(pick_a.get("stat_display", ""))
    cat_b = _stat_category(pick_b.get("stat_display", ""))
    side_a = (pick_a.get("selectedSide") or "").lower()
    side_b = (pick_b.get("selectedSide") or "").lower()
    if (pick_a.get("player_name") or "").lower() == (pick_b.get("player_name") or "").lower():
        if cat_a in ("batter_counting", "batter_k") and cat_b in ("batter_counting", "batter_k"):
            if "walk" in pick_a.get("stat_display", "").lower() and "rbi" in pick_b.get("stat_display", "").lower():
                return "negative"
            if "rbi" in pick_a.get("stat_display", "").lower() and "walk" in pick_b.get("stat_display", "").lower():
                return "negative"
            return "positive"
        return "positive"
    if cat_a == "pitcher" and cat_b == "batter_counting":
        if "strikeout" in pick_a.get("stat_display", "").lower() and side_a == "higher" and side_b == "lower":
            return "negative"
        if ("earned" in pick_a.get("stat_display", "").lower() or "hit" in pick_a.get("stat_display", "").lower()) and side_a == "higher" and side_b == "higher":
            return "positive"
        return "neutral"
    if cat_a == "batter_counting" and cat_b == "pitcher":
        if "strikeout" in pick_b.get("stat_display", "").lower() and side_b == "higher" and side_a == "lower":
            return "negative"
        if ("earned" in pick_b.get("stat_display", "").lower() or "hit" in pick_b.get("stat_display", "").lower()) and side_b == "higher" and side_a == "higher":
            return "positive"
        return "neutral"
    if cat_a == "batter_counting" and cat_b == "batter_counting":
        if pick_a.get("player_team", "").upper() == pick_b.get("player_team", "").upper() and side_a == side_b:
            return "positive"
        return "neutral"
    if cat_a == "goal" and cat_b == "goal":
        if pick_a.get("player_team", "").upper() == pick_b.get("player_team", "").upper() and side_a == "higher" and side_b == "higher":
            return "positive"
        return "neutral"
    if cat_a == "scoring" and cat_b == "scoring":
        if pick_a.get("player_team", "").upper() == pick_b.get("player_team", "").upper() and side_a == side_b:
            return "positive"
        return "neutral"
    return "neutral"

def _annotate_correlations(picks: list[dict]) -> list[dict]:
    for p in picks:
        p["gameKey"] = _game_key(p)
    game_groups: dict[str, list[dict]] = {}
    for p in picks:
        gk = p.get("gameKey")
        if gk:
            if gk not in game_groups: game_groups[gk] = []
            game_groups[gk].append(p)
    for p in picks:
        gk = p.get("gameKey")
        peers = [q for q in game_groups.get(gk, []) if q is not p] if gk else []
        corr_list = []
        for peer in peers:
            corr_type = _compute_correlation(p, peer)
            corr_list.append({
                "player": peer.get("player_name", ""),
                "stat": peer.get("stat_display", ""),
                "side": peer.get("selectedSide", ""),
                "correlationType": corr_type,
            })
        p["correlations"] = corr_list
        p["sameGamePicks"] = [f"{c['player']} {c['stat']}" for c in corr_list]
        p["maxPositiveCorr"] = sum(1 for c in corr_list if c["correlationType"] == "positive")
        p["maxNegativeCorr"] = sum(1 for c in corr_list if c["correlationType"] == "negative")
    return picks


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 10: +EV Composite Score
# ─────────────────────────────────────────────────────────────────────────────

def _compute_composite_ev(pick: dict) -> dict:
    model_prob  = (pick.get("modelProb") or 50) / 100
    edge        = (pick.get("edge") or 0) / 100  # now uses corrected raw edge
    signals     = pick.get("signals") or 0
    volatility  = (pick.get("volatility") or "medium").lower()
    kelly       = pick.get("kelly") or {}
    ev_dollar   = kelly.get("ev_per_dollar") if isinstance(kelly, dict) else None
    sample_size = len((pick.get("_playerStats") or {}).get("last10") or
                      (pick.get("_playerStats") or {}).get("last5") or [])
    model_component = min(40, ((model_prob - 0.50) / 0.30) * 40)
    edge_component = min(35, (edge / 0.20) * 35)
    signal_component = min(20, signals * 4)
    kelly_component = min(10, ev_dollar * 20) if ev_dollar is not None and ev_dollar > 0 else 0
    vol_penalty = {"high": 12, "medium": 4, "low": 0}.get(volatility, 4)
    sample_bonus = min(5, sample_size * 0.5)
    raw_score = model_component + edge_component + signal_component + kelly_component + sample_bonus - vol_penalty
    composite = max(0, min(100, raw_score))
    if composite >= 75: label = "Strong Play"
    elif composite >= 60: label = "Play"
    elif composite >= 40: label = "Lean"
    else: label = "Below Threshold"
    return {"compositeEV": round(composite, 1), "compositeEVLabel": label}


def score_props(props: list[dict], min_edge: float = 0.05) -> dict:
    """Score a batch of enriched props and separate into picks and passes."""
    picks = []
    passes = []
    projection_health = {
        "enabled": True,
        "sport": (props[0].get("_sport_key") or props[0].get("sport") or "") if props else "",
        "propsReceived": len(props),
        "projectionAttempted": len(props),
        "projectionMatched": 0,
        "projectionUnavailable": 0,
        "provider": "railway_prediction_api",
        "unavailableReasons": {
            "player_not_matched": 0,
            "stat_not_supported": 0,
            "no_game_logs": 0,
            "no_season_stats": 0,
            "no_stat_history": 0,
            "projection_exception": 0,
        },
        "sampleFailures": [],
        "errors": [],
    }

    for prop in props:
        scored = score_prop(prop)
        if scored is None:
            try:
                projection = build_projection_metadata(prop)
            except Exception as e:
                projection = {
                    "projection": None,
                    "projectionEdge": None,
                    "projectionSource": None,
                    "projectionAvailable": False,
                    "projectionUnavailableReason": "projection_exception",
                    "unavailableReason": "projection_exception",
                }
                projection_health["errors"].append({"error": str(e)})
            _record_projection_health(projection_health, prop, projection)
            passes.append({**prop, **projection, "verdict": "SKIP", "reason": "Insufficient data"})
            continue

        try:
            projection = build_projection_metadata(prop, scored)
        except Exception as e:
            projection = {
                "projection": None,
                "projectionEdge": None,
                "projectionSource": None,
                "projectionAvailable": False,
                "projectionUnavailableReason": "projection_exception",
                "unavailableReason": "projection_exception",
            }
            projection_health["errors"].append({"error": str(e)})
        _record_projection_health(projection_health, prop, projection)

        result = {**prop, **projection, **scored}
        composite = _compute_composite_ev(result)
        result.update(composite)

        if scored["edge"] >= min_edge * 100 and scored["verdict"] != "SKIP":
            picks.append(result)
        else:
            passes.append(result)

    if picks:
        picks = _annotate_correlations(picks)

    return {
        "picks": picks,
        "passes": passes,
        "stats_context": f"Scored {len(props)} props: {len(picks)} actionable, {len(passes)} passed",
        "projectionHealth": projection_health,
    }


def filter_prop(prop: dict) -> dict:
    """
    Filter a prop before scoring.
    Returns { status: 'pass' | 'hard_reject' | 'warn', reason: str }
    """
    stats = prop.get("_playerStats")
    line = float(prop.get("line", 0))

    # Esports props: no game log data available — use market-implied scoring only
    sport_key = prop.get("_sport_key", "")
    if sport_key == "esports" or prop.get("_enrichment_source") == "none":
        return {"status": "pass", "reason": "esports_math_only"}

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
        # Soccer/tournament players may have only 1-2 games — allow if seasonAvg present
        # or if we have at least 1 game log entry (tournament context)
        # Soccer: WC players have few games — need 2+ to avoid 0/1 binary hit rate artifacts
        # Other sports: 3+ games required
        min_games = 1 if prop.get("_sport_key") == "soccer" else 3  # WC 2026: players may have 1-3 games
        if len(last5) < min_games and stats.get("seasonAvg") is None:
            return {"status": "hard_reject", "reason": f"Insufficient player data (need {min_games}+ recent games, found {len(last5)})"}

    return {"status": "pass", "reason": ""}
