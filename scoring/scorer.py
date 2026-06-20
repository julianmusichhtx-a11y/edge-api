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
import math
from utils.odds_math import american_to_implied, classify_volatility, calculate_edge


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
    min_required = 2 if sport_key_inner == "soccer" else 3
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

    if line <= 1.5 and use_lambda:
        # Poisson base anchored to season average
        poisson_higher = _poisson_prob_higher(season_avg, line)
        poisson_lower = _poisson_prob_lower(season_avg, line)

        # Recent form as a 30% adjustment (not replacement) relative to Poisson base
        # This captures hot/cold streaks without letting a 4/5 L5 dominate
        adj_higher = (recent_higher - poisson_higher) * 0.30
        adj_lower = (recent_lower - poisson_lower) * 0.30

        smoothed_higher = poisson_higher + adj_higher
        smoothed_lower = poisson_lower + adj_lower

    else:
        # For higher lines: blend season-avg signal with recent form
        # Season avg contributes ~45%, recent form contributes ~55%
        avg_signal_higher = 0.0
        avg_signal_lower = 0.0
        if use_lambda and line > 0:
            gap = (season_avg - line) / max(line, 0.5)
            avg_signal_higher = min(max(gap * 0.25, -0.12), 0.12)
            avg_signal_lower = -avg_signal_higher

        smoothed_higher = recent_higher * 0.55 + (0.5 + avg_signal_higher) * 0.45
        smoothed_lower = recent_lower * 0.55 + (0.5 + avg_signal_lower) * 0.45

    # ── Pre-selection ceiling (loose) ──
    # Cap at 0.80 before side selection; direction-aware ceiling applied below.
    smoothed_higher = max(0.20, min(0.80, smoothed_higher))
    smoothed_lower = max(0.20, min(0.80, smoothed_lower))

    # ── Market probabilities from odds ──
    higher_odds = prop.get("higher_american_odds") or prop.get("american_odds")
    lower_odds = prop.get("lower_american_odds") or prop.get("lower_odds")

    higher_market = american_to_implied(higher_odds) if higher_odds else 0.5
    lower_market = american_to_implied(lower_odds) if lower_odds else 0.5

    # ── Market-anchoring dampener ──
    # If our model deviates from market by more than 28 percentage points,
    # pull it 35% back toward market. This is a safety valve for cases where
    # our model is reacting to a small sample and the market likely has better data.
    # We keep 65% of our signal — enough to still have edge, but capped.
    DAMPENER_THRESHOLD = 0.28
    DAMPENER_PULL = 0.35

    mkt_h = higher_market or 0.5
    mkt_l = lower_market or 0.5

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
        big_avg_gap = use_lambda and season_avg > line * 1.1 and (season_avg - line) / max(line, 0.5) > 0.3
    else:
        if last5 and lower_l5 >= 0.6: raw_signals_estimate += 1
        if last10 and lower_l10 >= 0.6: raw_signals_estimate += 1
        if use_lambda and season_avg < line * 0.9: raw_signals_estimate += 1
        big_avg_gap = use_lambda and season_avg < line * 0.9 and (line - season_avg) / max(line, 0.5) > 0.3
    signal_ceiling = {0: 0.65, 1: 0.67, 2: 0.70, 3: 0.74, 4: 0.77, 5: 0.80}.get(
        min(raw_signals_estimate + (2 if big_avg_gap else 0), 5), 0.72
    )
    model_prob = max(0.20, min(signal_ceiling, model_prob))

    # ── Suppress no-odds noise picks ──
    # When the market defaults to exactly 50% (no real odds data), the "edge"
    # is entirely model-driven with no market anchor. For soccer props with
    # small samples (1-2 games), this produces hundreds of spurious 65%/50%
    # picks. Require real market odds OR a higher model confidence threshold.
    no_real_odds = (not prop.get("higher_american_odds") and not prop.get("lower_american_odds")
                    and not prop.get("american_odds") and not prop.get("lower_odds"))
    sport_key = prop.get("_sport_key", "")
    if no_real_odds and sport_key == "soccer" and model_prob < 0.70:
        # Not enough model confidence to call edge without real market data
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


def score_props(props: list[dict], min_edge: float = 0.05) -> dict:
    """Score a batch of enriched props and separate into picks and passes."""
    picks = []
    passes = []

    for prop in props:
        scored = score_prop(prop)
        if scored is None:
            passes.append({**prop, "verdict": "SKIP", "reason": "Insufficient data"})
            continue

        result = {**prop, **scored}
        if scored["edge"] >= min_edge * 100 and scored["verdict"] != "SKIP":
            picks.append(result)
        else:
            passes.append(result)

    return {
        "picks": picks,
        "passes": passes,
        "stats_context": f"Scored {len(props)} props: {len(picks)} actionable, {len(passes)} passed",
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
        min_games = 2 if prop.get("_sport_key") == "soccer" else 3
        if len(last5) < min_games and stats.get("seasonAvg") is None:
            return {"status": "hard_reject", "reason": f"Insufficient player data (need {min_games}+ recent games, found {len(last5)})"}

    return {"status": "pass", "reason": ""}