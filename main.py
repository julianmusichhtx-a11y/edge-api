"""
Underdog Edge AI — Prediction API
Replaces the enrichment + scoring layer from generateAnalysis.js.

One endpoint: POST /analyze
Takes props + sport → returns scored picks with player stats.
"""
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from adapters.mlb_adapter import MLBAdapter
from adapters.sport_adapters import (
    WNBAAdapter, NBAAdapter, NFLAdapter, NHLAdapter,
    SoccerAdapter, MMAAdapter,
)
from scoring.scorer import score_prop, filter_prop
from utils.rate_limiter import rate_limiter
from cache import cache_manager as cache

app = FastAPI(
    title="Underdog Edge AI — Prediction API",
    description="Sport-agnostic prop scoring and player stats enrichment",
    version="1.0.0",
)

# Allow requests from your Base44 app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Adapter Registry ───────────────────────────────────────────────────────
# Each sport has its own adapter instance (stateful — caches rosters/game logs)
ADAPTERS = {
    "mlb":    MLBAdapter(),
    "wnba":  WNBAAdapter(),
    "nba":   NBAAdapter(),
    "nfl":   NFLAdapter(),
    "nhl":   NHLAdapter(),
    "soccer": SoccerAdapter(),
    "mma":   MMAAdapter(),
}


# ─── Request/Response Models ────────────────────────────────────────────────
class PropInput(BaseModel):
    player_name: str
    stat_display: str
    line: float
    higher_american_odds: Optional[int] = None
    lower_american_odds: Optional[int] = None
    player_team: Optional[str] = ""
    home_team: Optional[str] = ""
    away_team: Optional[str] = ""
    source: Optional[str] = "underdog"
    category: Optional[str] = ""


class AnalyzeRequest(BaseModel):
    sport: str                          # "mlb", "wnba", "nba", "nfl", etc.
    props: list[PropInput]              # Raw props from Apify/ParlayAPI
    platform: Optional[str] = "underdog"


class ScoredPick(BaseModel):
    player_name: str
    stat_display: str
    line: float
    selectedSide: str
    modelProb: float
    marketProb: float
    edge: float
    verdict: str
    tier: str
    volatility: str
    signals: int
    hitRates: dict
    seasonAvg: Optional[float] = None
    higher_american_odds: Optional[int] = None
    lower_american_odds: Optional[int] = None
    home_team: Optional[str] = ""
    away_team: Optional[str] = ""


class AnalyzeResponse(BaseModel):
    sport: str
    platform: str
    picks: list[ScoredPick]
    passes: list[dict]
    stats_context: str                  # Text block for AI narrative
    summary: dict
    elapsed_ms: int


# ─── Main Endpoint ──────────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Score and enrich player props for any supported sport.

    1. Fetches player stats (game logs, season averages) via sport adapter
    2. Scores each prop (model probability, market probability, edge)
    3. Returns picks (actionable) and passes (filtered out)
    """
    start = time.time()
    sport = req.sport.lower().strip()

    adapter = ADAPTERS.get(sport)
    if not adapter:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported sport: {sport}. Supported: {list(ADAPTERS.keys())}"
        )

    # Convert Pydantic models to dicts for the adapter
    props = [p.model_dump() for p in req.props]

    # Step 1: Enrich props with player stats
    try:
        props = await adapter.enrich_props(props)
    except Exception as e:
        print(f"[{sport.upper()}] Enrichment error: {e}")
        # Continue with whatever we have — partial data is better than none

    # Step 2: Filter and score
    picks = []
    passes = []
    stats_enriched = 0

    for prop in props:
        # Filter first
        filter_result = filter_prop(prop)
        if filter_result["status"] == "hard_reject":
            passes.append({
                "player_name": prop.get("player_name", ""),
                "stat_display": prop.get("stat_display", ""),
                "line": prop.get("line"),
                "reason": filter_result["reason"],
            })
            continue

        # Score
        scoring = score_prop(prop)
        if not scoring:
            passes.append({
                "player_name": prop.get("player_name", ""),
                "stat_display": prop.get("stat_display", ""),
                "line": prop.get("line"),
                "reason": "Insufficient data for scoring",
            })
            continue

        stats_enriched += 1

        # Negative edge → pass
        if scoring["edge"] <= 0:
            passes.append({
                "player_name": prop.get("player_name", ""),
                "stat_display": prop.get("stat_display", ""),
                "line": prop.get("line"),
                "reason": f"Negative edge ({scoring['edge']:.1f}%)",
            })
            continue

        # Skip verdict → pass
        if scoring["verdict"] == "SKIP":
            passes.append({
                "player_name": prop.get("player_name", ""),
                "stat_display": prop.get("stat_display", ""),
                "line": prop.get("line"),
                "reason": f"Below threshold (edge: {scoring['edge']:.1f}%, tier: {scoring['tier']})",
            })
            continue

        picks.append(ScoredPick(
            player_name=prop.get("player_name", ""),
            stat_display=prop.get("stat_display", ""),
            line=float(prop.get("line", 0)),
            home_team=prop.get("home_team", ""),
            away_team=prop.get("away_team", ""),
            higher_american_odds=prop.get("higher_american_odds"),
            lower_american_odds=prop.get("lower_american_odds"),
            **scoring,
        ))

    # Sort picks by edge descending
    picks.sort(key=lambda p: p.edge, reverse=True)

    # Build stats context block for AI narrative
    stats_context = _build_stats_context(picks, sport)

    elapsed = int((time.time() - start) * 1000)

    return AnalyzeResponse(
        sport=sport.upper(),
        platform=req.platform or "underdog",
        picks=picks,
        passes=passes,
        stats_context=stats_context,
        summary={
            "total_props": len(props),
            "enriched": stats_enriched,
            "actionable_picks": len(picks),
            "passes": len(passes),
            "verdicts": _count_verdicts(picks),
            "tiers": _count_tiers(picks),
        },
        elapsed_ms=elapsed,
    )


# ─── Utility Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "sports": list(ADAPTERS.keys())}


@app.get("/quota")
async def quota():
    """Check current API usage against limits."""
    return {
        "usage": rate_limiter.get_usage(),
        "cache": cache.stats(),
    }


@app.get("/schedule/{sport}")
async def schedule(sport: str):
    """Get today's games for a sport."""
    adapter = ADAPTERS.get(sport.lower())
    if not adapter:
        raise HTTPException(status_code=400, detail=f"Unsupported sport: {sport}")
    games = await adapter.get_todays_games()
    return {"sport": sport, "games": [g.__dict__ for g in games]}


# ─── Helpers ────────────────────────────────────────────────────────────────
def _build_stats_context(picks: list[ScoredPick], sport: str) -> str:
    """Build a text block summarizing player stats for AI narrative generation."""
    if not picks:
        return ""

    lines = []
    for p in picks[:15]:
        line = (
            f"PICK: {p.player_name} — {p.selectedSide} {p.stat_display} {p.line} | "
            f"Verdict: {p.verdict} | Tier: {p.tier} | Vol: {p.volatility} | "
            f"Signals: {p.signals} | Model: {p.modelProb}% | Market: {p.marketProb}% | "
            f"Edge: {'+' if p.edge >= 0 else ''}{p.edge}% | "
            f"L5HR: {p.hitRates.get('l5Higher', '?')}% | L10HR: {p.hitRates.get('l10Higher', '?')}% | "
            f"Game: {p.away_team or '?'} @ {p.home_team or '?'}"
        )
        if p.seasonAvg is not None:
            line += f" | SeasonAvg: {p.seasonAvg}"
        lines.append(line)

    return "\n".join(lines)


def _count_verdicts(picks: list[ScoredPick]) -> dict:
    counts = {}
    for p in picks:
        counts[p.verdict] = counts.get(p.verdict, 0) + 1
    return counts


def _count_tiers(picks: list[ScoredPick]) -> dict:
    counts = {}
    for p in picks:
        counts[p.tier] = counts.get(p.tier, 0) + 1
    return counts


# ─── Run with: uvicorn main:app --host 0.0.0.0 --port 8000 ─────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
