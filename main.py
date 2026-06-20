"""
Edge API - Predictive Player Prop Analysis Service
Replaces browser-side scoring/enrichment in Base44 app.
POST /analyze accepts props from any sport and returns scored, enriched picks.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from adapters import get_adapter
from scoring.scorer import score_props
from cache.cache_manager import cache
from utils.odds_math import american_to_implied_prob

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Edge API - Sports Prop Predictor",
    description="Multi-sport predictive analysis for DFS props (PrizePicks, Underdog, etc.)",
    version="0.1.0"
)

# Add CORS middleware so Base44 preview/sandbox can call it directly during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://preview--edgelabai.base44.app",
        "https://*.base44.app",
        "https://*.base44-preview.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "*"  # Allow all during development - tighten in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PropInput(BaseModel):
    player: str = Field(..., description="Player full name")
    stat: str = Field(..., description="Stat type e.g. Points, Rebounds, Strikeouts")
    line: float = Field(..., description="The prop line")
    sport: str = Field(..., description="nba, mlb, wnba, nfl, nhl, etc.")
    platform: Optional[str] = Field("PrizePicks", description="PrizePicks or Underdog Fantasy")
    team: Optional[str] = None
    opponent: Optional[str] = None
    game_time: Optional[str] = None  # ISO or whatever
    # Any extra context from frontend
    extra: Optional[Dict[str, Any]] = None

class AnalyzeRequest(BaseModel):
    props: List[PropInput]
    platform: str = "PrizePicks"
    persona: Optional[str] = "hybrid"  # for future context
    min_edge: float = 0.05  # 5% default

class ScoredPick(BaseModel):
    player: str
    stat: str
    line: float
    sport: str
    model_prob: float  # Our predicted probability of hitting Over
    market_prob: float  # Implied from line odds (assume -110 default if not provided)
    edge: float
    verdict: str  # PLAY, LEAN, PASS
    tier: str  # A/B/C or High/Med/Low
    player_stats: Dict[str, Any]  # Enriched stats from adapter
    key_factors: List[str]
    analysis_context: str  # Text blob for the AI narrative generator
    confidence: float  # 0-1

class AnalyzeResponse(BaseModel):
    picks: List[ScoredPick]
    summary: Dict[str, Any]
    cached_hits: int
    api_calls_made: int
    processing_time_ms: int

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "supported_sports": list(settings.SPORT_ADAPTERS.keys()),
        "version": "0.1.0"
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    start = datetime.utcnow()
    logger.info(f"Received analyze request for {len(request.props)} props on {request.platform}")

    enriched_props = []
    api_calls = 0
    cached = 0
    enriched_count = 0

    for prop in request.props:
        adapter = get_adapter(prop.sport.lower())
        player_stats = {"note": "No adapter - using generic recent form"}

        if adapter:
            try:
                player_stats, calls = await adapter.get_player_stats(
                    player_name=prop.player,
                    stat_type=prop.stat,
                    opponent=prop.opponent,
                    game_date=prop.game_time
                )
                api_calls += calls
                if calls == 0:
                    cached += 1
                # Count as enriched if we got real-looking data (not just stub note)
                if "source" in player_stats or "season_avg" in player_stats:
                    enriched_count += 1
            except Exception as e:
                logger.error(f"Adapter error for {prop.player} ({prop.sport} {prop.stat}): {e}")
                player_stats = {"error": str(e), "note": "Stats fetch failed - using conservative defaults"}

        enriched_props.append({
            **prop.model_dump(),
            "player_stats": player_stats
        })

    # Score all props
    scored = score_props(
        enriched_props,
        platform=request.platform,
        min_edge=request.min_edge,
        persona=request.persona or "hybrid"
    )

    processing_time = int((datetime.utcnow() - start).total_seconds() * 1000)

    # Background: log usage for future quota monitoring
    background_tasks.add_task(log_usage, len(request.props), api_calls, cached)

    return AnalyzeResponse(
        picks=scored,
        summary={
            "total_props": len(request.props),
            "enriched": enriched_count,
            "playable": len([p for p in scored if p.verdict == "PLAY"]),
            "avg_edge": round(sum(p.edge for p in scored) / len(scored), 4) if scored else 0,
            "sports_covered": list(set(p.sport for p in request.props))
        },
        cached_hits=cached,
        api_calls_made=api_calls,
        processing_time_ms=processing_time
    )

def log_usage(props_count: int, api_calls: int, cached: int):
    logger.info(f"Usage: {props_count} props | {api_calls} API calls | {cached} from cache")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)