from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import traceback
import logging

from adapters import get_adapter
from scoring.scorer import score_props
from cache.cache_manager import get, put

app = FastAPI(title="EdgeLab Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PropInput(BaseModel):
    player_name: str
    stat_display: str
    line: float
    higher_american_odds: Optional[float] = None
    lower_american_odds: Optional[float] = None
    player_team: Optional[str] = ""
    home_team: Optional[str] = ""
    away_team: Optional[str] = ""
    source: Optional[str] = "underdog"
    category: Optional[str] = ""


class PredictRequest(BaseModel):
    sport: str
    platform: str = "underdog"
    props: List[PropInput]
    min_edge: float = 0.05


@app.get("/health")
async def health():
    return {"status": "ok", "service": "prediction-api"}


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        adapter = get_adapter(request.sport)
        if not adapter:
            raise HTTPException(status_code=400, detail=f"Unsupported sport: {request.sport}")

        enriched_props = []
        enrichment_errors = []

        for prop in request.props:
            try:
                enriched = adapter.enrich_prop(prop.dict())
                if enriched:
                    enriched_props.append(enriched)
            except Exception as e:
                enrichment_errors.append({
                    "player": prop.player_name,
                    "stat": prop.stat_display,
                    "error": str(e)
                })

        if not enriched_props:
            return {
                "picks": [],
                "passes": [],
                "stats_context": "",
                "errors": enrichment_errors,
                "summary": {
                    "actionable_picks": 0,
                    "passes": 0,
                    "enriched": 0,
                    "total": len(request.props)
                }
            }

        # Score using the batch function
        scored_result = score_props(enriched_props, min_edge=request.min_edge)

        return {
            "picks": scored_result.get("picks", []),
            "passes": scored_result.get("passes", []),
            "stats_context": scored_result.get("stats_context", ""),
            "errors": enrichment_errors,
            "summary": {
                "actionable_picks": len(scored_result.get("picks", [])),
                "passes": len(scored_result.get("passes", [])),
                "enriched": len(enriched_props),
                "total": len(request.props)
            }
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction service error",
                "message": str(e),
                "type": type(e).__name__
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)